"""Per-game master table for the HatTrick game model and team ratings.

One row per regular-season game 2005-06 .. 2025-26 (plus a helper that builds
the same context columns for any future schedule, e.g. 2026-27).

Columns
  gid           HatTrick row key (int, chronological)
  game_id       NHL game id where known (see ``nhl_ids``), else <NA>
  date, season_end, home, away
  home_g, away_g, extra ('REG'|'OT'|'SO'), home_win
  reg_h, reg_a  regulation goals (for OT/SO games both = the loser's final)
  rest_h, rest_a    days since the team's previous regular-season game (cap 9)
  km_h, km_a, km3d_h, km3d_a, dtz_h, dtz_a   travel (great-circle km from the
                previous venue, km over 3 days, time-zone change in hours)
  sh_h, sh_a    shots on goal (NaN where unknown)
  gk_h, gk_a    starting goalie NHL player id (NaN where unknown)

Sources and coverage
  results/regulation goals: data/processed/games.csv (snapshot)  2006-2026
  rest/travel:              data/processed/travel_games.csv       2006-2026
  NHL game ids:             fastRhockey schedules 2011-2024(partial); for the
                            rest of 2023-24 .. 2025-26 the ids of NeurHL's
                            prediction files are matched to games.csv by
                            date, result, regulation score and (for ties)
                            Elo agreement (``nhl_ids``; match report printed
                            by ``python3 -m hattrick.gametable``)
  shots:                    fastRhockey team boxes 2011-2024(partial); the
                            realised shots columns (y_sogf_*) of NeurHL's
                            g_gate/g_seal files fill 2018-19..2025-26.  Those
                            columns are observed box-score facts, not model
                            output; they are only ever used AFTER the game.
  starting goalies:         fastRhockey player boxes, 2011-2023 complete,
                            2023-24 only through 2023-11-07.
"""
from __future__ import annotations

import functools

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from hattrick import config as C
from hattrick import data as D
from hattrick.snapshot import path as snap

PREDS = C.ROOT / "neurhl" / "output" / "preds"
TABLE_CACHE = C.CACHE / "gametable.parquet"


# ---------------------------------------------------------------------------
# Rest / travel for an arbitrary schedule (same algorithm as travel_games.csv)
# ---------------------------------------------------------------------------
def _haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    x = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(x))


def schedule_context(sched: pd.DataFrame, season_end: int) -> pd.DataFrame:
    """Rest days and travel for every game of a schedule.

    ``sched`` needs date, home, away (one season, regular season only).
    Returns the schedule with rest_h, rest_a, km_h, km_a, km3d_h, km3d_a,
    dtz_h, dtz_a, reproducing data/processed/travel_games.csv conventions
    (first game of a season, or after a >30-day gap: rest 9, no travel).
    """
    ar = pd.read_csv(snap("data/raw/arenas.csv"))
    ar = ar[(ar.start_end <= season_end) & (ar.end_end >= season_end)]
    ar = ar.drop_duplicates("team").set_index("team")
    s = sched.sort_values("date", kind="stable").reset_index(drop=True)
    state: dict = {}
    out = {k: np.zeros(len(s)) for k in
           ("rest_h", "rest_a", "km_h", "km_a", "km3d_h", "km3d_a", "dtz_h", "dtz_a")}
    for i, (d, h, a) in enumerate(zip(s.date, s.home, s.away)):
        vlat, vlon, vutc = ar.at[h, "lat"], ar.at[h, "lon"], ar.at[h, "utc_std"]
        for side, t in (("h", h), ("a", a)):
            last = state.get(t)
            if last is None or (d - last[0]).days > 30:
                km, dtz, rest, km3 = 0.0, 0, 9, 0.0
                legs = []
            else:
                km = float(_haversine(last[1], last[2], vlat, vlon))
                dtz = vutc - last[3]
                rest = min((d - last[0]).days, 9)
                legs = [(dd, k) for dd, k in last[4] if (d - dd).days <= 3]
                km3 = sum(k for _, k in legs) + km
            out[f"rest_{side}"][i] = rest
            out[f"km_{side}"][i] = km
            out[f"km3d_{side}"][i] = km3
            out[f"dtz_{side}"][i] = dtz
            state[t] = (d, vlat, vlon, vutc, legs + [(d, km)])
    for k, v in out.items():
        s[k] = v
    return s


# ---------------------------------------------------------------------------
# NHL game ids
# ---------------------------------------------------------------------------
def _neurhl_elo_d(g: pd.DataFrame) -> pd.Series:
    """Home-minus-away pregame Elo (NeurHL's src/engine.run_elo recipe:
    K=8, home 30, carryover 0.7, MOV, playoffs included). Used ONLY to break
    ties when matching NeurHL game ids to games; never a model input."""
    r: dict = {}
    cur = None
    d_out = np.zeros(len(g))
    for i, gm in enumerate(g.itertuples(index=False)):
        if cur is not None and gm.season_end != cur:
            for t in r:
                r[t] = 1505 + 0.7 * (r[t] - 1505)
        cur = gm.season_end
        for t in (gm.home, gm.away):
            r.setdefault(t, 1505.0 if gm.season_end <= 2006 else 1470.0)
        rh, ra = r[gm.home], r[gm.away]
        neutral = gm.season_end == 2020 and gm.game_type == "P"
        dd = rh + (0 if neutral else 30) - ra
        d_out[i] = rh - ra
        e = 1 / (1 + 10 ** (-dd / 400))
        hw = gm.home_g > gm.away_g
        gd = abs(gm.home_g - gm.away_g)
        wd = dd if hw else -dd
        mov = np.log(gd + 1) * (2.2 / (2.2 + 0.001 * wd))
        delta = 8.0 * mov * (float(hw) - e)
        r[gm.home], r[gm.away] = rh + delta, ra - delta
    return pd.Series(d_out, index=g.index)


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


@functools.lru_cache(maxsize=None)
def nhl_ids() -> pd.DataFrame:
    """game_id -> (date, home, away) for every regular-season game we can map.

    1. fastRhockey schedules (2011 .. 2024 partial): direct.
    2. Remaining NeurHL prediction ids (hier_restatement carries the date;
       g_gate / g_seal carry the regulation score and 4-way outcome): matched
       within each date to the unmapped games by an assignment problem whose
       cost penalises any disagreement in home win, REG vs OT/SO, and
       regulation score, plus |logit p_elo - fitted logit(Elo d)| as a
       tie-breaker.
    Returns game_id, date, home, away, season_end, src ('fr'|'match'),
    ambiguous (True when a zero-mismatch alternative assignment existed and
    Elo agreement had to decide).
    """
    g = D.games()
    reg = g[g.game_type == "R"].copy()
    fr = D.fr_schedule()[["game_id", "date", "home", "away", "season_end"]].copy()
    fr["src"] = "fr"
    fr["ambiguous"] = False
    # Validate fastRhockey against games.csv.
    chk = fr.merge(reg[["date", "home", "away"]], on=["date", "home", "away"],
                   how="left", indicator=True)
    fr = fr[chk["_merge"].to_numpy() == "both"]

    h = pd.read_csv(PREDS / "hier_restatement_games.csv", parse_dates=["date"])
    gg = pd.read_csv(PREDS / "g_gate_games.csv")
    gs = pd.read_csv(PREDS / "g_seal_games.csv")
    info = pd.concat([
        gg[["game_id", "outcome4", "gh_reg", "ga_reg"]],
        gs[["game_id", "outcome4"]].assign(gh_reg=gs.y_goals_h, ga_reg=gs.y_goals_a),
    ]).drop_duplicates("game_id")
    todo = h[~h.game_id.isin(fr.game_id)].merge(info, on="game_id", how="left")

    # Elo d for every regular-season game (all games incl. playoffs processed).
    elo_d = _neurhl_elo_d(g.sort_values(["date", "home"]).reset_index(drop=True))
    gs_sorted = g.sort_values(["date", "home"]).reset_index(drop=True)
    gs_sorted["elo_d"] = elo_d.to_numpy()
    cand = gs_sorted[gs_sorted.game_type == "R"].merge(
        reg[["date", "home", "away"]], on=["date", "home", "away"])
    cand = cand[~cand.set_index(["date", "home", "away"]).index.isin(
        fr.set_index(["date", "home", "away"]).index)]
    cand["reg_h"] = np.where(cand.extra == "REG", cand.home_g,
                             np.minimum(cand.home_g, cand.away_g))
    cand["reg_a"] = np.where(cand.extra == "REG", cand.away_g,
                             np.minimum(cand.home_g, cand.away_g))
    # calibrate logit(p_elo) ~ a + b*d on fr-mapped hier games (for the tie-break)
    hm = h.merge(fr, on="game_id", suffixes=("", "_fr"))
    hm = hm.merge(gs_sorted[["date", "home", "away", "elo_d"]],
                  left_on=["date_fr", "home", "away"],
                  right_on=["date", "home", "away"], suffixes=("", "_g"))
    b = np.polyfit(hm.elo_d.to_numpy(), _logit(hm.p_elo), 1)

    rows = []
    for d, ids in todo.groupby("date"):
        cg = cand[cand.date == d].reset_index(drop=True)
        if not len(cg):
            continue
        n, m = len(ids), len(cg)
        cost = np.zeros((n, m))
        exact = np.zeros((n, m), bool)
        for i, r in enumerate(ids.itertuples(index=False)):
            mis = (cg.home_win.to_numpy() != r.y).astype(float) * 100
            if not np.isnan(r.outcome4):
                ot = r.outcome4 >= 2
                mis += ((cg.extra.to_numpy() != "REG") != ot) * 50
                if not (np.isnan(r.gh_reg) or np.isnan(r.ga_reg)):
                    mis += (np.abs(cg.reg_h.to_numpy() - r.gh_reg)
                            + np.abs(cg.reg_a.to_numpy() - r.ga_reg)) * 10
            exact[i] = mis == 0
            soft = np.abs(_logit(r.p_elo) - np.polyval(b, cg.elo_d.to_numpy()))
            cost[i] = mis + soft
        ri, ci = linear_sum_assignment(cost)
        for i, j in zip(ri, ci):
            r = ids.iloc[i]
            amb = exact[i].sum() > 1
            ok = cost[i, j] < 10       # no hard mismatch
            rows.append((r.game_id, cg.date[j], cg.home[j], cg.away[j],
                         int(cg.season_end[j]), "match" if ok else "bad", amb))
    mt = pd.DataFrame(rows, columns=["game_id", "date", "home", "away",
                                     "season_end", "src", "ambiguous"])
    mt = mt[mt.src == "match"]
    out = pd.concat([fr, mt], ignore_index=True)
    return out.drop_duplicates("game_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Shots and starting goalies by game_id
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def shots_by_id() -> pd.DataFrame:
    """game_id -> sh_h, sh_a (shots on goal incl. OT)."""
    names = D.team_names()
    sch = D.fr_schedule()[["game_id", "home", "away"]]
    frames = []
    for y in range(2011, 2025):
        f = D.FR / f"team_box_{y}.parquet"
        if not f.exists():
            continue
        tb = pd.read_parquet(f, columns=["team_name", "shots", "game_id"])
        tb = tb[(tb.game_id // 10000) % 100 == 2]
        tb["team"] = tb.team_name.map(names)
        frames.append(tb)
    tb = pd.concat(frames, ignore_index=True)
    m = sch.merge(tb.rename(columns={"team": "home", "shots": "sh_h"})[
        ["game_id", "home", "sh_h"]], on=["game_id", "home"], how="inner")
    m = m.merge(tb.rename(columns={"team": "away", "shots": "sh_a"})[
        ["game_id", "away", "sh_a"]], on=["game_id", "away"], how="inner")
    m = m[["game_id", "sh_h", "sh_a"]]
    gg = pd.read_csv(PREDS / "g_gate_games.csv", usecols=["game_id", "y_sogf_h", "y_sogf_a"])
    gs = pd.read_csv(PREDS / "g_seal_games.csv", usecols=["game_id", "y_sogf_h", "y_sogf_a"])
    ng = pd.concat([gg, gs]).rename(columns={"y_sogf_h": "sh_h", "y_sogf_a": "sh_a"})
    ng = ng[~ng.game_id.isin(m.game_id)]
    return pd.concat([m, ng], ignore_index=True).drop_duplicates("game_id")


@functools.lru_cache(maxsize=None)
def goalies_by_id() -> pd.DataFrame:
    """game_id -> gk_h, gk_a (starting goalie ids), gsv_h/gsv_a/gsa_h/gsa_a
    (the starter's saves and shots faced)."""
    st = D.goalie_games()
    h = st[st.home_away == "Home"].set_index("game_id")
    a = st[st.home_away == "Away"].set_index("game_id")
    out = pd.DataFrame({"gk_h": h.goalie_id, "gk_a": a.goalie_id})
    return out.reset_index()


@functools.lru_cache(maxsize=None)
def goalie_game_log() -> pd.DataFrame:
    """Every goalie appearance (not only starters) with shots/saves, keyed by
    game_id: game_id, team_side ('Home'|'Away'), goalie_id, toi, shots, saves.
    Seasons 2011-2023 (+2024 partial)."""
    frames = []
    for y in range(2011, 2025):
        f = D.FR / f"player_box_{y}.parquet"
        if not f.exists():
            continue
        p = pd.read_parquet(f, columns=["player_id", "position_code", "home_away",
                                        "game_id", "goalie_stats_time_on_ice",
                                        "goalie_stats_shots", "goalie_stats_saves"])
        p = p[(p.position_code == "G") & ((p.game_id // 10000) % 100 == 2)]
        frames.append(p)
    p = pd.concat(frames, ignore_index=True)
    p["toi"] = D._mmss(p.goalie_stats_time_on_ice)
    p = p[p.toi > 0]
    return p.rename(columns={"player_id": "goalie_id", "home_away": "side",
                             "goalie_stats_shots": "shots",
                             "goalie_stats_saves": "saves"})[
        ["game_id", "side", "goalie_id", "toi", "shots", "saves"]]


# ---------------------------------------------------------------------------
# Master table
# ---------------------------------------------------------------------------
def build() -> pd.DataFrame:
    g = D.regular_season().copy()
    g["reg_h"] = np.where(g.extra == "REG", g.home_g, np.minimum(g.home_g, g.away_g))
    g["reg_a"] = np.where(g.extra == "REG", g.away_g, np.minimum(g.home_g, g.away_g))
    t = D.travel().rename(columns={
        "home_rest": "rest_h", "away_rest": "rest_a", "home_km": "km_h",
        "away_km": "km_a", "home_km3d": "km3d_h", "away_km3d": "km3d_a",
        "home_dtz": "dtz_h", "away_dtz": "dtz_a"}).drop(columns="season_end")
    g = g.merge(t, on=["date", "home", "away"], how="left")
    ids = nhl_ids()[["game_id", "date", "home", "away"]]
    ids = ids.drop_duplicates(["date", "home", "away"])
    g = g.merge(ids, on=["date", "home", "away"], how="left")
    assert len(g) == len(D.regular_season()), "id merge duplicated games"
    g = g.merge(shots_by_id(), on="game_id", how="left")
    g = g.merge(goalies_by_id(), on="game_id", how="left")
    g = g.sort_values(["date", "home"], kind="stable").reset_index(drop=True)
    g["gid"] = np.arange(len(g))
    g["game_id"] = g.game_id.astype("Int64")
    cols = ["gid", "game_id", "date", "season_end", "home", "away", "home_g",
            "away_g", "extra", "home_win", "reg_h", "reg_a", "rest_h", "rest_a",
            "km_h", "km_a", "km3d_h", "km3d_a", "dtz_h", "dtz_a", "sh_h", "sh_a",
            "gk_h", "gk_a"]
    return g[cols]


@functools.lru_cache(maxsize=None)
def table(refresh: bool = False) -> pd.DataFrame:
    """Cached master table (hattrick/cache/gametable.parquet)."""
    if TABLE_CACHE.exists() and not refresh:
        return pd.read_parquet(TABLE_CACHE)
    g = build()
    TABLE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    g.to_parquet(TABLE_CACHE, index=False)
    return g


def match_report() -> dict:
    """How NeurHL's game ids were mapped to games, and whether the mapped
    results agree with NeurHL's own columns: y (home win) and, where the file
    has it, the date (hier_restatement) and the regulation score
    (g_gate gh_reg/ga_reg, g_seal y_goals_h/y_goals_a)."""
    ids = nhl_ids().rename(columns={"date": "date_map"}).drop(columns="season_end")
    g = table()[["date", "home", "away", "home_win", "reg_h", "reg_a", "extra"]]
    rep = {}
    for name, f, scol, gcols in [
            ("hier_restatement", "hier_restatement_games.csv", "season", None),
            ("g_gate", "g_gate_games.csv", "season_end", ("gh_reg", "ga_reg")),
            ("g_seal", "g_seal_games.csv", "season_end", ("y_goals_h", "y_goals_a"))]:
        p = pd.read_csv(PREDS / f)
        m = p.merge(ids, on="game_id", how="left")
        m = m.merge(g, left_on=["date_map", "home", "away"],
                    right_on=["date", "home", "away"], how="left",
                    suffixes=("", "_g"))
        mapped = m.home_win.notna()
        r = {"n": len(p), "mapped": int(mapped.sum()),
             "y_equals_home_win": float((m.home_win[mapped] == m.y[mapped]).mean())}
        if "date" in p:
            r["date_agrees"] = float((pd.to_datetime(m.date[mapped])
                                      == m.date_map[mapped]).mean())
        if gcols:
            ok = mapped & (m.extra == "REG")
            r["reg_score_agrees_REG_games"] = float(
                ((m[gcols[0]] == m.reg_h) & (m[gcols[1]] == m.reg_a))[ok].mean())
        by = m.groupby(scol).apply(lambda d: pd.Series({
            "n": len(d), "mapped": int(d.home_win.notna().sum()),
            "via_fr": int((d.src == "fr").sum()), "via_match": int((d.src == "match").sum()),
            "ambiguous_resolved_by_elo": int(d.ambiguous.fillna(False).astype(bool).sum())}),
            include_groups=False)
        r["by_season"] = by.reset_index().to_dict("records")
        rep[name] = r
    return rep


if __name__ == "__main__":
    t = table(refresh=True)
    print(f"{len(t)} games; game_id known {t.game_id.notna().mean():.3f}; "
          f"shots known {t.sh_h.notna().mean():.3f}; goalies known {t.gk_h.notna().mean():.3f}")
    print(t.groupby("season_end")[["game_id", "sh_h", "gk_h"]].count())
    import json
    print(json.dumps(match_report(), indent=1, default=str))
