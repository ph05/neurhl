"""Lineup-aware game forecasts (ORR 1.1; pre-registered item X1 in PLAN_1_1.md).

Each team's goal rates are adjusted for who actually dresses: the summed
on-ice value of the dressed skaters minus the team's EXPECTED lineup value,
plus the starting goalies through the existing goalie layer. The adjustment
is a log offset on regulation goals that enters both the filter's pregame
prediction and its measurement update (``ratings.run_filter(lineup=...)``,
``ratings.InSeasonFilter.update_day`` columns ``lo_h`` / ``lo_a``); the shots
rows are unchanged.

Player value (preseason projection ``players.project(V)``, seasons < V only):
    v_xgf = tpg_ev * rel_xgf60 / 60      5v5 on-ice xGF impact per game
    v_xga = tpg_ev * rel_xga60 / 60      5v5 on-ice xGA impact (+ = bad)
    v_ppf, v_pka                         special-teams value (weight 0 in X1)
    toi   = projected minutes per game, all situations
Expected lineup: the team's exponentially weighted mean (half-life
``halflife`` team games) of its EARLIER lineups this season; a team's first
known lineup carries no signal (delta 0). Offsets, with D = dressed - expected:
    lo_h = beta_x * (D_xgf_h + D_xga_a) / xg_pg + beta_st * (D_ppf_h + D_pka_a) / xg_pg
         + beta_t * (D_toi_h - D_toi_a) / toi_scale          (lo_a symmetric)
xg_pg is the league xGF per team-game of ``players.project_league(V)``'s
reference season.

The configuration ``X1`` was fixed on 2012 and 2014-17 and tested once on the
gate games 2019-24 (orr/experiments/X1/; accepted). Backtests take dressed
skaters from the box scores (``backtest_offsets``); live, ``live_lineups``
reads NeurHL's committed pregame lineup files (DailyFaceoff) and
``live_offsets`` turns them into offsets.
"""
from __future__ import annotations

import functools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from orr import config as C

COMPS = ("v_xgf", "v_xga", "v_ppf", "v_pka", "toi")
BOX_SEASONS = range(2011, 2025)
NEURHL_LIVE = C.ROOT / "neurhl" / "output" / "live" / str(C.TARGET_SEASON)
# file kinds in NeurHL's live folder, latest information first
FILE_PRIORITY = (("pregame_*_lineups.json", 0), ("morning_lineups.json", 1), ("preview_lineups.json", 2))


@dataclass(frozen=True)
class LineupHP:
    beta_x: float = 0.45          # 5v5 on-ice xG value
    beta_st: float = 0.0          # special-teams value (dropped in tuning)
    beta_t: float = 0.6           # projected ice time
    toi_scale: float = 300.0      # minutes
    halflife: float = 10.0        # expected-lineup memory, team games
    use_goalie: bool = True       # known starters through the goalie layer


X1 = LineupHP()                   # the configuration fixed before X1's test


# ---------------------------------------------------------------------------
# Player values
# ---------------------------------------------------------------------------
def values_from_projection(pj: pd.DataFrame) -> pd.DataFrame:
    """player_id and the value components from a players.project frame."""
    from orr import players as PL
    out = pd.DataFrame({"player_id": pj.player_id.to_numpy()})
    out["v_xgf"] = (pj.tpg_ev * pj.rel_xgf60 / 60.0).to_numpy()
    out["v_xga"] = (pj.tpg_ev * pj.rel_xga60 / 60.0).to_numpy()
    ell_pp = pj.pp_onice_60 / (1.0 + pj.pp_onice_rel) / 60.0       # league PP xGF per minute
    ell_sh = pj.sh_onice_60 / (1.0 + pj.sh_onice_rel) / 60.0
    out["v_ppf"] = (pj.tpg_pp * pj.pp_onice_rel * ell_pp).to_numpy()
    out["v_pka"] = (pj.tpg_sh * pj.sh_onice_rel * ell_sh).to_numpy()
    out["toi"] = sum(pj[f"tpg_{k}"] for k in PL.SITS).to_numpy()
    out["has_hist"] = pj.has_hist.to_numpy()
    return out


def player_values_for(V: int, ids) -> pd.DataFrame:
    """Preseason values (seasons < V only) of the given skaters; players the
    projection does not cover are average (0)."""
    from orr import players as PL
    ids = sorted(set(int(i) for i in ids))
    pj = PL.project(V, PL.load_params(), ids=ids)
    out = values_from_projection(pj)
    out["season_end"] = V
    miss = sorted(set(ids) - set(out.player_id))
    if miss:
        z = pd.DataFrame({"player_id": miss, "season_end": V, "has_hist": False})
        for c in COMPS:
            z[c] = 0.0
        out = pd.concat([out, z], ignore_index=True)
    return out.fillna({c: 0.0 for c in COMPS})


@functools.lru_cache(maxsize=None)
def league_xg_pg(V: int) -> float:
    """League xGF per team-game (all situations) in the reference season of V
    (the last intact season before V, as players.project_league)."""
    from orr import data as D
    from orr import players as PL
    ref = int(PL.project_league(V).name)
    t = D.team_seasons()
    t = t[t.season_end == ref]
    return float(t.xgf_all.sum() / t.gp.sum())


# ---------------------------------------------------------------------------
# Expected lineup and offsets (shared by backtest and live)
# ---------------------------------------------------------------------------
def expected_lineup(s: pd.DataFrame, halflife: float, group=("team", "season_end"),
                    order=("team", "season_end", "date", "gid")) -> pd.DataFrame:
    """Adds E_<comp> (EWMA of the team's earlier lineups, half-life in team
    games; a team's first lineup is its own expectation) and D_<comp> = L - E
    to a frame of per-team-game dressed sums L_<comp>."""
    s = s.sort_values(list(order)).reset_index(drop=True)
    lam = 0.5 ** (1.0 / halflife) if np.isfinite(halflife) else 1.0
    L = s[[f"L_{c}" for c in COMPS]].to_numpy()
    E = np.zeros_like(L)
    k = np.zeros(len(s), int)
    for _, idx in s.groupby(list(group)).indices.items():
        num = np.zeros(len(COMPS))
        den = 0.0
        for j, i in enumerate(idx):
            E[i] = L[i] if den == 0 else num / den
            k[i] = j
            num = lam * num + L[i]
            den = lam * den + 1.0
    for j, c in enumerate(COMPS):
        s[f"E_{c}"] = E[:, j]
        s[f"D_{c}"] = L[:, j] - E[:, j]
    s["k_team_game"] = k
    return s


def offsets(w: pd.DataFrame, beta_x: float, beta_st: float = 0.0, beta_t: float = 0.0,
            toi_scale: float = 300.0, key: str = "gid") -> pd.DataFrame:
    """Log offsets on regulation goals (lo_h on home goals, lo_a on away goals)
    from a one-row-per-game frame with D_<comp>_h, D_<comp>_a and xg_pg."""
    x = w.xg_pg.to_numpy()
    lo_h = (beta_x * (w.D_v_xgf_h + w.D_v_xga_a) + beta_st * (w.D_v_ppf_h + w.D_v_pka_a)) / x \
        + beta_t * (w.D_toi_h - w.D_toi_a) / toi_scale
    lo_a = (beta_x * (w.D_v_xgf_a + w.D_v_xga_h) + beta_st * (w.D_v_ppf_a + w.D_v_pka_h)) / x \
        + beta_t * (w.D_toi_a - w.D_toi_h) / toi_scale
    return pd.DataFrame({key: w[key].to_numpy(), "lo_h": lo_h.to_numpy(), "lo_a": lo_a.to_numpy()})


# ---------------------------------------------------------------------------
# Backtest: dressed skaters from the box scores (2011-2024)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def dressed() -> pd.DataFrame:
    """gid, season_end, date, side ('h'|'a'), team, player_id, pos for every
    dressed skater of every regular-season game with a box score."""
    from orr import data as D
    from orr import structural as S
    frames = []
    for y in BOX_SEASONS:
        f = D.FR / f"player_box_{y}.parquet"
        if not f.exists():
            continue
        p = pd.read_parquet(f, columns=["player_id", "position_code", "home_away", "game_id"])
        p = p[(p.position_code != "G") & ((p.game_id // 10000) % 100 == 2)]
        frames.append(p)
    p = pd.concat(frames, ignore_index=True)
    g = S.game_frame()
    g = g[g.game_id.notna()][["gid", "game_id", "season_end", "date", "home", "away"]].copy()
    g["game_id"] = g.game_id.astype("int64")
    p = p.merge(g, on="game_id", how="inner")
    p["side"] = np.where(p.home_away.str.lower() == "home", "h", "a")
    p["team"] = np.where(p.side == "h", p.home, p.away)
    p["pos"] = np.where(p.position_code == "D", "D", "F")
    return p[["gid", "season_end", "date", "side", "team", "player_id", "pos"]].reset_index(drop=True)


@functools.lru_cache(maxsize=None)
def player_values(V: int) -> pd.DataFrame:
    """Preseason values (seasons < V only) of every skater who dressed in V."""
    return player_values_for(V, dressed().query("season_end == @V").player_id)


@functools.lru_cache(maxsize=None)
def _lineup_sums() -> pd.DataFrame:
    d = dressed()
    vals = pd.concat([player_values(V) for V in sorted(set(d.season_end))], ignore_index=True)
    d = d.merge(vals, on=["player_id", "season_end"], how="left")
    for c in COMPS:
        d[c] = d[c].fillna(0.0)
    return d.groupby(["gid", "season_end", "date", "side", "team"], as_index=False).agg(
        n_sk=("player_id", "size"), **{f"L_{c}": (c, "sum") for c in COMPS})


@functools.lru_cache(maxsize=None)
def lineup_table(halflife: float = X1.halflife) -> pd.DataFrame:
    """Per box-score game and side: dressed sums L, expected E and D = L - E."""
    s = expected_lineup(_lineup_sums(), halflife)
    s["xg_pg"] = s.season_end.map({V: league_xg_pg(V) for V in sorted(set(s.season_end))})
    return s


def _wide(t: pd.DataFrame, key: str) -> pd.DataFrame:
    cols = [f"D_{c}" for c in COMPS]
    h = t[t.side == "h"].set_index(key)[cols + ["xg_pg"]]
    a = t[t.side == "a"].set_index(key)[cols]
    return h.join(a, lsuffix="_h", rsuffix="_a", how="inner").reset_index()


def backtest_offsets(hp: LineupHP = X1) -> pd.DataFrame:
    """gid, lo_h, lo_a for every regular-season game 2011-2024 with a box
    score (games without one get no offset in run_filter)."""
    w = _wide(lineup_table(hp.halflife), "gid")
    return offsets(w, hp.beta_x, hp.beta_st, hp.beta_t, hp.toi_scale)


# ---------------------------------------------------------------------------
# Live: NeurHL's committed pregame lineup files
# ---------------------------------------------------------------------------
def live_lineups(through, root: Path | str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """Lineups of every game dated on or before ``through`` found in NeurHL's
    live folder (<root>/<YYYY-MM-DD>/...): per game the latest pregame file,
    else the morning file, else the preview file.

    Returns (skaters, goalies, files):
      skaters: game_id, date, side ('h'|'a'), team, player_id, pos, lineup_source
               (lineups whose source is a FALLBACK are dropped)
      goalies: game_id, goalie_home, goalie_away (NaN where the source is a
               FALLBACK or missing), goalie_source_home/_away
      files:   the files read (paths), for provenance."""
    root = Path(root) if root is not None else NEURHL_LIVE
    through = pd.Timestamp(through)
    best: dict = {}
    files = []
    if root.exists():
        for ddir in sorted(p for p in root.iterdir() if p.is_dir()):
            try:
                if pd.Timestamp(ddir.name) > through:
                    continue
            except ValueError:
                continue
            for pat, pri in FILE_PRIORITY:
                for f in sorted(ddir.glob(pat)):
                    js = json.loads(f.read_text())
                    files.append(str(f))
                    for gid, g in js.items():
                        # latest as_of first within a file kind; a missing as_of ranks last
                        rank = (pri, float("inf") if g.get("as_of") is None
                                else -float(pd.Timestamp(g["as_of"]).value))
                        if gid not in best or rank < best[gid][0]:
                            best[gid] = (rank, g)
    sk, gk = [], []
    for gid, (_, g) in best.items():
        if pd.Timestamp(g["date"]) > through:
            continue
        row = {"game_id": int(g["game_id"])}
        for side, key in (("h", "home"), ("a", "away")):
            s = g.get(key) or {}
            src = str(s.get("goalie_source") or "")
            ok = s.get("goalie") is not None and not src.upper().startswith("FALLBACK")
            row[f"goalie_{key}"] = float(s["goalie"]) if ok else np.nan
            row[f"goalie_source_{key}"] = src
            lsrc = str(s.get("lineup_source") or "")
            if s.get("skaters") and not lsrc.upper().startswith("FALLBACK"):
                dmen = set(int(x) for x in (s.get("defense") or []))
                for pid in s["skaters"]:
                    sk.append((int(g["game_id"]), g["date"], side, s.get("team"), int(pid),
                               "D" if int(pid) in dmen else "F", lsrc))
        gk.append(row)
    skaters = pd.DataFrame(sk, columns=["game_id", "date", "side", "team", "player_id", "pos",
                                        "lineup_source"])
    skaters["date"] = pd.to_datetime(skaters.date)
    goalies = pd.DataFrame(gk, columns=["game_id", "goalie_home", "goalie_away",
                                        "goalie_source_home", "goalie_source_away"])
    return skaters, goalies, files


def live_values(V: int, skaters: pd.DataFrame) -> pd.DataFrame:
    """Values of the skaters in live lineups: players.project(V) where the
    MoneyPuck panel covers them; otherwise the frozen projection
    (freeze_<V>/skaters_<V>.csv, which projects true rookies too); otherwise
    the average no-history skater of the same position in the freeze."""
    from orr import players as PL
    ids = sorted(set(int(i) for i in skaters.player_id))
    out = values_from_projection(PL.project(V, PL.load_params(), ids=ids))
    out["source"] = "project"
    miss = sorted(set(ids) - set(out.player_id))
    if miss:
        fz = pd.read_csv(C.OUT / f"freeze_{V}" / f"skaters_{V}.csv")
        tpg = sum(fz[f"tpg_{k}"] for k in PL.SITS)
        fv = pd.DataFrame({"player_id": fz.player_id, "pos": fz.pos,
                           "v_xgf": fz.tpg_ev * fz.rel_xgf60 / 60.0, "v_xga": fz.tpg_ev * fz.rel_xga60 / 60.0,
                           "v_ppf": 0.0, "v_pka": 0.0, "toi": tpg, "has_hist": fz.has_hist.astype(bool)})
        hit = fv[fv.player_id.isin(miss)].drop(columns="pos").assign(source="freeze")
        rest = sorted(set(miss) - set(hit.player_id))
        rows = [hit]
        if rest:
            pos = skaters.drop_duplicates("player_id").set_index("player_id").get("pos")
            new = fv[~fv.has_hist].groupby("pos")[list(COMPS)].mean()
            z = pd.DataFrame({"player_id": rest})
            pz = z.player_id.map(pos) if pos is not None else pd.Series("F", index=z.index)
            for c in COMPS:
                z[c] = pz.fillna("F").map(new[c]).fillna(0.0).to_numpy()
            rows.append(z.assign(has_hist=False, source="rookie_default"))
        out = pd.concat([out, *rows], ignore_index=True)
    return out.fillna({c: 0.0 for c in COMPS})


def live_offsets(skaters: pd.DataFrame, hp: LineupHP = X1, V: int = C.TARGET_SEASON,
                 values: pd.DataFrame | None = None) -> pd.DataFrame:
    """game_id, lo_h, lo_a, known_h, known_a from live lineups (one row per
    game with at least one known side). A side without a lineup counts as its
    expected lineup (D = 0); a team's first known lineup carries no signal."""
    cols = ["game_id", "lo_h", "lo_a", "known_h", "known_a"]
    if skaters is None or not len(skaters):
        return pd.DataFrame(columns=cols)
    if values is None:
        values = live_values(V, skaters)
    d = skaters.merge(values[["player_id", *COMPS]], on="player_id", how="left")
    for c in COMPS:
        d[c] = d[c].fillna(0.0)
    s = d.groupby(["game_id", "date", "side", "team"], as_index=False).agg(
        n_sk=("player_id", "size"), **{f"L_{c}": (c, "sum") for c in COMPS})
    s["season_end"] = V
    s = expected_lineup(s, hp.halflife, order=("team", "season_end", "date", "game_id"))
    s["xg_pg"] = league_xg_pg(V)
    out = []
    for gid, x in s.groupby("game_id"):
        r = {"game_id": gid, "xg_pg": float(x.xg_pg.iloc[0])}
        for side in ("h", "a"):
            y = x[x.side == side]
            r[f"known_{side}"] = bool(len(y))
            for c in COMPS:
                r[f"D_{c}_{side}"] = float(y[f"D_{c}"].iloc[0]) if len(y) else 0.0
        out.append(r)
    w = pd.DataFrame(out)
    o = offsets(w, hp.beta_x, hp.beta_st, hp.beta_t, hp.toi_scale, key="game_id")
    o["known_h"], o["known_a"] = w.known_h.to_numpy(), w.known_a.to_numpy()
    return o[cols]


def settings(hp: LineupHP = X1) -> dict:
    return asdict(hp)
