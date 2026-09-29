"""NeurHL side rates: per-player per-60 rates for the box-score stats that the
unified season simulator does not model itself.

The simulator draws each skater's ice time per game and already models shots,
attempts, individual xG, goals and assists. This table supplies the rest as
rates per 60 minutes of total (all-strength) TOI, so a team-game total is the
sum over its skaters of rate x TOI / 60:

  hits60, blocks60, giveaways60, takeaways60    credited player
  fo_taken60, fo_win_share                      faceoffs won + lost; share won
  pim60                                         penalty minutes charged
  pen_taken60, pen_drawn60                      MINOR penalties (a double minor
                                                counts 2), taken / drawn

Source: NHL play-by-play JSON (data/raw/pbp/<s>/*.json.gz), regular season,
periods 1-4, 2012-2026, counted per player-game and joined to shift TOI in
neurhl/data/tensors/player_games_<s>.parquet (so rates are per 60 of actual
TOI). Actors: hittingPlayerId, blockingPlayerId, giveaway/takeaway playerId,
winning/losingPlayerId, committedBy/drawnByPlayerId. Blocks exclude shots
blocked by the shooter's own teammate ('teammate-blocked'/'other-block', logged
from 2023-24; none before). PIM is the penalty duration charged to the
committer; bench minors have no committer and appear only in the league block.
Season totals reproduce the NHL skater realtime / faceoff / summary reports.
The HTM era (2008-2011) is not used: its shards carry no penalty duration, its
faceoff winner matches the NHL report for only ~60% of players (hits, blocks,
giveaways, takeaways do match), and every vantage here needs only 2012+.

Recording eras. Hits, blocks, giveaways and takeaways are scorer judgements.
Through 2023-24 each rink's scorer left a large, persistent footprint (venue
factor SD ~0.2 hits, ~0.13 blocks, ~0.3-0.4 giveaways/takeaways); from 2024-25
the venue spread collapses to ~0.05 and the league levels shift (giveaways x2,
takeaways -1/3, defensemen's share of hits down and of takeaways up). So:
  - seasons <= RINK_ERA_END are rink-neutralised: venue factor = events per
    game at a rink (both teams) over the home team's road games, and each
    player-season is divided by the TOI-weighted factor of his venues;
  - every season is rescaled to the V-1 league level per position group (F/D);
  - validation targets (2023, 2024) are still rink-biased, so there the neutral
    rate is multiplied by the player's expected venue factor (half his team's
    home factor, forecast from V-1 with the slope of log factor on its lag,
    half the road average). The 2027 table is left neutral (central era).

Estimator, per player at vantage V (seasons < V only):
  1. rink-neutral, level-rescaled season counts (above);
  2. exponentially weighted per-60 rate over V-1, V-2, V-3 (weights 1, d, d^2,
     d per stat);
  3. empirical-Bayes shrinkage toward the position mean, prior strength k
     minutes: r = (n*raw + k*mu) / (n + k), n = weighted minutes. Position is
     F/D; fo_taken60 shrinks toward the center / winger / defense mean;
     fo_win_share = (W + k_fo/2) / (W + L + k_fo), W, L season-weighted;
  4. players with no regular-season minutes in V-3..V-1 get the newcomer mean
     of their group (skaters in their first season after a 3-season gap),
     including the newcomer faceoff win share (~0.45 for centers).
d and k per stat (and d, k_fo) are chosen by Poisson / binomial deviance on
V = 2015..2022 only (k_fo on listed centers, for whom the 0.5 prior holds);
2023 and 2024 are then out-of-sample validation years (2025, 2026 reserved).
toi_minutes_basis weights minutes 1, 0.6, 0.36; min_<s> are raw minutes.

Outputs:
  neurhl/output/neurhl_1_0/player_rates_<V>.csv, one row per skater on any
      roster snapshot (data/raw/rosters/*/rosters.csv) or with a V-3..V-1 NHL
      game: player_id, name, team, pos (C/L/R/D), pos_group (0 F, 1 D),
      toi_minutes_basis, the rates above, fo_basis (weighted faceoffs behind
      the share), no_history, on_roster (latest snapshot), last_season (0 =
      no NHL regular season since 2012), min_<s> (raw minutes per season)
  neurhl/output/neurhl_1_0/player_rates_validation.json   (--validate):
      settings and tuning, validation on 2023 and 2024 against both
      baselines, league per-60 by group, venue-factor SDs, league
      per-team-game averages 2024-2026 split skaters / goalies / bench, the
      2027 priors, and the table's implied team-game totals at 2026 minutes

Usage: ... python neurhl/data/build_player_rates.py [--vantage 2027] [--validate]
       [--cache DIR]   (optional scratch dir for the per-season extraction)
"""
import argparse
import gzip
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import NOUT, RAW, TENSORS  # noqa: E402

OUT = NOUT / "neurhl_1_0"
FIRST, LAST = 2012, 2026             # API play-by-play seasons on disk
EXCLUDE_GAMES = {2012020660}         # degenerate game (EDA-01)
TUNE_V = list(range(2015, 2023))     # k chosen here; targets <= 2022
VALID_V = (2023, 2024)
DECAY = 0.6                          # reference weights 1, .6, .36 (basis)
DECAYS = (0.3, 0.45, 0.6, 0.75, 0.9)  # per-stat decay grid, tuned with k
LOOKBACK = 3
RINK = ["hits", "blocks", "give", "take"]   # scorer-judged counts
RINK_ERA_END = 2024                  # last season with rink-scorer footprints
K_GRID = np.unique(np.round(np.geomspace(25, 20000, 49)))
KFO_GRID = np.unique(np.round(np.geomspace(10, 5000, 45)))
MIN_EVAL = 500.0                     # minutes in V for the validation set
MIN_EVAL_FO = 200                    # faceoffs in V for the win-share set

# per player-game counts taken from the play-by-play
COUNTS = ["hits", "blocks", "give", "take", "fo_w", "fo_l", "pim", "minors",
          "minors_drawn", "pens", "pens_drawn", "majors", "blocks_tm"]
# output rate -> season-table count (fo = fo_w + fo_l)
RATES = {"hits60": "hits", "blocks60": "blocks", "giveaways60": "give",
         "takeaways60": "take", "fo_taken60": "fo", "pim60": "pim",
         "pen_taken60": "minors", "pen_drawn60": "minors_drawn"}
SHARE = "fo_win_share"
# per-game league totals, including events with no credited player
LG = ["n_hits", "n_blocks", "n_blocks_tm", "n_give", "n_take", "n_fo",
      "n_pens", "pim_all", "minors_all", "bench_minors", "bench_pim"]


def num_col(c: str) -> str:
    """Season-table column that feeds the estimator (rink-neutral if judged)."""
    return c + "_n" if c in RINK else c


# ------------------------------------------------------------------ extract
def game_counts(path):
    """One regular-season game -> (player rows, league row, bio rows)."""
    with gzip.open(path, "rt") as f:
        g = json.load(f)
    if g.get("gameType") != 2 or g["id"] in EXCLUDE_GAMES:
        return None
    gid = g["id"]
    abbrev = {t["id"]: t.get("abbrev", "") for t in (g["homeTeam"], g["awayTeam"])}
    team, bio = {}, []
    for r in g.get("rosterSpots", []):
        pid = r.get("playerId")
        if not pid:
            continue
        team[pid] = r.get("teamId")
        name = " ".join(x for x in ((r.get("firstName") or {}).get("default"),
                                    (r.get("lastName") or {}).get("default")) if x)
        bio.append((gid, pid, abbrev.get(r.get("teamId"), ""),
                    r.get("positionCode") or "", name))
    ix = {c: i for i, c in enumerate(COUNTS)}
    cnt: dict = {}
    lg = dict.fromkeys(LG, 0.0)

    def add(pid, col, v=1.0):
        if not pid:
            return
        if pid not in cnt:
            cnt[pid] = np.zeros(len(COUNTS))
        cnt[pid][ix[col]] += v

    for p in g.get("plays", []):
        if (p.get("periodDescriptor") or {}).get("number", 0) >= 5:
            continue                                   # shootout
        k = p.get("typeDescKey")
        d = p.get("details") or {}
        if k == "hit":
            lg["n_hits"] += 1
            add(d.get("hittingPlayerId"), "hits")
        elif k == "blocked-shot":
            b, s = d.get("blockingPlayerId"), d.get("shootingPlayerId")
            own = (d.get("reason") in ("teammate-blocked", "other-block")
                   or (b and s and team.get(b) == team.get(s)))
            if own:
                lg["n_blocks_tm"] += 1
                add(b, "blocks_tm")
            else:
                lg["n_blocks"] += 1
                add(b, "blocks")
        elif k in ("giveaway", "takeaway"):
            lg["n_give" if k == "giveaway" else "n_take"] += 1
            add(d.get("playerId"), "give" if k == "giveaway" else "take")
        elif k == "faceoff":
            lg["n_fo"] += 1
            add(d.get("winningPlayerId"), "fo_w")
            add(d.get("losingPlayerId"), "fo_l")
        elif k == "penalty":
            tc, dur = d.get("typeCode"), float(d.get("duration") or 0)
            minors = dur / 2.0 if tc == "MIN" else 0.0
            lg["n_pens"] += 1
            lg["pim_all"] += dur
            lg["minors_all"] += minors
            cb, db = d.get("committedByPlayerId"), d.get("drawnByPlayerId")
            if tc == "BEN" or not cb:
                lg["bench_minors"] += tc == "BEN"
                lg["bench_pim"] += dur
            add(cb, "pim", dur)
            add(cb, "pens")
            add(cb, "minors", minors)
            add(cb, "majors", float(tc == "MAJ"))
            add(db, "pens_drawn")
            add(db, "minors_drawn", minors)
    rows = [(gid, pid, *v) for pid, v in cnt.items()]
    return rows, (gid, *lg.values()), bio


def extract_season(se: int, pool) -> tuple:
    files = sorted((RAW / "pbp" / str(se)).glob("*.json.gz"))
    rows, lgs, bios = [], [], []
    for res in pool.imap(game_counts, files, chunksize=16):
        if res:
            rows += res[0]
            lgs.append(res[1])
            bios += res[2]
    pc = pd.DataFrame(rows, columns=["game_id", "player_id"] + COUNTS)
    lg = pd.DataFrame(lgs, columns=["game_id"] + LG)
    bio = pd.DataFrame(bios, columns=["game_id", "player_id", "team", "pos",
                                      "name"])
    return pc, lg, bio


def load_season(se: int, pool, cache: Path | None) -> dict:
    """Per player-game counts joined to shift TOI (regular season)."""
    if cache is not None and (cache / f"pr_pg_{se}.parquet").exists():
        return {k: pd.read_parquet(cache / f"pr_{k}_{se}.parquet")
                for k in ("pg", "lg", "bio")}
    pc, lg, bio = extract_season(se, pool)
    pg = pd.read_parquet(TENSORS / f"player_games_{se}.parquet",
                         columns=["game_id", "game_type", "player_id", "is_home",
                                  "pos_group", "toi_sec"])
    pg = pg[pg.game_type == 2].drop(columns="game_type")
    out = pg.merge(pc, on=["game_id", "player_id"], how="left")
    out[COUNTS] = out[COUNTS].fillna(0.0).astype("float32")
    out = out.merge(bio[["game_id", "player_id", "team"]],
                    on=["game_id", "player_id"], how="left")
    venue = out[out.is_home].groupby("game_id").team.first()
    out["venue"] = out.game_id.map(venue)
    # events credited to a player with no player_games row (should be ~0)
    lost = pc.merge(pg[["game_id", "player_id"]], how="left", indicator=True)
    lg["lost_events"] = lg.game_id.map(
        lost[lost._merge == "left_only"].groupby("game_id")[COUNTS[:6]]
        .sum().sum(axis=1)).fillna(0.0)
    out["season"] = se
    d = {"pg": out, "lg": lg.assign(season=se), "bio": bio.assign(season=se)}
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        for k, v in d.items():
            v.to_parquet(cache / f"pr_{k}_{se}.parquet", index=False)
    return d


# ------------------------------------------------------------------ tables
def rink_factors(pg: pd.DataFrame) -> pd.DataFrame:
    """(season, venue) -> factor per RINK stat: events per game at the rink,
    both teams, over the home team's road games (both teams); mean 1."""
    g = pg.groupby(["season", "game_id"]).agg(
        venue=("venue", "first"), **{c: (c, "sum") for c in RINK})
    away = pg[~pg.is_home].groupby("game_id").team.first()
    g["away"] = g.index.get_level_values("game_id").map(away)
    parts = []
    for se, d in g.groupby(level="season"):
        rf = d.groupby("venue")[RINK].mean() / d.groupby("away")[RINK].mean()
        rf = (rf / rf.mean()).dropna()
        rf.index.name = "venue"
        parts.append(rf.assign(season=se).reset_index())
    return pd.concat(parts).set_index(["season", "venue"])


def season_table(pg: pd.DataFrame, bio: pd.DataFrame,
                 rf: pd.DataFrame) -> pd.DataFrame:
    """(season, player) totals over skater-games: minutes, counts, position,
    teams, venue factor per RINK stat and rink-neutral counts."""
    sk = pg[pg.pos_group != 2].copy()
    ap = rf.copy()
    ap.loc[ap.index.get_level_values("season") > RINK_ERA_END] = 1.0
    f = ap.reindex(pd.MultiIndex.from_arrays([sk.season, sk.venue])).to_numpy()
    for i, c in enumerate(RINK):
        sk["wf_" + c] = sk.toi_sec * np.nan_to_num(f[:, i], nan=1.0)
    sk = sk.sort_values(["season", "game_id"])
    st = sk.groupby(["season", "player_id"]).agg(
        toi=("toi_sec", "sum"), gp=("game_id", "size"),
        pos_group=("pos_group", lambda s: int(s.mode().iloc[0])),
        team_first=("team", "first"), team_last=("team", "last"),
        **{c: (c, "sum") for c in COUNTS},
        **{"wf_" + c: ("wf_" + c, "sum") for c in RINK})
    for c in RINK:
        fac = (st["wf_" + c] / st.toi).where(st.toi > 0, 1.0)
        st["f_" + c] = fac
        st[c + "_n"] = st[c] / fac
    st = st.drop(columns=["wf_" + c for c in RINK])
    st["toi"] = st.toi / 60.0
    st["fo"] = st.fo_w + st.fo_l
    # listed position: most frequent code that season (C/L/R/D)
    b = bio[bio.pos.isin(["C", "L", "R", "D"])]
    code = b.groupby(["season", "player_id"]).pos.agg(lambda s: s.mode().iloc[0])
    st["pos"] = code.reindex(st.index)
    st["pos"] = st.pos.fillna(pd.Series(np.where(st.pos_group == 1, "D", "C"),
                                        index=st.index))
    return st.reset_index()


def league_levels(st: pd.DataFrame) -> pd.DataFrame:
    """League per-60 of each (rink-neutral) stat by season and group (0 F, 1 D)."""
    g = st.groupby(["season", "pos_group"])
    return pd.DataFrame({r: g[num_col(c)].sum() / g.toi.sum() * 60
                         for r, c in RATES.items()})


def fo_group(pos: pd.Series) -> pd.Series:
    return pos.map({"C": "C", "L": "W", "R": "W", "D": "D"}).fillna("W")


def window(st: pd.DataFrame, lv: pd.DataFrame, V: int, decay: float):
    """Season-weighted sums over V-LOOKBACK..V-1 at the V-1 level of the
    player's group."""
    w = st[(st.season < V) & (st.season >= V - LOOKBACK)].copy()
    w["w"] = decay ** (V - 1 - w.season)
    key = pd.MultiIndex.from_arrays([w.season, w.pos_group])
    now = pd.MultiIndex.from_arrays([np.full(len(w), V - 1), w.pos_group])
    for r, c in RATES.items():
        scale = lv[r].reindex(now).to_numpy() / lv[r].reindex(key).to_numpy()
        w["n_" + r] = w[num_col(c)] * scale * w.w
    w["wm"] = w.toi * w.w
    w["wfo_w"] = w.fo_w * w.w
    w["wfo"] = w.fo * w.w
    return w


AGG = ["wm", "wfo_w", "wfo"] + ["n_" + r for r in RATES]


def priors(st: pd.DataFrame, w: pd.DataFrame) -> dict:
    """Position means and newcomer means per stat from a window frame."""
    w = w.assign(F=np.where(w.pos_group == 1, "D", "F"), fog=fo_group(w.pos))
    mu = {}
    for r in RATES:
        grp = "fog" if r == "fo_taken60" else "F"
        s = w.groupby(grp)[["n_" + r, "wm"]].sum()
        mu[r] = (s["n_" + r] / s.wm * 60).to_dict()
    # newcomers: skaters in season s with no minutes in s-3..s-1 (s >= 2015)
    seen = st[st.toi > 0].groupby("player_id").season.apply(set)
    nw = w[w.season >= FIRST + LOOKBACK]
    fresh = np.array([not any(s - j in seen.get(p, set()) for j in (1, 2, 3))
                      for p, s in zip(nw.player_id, nw.season)], dtype=bool)
    nw = nw[fresh]
    new = {}
    for r in RATES:
        grp = "fog" if r == "fo_taken60" else "F"
        s = nw.groupby(grp)[["n_" + r, "wm"]].sum()
        s = s[s.wm > 2000]                      # need a real sample
        new[r] = {g: (s.loc[g, "n_" + r] / s.loc[g, "wm"] * 60
                      if g in s.index else mu[r][g]) for g in mu[r]}
    s = nw.groupby("fog")[["wfo_w", "wfo"]].sum()
    new[SHARE] = {g: (float(s.loc[g, "wfo_w"] / s.loc[g, "wfo"])
                      if g in s.index and s.loc[g, "wfo"] >= 1000 else 0.5)
                  for g in ("C", "W", "D")}
    return {"mu": mu, "new": new}


def estimate(st, lv, V, k: dict, decay: dict, pos_now: pd.Series | None = None,
             players=None) -> pd.DataFrame:
    """Per-player shrunk rink-neutral rates at vantage V from seasons < V.

    k, decay: per rate name plus SHARE. pos_now: optional listed position
    (C/L/R/D) overriding history (the roster at V); players: ids to cover
    (no-history rows get the newcomer mean)."""
    cache = {}

    def win(d):
        if d not in cache:
            w = window(st, lv, V, d)
            cache[d] = (w.groupby("player_id")[AGG].sum(), priors(st, w), w)
        return cache[d]

    agg0, _, w0 = win(DECAY)
    pos = w0.sort_values("season").groupby("player_id").pos.last()
    if pos_now is not None:
        pos = pos_now.combine_first(pos)
    ids = agg0.index if players is None else pd.Index(players).unique()
    out = pd.DataFrame(index=ids)
    out.index.name = "player_id"
    out["pos"] = pos.reindex(ids)
    out["pos_group"] = np.where(out.pos == "D", 1, 0)
    out["toi_minutes_basis"] = agg0.wm.reindex(ids).fillna(0.0)
    out["no_history"] = (out.toi_minutes_basis <= 0).astype(int)
    F = np.where(out.pos_group == 1, "D", "F")
    fog = fo_group(out.pos).to_numpy()
    for r in RATES:
        agg, pr, _ = win(decay[r])
        grp = fog if r == "fo_taken60" else F
        mu = np.array([pr["mu"][r][g] for g in grp])
        new = np.array([pr["new"][r][g] for g in grp])
        n = agg.wm.reindex(ids).fillna(0.0).to_numpy()
        num = agg["n_" + r].reindex(ids).fillna(0.0).to_numpy() * 60
        est = (num + k[r] * mu) / np.maximum(n + k[r], 1e-9)
        out[r] = np.where(n > 0, est, new)
    agg, pr, _ = win(decay[SHARE])
    W = agg.wfo_w.reindex(ids).fillna(0.0).to_numpy()
    T = agg.wfo.reindex(ids).fillna(0.0).to_numpy()
    new = np.array([pr["new"][SHARE][g] for g in fog])
    out[SHARE] = np.where(out.no_history == 1, new,
                          (W + 0.5 * k[SHARE]) / (T + k[SHARE]))
    out["fo_basis"] = T
    return out


def venue_forecast(rf: pd.DataFrame, V: int) -> dict:
    """Expected venue factor for a player of each team in season V (rink era):
    half the home factor forecast from V-1 (log slope on its lag, fitted on
    seasons < V), half the average of the other rinks."""
    out = {}
    for c in RINK:
        x, y = [], []
        for s in range(FIRST + 1, V):
            a = rf[c].xs(s - 1, level="season")
            b = rf[c].xs(s, level="season")
            j = pd.concat([a, b], axis=1, join="inner")
            x += list(np.log(j.iloc[:, 0]))
            y += list(np.log(j.iloc[:, 1]))
        x, y = np.array(x), np.array(y)
        slope = float(np.sum(x * y) / np.sum(x * x))
        home = np.exp(slope * np.log(rf[c].xs(V - 1, level="season")))
        n = len(home)
        road = (home.sum() - home) / (n - 1)
        out[c] = {"slope": slope, "factor": 0.5 * home + 0.5 * road}
    return out


# ------------------------------------------------------------------ tuning
def poisson_dev(y, mu):
    mu = np.maximum(mu, 1e-9)
    t = np.where(y > 0, y * np.log(np.maximum(y, 1e-12) / mu), 0.0)
    return 2.0 * float(np.sum(t - (y - mu)))


def deviance_curves(st, lv, d: float) -> tuple:
    """Deviance over V in TUNE_V for every k on the grid, at decay d.

    Targets are season-V rink-neutral counts rescaled to the V-1 level of the
    player's group, so k is not confounded with drift or scorers; only players
    with history enter. The win share is scored on listed centers, the group
    for which the 0.5 prior is right (they take ~85% of faceoffs)."""
    dev = {r: np.zeros(len(K_GRID)) for r in RATES}
    dev_fo = np.zeros(len(KFO_GRID))
    for V in TUNE_V:
        w = window(st, lv, V, d)
        pr = priors(st, w)
        agg = w.groupby("player_id")[AGG].sum()
        agg = agg[agg.wm > 0]
        tgt = st[(st.season == V) & (st.toi > 0)].set_index("player_id")
        ids = agg.index.intersection(tgt.index)
        pos = w.sort_values("season").groupby("player_id").pos.last().reindex(ids)
        F = np.where(pos == "D", "D", "F")
        fog = fo_group(pos).to_numpy()
        n = agg.wm.reindex(ids).to_numpy()
        m = tgt.toi.reindex(ids).to_numpy()
        grp = tgt.pos_group.reindex(ids).to_numpy()
        for r, c in RATES.items():
            mu = np.array([pr["mu"][r][g]
                           for g in (fog if r == "fo_taken60" else F)])
            num = agg["n_" + r].reindex(ids).to_numpy() * 60
            lvl = (lv[r].reindex(pd.MultiIndex.from_arrays(
                [np.full(len(ids), V - 1), grp])).to_numpy()
                / lv[r].reindex(pd.MultiIndex.from_arrays(
                    [np.full(len(ids), V), grp])).to_numpy())
            y = tgt[num_col(c)].reindex(ids).to_numpy() * lvl
            for i, kk in enumerate(K_GRID):
                rate = (num + kk * mu) / (n + kk)
                dev[r][i] += poisson_dev(y, rate * m / 60)
        W = agg.wfo_w.reindex(ids).to_numpy()
        T = agg.wfo.reindex(ids).to_numpy()
        yw = tgt.fo_w.reindex(ids).to_numpy()
        yl = tgt.fo_l.reindex(ids).to_numpy()
        ok = (T > 0) & (yw + yl > 0) & (fog == "C")
        for i, kk in enumerate(KFO_GRID):
            p = np.clip((W[ok] + 0.5 * kk) / (T[ok] + kk), 1e-6, 1 - 1e-6)
            dev_fo[i] += -2 * float(np.sum(yw[ok] * np.log(p)
                                           + yl[ok] * np.log(1 - p)))
    return dev, dev_fo


def tune(st, lv) -> tuple:
    """(decay, k) per stat and for the win share, by deviance on TUNE_V."""
    table = {}
    for d in DECAYS:
        dev, dev_fo = deviance_curves(st, lv, d)
        for r in RATES:
            i = int(np.argmin(dev[r]))
            table.setdefault(r, {})[d] = (float(dev[r][i]), float(K_GRID[i]))
        i = int(np.argmin(dev_fo))
        table.setdefault(SHARE, {})[d] = (float(dev_fo[i]), float(KFO_GRID[i]))
    k, decay = {}, {}
    for r, t in table.items():
        decay[r] = min(t, key=lambda d: t[d][0])
        k[r] = t[decay[r]][1]
    return k, decay, table


# ------------------------------------------------------------------ validation
def _metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    r = float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 0 else None
    return {"corr": None if r is None else round(r, 4),
            "mae": round(float(np.mean(np.abs(y - p))), 4),
            "bias": round(float(np.mean(p - y)), 4)}


def validate(st, lv, rf, k, decay) -> dict:
    """Predict V from seasons < V; skaters with >= MIN_EVAL minutes in V,
    scored against realised (recorded) per-60 rates.

    Baselines: (a) last season's raw per-60 (position mean when the player had
    no V-1 minutes), (b) the V-1 position mean (F/D; C/W/D for faceoffs)."""
    out = {}
    for V in VALID_V:
        tgt = st[st.season == V].set_index("player_id")
        prev = st[st.season == V - 1].set_index("player_id")
        ev = tgt[tgt.toi >= MIN_EVAL]
        est = estimate(st, lv, V, k, decay, pos_now=tgt.pos, players=ev.index)
        vf = venue_forecast(rf, V) if V <= RINK_ERA_END else None
        F = pd.Series(np.where(est.pos == "D", "D", "F"), index=ev.index)
        fog = fo_group(est.pos)
        pm = prev.toi.reindex(ev.index).fillna(0.0)
        hp = (pm > 0).to_numpy()
        res = {"n": len(ev), "n_with_prev": int(hp.sum()),
               "n_no_history": int(est.no_history.sum()), "stats": {}}
        if vf:
            res["venue_slope"] = {c: round(vf[c]["slope"], 3) for c in RINK}
        for r, c in RATES.items():
            grp = fog if r == "fo_taken60" else F
            pg_ = prev.assign(g=fo_group(prev.pos) if r == "fo_taken60"
                              else np.where(prev.pos == "D", "D", "F"))
            s = pg_.groupby("g")[[c, "toi"]].sum()
            base_b = grp.map((s[c] / s.toi * 60).to_dict()).to_numpy()
            raw_a = (prev[c].reindex(ev.index) / pm.where(pm > 0) * 60).to_numpy()
            base_a = np.where(hp, raw_a, base_b)
            y = (ev[c] / ev.toi * 60).to_numpy()
            neutral = est[r].to_numpy()
            model = neutral
            if vf and c in RINK:
                fac = ev.team_first.map(vf[c]["factor"]).fillna(1.0).to_numpy()
                model = neutral * fac
            m = {"model": _metrics(y, model), "last_season": _metrics(y, base_a),
                 "position_mean": _metrics(y, base_b),
                 "with_prev": {"model": _metrics(y[hp], model[hp]),
                               "last_season": _metrics(y[hp], base_a[hp])}}
            if vf and c in RINK:
                m["model_without_venue_factor"] = _metrics(y, neutral)
            res["stats"][r] = m
        # faceoff win share: players with enough faceoffs in V
        fo = tgt[tgt.fo >= MIN_EVAL_FO]
        e2 = estimate(st, lv, V, k, decay, pos_now=tgt.pos, players=fo.index)
        y = (fo.fo_w / fo.fo).to_numpy()
        pf = prev.fo.reindex(fo.index).fillna(0.0)
        last = np.where(pf > 0, prev.fo_w.reindex(fo.index) / pf.where(pf > 0), 0.5)
        res["stats"]["fo_win_share"] = {
            "n": len(fo), "model": _metrics(y, e2.fo_win_share),
            "last_season": _metrics(y, last),
            "position_mean": _metrics(y, np.full(len(y), 0.5))}
        out[str(V)] = res
    return out


def league_block(data: dict, seasons) -> dict:
    """Per-team-game averages by season, split by who is credited."""
    out = {}
    for se in seasons:
        pg, lg = data[se]["pg"], data[se]["lg"]
        tg = 2.0 * len(lg)
        sk = pg[pg.pos_group != 2][COUNTS].sum()
        gk = pg[pg.pos_group == 2][COUNTS].sum()
        tot = lg[LG].sum()
        d = {"games": len(lg),
             "skater_minutes": round(float(pg[pg.pos_group != 2].toi_sec.sum())
                                     / 60 / tg, 2)}
        for name, col, c in (("hits", "n_hits", "hits"),
                             ("blocks", "n_blocks", "blocks"),
                             ("giveaways", "n_give", "give"),
                             ("takeaways", "n_take", "take")):
            d[name] = {"all": round(float(tot[col]) / tg, 3),
                       "skaters": round(float(sk[c]) / tg, 3),
                       "goalies": round(float(gk[c]) / tg, 3)}
        d["blocks_by_teammate_excluded"] = round(float(tot.n_blocks_tm) / tg, 3)
        d["faceoffs"] = {"taken_per_team_game": round(float(tot.n_fo) / tg * 2, 3),
                         "won_per_team_game": round(float(tot.n_fo) / tg, 3),
                         "skater_fo_won": round(float(sk.fo_w) / tg, 3),
                         "skater_fo_lost": round(float(sk.fo_l) / tg, 3)}
        d["pim"] = {"all": round(float(tot.pim_all) / tg, 3),
                    "skaters": round(float(sk.pim) / tg, 3),
                    "goalies": round(float(gk.pim) / tg, 3),
                    "bench": round(float(tot.bench_pim) / tg, 3)}
        d["minor_penalties"] = {
            "all_players": round(float(tot.minors_all) / tg, 3),
            "skaters_taken": round(float(sk.minors) / tg, 3),
            "goalies_taken": round(float(gk.minors) / tg, 3),
            "bench_minors": round(float(tot.bench_minors) / tg, 3),
            "skaters_drawn": round(float(sk.minors_drawn) / tg, 3)}
        d["penalties_any_type"] = {
            "all": round(float(tot.n_pens) / tg, 3),
            "skaters_taken": round(float(sk.pens) / tg, 3),
            "skaters_drawn": round(float(sk.pens_drawn) / tg, 3),
            "skater_majors": round(float(sk.majors) / tg, 3)}
        d["credited_player_not_in_player_games"] = int(lg.lost_events.sum())
        out[str(se)] = d
    return out


# ------------------------------------------------------------------ output
def roster_players() -> pd.DataFrame:
    """Union of roster snapshots; team/pos/name from the latest one listing
    the player; `latest` flags presence in the newest snapshot."""
    snaps = sorted((RAW / "rosters").glob("*/rosters.csv"))
    ro = pd.concat([pd.read_csv(p).assign(snap=p.parent.name) for p in snaps])
    ro = ro.sort_values("snap").groupby("player_id").last()
    ro["name"] = (ro["first"].fillna("") + " " + ro["last"].fillna("")).str.strip()
    ro["latest"] = (ro.snap == snaps[-1].parent.name).astype(int)
    return ro[ro.pos != "G"][["team", "pos", "name", "latest"]]


def build_table(data, st, lv, V, k, decay) -> pd.DataFrame:
    ro = roster_players()
    recent = []
    for se in range(V - LOOKBACK, V):          # any game type, skaters
        pg = pd.read_parquet(TENSORS / f"player_games_{se}.parquet",
                             columns=["player_id", "pos_group"])
        recent.append(pg[pg.pos_group != 2])
    recent = pd.concat(recent).groupby("player_id").pos_group.max()
    ids = pd.Index(sorted(set(ro.index) | set(recent.index)))
    bio = pd.concat([data[s]["bio"] for s in sorted(data) if s < V])
    bio = bio.sort_values(["season", "game_id"]).groupby("player_id").last()
    listed = bio.pos.reindex(ids)
    fallback = pd.Series(np.where(recent.reindex(ids) == 1, "D", "C"), index=ids)
    pos_now = ro.pos.reindex(ids).combine_first(
        listed.where(listed.isin(["C", "L", "R", "D"]))).combine_first(fallback)
    est = estimate(st, lv, V, k, decay, pos_now=pos_now, players=ids)
    est.insert(0, "name", ro.name.reindex(ids).combine_first(
        bio.name.reindex(ids)).fillna(""))
    est.insert(1, "team", ro.team.reindex(ids).combine_first(
        bio.team.reindex(ids)).fillna(""))
    last = st[(st.season < V) & (st.toi > 0)].groupby("player_id").season.max()
    est["last_season"] = last.reindex(ids).fillna(0).astype(int)
    mins = st[(st.season < V) & (st.season >= V - LOOKBACK)].pivot_table(
        index="player_id", columns="season", values="toi", aggfunc="sum")
    for se in range(V - LOOKBACK, V):
        col = mins[se] if se in mins.columns else pd.Series(dtype=float)
        est[f"min_{se}"] = col.reindex(ids).fillna(0.0).round(1)
    est["on_roster"] = ro.latest.reindex(ids).fillna(0).astype(int)
    cols = ["name", "team", "pos", "pos_group", "toi_minutes_basis", "hits60",
            "blocks60", "giveaways60", "takeaways60", "fo_taken60",
            "fo_win_share", "fo_basis", "pim60", "pen_taken60", "pen_drawn60",
            "no_history", "on_roster", "last_season"] + \
        [f"min_{se}" for se in range(V - LOOKBACK, V)]
    out = est[cols].copy()
    for c in ["toi_minutes_basis", "fo_basis"]:
        out[c] = out[c].round(1)
    for c in list(RATES) + ["fo_win_share"]:
        out[c] = out[c].round(4)
    return out.reset_index()


def table_check(tab: pd.DataFrame, st: pd.DataFrame, se: int, tg: float) -> dict:
    """Per-team-game totals implied by the table at each skater's season-se
    minutes, against the realised season-se skater totals (tg team-games)."""
    s = st[st.season == se].set_index("player_id")
    t = tab.set_index("player_id").reindex(s.index)
    m = s.toi.to_numpy()
    out = {}
    for r, c in RATES.items():
        imp = np.nansum(t[r].to_numpy() * m) / 60 / tg
        out[r] = {"implied": round(float(imp), 3),
                  "realised": round(float(s[c].sum()) / tg, 3)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vantage", type=int, default=2027)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--cache", type=Path, default=None,
                    help="optional dir for per-season extraction parquet")
    args = ap.parse_args()
    V = args.vantage
    if not TUNE_V[-1] < V <= LAST + 1:
        sys.exit(f"--vantage must be in {TUNE_V[-1] + 1}..{LAST + 1}")
    # the estimator only reads seasons < V; later ones feed validation and
    # the league block
    seasons = list(range(FIRST, (LAST if args.validate else V - 1) + 1))
    with Pool(8) as pool:
        data = {se: load_season(se, pool, args.cache) for se in seasons}
    pg = pd.concat([data[s]["pg"] for s in seasons], ignore_index=True)
    bio = pd.concat([data[s]["bio"] for s in seasons], ignore_index=True)
    rf = rink_factors(pg)
    st = season_table(pg, bio, rf)
    lv = league_levels(st)
    print("league per-60 by season and group (rink-neutral):")
    print(lv.round(3).to_string())
    k, decay, table = tune(st, lv)
    for r in k:
        print(f"  {r:<13} decay {decay[r]:.2f}  k {k[r]:g}")

    tab = build_table(data, st, lv, V, k, decay)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"player_rates_{V}.csv"
    if V == 2027:
        from common import refuse_if_frozen_1_0
        refuse_if_frozen_1_0(path.name)
    tab.to_csv(path, index=False)
    print(f"{path.name}: {len(tab)} players, {int(tab.no_history.sum())} without "
          f"history, {int(tab.on_roster.sum())} on rosters")

    if not args.validate:
        return
    val = validate(st, lv, rf, k, decay)
    sd = rf.groupby(level="season").std()
    rep = {
        "definition": {
            "source": "NHL play-by-play JSON, regular season, periods 1-4, per "
                      "player-game joined to shift TOI (player_games)",
            "seasons_used": [FIRST, LAST],
            "rates": "per 60 minutes of total TOI; 2027 table is rink-neutral",
            "pen_taken60/pen_drawn60": "minor penalties (typeCode MIN; a double "
                                       "minor counts 2)",
            "blocks60": "excludes shots blocked by the shooter's teammate",
            "lookback_seasons": LOOKBACK,
            "toi_minutes_basis": f"minutes weighted 1, {DECAY}, {DECAY**2:.2f} "
                                 "for V-1, V-2, V-3",
            "rink_neutralised_stats": RINK, "rink_era_end": RINK_ERA_END,
            "level_normalisation": "per season and position group (F/D) to V-1",
            "decay": decay,
            "k": {r: v for r, v in k.items()},
            "k_units": "minutes of TOI; faceoffs taken for fo_win_share",
            "fo_win_share_prior": 0.5, "tuned_on_vantages": TUNE_V,
            "tuning_deviance_by_decay": {
                r: {str(d): {"deviance": round(v[0], 1), "k": v[1]}
                    for d, v in t.items()} for r, t in table.items()},
            "eval_min_minutes": MIN_EVAL, "eval_min_faceoffs": MIN_EVAL_FO,
            "validation_notes": (
                "Targets are realised (recorded) per-60 rates in V for skaters "
                "with >= 500 minutes (win share: >= 200 faceoffs). model = "
                "shrunk rate from seasons < V; for rink-judged stats it is "
                "multiplied by the expected venue factor of the team the player "
                "opened V with (first game). last_season = raw V-1 rate "
                "(position mean if no V-1 minutes); position_mean = V-1 "
                "F/D mean (C/W/D for faceoffs; 0.5 for win share). with_prev "
                "restricts to players with V-1 minutes. bias = mean(pred - y).")},
        "league_per60_by_group": {
            str(s): {("F" if g == 0 else "D"): {r: round(float(v), 4)
                                                for r, v in lv.loc[(s, g)].items()}
                     for g in (0, 1)} for s in sorted(set(lv.index.get_level_values(0)))},
        "rink_factor_sd": {str(s): {c: round(float(v), 3)
                                    for c, v in sd.loc[s].items()} for s in sd.index},
        "validation": val,
    }
    lb = league_block(data, [s for s in (2024, 2025, 2026) if s in data])
    # flat key read by the season simulator: faceoffs per game, 2024-2026
    lb["faceoffs_per_game"] = round(float(np.average(
        [d["faceoffs"]["taken_per_team_game"] for d in lb.values()],
        weights=[d["games"] for d in lb.values()])), 3)
    rep["league_per_team_game"] = lb
    pri = {r: priors(st, window(st, lv, V, decay[r])) for r in k}
    rep[f"priors_{V}"] = {
        "position_mean": {r: {g: round(float(v), 4)
                              for g, v in pri[r]["mu"][r].items()} for r in RATES},
        "newcomer_mean": {r: {g: round(float(v), 4)
                              for g, v in pri[r]["new"][r].items()} for r in k}}
    if V - 1 in data:
        rep[f"table_check_at_{V - 1}_minutes"] = table_check(
            tab, st, V - 1, 2.0 * len(data[V - 1]["lg"]))
    vp = OUT / "player_rates_validation.json"
    vp.write_text(json.dumps(rep, indent=1))
    print(f"wrote {vp.name}")

    def f(x):
        return "    -" if x is None else f"{x:.3f}"
    for V_, res in val.items():
        print(f"\n{V_}: n={res['n']} (with V-1 minutes {res['n_with_prev']})")
        print(f"{'stat':<14}{'model r':>8}{'mae':>7}{'last r':>8}{'mae':>7}"
              f"{'posmean r':>10}{'mae':>7}")
        for r, m in res["stats"].items():
            print(f"{r:<14}{f(m['model']['corr']):>8}{m['model']['mae']:>7.3f}"
                  f"{f(m['last_season']['corr']):>8}{m['last_season']['mae']:>7.3f}"
                  f"{f(m['position_mean']['corr']):>10}"
                  f"{m['position_mean']['mae']:>7.3f}")


if __name__ == "__main__":
    main()
