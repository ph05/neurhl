"""NeurHL — game-level tensorization (PLAN_NeurHL Phase 1).

Produces, per season_end (2012-2026), two normalized parquet tables under
neurhl/data/tensors/ (gitignored):

  games_ctx_<se>.parquet    one row per game: teams, walk-forward era vector,
                            context (rest/travel/b2b/days-in), labels
                            (goals, outcome4, 5v5 SAT for/against)
  player_games_<se>.parquet one row per dressed player per game: side, position
                            group, shift TOI, goalie-starter flag, and per-game
                            counting stats (H4 labels / form-feature source)

Label provenance: games.csv (EDA-06: 100% join, 0 score mismatches). Era vector
uses PRIOR-season league stats only (P5). Travel/rest from travel_games.csv.
The roster[2,20] input tensor is assembled by the training dataloader from
player_games rows — these tables stay normalized.
"""
import gzip
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PROC, RAW, TENSORS  # noqa: E402
from data.build_onice import mmss  # noqa: E402

MAPS = TENSORS / "maps.json"
REMAP = {"PHX": "UTA", "ARI": "UTA", "ATL": "WPG"}   # PBP era-code -> franchise
POS_GROUP = {"C": 0, "L": 0, "R": 0, "D": 1, "G": 2}
SOG_T, MISS_T, BLOCK_T, GOAL_T, FO_T, PEN_T = ("shot-on-goal", "missed-shot",
                                               "blocked-shot", "goal",
                                               "faceoff", "penalty")


def roster_toi_one(args):
    """Per game: dressed players with side/pos/TOI + starting goalies."""
    pbp_path, shift_path = args
    with gzip.open(pbp_path, "rt") as f:
        g = json.load(f)
    gid, home_id = g["id"], g["homeTeam"]["id"]
    toi: dict = {}
    first_start: dict = {}
    if shift_path.exists():
        with gzip.open(shift_path, "rt") as f:
            recs = json.load(f)
        if isinstance(recs, dict):
            recs = recs.get("data", [])
        for r in recs:
            if r.get("typeCode") != 517 or not r.get("startTime") or not r.get("endTime"):
                continue
            if r["period"] >= 5:
                continue
            dur = ((r["period"] - 1) * 1200 + mmss(r["endTime"])) - \
                  ((r["period"] - 1) * 1200 + mmss(r["startTime"]))
            if dur <= 0:
                continue
            pid = r["playerId"]
            toi[pid] = toi.get(pid, 0) + dur
            t0 = (r["period"] - 1) * 1200 + mmss(r["startTime"])
            first_start[pid] = min(first_start.get(pid, 1 << 30), t0)
    rows = []
    starters = {}
    for r in g.get("rosterSpots", []):
        pid = r.get("playerId")
        if not pid:
            continue
        is_home = r["teamId"] == home_id
        pos = POS_GROUP.get(r.get("positionCode"), 0)
        rows.append((gid, g["gameType"], pid, is_home, pos,
                     toi.get(pid, 0), first_start.get(pid, -1)))
        if pos == 2:
            key = is_home
            cand = starters.get(key)
            fs = first_start.get(pid, 1 << 30)
            if cand is None or fs < cand[1]:
                starters[key] = (pid, fs)
    starter_ids = {v[0] for v in starters.values()}
    return [(gid, gt, pid, ih, pos, t, int(pid in starter_ids))
            for gid, gt, pid, ih, pos, t, _ in rows]


def era_table() -> pd.DataFrame:
    games = pd.read_csv(PROC / "games.csv")
    ts = pd.read_csv(PROC / "team_seasons.csv")
    r = games[games.game_type == "R"]
    per = r.groupby("season_end").apply(lambda d: pd.Series({
        "gpg": (d.home_g + d.away_g).mean(),
        "ot_share": d.went_ot.astype(bool).mean(),
        "so_share": d.went_so.astype(bool).mean(),
        "margin_abs": d.margin.abs().mean()}), include_groups=False)
    per["parity"] = ts.groupby("season_end").pts_pct.std()
    prior = per.shift(1).add_prefix("prior_")          # season V uses V-1 stats
    prior["flag_3v3"] = (prior.index >= 2016).astype(float)
    prior["flag_covid"] = prior.index.isin([2020, 2021]).astype(float)
    prior["season_scaled"] = (prior.index - 2006) / 20.0
    return prior


ERA_COLS = ["prior_gpg", "prior_ot_share", "prior_so_share", "prior_margin_abs",
            "prior_parity", "flag_3v3", "flag_covid", "season_scaled"]


def main():
    maps = json.loads(MAPS.read_text())
    team_idx = maps["team"]
    games = pd.read_csv(PROC / "games.csv")
    travel = pd.read_csv(PROC / "travel_games.csv")
    era = era_table()

    for sdir in sorted((RAW / "pbp").iterdir()):
        se = int(sdir.name)
        out_g = TENSORS / f"games_ctx_{se}.parquet"
        out_p = TENSORS / f"player_games_{se}.parquet"
        if out_g.exists() and out_p.exists():
            print(f"{se}: exists, skipping")
            continue

        # --- player-game table (rosters + TOI + starters)
        jobs = []
        for src in ("pbp", "pbp_po"):
            d = RAW / src / str(se)
            if d.exists():
                jobs += [(p, RAW / "shifts" / str(se) / p.name)
                         for p in sorted(d.glob("*.json.gz"))]
        prows = []
        with Pool(8) as pool:
            for rows in pool.imap(roster_toi_one, jobs, chunksize=32):
                prows.extend(rows)
        pg = pd.DataFrame(prows, columns=["game_id", "game_type", "player_id",
                                          "is_home", "pos_group", "toi_sec",
                                          "goalie_start"])

        # --- per-player counting stats from the events shard
        ev = pd.read_parquet(TENSORS / f"events_{se}.parquet",
                             columns=["game_id", "event_type", "strength",
                                      "home_event", "p1", "p2", "p3"])
        et_inv = {v: k for k, v in maps["event_type"].items()}
        ev["etn"] = ev.event_type.map(et_inv)

        def count(mask, col):
            c = ev[mask].groupby(["game_id", col]).size()
            c.index.names = ["game_id", "player_id"]
            return c

        stats = pd.DataFrame({
            "goals": count(ev.etn == GOAL_T, "p1"),
            "a1": count(ev.etn == GOAL_T, "p2"),
            "a2": count(ev.etn == GOAL_T, "p3"),
            "sog": count(ev.etn.isin([SOG_T, GOAL_T]), "p1"),
            "att": count(ev.etn.isin([SOG_T, GOAL_T, MISS_T, BLOCK_T]), "p1"),
            "blocks": count(ev.etn == BLOCK_T, "p2"),
            "pen": count(ev.etn == PEN_T, "p1"),
            "fo_w": count(ev.etn == FO_T, "p1"),
            "fo_l": count(ev.etn == FO_T, "p2"),
        }).fillna(0).astype(int).reset_index()
        stats = stats[stats.player_id > 0]
        stats["assists"] = stats.a1 + stats.a2
        pg = pg.merge(stats.drop(columns=["a1", "a2"]),
                      on=["game_id", "player_id"], how="left").fillna(0)
        for c in ("goals", "sog", "att", "blocks", "pen", "fo_w", "fo_l", "assists"):
            pg[c] = pg[c].astype("int16")
        pg.to_parquet(out_p, index=False)

        # --- game context table
        sat = (ev[(ev.strength == 1551)
                  & ev.etn.isin([SOG_T, GOAL_T, MISS_T, BLOCK_T])]
               .groupby(["game_id", "home_event"]).size().unstack(fill_value=0))
        gsum = pd.read_parquet(TENSORS / "_edacache" / "game_summary.parquet")
        gsum = gsum[gsum.season_end == se].copy()
        gsum["home_m"] = gsum.home.replace(REMAP)
        gsum["away_m"] = gsum.away.replace(REMAP)
        lab = games[games.season_end == se]
        m = gsum.merge(lab, left_on=["date", "home_m", "away_m"],
                       right_on=["date", "home", "away"], suffixes=("", "_hr"))
        m = m.merge(travel, left_on=["date", "home_m", "away_m"],
                    right_on=["date", "home", "away"], how="left",
                    suffixes=("", "_tv"))
        extra = m.went_ot.astype(bool) | m.went_so.astype(bool)
        home_won = m.home_g > m.away_g
        m["outcome4"] = np.select(
            [home_won & ~extra, ~home_won & ~extra, home_won & extra],
            [0, 1, 2], default=3).astype("uint8")
        first_date = pd.to_datetime(m[m.game_type == 2].date).min()
        m["days_in"] = (pd.to_datetime(m.date) - first_date).dt.days
        m["home_idx"] = m.home.map(team_idx).fillna(0).astype("uint8")
        m["away_idx"] = m.away.map(team_idx).fillna(0).astype("uint8")
        m["sat5_h"] = m.game_id.map(sat.get(1, pd.Series(dtype=int))).fillna(0)
        m["sat5_a"] = m.game_id.map(sat.get(0, pd.Series(dtype=int))).fillna(0)
        for c in ERA_COLS:
            m[c] = era.loc[se, c] if se in era.index else np.nan
        keep = (["game_id", "game_type", "date", "home_idx", "away_idx",
                 "home_g", "away_g", "outcome4", "sat5_h", "sat5_a", "days_in",
                 "home_rest", "away_rest", "home_km3d", "away_km3d",
                 "home_dtz", "away_dtz"] + ERA_COLS)
        m[keep].to_parquet(out_g, index=False)
        print(f"{se}: {len(m)} games ({len(pg):,} player-games) "
              f"[travel joined {m.home_rest.notna().mean():.3f}]")
        sys.stdout.flush()
    print("done")


if __name__ == "__main__":
    main()
