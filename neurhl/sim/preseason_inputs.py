"""NeurHL Tier-3b — preseason (h1) model inputs for a target season T.

Builds game-model input tensors for season T's REAL schedule using ONLY
information available at vantage T-1 (P1): each team's roster is its
prior-season top-18 skaters by TOI + top-2 goalies by starts (players' modal
team over their last games of T-1); form features are each player's EWMA state
entering season T (no shift — the state after their last T-1 game); era vector
is T's walk-forward vector; rest/travel come from the schedule itself
(knowable preseason). Rookies/new signings are unknown by construction — the
same honesty as the house h1 protocol.

Returns (tensors, games_df) where tensors match train_game's batch schema and
games_df carries (date, home, away) in franchise codes for the bridge/sim.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PROC, TENSORS  # noqa: E402
from train.train_game import (EWMA_ALPHA, era_table_row,  # noqa: E402
                              load_frames)


def era_abbrev(franchise: str, se: int) -> str:
    if franchise == "UTA":
        return "PHX" if se <= 2014 else ("ARI" if se <= 2024 else "UTA")
    if franchise == "WPG" and se <= 2011:
        return "ATL"
    return franchise


def final_team_form(gc: pd.DataFrame, season: int, n: int = 25) -> dict:
    """{team_idx: (gf, ga, pts)} over each team's last n games of `season`.

    The h1 analogue of train_game.team_form: a preseason projection knows the
    prior season's results and nothing of the season being projected.
    """
    g = gc[gc.season_end == season].sort_values("date")
    hist: dict = {}
    for r in g.itertuples(index=False):
        home_win = r.home_g > r.away_g
        extra = r.outcome4 >= 2
        hist.setdefault(r.home_idx, []).append(
            (r.home_g, r.away_g, 2 if home_win else (1 if extra else 0)))
        hist.setdefault(r.away_idx, []).append(
            (r.away_g, r.home_g, 2 if not home_win else (1 if extra else 0)))
    out = {}
    for tid, rows in hist.items():
        last = rows[-n:]
        out[int(tid)] = (float(np.mean([x[0] for x in last])),
                         float(np.mean([x[1] for x in last])),
                         float(np.mean([x[2] for x in last])))
    return out


def final_form(pg: pd.DataFrame) -> pd.DataFrame:
    """Per player: EWMA state entering next season (last row, unshifted)."""
    pg = pg.sort_values(["player_id", "date", "game_id"])
    grp = pg.groupby("player_id")
    pg = pg.assign(
        ewma_toi=grp.toi_sec.transform(
            lambda s: s.ewm(alpha=EWMA_ALPHA).mean()) / 60.0,
        gp_todate=grp.cumcount() + 1)
    return pg.groupby("player_id").tail(1)[
        ["player_id", "ewma_toi", "gp_todate", "pos_group"]]


def build_preseason(T: int, train_from: int = 2009):
    maps = json.loads((TENSORS / "maps.json").read_text())
    team_idx = maps["team"]
    seasons = list(range(train_from, T))
    gc, pg = load_frames(seasons)          # <= T-1 only
    prior = pg[pg.season_end == T - 1].copy()

    # player -> team idx (modal over last 10 games of T-1)
    gmeta = gc[gc.season_end == T - 1].set_index("game_id")
    prior["team_i"] = np.where(
        prior.is_home, prior.game_id.map(gmeta.home_idx),
        prior.game_id.map(gmeta.away_idx))
    last10 = (prior.sort_values("date").groupby("player_id").tail(10)
              .groupby("player_id").team_i.agg(lambda s: s.mode().iat[0]))
    toi_tot = prior.groupby("player_id").toi_sec.sum()
    starts = prior.groupby("player_id").goalie_start.sum()
    form = final_form(pg).set_index("player_id")

    # rosters per era-team index
    rosters = {}
    pl = pd.DataFrame({"team_i": last10, "toi": toi_tot, "starts": starts})
    pl = pl.join(form)
    for ti, grp in pl.groupby("team_i"):
        gk = grp[grp.pos_group == 2].sort_values("starts",
                                                 ascending=False).head(2)
        sk = grp[grp.pos_group < 2].sort_values("toi", ascending=False).head(18)
        rosters[int(ti)] = (gk, sk)

    # season T schedule + context (franchise codes)
    games = pd.read_csv(PROC / "games.csv")
    sched = games[(games.season_end == T) & (games.game_type == "R")].copy()
    travel = pd.read_csv(PROC / "travel_games.csv")
    sched = sched.merge(travel, on=["date", "home", "away"], how="left",
                        suffixes=("", "_tv"))
    first_date = pd.to_datetime(sched.date).min()
    sched["days_in"] = (pd.to_datetime(sched.date) - first_date).dt.days

    era_vec = era_table_row(T)

    npz = np.load(TENSORS / f"embeddings_v{T}.npz")
    row = {int(p): i for i, p in enumerate(npz["ids"])}
    emb_m = npz["emb"]
    d = emb_m.shape[1]
    bios = pd.read_parquet(TENSORS / "career_bios.parquet")
    pos_mean = np.zeros((3, d), np.float32)
    for pgi in range(3):
        sel = [row[p] for p in bios[bios.pos_group == pgi].player_id
               if p in row]
        if sel:
            pos_mean[pgi] = emb_m[sel].mean(0)

    N = len(sched)
    out = {"emb": np.zeros((N, 2, 20, d), np.float32),
           "pctx": np.zeros((N, 2, 20, 6), np.float32),
           "pad": np.ones((N, 2, 20), bool),
           "ctx": np.zeros((N, 16), np.float32),
           "era": np.tile(era_vec, (N, 1)).astype(np.float32)}
    cfg = json.loads((TENSORS.parents[1] / "configs" / "game_model.json")
                     .read_text())
    tform = final_team_form(gc, T - 1) if cfg.get("use_team_form") else None
    n_missing_roster = 0
    for i, g in enumerate(sched.itertuples(index=False)):
        team_i = {}
        for t, fr in ((0, g.home), (1, g.away)):
            ab = era_abbrev(fr, T)
            ti = team_idx.get(ab, 0)
            team_i[t] = ti
            ros = rosters.get(ti)
            if ros is None:
                n_missing_roster += 1
                continue
            gk, sk = ros
            sel = pd.concat([gk, sk])
            for j, (pid, r) in enumerate(sel.iterrows()):
                if j >= 20:
                    break
                pg_i = 0 if pd.isna(r.pos_group) else int(r.pos_group)
                ridx = row.get(int(pid))
                out["emb"][i, t, j] = (emb_m[ridx] if ridx is not None
                                       else pos_mean[pg_i])
                pos1h = np.zeros(3, np.float32)
                pos1h[pg_i] = 1
                starter = float(pg_i == 2 and j == 0)
                out["pctx"][i, t, j] = [*pos1h,
                                        np.nan_to_num(float(r.ewma_toi)) / 20.0,
                                        np.log1p(np.nan_to_num(
                                            float(r.gp_todate))) / 5.0,
                                        starter]
                out["pad"][i, t, j] = False
        out["ctx"][i, :9] = [min(g.home_rest, 7) / 7, min(g.away_rest, 7) / 7,
                             float(g.home_rest <= 1), float(g.away_rest <= 1),
                             g.home_km3d / 1000, g.away_km3d / 1000,
                             g.home_dtz / 3, g.away_dtz / 3, g.days_in / 200]
        if tform is not None:
            hg, ha, hp = tform.get(team_i[0], (2.7, 2.7, 1.1))
            ag, aa, ap = tform.get(team_i[1], (2.7, 2.7, 1.1))
            out["ctx"][i, 9:15] = [hg / 3.0, ha / 3.0, hp / 2.0,
                                   ag / 3.0, aa / 3.0, ap / 2.0]
    out["ctx"] = np.nan_to_num(out["ctx"])
    if n_missing_roster:
        print(f"  preseason {T}: {n_missing_roster} team-games without a "
              f"prior roster (expansion debut)")
    import torch
    tensors = {k: torch.as_tensor(v) for k, v in out.items()}
    return tensors, sched[["date", "home", "away"]].reset_index(drop=True)
