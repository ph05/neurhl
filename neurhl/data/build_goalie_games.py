"""NeurHL-3 — D6: per goalie-game panel with walk-forward quality priors.

One row per (game_id, goalie): start flag, shots faced, goals against, and two
STRICTLY PRE-GAME quality numbers —

  * `gq`  — EB-shrunk EWMA save% minus the expanding league save%, the exact
    construction Tier-0 validated (models/baseline_gbm.goalie_quality);
  * `gsax60` — EB-shrunk goals saved above expected per 60, from the house xG
    joined onto each shot's goalie (events carry `goalie` on >=99.4% of
    SOG+goal events back to 2008). Whether GSAx earns its place over `gq` is
    gate GQ1 (Steiger), not assumed.

Plus workload: starts in the trailing 7 days, days since last start, b2b flag.
Manifest-tracked: rebuilds when xg_shots (A9/A11) or the event shards move.

Run: uv run --no-project --python 3.12 --with numpy --with "pandas<3" \
     --with pyarrow python neurhl/data/build_goalie_games.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import TENSORS  # noqa: E402
import manifest as MAN  # noqa: E402

SEASONS = list(range(2008, 2027))
EWMA_N = 10.0                # shrink games, as in baseline_gbm.goalie_quality
ALPHA = 0.1
GSAX_K = 900.0               # EB prior weight in FENWICK shots (~20 starts)
GOAL_ID = 7                  # maps.json event_type id
SOG_ID = 14
MISS_ID = 9
CFG = {"ewma_n": EWMA_N, "alpha": ALPHA, "gsax_k": GSAX_K, "version": 3}


def season_shots(s: int) -> pd.DataFrame:
    """Per (game_id, goalie): SOG faced / goals against (save%% convention)
    and FENWICK xG faced (GSAx must compare goals to expected goals over the
    same population the xG was calibrated on — SOG-only xG understates
    expected goals on that subset and biases GSAx negative)."""
    ev = pd.read_parquet(TENSORS / f"events_{s}.parquet",
                         columns=["game_id", "game_type", "event_type",
                                  "period", "t", "p1", "goalie"])
    ev = ev[(ev.game_type == 2)
            & ev.event_type.isin([GOAL_ID, SOG_ID, MISS_ID])
            & (ev.goalie > 0) & (ev.period <= 4)]
    ev["ga"] = (ev.event_type == GOAL_ID).astype("int64")
    ev["on_goal"] = ev.event_type.isin([GOAL_ID, SOG_ID]).astype("int64")
    xp = TENSORS / f"xg_shots_{s}.parquet"
    if xp.exists():
        xg = pd.read_parquet(xp, columns=["game_id", "time", "period",
                                          "shooter", "xg"])
        xg = xg.rename(columns={"shooter": "p1"})
        # events t is absolute game seconds; xg time is MoneyPuck's in-game
        # seconds -- align on (game_id, period, p1, within-period seconds)
        ev["sec_in_p"] = (ev.t.astype("int64")
                          - (ev.period.astype("int64") - 1) * 1200)
        xg["sec_in_p"] = (xg.time.astype("int64")
                          - (xg.period.astype("int64") - 1) * 1200)
        m = ev.merge(xg[["game_id", "period", "p1", "sec_in_p", "xg"]],
                     on=["game_id", "period", "p1", "sec_in_p"], how="left")
        # unmatched shots (~0.2%) are filled WALK-FORWARD in build() from the
        # league running mean strictly before the game — a season-wide fill
        # here leaked future xG into past rows (caught by the causality audit)
        ev = m
    else:
        ev["xg"] = np.nan
    return ev.groupby(["game_id", "goalie"], as_index=False).agg(
        sf=("on_goal", "sum"), ga=("ga", "sum"), fen=("ga", "size"),
        xgf=("xg", "sum"), n_xg=("xg", "count"))


def build(s: int, carry: dict) -> pd.DataFrame:
    pg = pd.read_parquet(TENSORS / f"player_games_{s}.parquet",
                         columns=["game_id", "game_type", "player_id",
                                  "is_home", "pos_group", "toi_sec",
                                  "goalie_start"])
    pg = pg[(pg.game_type == 2) & (pg.pos_group == 2)]
    gc = pd.read_parquet(TENSORS / f"games_ctx_{s}.parquet",
                         columns=["game_id", "game_type", "date", "home_idx",
                                  "away_idx"])
    gc = gc[gc.game_type == 2].copy()
    gc["date"] = pd.to_datetime(gc.date)
    pg = pg.merge(gc, on="game_id")
    pg["team"] = np.where(pg.is_home, pg.home_idx, pg.away_idx)
    sh = season_shots(s)
    pg = pg.merge(sh, left_on=["game_id", "player_id"],
                  right_on=["game_id", "goalie"], how="left")
    for c in ("sf", "ga", "fen", "xgf", "n_xg"):
        pg[c] = pg[c].fillna(0.0)
    pg = pg.sort_values(["date", "game_id"], kind="stable").reset_index(drop=True)

    # walk-forward state (carried across seasons through `carry`)
    lg_sv_n = carry.setdefault("lg_n", 0.0)
    lg_sv_s = carry.setdefault("lg_s", 0.0)
    st = carry.setdefault("g", {})   # pid -> dict(ewma_sv, n, gsax_sum, sf_sum, last_start, starts)
    rows = []
    for r in pg.itertuples():
        g = st.setdefault(int(r.player_id),
                          {"ewma_sv": None, "n": 0.0, "gsax": 0.0,
                           "sf": 0.0, "fen": 0.0, "toi": 0.0, "last": None,
                           "starts": []})
        lg_sv = (carry["lg_s"] / carry["lg_n"]) if carry["lg_n"] > 0 else 0.905
        w = g["n"] / (g["n"] + EWMA_N)
        gq = (w * ((g["ewma_sv"] if g["ewma_sv"] is not None else lg_sv)
                   - lg_sv))
        gsax60 = 3600.0 * g["gsax"] / g["toi"] if g["toi"] > 0 else 0.0
        gsax60 *= g["fen"] / (g["fen"] + GSAX_K)
        starts7 = sum(1 for d in g["starts"]
                      if 0 < (r.date - d).days <= 7)
        rest = (r.date - g["last"]).days if g["last"] is not None else -1
        rows.append((r.game_id, int(r.player_id), int(r.team),
                     bool(r.is_home), int(r.goalie_start), int(r.toi_sec),
                     float(r.sf), float(r.ga), float(r.fen), float(r.xgf),
                     float(gq), float(gsax60), int(starts7), int(rest)))
        # post-game state update
        if r.sf > 0:
            sv = 1.0 - r.ga / r.sf
            g["ewma_sv"] = (sv if g["ewma_sv"] is None
                            else ALPHA * sv + (1 - ALPHA) * g["ewma_sv"])
            g["n"] += 1.0
            g["sf"] += float(r.sf)
            carry["lg_n"] += float(r.sf)
            carry["lg_s"] += float(r.sf - r.ga)
        if r.fen > 0:
            # fill this game's unmatched shots with the league running mean
            # xG per shot STRICTLY BEFORE this game (walk-forward)
            run_mean = (carry["xg_s"] / carry["xg_n"]
                        if carry.get("xg_n", 0.0) > 0 else 0.0)
            xgf_use = float(r.xgf) + (float(r.fen) - float(r.n_xg)) * run_mean
            g["gsax"] += xgf_use - float(r.ga)
            g["fen"] += float(r.fen)
            g["toi"] += float(max(r.toi_sec, 0))
            carry["xg_n"] = carry.get("xg_n", 0.0) + float(r.n_xg)
            carry["xg_s"] = carry.get("xg_s", 0.0) + float(r.xgf)
        if r.goalie_start == 1:
            g["last"] = r.date
            g["starts"].append(r.date)
            g["starts"] = g["starts"][-30:]
    return pd.DataFrame(rows, columns=[
        "game_id", "player_id", "team", "is_home", "goalie_start", "toi_sec",
        "sf", "ga", "fen", "xgf", "gq", "gsax60", "starts_7d", "rest_days"])


def main():
    carry = {}
    for s in SEASONS:
        src = [TENSORS / f"player_games_{s}.parquet",
               TENSORS / f"games_ctx_{s}.parquet",
               TENSORS / f"events_{s}.parquet"]
        xp = TENSORS / f"xg_shots_{s}.parquet"
        if xp.exists():
            src.append(xp)
        if not all(p.exists() for p in src[:3]):
            print(f"{s}: missing sources, skipping")
            continue
        out = TENSORS / f"goalie_games_{s}.parquet"
        d = build(s, carry)          # carry must advance even on fresh skips,
        if not MAN.is_fresh(out, src, CFG):   # so build always runs
            d.to_parquet(out, index=False)
            MAN.write_manifest(out, src, CFG, {"n_rows": len(d)})
        starters = d[d.goalie_start == 1]
        faced = d[d.fen > 0]
        xcov = float((faced.xgf > 0).mean()) if len(faced) else 0.0
        print(f"{s}: {len(d):,} goalie-games, {len(starters):,} starts, "
              f"xg joined {xcov:.1%} of shot-facing rows, starter sv "
              f"{1 - starters.ga.sum() / max(starters.sf.sum(), 1):.4f}, "
              f"mean starter gsax60 {starters.gsax60.mean():+.3f}")
        sys.stdout.flush()
    print("done")


if __name__ == "__main__":
    main()
