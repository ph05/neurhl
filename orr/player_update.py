"""In-season updating of skater per-game rates (ORR 1.2).

A skater's preseason projection gives per-game rates for goals, assists and
shots on goal. As the season goes on, each rate is updated with his own
games so far, as a conjugate Gamma-Poisson posterior mean:

    rate = (n0 * prior_rate + count_so_far) / (n0 + games_so_far)

n0 (pseudo-games of prior weight) is set per statistic by the walk-forward
backtest orr/backtest/player_update_bt.py (tuned on 2012-2017 only). The
daily loop (orr/inseason.py) uses the updated rates for its per-game player
lines when box scores are available (orr/output/live/boxes_2027.csv, written
by orr.ingest from the NHL API).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from orr import config as C

STATS = ("g", "a", "sog")
PARAMS = C.PARAMS / "player_update.json"
BOXES_LIVE = C.OUT / "live" / f"boxes_{C.TARGET_SEASON}.csv"
DEFAULT_N0 = {"g": 60.0, "a": 60.0, "sog": 30.0}


def load_n0() -> dict:
    if PARAMS.exists():
        return {k: float(v) for k, v in json.loads(PARAMS.read_text())["n0"].items()}
    return dict(DEFAULT_N0)


def season_boxes(V: int) -> pd.DataFrame:
    """Skater rows of the regular-season box scores of season V (fastRhockey):
    game_id, player_id, g, a, sog, toi (minutes)."""
    f = C.ROOT / "data" / "raw" / "fastrhockey" / f"player_box_{V}.parquet"
    p = pd.read_parquet(f, columns=["game_id", "player_id", "position_code", "skater_stats_goals",
                                    "skater_stats_assists", "skater_stats_shots", "skater_stats_time_on_ice"])
    p = p[(p.position_code != "G") & ((p.game_id // 10000) % 100 == 2)].copy()
    t = p.skater_stats_time_on_ice.fillna("0:00").astype(str).str.split(":", expand=True)
    p["toi"] = t[0].astype(float) + t[1].astype(float) / 60.0
    p = p[p.toi > 0]
    return pd.DataFrame({"game_id": p.game_id.astype(int), "player_id": p.player_id.astype(int),
                         "g": p.skater_stats_goals.fillna(0).astype(float),
                         "a": p.skater_stats_assists.fillna(0).astype(float),
                         "sog": p.skater_stats_shots.fillna(0).astype(float), "toi": p.toi})


def totals(boxes: pd.DataFrame) -> pd.DataFrame:
    """player_id -> n (games), g, a, sog."""
    t = boxes.groupby("player_id").agg(n=("game_id", "size"), g=("g", "sum"), a=("a", "sum"), sog=("sog", "sum"))
    return t.reset_index()


def posterior(prior: pd.DataFrame, obs: pd.DataFrame, n0: dict) -> pd.DataFrame:
    """prior: player_id, g_pg, a_pg, sog_pg. obs: player_id, n, g, a, sog.
    Returns player_id with updated g_pg, a_pg, sog_pg (prior rate where no games)."""
    m = prior.merge(obs, on="player_id", how="left").fillna({"n": 0, "g": 0, "a": 0, "sog": 0})
    out = m[["player_id"]].copy()
    for s in STATS:
        k = n0[s]
        out[f"{s}_pg"] = (k * m[f"{s}_pg"] + m[s]) / (k + m.n) if np.isfinite(k) else m[f"{s}_pg"]
    out["n"] = m.n.to_numpy()
    return out


def live_rates(skaters: pd.DataFrame, n0: dict | None = None, path: Path = BOXES_LIVE,
               before: pd.Timestamp | None = None) -> pd.DataFrame | None:
    """Updated per-game rates for the 2026-27 skaters from the live box scores
    (games strictly before ``before``); None if there are no box scores."""
    if not Path(path).exists():
        return None
    b = pd.read_csv(path)
    if before is not None and "date" in b:
        b = b[pd.to_datetime(b.date) < before]
    b = b[b.toi > 0] if "toi" in b else b
    b = b[b.pos != "G"] if "pos" in b else b
    if not len(b):
        return None
    gp = skaters.gp.clip(lower=1)
    prior = pd.DataFrame({"player_id": skaters.player_id, "g_pg": skaters.g / gp,
                          "a_pg": skaters.a / gp, "sog_pg": skaters.sog / gp})
    return posterior(prior, totals(b), n0 or load_n0())


CAL_PARAMS = C.PARAMS / "player_prob.json"


def load_calibration() -> dict | None:
    """Platt parameters for per-game P(goal) and P(point), or None when the
    calibration backtest (orr/backtest/player_prob_bt.py) did not adopt them."""
    if not CAL_PARAMS.exists():
        return None
    d = json.loads(CAL_PARAMS.read_text())
    return {"goal": d["goal"], "point": d["point"]} if d.get("adopted") else None


def platt(p: np.ndarray, w) -> np.ndarray:
    x = np.log(p / (1 - p))
    return 1 / (1 + np.exp(-(w[0] + w[1] * x)))
