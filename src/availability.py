"""Beta-binomial player availability model.

Empirics (this panel): next-season GP of established regulars is ~24x overdispersed vs
binomial; P(miss 15+) rises 29% -> 42% from age<24 to 36+. Model: per age bucket,
gp_next ~ BetaBinomial(82, a, b) fit by moments on an expanding causal window.

Used in simulation as ZERO-MEAN team strength noise: draws are centered by the bucket
expectation, so the mean age->availability effect stays with the (already gated) age
features and no double-counting occurs; this term contributes variance and tails only.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import players as P

BUCKETS = [(0, 24), (24, 28), (28, 32), (32, 36), (36, 99)]
N_GAMES = 82.0
TOP_N = 9  # players whose absence is modeled per team


def _bucket_idx(ages: np.ndarray) -> np.ndarray:
    idx = np.zeros(len(ages), dtype=int)
    for i, (lo, hi) in enumerate(BUCKETS):
        idx[(ages >= lo) & (ages < hi)] = i
    return idx


def fit_availability(skaters: pd.DataFrame, bios: pd.DataFrame, max_season: int) -> list[dict]:
    """Per-bucket beta-binomial (a, b) by moments. Causal: pairs with target <= max_season,
    82-game target seasons only."""
    ok_targets = [s for s in range(2010, max_season + 1) if s not in (2013, 2020, 2021)]
    cur = skaters[(skaters.toi_min >= 1200)][["playerId", "season_end"]].copy()
    nxt = skaters[["playerId", "season_end", "gp"]].copy()
    nxt["season_end"] -= 1
    m = cur.merge(nxt, on=["playerId", "season_end"])
    m = m[(m.season_end + 1).isin(ok_targets)]
    ref = pd.to_datetime((m.season_end + 1).astype(str) + "-01-01")
    bd = bios.birthDate.reindex(m.playerId).to_numpy()
    m["age"] = (ref.to_numpy() - bd).astype("timedelta64[D]").astype(float) / 365.25
    m = m.dropna(subset=["age"])
    m["b"] = _bucket_idx(m.age.to_numpy())
    params = []
    for i in range(len(BUCKETS)):
        d = m[m.b == i]
        pr = (d.gp.clip(0, N_GAMES) / N_GAMES).to_numpy()
        mean, var = pr.mean(), pr.var(ddof=1)
        rho = max((var * N_GAMES / (mean * (1 - mean)) - 1) / (N_GAMES - 1), 1e-4)
        ab = 1.0 / rho - 1.0
        params.append({"a": mean * ab, "b": (1 - mean) * ab, "mean": mean, "n": len(d)})
    return params


def make_extra_noise(teams: list[str], rosters: dict, params: list[dict],
                     k_elo: float, ppg_scale: float = 82.0):
    """Returns extra_noise(m, rng) -> (m, n_teams) zero-mean Elo noise from availability draws.

    rosters: team -> list of (age, value_goals) for its TOP_N most valuable skaters.
    Player full-season value in Elo: value_goals * k_elo / 82; absence fraction scales it.
    """
    per_team = []
    for t in teams:
        arr = rosters.get(t, [])[:TOP_N]
        if not arr:
            per_team.append((np.zeros(0), np.zeros(0), np.zeros(0, dtype=int)))
            continue
        ages = np.array([a for a, _ in arr])
        vals = np.array([v for _, v in arr]) * k_elo / ppg_scale
        per_team.append((ages, vals, _bucket_idx(ages)))

    a_vec = np.array([p["a"] for p in params])
    b_vec = np.array([p["b"] for p in params])
    mean_vec = np.array([p["mean"] for p in params])

    def extra_noise(m: int, rng: np.random.Generator) -> np.ndarray:
        out = np.zeros((m, len(teams)), dtype=float)
        for j, (ages, vals, bidx) in enumerate(per_team):
            if len(vals) == 0:
                continue
            pdraw = rng.beta(a_vec[bidx], b_vec[bidx], size=(m, len(vals)))
            gp = rng.binomial(int(N_GAMES), pdraw) / N_GAMES
            out[:, j] = ((gp - mean_vec[bidx]) * vals).sum(axis=1)
        return out

    return extra_noise
