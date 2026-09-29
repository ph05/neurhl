"""NeurHL 1.1 C3: a filtered team rating (PLAN_NeurHL_1_1, C3).

Each team carries a strength mean r and variance P on the logit scale. Before a
game the home-win logit is z = H + r_home - r_away. After it, an extended
Kalman step for a logistic observation moves both teams toward the result:

    p = sigmoid(z),  I = p (1 - p),  S = P_home + P_away
    r_home += P_home * w * (y - p) / (1 + S I),  r_away -= P_away * w * (y - p) / (1 + S I)
    P_side -= P_side^2 I / (1 + S I)

- **Result y is outcome-aware** (Whelan and Klein 2021): 1 for a regulation win,
  p_ot for an overtime or shootout win, 1 - p_ot for an overtime loss, 0 for a
  regulation loss.
- **w is a margin factor** for regulation results: w = |goal diff|^m,
  so m = 0 ignores the margin.
- **Variance grows with time** (Elo's "development coefficient"): P += q per
  day between a team's games.
- **At each season start**, r is regressed toward 0 by phi and P is raised
  by s2.

Walk-forward by construction: every rating used for a game comes from earlier
games only. The frozen Elo comparator (sim/game_model.run_elo and the stored
elo_logit) is not touched.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import TENSORS  # noqa: E402

SEALED = {2025, 2026}
DEFAULT = {"H": 0.15, "q": 2e-4, "s2": 0.02, "phi": 0.7, "P0": 0.05, "p_ot": 0.6, "m": 0.0}


def load_games(seasons, allow_sealed=False) -> pd.DataFrame:
    if not allow_sealed:
        bad = set(seasons) & SEALED
        assert not bad, f"sealed seasons requested: {sorted(bad)}"
    frames = []
    for s in sorted(seasons):
        p = TENSORS / f"games_ctx_{s}.parquet"
        if p.exists():
            g = pd.read_parquet(p, columns=["game_id", "game_type", "date", "home_idx", "away_idx",
                                            "home_g", "away_g", "outcome4"])
            frames.append(g[g.game_type == 2].assign(season_end=s))
    g = pd.concat(frames, ignore_index=True)
    g["date"] = pd.to_datetime(g.date)
    return g.sort_values(["season_end", "date", "game_id"], kind="stable").reset_index(drop=True)


def run(g: pd.DataFrame, prm: dict = None) -> pd.DataFrame:
    """Pre-game home-win logit z and the two teams' variances for every game."""
    p_ = {**DEFAULT, **(prm or {})}
    H, q, s2, phi, P0, p_ot, m = (p_[k] for k in ("H", "q", "s2", "phi", "P0", "p_ot", "m"))
    hi, ai = g.home_idx.to_numpy(int), g.away_idx.to_numpy(int)
    se = g.season_end.to_numpy(int)
    day = (g.date - g.date.min()).dt.days.to_numpy()
    o4 = g.outcome4.to_numpy(int)
    diff = np.abs(g.home_g.to_numpy() - g.away_g.to_numpy())
    n = int(max(hi.max(), ai.max())) + 1
    r, P, last = np.zeros(n), np.full(n, P0), np.full(n, -1)
    z, ph, pa = np.empty(len(g)), np.empty(len(g)), np.empty(len(g))
    cur = None
    yv = np.array([1.0, 0.0, p_ot, 1.0 - p_ot])
    for i in range(len(g)):
        if se[i] != cur:
            if cur is not None:
                r *= phi
                P = phi * phi * P + s2
            cur = se[i]
        h, a = hi[i], ai[i]
        for t in (h, a):
            if last[t] >= 0:
                P[t] += q * (day[i] - last[t])
            last[t] = day[i]
        zi = H + r[h] - r[a]
        z[i], ph[i], pa[i] = zi, P[h], P[a]
        p = 1.0 / (1.0 + np.exp(-zi))
        info = p * (1 - p)
        S = P[h] + P[a]
        w = (diff[i] ** m) if (m and o4[i] < 2) else 1.0
        d = w * (yv[o4[i]] - p) / (1 + S * info)
        r[h] += P[h] * d
        r[a] -= P[a] * d
        P[h] -= P[h] ** 2 * info / (1 + S * info)
        P[a] -= P[a] ** 2 * info / (1 + S * info)
    return pd.DataFrame({"game_id": g.game_id.to_numpy(), "season_end": se, "z_v2": z,
                         "var_h": ph, "var_a": pa})
