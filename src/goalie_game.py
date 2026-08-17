"""I1: goalie game layer — per-game starter rotation in the simulator (PLAN_V4).

Per team: tandem (G1, G2) by projected workload; s1 = G1 start share, EB-shrunk
gp-share history toward the causal league starter mean. Per simulated game the starter
is drawn Bernoulli(p1); on second-of-back-to-back nights p1 = BETA_B2B * s1 (declared
constant, no game-level start data in repo; sensitivity reported by the gate script).
Per-game Elo delta = k * SHOTS_PG * (theta_started - theta_tandem_mean): ZERO-MEAN by
construction, so the mean goalie effect stays with the gated tandem_gsax feature and
this layer contributes variance shape + the b2b/backup-quality interaction only.

Start-share proxy: goalie games_played per team (relief appearances overcount starts
slightly; documented). Gate G in backtest4.py decides shipping.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import players as P

SHOTS_PG = 30.0
BETA_B2B = 0.5   # declared, not fit (PLAN_V4 I1); sensitivity {0.3, 0.7} reported
LEAGUE_S1 = None  # computed causally


def league_starter_share(got: pd.DataFrame, ts: pd.DataFrame, vantage: int) -> float:
    """Causal league mean of the top goalie's share of team games."""
    d = got[got.season_end <= vantage]
    shares = []
    for (s, team), grp in d.groupby(["season_end", "team"]):
        tg = ts.loc[(ts.season_end == s) & (ts.team == team), "gp"]
        if len(tg) and grp.games_played.sum() > 0:
            shares.append(grp.games_played.max() / float(tg.iloc[0]))
    return float(np.mean(shares))


def team_tandems(got: pd.DataFrame, ts: pd.DataFrame, goalie_proj: pd.DataFrame,
                 vantage: int, n0_g: float) -> pd.DataFrame:
    """Per-team tandem at vantage: G1/G2 ids, thetas, EB start share s1.
    G1/G2 ranked by season workload (ongoal); s1 shrunk toward the causal league mean."""
    proj = goalie_proj.set_index("playerId")
    mu = league_starter_share(got, ts, vantage)
    gt = got[got.season_end == vantage]
    rows = []
    for team, d in gt.groupby("team"):
        d = d.sort_values("ongoal", ascending=False)
        g1 = d.iloc[0]
        th1 = float(proj.theta.get(g1.playerId, 0.0))
        if len(d) > 1:
            g2 = d.iloc[1]
            th2 = float(proj.theta.get(g2.playerId, 0.0))
            gp2 = float(g2.games_played)
        else:
            th2, gp2 = 0.0, 0.0  # league-average fill-in backup
        tg = ts.loc[(ts.season_end == vantage) & (ts.team == team), "gp"]
        team_g = float(tg.iloc[0]) if len(tg) else 82.0
        # EB shrink in share space: s1 = (starts + n0*mu) / (team_games + n0)
        s1 = (float(g1.games_played) + n0_g * mu) / (team_g + n0_g)
        s1 = float(np.clip(s1, 0.40, 0.85))
        rows.append({"team": team, "g1": int(g1.playerId), "g2": int(g2.playerId) if len(d) > 1 else -1,
                     "theta1": th1, "theta2": th2, "s1": s1,
                     "gap_gpg": abs(th1 - th2) * SHOTS_PG})  # goals/game; ×k for Elo
    return pd.DataFrame(rows).set_index("team")


def make_game_noise(sched: pd.DataFrame, tandems: pd.DataFrame, k_elo: float,
                    beta_b2b: float = BETA_B2B):
    """Per-game goalie-start noise closure for engine.simulate_season(game_noise=...).

    sched needs home/away and (optionally) hb2b/ab2b columns; missing flags = 0.
    Returns callable(m, rng) -> (m, n_games) home-minus-away Elo adjustment,
    zero-mean over start draws by construction.
    """
    teams = list(tandems.index)
    tidx = {t: i for i, t in enumerate(teams)}
    th1 = tandems.theta1.to_numpy()
    th2 = tandems.theta2.to_numpy()
    s1 = tandems.s1.to_numpy()
    tbar = s1 * th1 + (1 - s1) * th2
    # per-team per-start adjustments (Elo), zero-mean under p1 = s1
    a1 = k_elo * SHOTS_PG * (th1 - tbar)
    a2 = k_elo * SHOTS_PG * (th2 - tbar)

    hi = sched.home.map(tidx).to_numpy()
    ai = sched.away.map(tidx).to_numpy()
    hb = sched.hb2b.to_numpy() if "hb2b" in sched.columns else np.zeros(len(sched))
    ab = sched.ab2b.to_numpy() if "ab2b" in sched.columns else np.zeros(len(sched))
    # start prob for G1 by game (b2b nights hand the net to the backup more often)
    p1_h = np.where(hb > 0, beta_b2b * s1[hi], s1[hi])
    p1_a = np.where(ab > 0, beta_b2b * s1[ai], s1[ai])
    # centering must match the game-specific p1, else b2b games acquire a mean shift
    mean_h = p1_h * a1[hi] + (1 - p1_h) * a2[hi]
    mean_a = p1_a * a1[ai] + (1 - p1_a) * a2[ai]

    def game_noise(m: int, rng: np.random.Generator) -> np.ndarray:
        u_h = rng.random((m, len(hi)))
        u_a = rng.random((m, len(ai)))
        adj_h = np.where(u_h < p1_h[None, :], a1[hi][None, :], a2[hi][None, :]) - mean_h[None, :]
        adj_a = np.where(u_a < p1_a[None, :], a1[ai][None, :], a2[ai][None, :]) - mean_a[None, :]
        return adj_h - adj_a

    return game_noise


def prod_tandems(ros: pd.DataFrame, got: pd.DataFrame, ts: pd.DataFrame,
                 goalie_proj: pd.DataFrame, vantage: int, n0_g: float) -> pd.DataFrame:
    """Production tandems from August API rosters (goalies move in the offseason):
    G1/G2 = top-2 by projection-window shots; s1 from G1's own last-season gp share
    (any team), EB-shrunk to the causal league mean."""
    proj = goalie_proj.set_index("playerId")
    mu = league_starter_share(got, ts, vantage)
    last = got[got.season_end == vantage].groupby("playerId").games_played.sum()
    rows = []
    for team, d in ros[ros.position == "G"].groupby("team"):
        d = d.copy()
        d["shots_win"] = d.playerId.map(proj.shots_win).fillna(0.0)
        d = d.sort_values("shots_win", ascending=False)
        g1 = d.iloc[0]
        th1 = float(proj.theta.get(g1.playerId, 0.0))
        th2 = float(proj.theta.get(d.iloc[1].playerId, 0.0)) if len(d) > 1 else 0.0
        gp1 = float(last.get(g1.playerId, mu * 82.0))
        s1 = float(np.clip((gp1 + n0_g * mu) / (82.0 + n0_g), 0.40, 0.85))
        rows.append({"team": team, "g1": int(g1.playerId),
                     "g2": int(d.iloc[1].playerId) if len(d) > 1 else -1,
                     "theta1": th1, "theta2": th2, "s1": s1,
                     "gap_gpg": abs(th1 - th2) * SHOTS_PG})
    return pd.DataFrame(rows).set_index("team")


def tune_n0_g(got: pd.DataFrame, ts: pd.DataFrame, grid=(2.0, 4.0, 8.0),
              vantages=range(2011, 2017)) -> tuple[float, pd.DataFrame]:
    """Train-only: predict next-season G1 share (same goalie, same team) from the
    EB-shrunk share; weighted MSE by next-season team games."""
    rows = []
    for n0 in grid:
        se, wt = [], []
        for V in vantages:
            gt = got[got.season_end == V]
            mu = league_starter_share(got, ts, V)
            nxt = got[got.season_end == V + 1]
            for team, d in gt.groupby("team"):
                d = d.sort_values("ongoal", ascending=False)
                g1 = d.iloc[0]
                tg = ts.loc[(ts.season_end == V) & (ts.team == team), "gp"]
                team_g = float(tg.iloc[0]) if len(tg) else 82.0
                s1_hat = (float(g1.games_played) + n0 * mu) / (team_g + n0)
                nx = nxt[(nxt.team == team) & (nxt.playerId == g1.playerId)]
                tgn = ts.loc[(ts.season_end == V + 1) & (ts.team == team), "gp"]
                if len(nx) and len(tgn):
                    team_gn = float(tgn.iloc[0])
                    real = float(nx.games_played.iloc[0]) / team_gn
                    se.append((s1_hat - real) ** 2 * team_gn)
                    wt.append(team_gn)
        rows.append((n0, sum(se) / sum(wt)))
    tab = pd.DataFrame(rows, columns=["n0_g", "wmse"]).sort_values("wmse")
    return float(tab.iloc[0].n0_g), tab
