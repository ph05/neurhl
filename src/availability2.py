"""I4: Availability 2.0 (PLAN_V4) — per-player persistence, zero-GP mass, goalie slot.

Extends v3's age-bucket beta-binomial availability with:
(a) per-player EB persistence: player availability prior = recency-weighted own GP-share
    history shrunk toward the age-bucket mean (weight n0_a, train-tuned);
    SCREEN S1: within-bucket YoY correlation of GP share on train >= 0.10;
(b) zero-GP mass: P0(bucket) = fraction of regulars with ZERO next-season NHL games
    (the catastrophic tail the review showed was excluded); mixture draw, centered at
    the mixture mean so the layer stays zero-mean;
(c) goalie slot: team's G1 joins the draw list, value = (theta1-theta2)*2500 goals
    (the replacement is the backup); goalie params fit on clear starters;
    SCREEN S2: goalie share overdispersion vs binomial >= 5x on train.

Gate A2 in backtest4.py decides shipping (coverage + fragility-tertile spread).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import players as P
from availability import BUCKETS, N_GAMES, TOP_N, _bucket_idx

DELTA_A = 0.7          # recency weight on own history (fixed; n0_a is the tuned knob)
GOALIE_SHOTS = 2500.0  # starter shots/season (overlay convention)


# ---------------------------------------------------------------- panel of pairs
def _pairs(skaters: pd.DataFrame, bios: pd.DataFrame, max_season: int,
           min_toi=1200.0) -> pd.DataFrame:
    """(player, season) regulars with next-season GP share target (0 if absent),
    82-game target seasons only, causal through max_season."""
    ok_targets = [s for s in range(2010, max_season + 1) if s not in (2013, 2020, 2021)]
    cur = skaters[skaters.toi_min >= min_toi][["playerId", "season_end"]].copy()
    nxt = skaters[["playerId", "season_end", "gp"]].copy()
    nxt["season_end"] -= 1
    m = cur.merge(nxt, on=["playerId", "season_end"], how="left")  # left: keep vanished
    m["gp"] = m.gp.fillna(0.0)
    m = m[(m.season_end + 1).isin(ok_targets)]
    ref = pd.to_datetime((m.season_end + 1).astype(str) + "-01-01")
    bd = bios.birthDate.reindex(m.playerId).to_numpy()
    m["age"] = (ref.to_numpy() - bd).astype("timedelta64[D]").astype(float) / 365.25
    m = m.dropna(subset=["age"])
    m["b"] = _bucket_idx(m.age.to_numpy())
    m["share"] = m.gp.clip(0, N_GAMES) / N_GAMES
    return m


def screen_persistence(skaters: pd.DataFrame, bios: pd.DataFrame,
                       max_season: int) -> dict:
    """S1: within-bucket YoY corr of GP share among regulars (both seasons played)."""
    a = skaters[skaters.toi_min >= 1200][["playerId", "season_end", "gp"]].copy()
    b = skaters[["playerId", "season_end", "gp"]].copy()
    b["season_end"] -= 1
    m = a.merge(b, on=["playerId", "season_end"], suffixes=("_t", "_n"))
    ok_targets = [s for s in range(2010, max_season + 1) if s not in (2013, 2020, 2021)]
    m = m[(m.season_end + 1).isin(ok_targets) & (m.season_end <= max_season - 1)]
    ref = pd.to_datetime((m.season_end + 1).astype(str) + "-01-01")
    bd = bios.birthDate.reindex(m.playerId).to_numpy()
    m["age"] = (ref.to_numpy() - bd).astype("timedelta64[D]").astype(float) / 365.25
    m = m.dropna(subset=["age"])
    m["b"] = _bucket_idx(m.age.to_numpy())
    m["s_t"] = m.gp_t.clip(0, N_GAMES) / N_GAMES
    m["s_n"] = m.gp_n.clip(0, N_GAMES) / N_GAMES
    m["s_t_c"] = m.s_t - m.groupby("b").s_t.transform("mean")
    m["s_n_c"] = m.s_n - m.groupby("b").s_n.transform("mean")
    r = float(np.corrcoef(m.s_t_c, m.s_n_c)[0, 1])
    return {"r_within_bucket": round(r, 3), "n": len(m), "pass": bool(r >= 0.10)}


def fit_availability2(skaters: pd.DataFrame, bios: pd.DataFrame, max_season: int) -> dict:
    """Bucket params: beta-binomial (a,b) among players who play, + zero-GP mass P0."""
    m = _pairs(skaters, bios, max_season)
    params = []
    for i in range(len(BUCKETS)):
        d = m[m.b == i]
        p0 = float((d.share == 0).mean())
        dd = d[d.share > 0]
        pr = dd.share.to_numpy()
        mean, var = pr.mean(), pr.var(ddof=1)
        rho = max((var * N_GAMES / (mean * (1 - mean)) - 1) / (N_GAMES - 1), 1e-4)
        ab = 1.0 / rho - 1.0
        params.append({"a": mean * ab, "b": (1 - mean) * ab, "mean": mean,
                       "p0": p0, "ab": ab, "n": len(d)})
    return {"buckets": params}


def player_mu(skaters: pd.DataFrame, bios: pd.DataFrame, vantage: int,
              params: dict, n0_a: float, window: int = 3) -> pd.Series:
    """EB availability prior per player: own recency-weighted GP-share history shrunk
    toward the age-bucket mean (bucket at vantage+1 age)."""
    hist = skaters[(skaters.season_end <= vantage)
                   & (skaters.season_end > vantage - window)]
    # effective games observed per season, recency-weighted
    by = hist.groupby("playerId")
    ages = P.age_of(bios, pd.Index(by.size().index), vantage + 1)
    bidx = _bucket_idx(np.nan_to_num(ages.to_numpy(), nan=27.0))
    bmean = np.array([params["buckets"][i]["mean"] for i in bidx])
    mus = {}
    for j, (pid, d) in enumerate(by):
        lag = vantage - d.season_end.to_numpy()
        w = DELTA_A ** lag * N_GAMES          # each season worth up to 82 obs
        x = d.gp.clip(0, N_GAMES).to_numpy() / N_GAMES
        mu = (float((w * x).sum()) + n0_a * bmean[j]) / (float(w.sum()) + n0_a)
        mus[pid] = float(np.clip(mu, 0.30, 0.99))
    return pd.Series(mus, name="mu")


def tune_n0_a(skaters: pd.DataFrame, bios: pd.DataFrame, grid=(4.0, 8.0, 16.0, 32.0),
              vantages=range(2011, 2017)) -> tuple[float, pd.DataFrame]:
    """Train-only: MSE of next-season GP share (players who play) vs EB prediction.
    n0_a in units of effective observed games (one full prior season = 82)."""
    rows = []
    for n0 in grid:
        se = []
        for V in vantages:
            params = fit_availability2(skaters, bios, V)
            mu = player_mu(skaters, bios, V, params, n0)
            m = _pairs(skaters, bios, V + 1)
            m = m[m.season_end == V]          # pairs targeting V+1 exactly
            mm = m[m.share > 0]
            pred = mu.reindex(mm.playerId).to_numpy()
            b_mean = np.array([params["buckets"][i]["mean"] for i in mm.b])
            pred = np.where(np.isnan(pred), b_mean, pred)
            se += list((pred - mm.share.to_numpy()) ** 2)
        rows.append((n0, float(np.mean(se))))
    tab = pd.DataFrame(rows, columns=["n0_a", "mse"]).sort_values("mse")
    # bucket-only baseline for context
    se0 = []
    for V in vantages:
        params = fit_availability2(skaters, bios, V)
        m = _pairs(skaters, bios, V + 1)
        m = m[(m.season_end == V) & (m.share > 0)]
        b_mean = np.array([params["buckets"][i]["mean"] for i in m.b])
        se0 += list((b_mean - m.share.to_numpy()) ** 2)
    tab.attrs["bucket_only_mse"] = float(np.mean(se0))
    return float(tab.iloc[0].n0_a), tab


# ---------------------------------------------------------------- goalie module
def screen_goalie_overdispersion(goalies_team: pd.DataFrame, ts: pd.DataFrame,
                                 max_season: int) -> dict:
    """S2: are clear starters' next-season game shares >=5x overdispersed vs binomial?"""
    rows = []
    for V in range(2010, max_season):
        if V + 1 in (2013, 2020, 2021):
            continue
        gt = goalies_team[goalies_team.season_end == V]
        nxt = goalies_team[goalies_team.season_end == V + 1]
        for team, d in gt.groupby("team"):
            d = d.sort_values("games_played", ascending=False)
            g1 = d.iloc[0]
            tg = ts.loc[(ts.season_end == V) & (ts.team == team), "gp"]
            if not len(tg) or g1.games_played / float(tg.iloc[0]) < 0.45:
                continue
            nx = nxt[nxt.playerId == g1.playerId]
            gpn = float(nx.games_played.sum())  # any team next season
            rows.append(min(gpn, N_GAMES) / N_GAMES)
    pr = np.array(rows)
    mean, var = pr.mean(), pr.var(ddof=1)
    binom_var = mean * (1 - mean) / N_GAMES
    od = float(var / binom_var)
    return {"overdispersion": round(od, 1), "mean_share": round(float(mean), 3),
            "n": len(pr), "pass": bool(od >= 5.0)}


def fit_goalie_availability(goalies_team: pd.DataFrame, ts: pd.DataFrame,
                            max_season: int) -> dict:
    """Beta-binomial (+P0) on clear starters' next-season game share, single pool
    (goalie starter counts are too thin for age buckets; age enters via the mean)."""
    shares = []
    for V in range(2010, max_season):
        if V + 1 in (2013, 2020, 2021):
            continue
        gt = goalies_team[goalies_team.season_end == V]
        nxt = goalies_team[goalies_team.season_end == V + 1]
        for team, d in gt.groupby("team"):
            d = d.sort_values("games_played", ascending=False)
            g1 = d.iloc[0]
            tg = ts.loc[(ts.season_end == V) & (ts.team == team), "gp"]
            if not len(tg) or g1.games_played / float(tg.iloc[0]) < 0.45:
                continue
            shares.append(min(float(nxt[nxt.playerId == g1.playerId].games_played.sum()),
                              N_GAMES) / N_GAMES)
    pr = np.array(shares)
    p0 = float((pr == 0).mean())
    prp = pr[pr > 0]
    mean, var = prp.mean(), prp.var(ddof=1)
    rho = max((var * N_GAMES / (mean * (1 - mean)) - 1) / (N_GAMES - 1), 1e-4)
    ab = 1.0 / rho - 1.0
    return {"a": mean * ab, "b": (1 - mean) * ab, "mean": mean, "p0": p0,
            "ab": ab, "n": len(pr)}


# ---------------------------------------------------------------- noise closure
def make_extra_noise2(teams: list[str], rosters: dict, params: dict,
                      goalie_rows: dict | None, goalie_par: dict | None,
                      k_elo: float):
    """Zero-mean team Elo noise from availability draws.

    rosters: team -> list of (bucket_idx, mu_i, value_goals) for TOP_N skaters.
    goalie_rows: team -> (mu_g1, value_goals_gap) or None if goalie slot unshipped.
    Mixture per player: with prob p0(bucket) season lost (share 0), else
    BetaBin(82, mu_i * ab_b, (1-mu_i) * ab_b); draws centered at the mixture mean.
    """
    bpar = params["buckets"]
    per_team = []
    for t in teams:
        arr = rosters.get(t, [])[:TOP_N]
        b_idx = np.array([b for b, _, _ in arr], dtype=int)
        mus = np.array([mu for _, mu, _ in arr])
        vals = np.array([v for _, _, v in arr]) * k_elo / 82.0
        p0 = np.array([bpar[i]["p0"] for i in b_idx])
        ab = np.array([bpar[i]["ab"] for i in b_idx])
        gk = goalie_rows.get(t) if goalie_rows else None
        per_team.append((mus, vals, p0, ab, gk))

    def extra_noise(m: int, rng: np.random.Generator) -> np.ndarray:
        out = np.zeros((m, len(teams)), dtype=float)
        for j, (mus, vals, p0, ab, gk) in enumerate(per_team):
            if len(vals):
                pdraw = rng.beta(np.maximum(mus * ab, 1e-3),
                                 np.maximum((1 - mus) * ab, 1e-3), size=(m, len(vals)))
                gp = rng.binomial(int(N_GAMES), pdraw) / N_GAMES
                lost = rng.random((m, len(vals))) < p0[None, :]
                gp = np.where(lost, 0.0, gp)
                mix_mean = (1 - p0) * mus
                out[:, j] += ((gp - mix_mean[None, :]) * vals[None, :]).sum(axis=1)
            if gk is not None:
                mu_g, val_g = gk
                ab_g, p0_g = goalie_par["ab"], goalie_par["p0"]
                pg = rng.beta(max(mu_g * ab_g, 1e-3), max((1 - mu_g) * ab_g, 1e-3),
                              size=m)
                gpg = rng.binomial(int(N_GAMES), pg) / N_GAMES
                lostg = rng.random(m) < p0_g
                gpg = np.where(lostg, 0.0, gpg)
                out[:, j] += (gpg - (1 - p0_g) * mu_g) * (val_g * k_elo / 82.0)
        return out

    return extra_noise
