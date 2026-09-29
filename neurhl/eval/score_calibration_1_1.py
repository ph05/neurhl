"""Score NeurHL 1.1 C1 (PLAN_NeurHL_1_1): the calibrated stat-sheet counts
against the frozen ones, on the same committed forecasts.

The calibrated forecast is a fixed function of numbers already frozen in each
pregame forecast file and of configs/calibration_1_1.json, which was committed
before the first 2026-27 game:

  team SOG     NB2(mu = sog_home / sog_away, r_hat)   vs  NB2(mu, r = 40)
  team goals   Poisson(m_t * M * (raw / M) ** b_hat)   vs  Poisson(m_t * raw)
               raw = goals_home_raw / goals_away_raw, m_t = goal_mult, M from
               configs/live_goal_calibration.json (target: regulation goals)

Forecast selection, validation and results reuse eval/score_live_g_2027.py, so
exactly the games PLAN_NeurHL4 LIVE counts are counted here (primary: pregame).
Primary tests, one Holm family of two, run once after the last regular-season
game: mean paired log-score difference per team-game (calibrated minus
frozen, higher is better), week-block bootstrap 95% interval (9,999 draws,
seed 711). Also reported: randomised-PIT 80% coverage for SOG, goal deciles,
and a team-clustered sensitivity. Interim scorecards are descriptive.

Writes neurhl/output/live/scorecard_1_1_2027.json.
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import special, stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PROJ  # noqa: E402
import eval.score_live_g_2027 as L  # noqa: E402

CARD = "neurhl/output/live/scorecard_1_1_2027.json"
CAL = "neurhl/configs/calibration_1_1.json"
GOALCAL = "neurhl/configs/live_goal_calibration.json"
R_FROZEN = 40.0


def nb_logpmf(y, mu, r):
    return stats.nbinom.logpmf(y, r, r / (r + mu))


def pois_logpmf(y, mu):
    return y * np.log(mu) - mu - special.gammaln(y + 1)


def pit_cov(y, mu, r, seed=711):
    p = r / (r + mu)
    f1, f0 = stats.nbinom.cdf(y, r, p), stats.nbinom.cdf(y - 1, r, p)
    u = f0 + np.random.default_rng(seed).random(len(y)) * (f1 - f0)
    return float(np.mean((u > 0.1) & (u < 0.9))) if len(y) else None


def holm(ps: dict) -> dict:
    items = sorted((p, k) for k, p in ps.items() if p is not None)
    out, run = {}, 0.0
    for i, (p, k) in enumerate(items):
        run = max(run, min(1.0, (len(items) - i) * p))
        out[k] = run
    return out


def boot_p(d, weeks, draws=L.DRAWS, seed=L.SEED):
    """Two-sided week-block bootstrap p for mean(d) = 0 (centred resampling)."""
    d = np.asarray(d, float)
    keys, inv = np.unique(np.asarray(weeks), return_inverse=True)
    idx = np.random.default_rng(seed).integers(0, len(keys), size=(draws, len(keys)))
    den = np.bincount(inv, minlength=len(keys))[idx].sum(1)
    num = np.bincount(inv, weights=d - d.mean(), minlength=len(keys))[idx].sum(1)
    b = num / den
    return float((np.sum(np.abs(b) >= abs(d.mean())) + 1) / (draws + 1))


def team_cluster_se(d, teams):
    s = pd.Series(np.asarray(d, float)).groupby(np.asarray(teams)).sum()
    n = len(d)
    return float(np.sqrt((s ** 2).sum()) / n) if n else None


def compare(new, old, weeks, teams, final):
    d = np.asarray(new) - np.asarray(old)
    if not len(d):
        return {"n_team_games": 0}
    out = {"n_team_games": int(len(d)), "mean_new": float(np.mean(new)),
           "mean_frozen": float(np.mean(old)), "diff": float(d.mean()),
           "se_team_clustered": team_cluster_se(d, teams)}
    if len(set(weeks)) >= 2:
        out["ci95_week_bootstrap"] = L.week_bootstrap([d], weeks)[0]
        if final:
            out["p_week_bootstrap"] = boot_p(d, weeks)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Score NeurHL 1.1 C1 (PLAN_NeurHL_1_1).")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--as-of")
    ap.add_argument("--root", default=str(PROJ))
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    as_of = dt.date.fromisoformat(a.as_of) if a.as_of else dt.datetime.now(L.ET).date()
    cal = json.loads((root / CAL).read_text())
    r_hat, b_hat = cal["sog_dispersion"]["r_hat"], cal["goal_slope"]["b_hat"]
    M = json.loads((root / GOALCAL).read_text())["M"]

    res = L.load_results(root, as_of)
    first, _, _, prefix = L.history(root)
    cand = L.candidates(root, first, prefix)
    _, pending = L.uncommitted(root, first)
    c = L.validate(cand, res, L.api_starts(root, sorted(set(res.date)), a.offline), root)
    g = L.choose(c, "pregame").merge(res.rename(columns={"date": "game_date"}), on="game_id")
    tabs = L.actuals(root)
    final = as_of > L.LAST and len(res) >= L.N_GAMES
    card = {"as_of": as_of.isoformat(), "status": "final" if final else "interim",
            "note": ("final: the one inference PLAN_NeurHL_1_1 C1 declares" if final else
                     "interim: descriptive only; inference once, after 2027-04-10"),
            "r_hat": r_hat, "b_hat": b_hat, "M": M, "games": int(len(g))}

    # goals: every chosen pregame forecast with a result
    teams = np.r_[g.home, g.away] if len(g) else np.array([])
    weeks = np.r_[g.week, g.week] if len(g) else np.array([])
    y = np.r_[g.reg_h, g.reg_a].astype(float) if len(g) else np.array([])
    raw = np.r_[g.goals_home_raw, g.goals_away_raw].astype(float) if len(g) else np.array([])
    m = np.r_[g.goal_mult, g.goal_mult].astype(float) if len(g) else np.array([])
    mu_old, mu_new = m * raw, m * M * (raw / M) ** b_hat
    ok = np.isfinite(y) & np.isfinite(mu_old) & (mu_old > 0)
    gl = compare(pois_logpmf(y[ok], mu_new[ok]), pois_logpmf(y[ok], mu_old[ok]),
                 weeks[ok], teams[ok], final)
    if ok.sum() >= 50:
        q = pd.qcut(mu_old[ok], 5, labels=False, duplicates="drop")
        dq = pd.DataFrame({"q": q, "frozen": mu_old[ok], "cal": mu_new[ok], "obs": y[ok]}).groupby("q").mean()
        gl["quintiles"] = {k: [round(float(v), 3) for v in dq[k]] for k in dq}
    card["goals"] = gl

    # SOG: needs the ingested team-game table
    t = tabs.get("tgx")
    if t is None or not len(g):
        card["sog"] = {"available": False}
    else:
        tt = t[~t.game_id.isin(tabs["degraded"])]
        side = {s: tt[tt.is_home == s].drop_duplicates("game_id").set_index("game_id")["sogf"]
                for s in (1, 0)}
        ys = np.r_[g.game_id.map(side[1]), g.game_id.map(side[0])].astype(float)
        mu = np.r_[g.sog_home, g.sog_away].astype(float)
        k = np.isfinite(ys) & np.isfinite(mu) & (mu > 0)
        sg = compare(nb_logpmf(ys[k], mu[k], r_hat), nb_logpmf(ys[k], mu[k], R_FROZEN),
                     weeks[k], teams[k], final)
        sg.update({"available": True, "coverage80_frozen": pit_cov(ys[k], mu[k], R_FROZEN),
                   "coverage80_cal": pit_cov(ys[k], mu[k], r_hat)})
        card["sog"] = sg
    # C1b (A6): team xG, gamma(k_hat) against the frozen gamma(9); its own family of one
    calb = root / "neurhl/configs/calibration_1_1b.json"
    if calb.exists() and t is not None and len(g):
        k_hat = json.loads(calb.read_text())["k_hat"]
        tt = t[~t.game_id.isin(tabs["degraded"])]
        side = {s_: tt[tt.is_home == s_].drop_duplicates("game_id").set_index("game_id")["xgf_all"]
                for s_ in (1, 0)}
        yx = np.r_[g.game_id.map(side[1]), g.game_id.map(side[0])].astype(float)
        mx = np.r_[g.xgf_home, g.xgf_away].astype(float)
        kx = np.isfinite(yx) & np.isfinite(mx) & (mx > 0) & (yx > 0)
        gl_ = lambda y_, m_, k_: stats.gamma.logpdf(y_, k_, scale=m_ / k_)  # noqa: E731
        xg = compare(gl_(yx[kx], mx[kx], k_hat), gl_(yx[kx], mx[kx], 9.0), weeks[kx], teams[kx], final)
        cov = lambda k_: float(np.mean((lambda u: (u > 0.1) & (u < 0.9))(stats.gamma.cdf(yx[kx], k_, scale=mx[kx] / k_))))  # noqa: E731
        xg.update({"available": True, "k_hat": k_hat,
                   "coverage80_frozen": cov(9.0) if kx.any() else None,
                   "coverage80_cal": cov(k_hat) if kx.any() else None})
        card["xg"] = xg
    else:
        card["xg"] = {"available": False}
    if final:
        card["holm"] = holm({"sog": card["sog"].get("p_week_bootstrap"),
                             "goals": card["goals"].get("p_week_bootstrap")})
    out = root / CARD
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(card, indent=1))
    print(json.dumps(card, indent=1)[:1500])
    return 0


if __name__ == "__main__":
    sys.exit(main())
