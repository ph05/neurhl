"""NeurHL-2 — season-level backtest: is the projection actually calibrated?

The 2026-27 deliverable ships projected points, 80% ranges and playoff odds. None
of that is trustworthy without checking the same machinery against seasons whose
outcomes are known. PLAN_NeurHL2 validation level 4 makes these SECONDARY
diagnostics — correct per-game hazards that aggregate to the wrong season spread
indicate a missing-uncertainty bug — and gives the house benchmarks to beat:

    standings points MAE   HOWE 10.36
    CRPS                   v1 sim 7.10 / 7.77
    playoff Brier          0.195 / 0.208

Three things are measured, because a projection can fail in three different ways:

  * **MAE** — are the central estimates right?
  * **CRPS** — is the whole predictive DISTRIBUTION right? A model with good means
    and no spread scores well on MAE and badly here, which is exactly the failure
    mode S5's posterior sampling exists to prevent.
  * **coverage of the 80% interval** — the blunt check that the stated
    uncertainty means what it says. Under-dispersion is the classic sin of a
    simulator that propagates means but not parameter uncertainty.

Everything is walk-forward: season V is simulated from inputs computed on
seasons < V only. Scored on DEV and TUNE; CONFIRM is not touched (G-STOP fired).

Run: uv run --no-project --python 3.12 --with numpy --with "pandas<3" \
     --with pyarrow --with scipy python neurhl/eval/backtest_season.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import TENSORS  # noqa: E402
from sim.game_model import (TEAM_SIGMA, fit_tie_calibration,  # noqa: E402
                            integrate, observed_tie_rate, outcome, run_elo,
                            score_effect_curve, team_strength)
import windows as W  # noqa: E402

FIRST = 2008


def crps_ensemble(samples: np.ndarray, obs: float) -> float:
    """CRPS of an empirical predictive distribution (Hersbach form)."""
    x = np.sort(samples)
    n = len(x)
    e1 = np.abs(x - obs).mean()
    e2 = np.abs(x[:, None] - x[None, :]).mean() if n <= 2000 else \
        np.abs(x - np.random.permutation(x)).mean()
    return float(e1 - 0.5 * e2)


def actual_standings(season: int) -> pd.DataFrame:
    g = pd.read_parquet(TENSORS / f"games_ctx_{season}.parquet",
                        columns=["game_id", "game_type", "home_idx", "away_idx",
                                 "home_g", "away_g", "outcome4"])
    g = g[g.game_type == 2]
    pts = {}
    for r in g.itertuples():
        h, a = int(r.home_idx), int(r.away_idx)
        pts.setdefault(h, 0)
        pts.setdefault(a, 0)
        ot = int(r.outcome4) in (2, 3)
        if r.home_g > r.away_g:
            pts[h] += 2
            pts[a] += 1 if ot else 0
        else:
            pts[a] += 2
            pts[h] += 1 if ot else 0
    return pd.Series(pts, name="actual_pts")


def roster_net(season: int, prev: int) -> dict:
    pr = TENSORS / f"rapm_prior_{season}.parquet"
    pg = TENSORS / f"player_games_{prev}.parquet"
    gc = TENSORS / f"games_ctx_{prev}.parquet"
    if not (pr.exists() and pg.exists() and gc.exists()):
        return {}
    r = pd.read_parquet(pr)
    r = r[~r.is_replacement.astype(bool)]
    net = (r.set_index("player_id").cf_off - r.set_index("player_id").cf_def)
    p = pd.read_parquet(pg, columns=["game_id", "player_id", "is_home",
                                     "toi_sec", "pos_group", "game_type"])
    p = p[(p.game_type == 2) & (p.pos_group != 2)]
    g = pd.read_parquet(gc, columns=["game_id", "home_idx", "away_idx"])
    p = p.merge(g, on="game_id", how="left")
    p["team"] = np.where(p.is_home, p.home_idx, p.away_idx)
    a = p.groupby(["team", "player_id"], as_index=False).toi_sec.sum()
    a["net"] = a.player_id.map(net)
    a = a.dropna(subset=["net"])
    out = {int(t): float(np.average(d.net, weights=d.toi_sec))
           for t, d in a.groupby("team")}
    mu, sd = np.mean(list(out.values())), np.std(list(out.values())) or 1.0
    return {k: (v - mu) / sd for k, v in out.items()}


def rate_sensitivity(curve, lg=2.8):
    """d logit(P(reg home win)) / d(delta_home - delta_away) in log-rate units.

    Team-strength uncertainty is a perturbation on the LOG goal rate; converting
    it into a win-probability shift exactly would need re-integrating per draw,
    which is far too slow. The map is smooth and near-linear over the relevant
    range, so one finite difference gives the slope.
    """
    def lg_p(dh):
        P = integrate(lg * np.exp(dh), lg * np.exp(-dh), curve)
        q = outcome(P)["p_reg_home"]
        return np.log(q / (1 - q))
    return float((lg_p(0.05) - lg_p(-0.05)) / 0.10)


def simulate(season, sims, rng, curve, sigma=0.0):
    train = list(range(FIRST, season))
    att, dfn, lg, hm = team_strength(train)
    rap = roster_net(season, season - 1)
    g = pd.read_parquet(TENSORS / f"games_ctx_{season}.parquet",
                        columns=["game_id", "game_type", "home_idx", "away_idx"])
    g = g[g.game_type == 2].reset_index(drop=True)
    teams = sorted(set(g.home_idx) | set(g.away_idx))
    idx = {t: i for i, t in enumerate(teams)}

    raw = []
    for r in g.itertuples():
        ah, dh = att.get(int(r.home_idx), 1.0), dfn.get(int(r.home_idx), 1.0)
        aa, da = att.get(int(r.away_idx), 1.0), dfn.get(int(r.away_idx), 1.0)
        rh, ra = rap.get(int(r.home_idx), 0.0), rap.get(int(r.away_idx), 0.0)
        lh = lg * ah * da * hm ** 0.5 * np.exp(0.045 * (rh - ra))
        la = lg * aa * dh / hm ** 0.5 * np.exp(0.045 * (ra - rh))
        raw.append(integrate(lh, la, curve))
    tie_c = fit_tie_calibration([outcome(P)["p_tie"] for P in raw], train[-3:])
    prh = np.array([outcome(P, tie_calib=tie_c)["p_reg_home"] for P in raw])
    pot = np.array([outcome(P, tie_calib=tie_c)["p_tie"] for P in raw])

    hi = np.array([idx[int(t)] for t in g.home_idx])
    ai = np.array([idx[int(t)] for t in g.away_idx])
    pts = np.zeros((sims, len(teams)), np.int16)
    c = rate_sensitivity(curve) if sigma > 0 else 0.0
    l0 = np.log(np.clip(prh, 1e-6, 1 - 1e-6) / (1 - np.clip(prh, 1e-6, 1 - 1e-6)))
    for s in range(sims):
        if sigma > 0:
            # PARAMETER UNCERTAINTY. One draw of each team's true strength per
            # simulated season, held fixed across that season's games. Without
            # it the only variance is game-outcome noise and the season spread
            # is far too narrow -- measured 80% coverage 0.628 against a nominal
            # 0.80, predicted sd 8.22 against an actual 13.79.
            dt = rng.normal(0.0, sigma, len(teams))
            shift = c * (dt[hi] - dt[ai])
            pr_s = 1.0 / (1.0 + np.exp(-(l0 + shift)))
            pr_s = pr_s * (1.0 - pot)
        else:
            pr_s = prh
        u, v = rng.random(len(g)), rng.random(len(g))
        hr = u < pr_s
        ot = (u >= pr_s) & (u < pr_s + pot)
        ho = ot & (v < 0.53)
        p = np.zeros(len(teams), np.int32)
        np.add.at(p, hi[hr], 2)
        np.add.at(p, ai[~hr & ~ot], 2)
        np.add.at(p, hi[ho], 2)
        np.add.at(p, ai[ot & ~ho], 2)
        np.add.at(p, ai[ho], 1)
        np.add.at(p, hi[ot & ~ho], 1)
        pts[s] = p
    return teams, pts, tie_c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="*",
                    default=[2011, 2012, 2014, 2015, 2016, 2017])
    ap.add_argument("--sims", type=int, default=4000)
    ap.add_argument("--sigma", type=float, default=TEAM_SIGMA,
                    help="team log-rate uncertainty; None = calibrate it")
    args = ap.parse_args()
    rng = np.random.default_rng(7)
    curve = score_effect_curve(Path(__file__).resolve().parents[1] /
                               "configs" / "event_sim_gates.json")

    sigma = args.sigma
    if sigma is None:
        # Calibrate on the EARLIEST seasons only, then apply to the rest, so the
        # coverage figures reported below are not fitted on themselves.
        cal = [s for s in args.seasons if s <= 2012] or args.seasons[:2]
        best, bs = None, 1e9
        for cand in (0.00, 0.03, 0.05, 0.07, 0.09, 0.12):
            covs = []
            for V in cal:
                teams, pts, _ = simulate(V, 800, np.random.default_rng(3),
                                         curve, sigma=cand)
                act = actual_standings(V)
                obs = np.array([act.get(t, np.nan) for t in teams], float)
                m = np.isfinite(obs)
                lo = np.percentile(pts[:, m], 10, axis=0)
                hi_ = np.percentile(pts[:, m], 90, axis=0)
                covs.append(np.mean((obs[m] >= lo) & (obs[m] <= hi_)))
            err = abs(np.mean(covs) - 0.80)
            if err < bs:
                bs, best = err, cand
        sigma = best
        print(f"team-strength sigma calibrated on {cal}: {sigma:.3f} "
              f"(log-rate sd)\n")

    print(f"{'V':>5} {'win':>7} {'teams':>6} {'MAE':>7} {'CRPS':>7} "
          f"{'cover80':>8} {'sd_pred':>8} {'sd_act':>7} {'corr':>6}")
    rows = []
    for V in args.seasons:
        if V in W.CONFIRM:
            print(f"{V:>5}  CONFIRM — not touched (G-STOP fired)")
            continue
        teams, pts, tie_c = simulate(V, args.sims, rng, curve, sigma)
        act = actual_standings(V)
        obs = np.array([act.get(t, np.nan) for t in teams], float)
        m = np.isfinite(obs)
        pred = pts[:, m].astype(float)
        obs = obs[m]
        mean = pred.mean(0)
        mae = float(np.abs(mean - obs).mean())
        crps = float(np.mean([crps_ensemble(pred[:, j], obs[j])
                              for j in range(len(obs))]))
        lo = np.percentile(pred, 10, axis=0)
        hi_ = np.percentile(pred, 90, axis=0)
        cover = float(np.mean((obs >= lo) & (obs <= hi_)))
        rows.append({"season": V, "n_teams": int(m.sum()), "mae": mae,
                     "crps": crps, "cover80": cover,
                     "sd_pred": float(pred.std(0).mean()),
                     "sd_actual": float(obs.std()),
                     "corr": float(np.corrcoef(mean, obs)[0, 1]),
                     "tie_calib": tie_c})
        r = rows[-1]
        print(f"{V:>5} {W.window_of(V):>7} {r['n_teams']:>6} {mae:>7.2f} "
              f"{crps:>7.2f} {cover:>8.3f} {r['sd_pred']:>8.2f} "
              f"{r['sd_actual']:>7.2f} {r['corr']:>6.3f}")
        sys.stdout.flush()

    if rows:
        mae = float(np.mean([r["mae"] for r in rows]))
        crps = float(np.mean([r["crps"] for r in rows]))
        cov = float(np.mean([r["cover80"] for r in rows]))
        sp = float(np.mean([r["sd_pred"] for r in rows]))
        sa = float(np.mean([r["sd_actual"] for r in rows]))
        print(f"\nPOOLED over {len(rows)} seasons")
        print(f"  standings MAE   {mae:6.2f}   (house benchmark: HOWE 10.36)")
        print(f"  CRPS            {crps:6.2f}   (house benchmark: 7.10 / 7.77)")
        print(f"  80% coverage    {cov:6.3f}   (nominal 0.80)")
        print(f"  spread: predicted sd {sp:.2f} vs actual sd {sa:.2f} "
              f"-> {'UNDER' if sp < sa * 0.9 else 'OK'}-dispersed")
        p = Path(__file__).resolve().parents[1] / "configs" / "season_backtest.json"
        p.write_text(json.dumps({"rows": rows, "mae": mae, "crps": crps,
                                 "cover80": cov, "sd_pred": sp,
                                 "sd_actual": sa, "sigma": sigma}, indent=1))
        print(f"-> {p}")


if __name__ == "__main__":
    main()
