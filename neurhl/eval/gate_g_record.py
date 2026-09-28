"""NeurHL-G gate record correction (PLAN_NeurHL4 A2): the declared quantities
that eval/gate_g.py did not compute, from the same saved G_GATE predictions.

  C   randomised PIT (Czado et al. 2009) and central 80% coverage for team
      regulation goals, team SOG and skater SOG, with the predictive
      distributions gate_g.py uses for its intervals (Poisson; negative
      binomial with r = 40 for team SOG). Declared rule: coverage within
      3 points of 0.80.
  T(b) team SOG and team regulation goals against the player-game-layer sum
      over dressed skaters (SOG: sum of predicted shots; goals: sum of
      -log(1 - P(goal)), the Poisson mean implied by P(goal >= 1)), on
      Poisson deviance.
  CRPS team SOG and goals for NeurHL-G, the log5 team-history baseline and the
      player-game-layer sum, under the same predictive distributions.
Nothing is refitted and no decision in the FREEZE section changes; the
predictions are read from the runner's cache exactly as gate_g.py reads them.
Writes output/g_gates_record.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import CONFIGS, NOUT, TENSORS  # noqa: E402
import windows as W  # noqa: E402
from eval.gate_g import cached_outputs  # noqa: E402
from eval.run_g import RUNS  # noqa: E402
from train.train_neurhl_g import Data  # noqa: E402

R_SOG = 40.0
SEED = 711


def dist(kind, mu):
    mu = np.clip(mu, 1e-6, None)
    if kind == "poisson":
        return stats.poisson(mu)
    return stats.nbinom(R_SOG, R_SOG / (R_SOG + mu))


def rpit(y, d, rng):
    lo = np.where(y > 0, d.cdf(y - 1), 0.0)
    return lo + rng.uniform(size=len(y)) * (d.cdf(y) - lo)


def crps_counts(y, kind, mu, kmax=160):
    """Discrete CRPS: sum over k of (F(k) - 1{y <= k})^2."""
    k = np.arange(kmax + 1)
    mu = np.clip(mu, 1e-6, None)[:, None]
    if kind == "poisson":
        F = stats.poisson.cdf(k[None, :], mu)
    else:
        F = stats.nbinom.cdf(k[None, :], R_SOG, R_SOG / (R_SOG + mu))
    ind = (y[:, None] <= k[None, :]).astype(float)
    return ((F - ind) ** 2).sum(1)


def pdev(y, mu):
    mu = np.clip(mu, 1e-6, None)
    return mu - y * np.log(mu)


def paired(d):
    se = d.std(ddof=1) / np.sqrt(len(d))
    return {"mean_diff": float(d.mean()), "se": float(se),
            "p": float(2 * stats.norm.sf(abs(d.mean() / se))), "better": bool(d.mean() < 0)}


def main():
    cfg = json.loads((CONFIGS / "neurhl_g" / "g1.json").read_text())
    seeds = list(range(cfg.get("seeds", 5)))
    ev = pd.read_parquet(RUNS / "g1_full_gate.parquet")
    D = Data("train")
    meta, A = D.meta, D.A
    rng = np.random.default_rng(SEED)

    # skater rows joined to the confirmed player-game layer, as gate_g.py builds them
    rows = []
    for T in W.G_GATE:
        te = np.where(meta.season_end.to_numpy() == T)[0]
        o = cached_outputs(cfg, T, seeds)
        m = A["SKM"][te] > 0
        gi = np.repeat(meta.game_id.to_numpy()[te][:, None, None], 2, 1).repeat(20, 2)
        side = np.broadcast_to(np.array([0, 1])[None, :, None], m.shape)
        g = pd.DataFrame({"game_id": gi[m], "side": side[m], "player_id": A["SKID"][te][m],
                          "g_sog": o["isog"][m]})
        pp = pd.read_parquet(TENSORS / f"player_game_preds_{T}.parquet")
        g = g.merge(pp[["game_id", "player_id", "shots", "shots_pred", "p_goal", "gp_todate"]],
                    on=["game_id", "player_id"], how="left")
        rows.append(g)
    J = pd.concat(rows, ignore_index=True)

    # C: randomised PIT and central 80% coverage
    out = {"config": "g1", "n_games": int(len(ev)), "seed": SEED, "C_declared": {}}
    pit_hist = {}
    for name, (k, yk, kind) in {"team_goals": ("goals", "gf_reg", "poisson"),
                                "team_sog": ("sogf", "sogf", "nbinom")}.items():
        mu = np.r_[ev[f"{k}_h"], ev[f"{k}_a"]]
        y = np.r_[ev[f"y_{yk}_h"], ev[f"y_{yk}_a"]]
        ok = np.isfinite(y) & np.isfinite(mu)
        u = rpit(y[ok], dist(kind, mu[ok]), rng)
        cov = float(((u >= 0.1) & (u <= 0.9)).mean())
        out["C_declared"][name] = {"n": int(ok.sum()), "coverage80": cov,
                                   "pass": bool(abs(cov - 0.8) <= 0.03)}
        pit_hist[name] = np.histogram(u, bins=10, range=(0, 1))[0].tolist()
    sk = J[J.gp_todate >= 1].dropna(subset=["shots"])
    u = rpit(sk.shots.to_numpy(), dist("poisson", sk.g_sog.to_numpy()), rng)
    cov = float(((u >= 0.1) & (u <= 0.9)).mean())
    out["C_declared"]["skater_sog"] = {"n": int(len(sk)), "coverage80": cov,
                                       "pass": bool(abs(cov - 0.8) <= 0.03)}
    pit_hist["skater_sog"] = np.histogram(u, bins=10, range=(0, 1))[0].tolist()
    out["C_declared"]["pass"] = all(v["pass"] for v in out["C_declared"].values()
                                    if isinstance(v, dict))
    out["pit_histograms"] = pit_hist

    # team sums of the player-game layer over dressed skaters (all dressed rows
    # with a layer prediction; skaters without one contribute nothing)
    J["lam_goal"] = -np.log(np.clip(1 - J.p_goal, 1e-9, 1))
    S = J.groupby(["game_id", "side"])[["shots_pred", "lam_goal"]].sum(min_count=1).reset_index()
    cover = J.groupby(["game_id", "side"]).shots_pred.apply(lambda s: s.notna().mean()).rename("cover")
    S = S.merge(cover.reset_index(), on=["game_id", "side"])

    names = D.names["tm_feat"]
    te_all = np.concatenate([np.where(meta.season_end.to_numpy() == T)[0] for T in W.G_GATE])
    tm = A["TM"][te_all]
    gids = meta.game_id.to_numpy()[te_all]
    ev = ev.set_index("game_id").loc[gids].reset_index()
    tb, crps_out = {}, {}
    for k, (fcol, acol, yk, pgcol, kind) in {
            "sogf": ("tm_sogf_pg_d95", "tm_soga_pg_d95", "sogf", "shots_pred", "nbinom"),
            "goals": ("tm_gf_pg_d95", "tm_ga_pg_d95", "gf_reg", "lam_goal", "poisson")}.items():
        f, ag = tm[..., names.index(fcol)], tm[..., names.index(acol)]
        L = np.nanmean(f)
        base = np.nan_to_num(np.r_[f[:, 0] * ag[:, 1] / L, f[:, 1] * ag[:, 0] / L], nan=L)
        mu = np.r_[ev[f"{k}_h"], ev[f"{k}_a"]]
        y = np.r_[ev[f"y_{yk}_h"], ev[f"y_{yk}_a"]]
        key = pd.DataFrame({"game_id": np.r_[gids, gids],
                            "side": np.r_[np.zeros(len(gids), int), np.ones(len(gids), int)]})
        pg = key.merge(S, on=["game_id", "side"], how="left")
        ok = np.isfinite(y) & pg[pgcol].notna().to_numpy() & (pg.cover.to_numpy() >= 0.9)
        pgm = pg[pgcol].to_numpy()
        tb[k] = {"n_team_games": int(ok.sum()),
                 "vs_pg_layer_sum": paired(pdev(y[ok], mu[ok]) - pdev(y[ok], pgm[ok]))}
        c_g = crps_counts(y[ok].astype(int), kind, mu[ok])
        c_b = crps_counts(y[ok].astype(int), kind, base[ok])
        c_p = crps_counts(y[ok].astype(int), kind, pgm[ok])
        crps_out[k] = {"neurhl_g": float(c_g.mean()), "log5_history": float(c_b.mean()),
                       "pg_layer_sum": float(c_p.mean()),
                       "vs_history": paired(c_g - c_b), "vs_pg_layer_sum": paired(c_g - c_p)}
    out["T_b_declared"] = tb
    out["CRPS_declared"] = crps_out
    out["note"] = ("Computed after the FREEZE from the same saved predictions; "
                   "PLAN_NeurHL4 A2 records the correction. No decision changes.")
    (NOUT / "g_gates_record.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "pit_histograms"}, indent=1))


if __name__ == "__main__":
    main()
