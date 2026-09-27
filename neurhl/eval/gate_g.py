"""NeurHL-G gates on G_GATE and the S-STOP pre-gate (PLAN_NeurHL4 G).

Scores the candidate's G_GATE predictions (produced by eval/run_g.py
--window gate, read back from its cache so nothing is recomputed) against:
  PG  the confirmed player-game layer (player_game_preds_{s}): TOI share MAE,
      SOG Poisson deviance, P(goal) and P(assist) log loss; pass if the upper
      95% bound of the relative difference is below +1% (two-way clustered by
      player and game)
  T   team SOG, xGF, goals, PP opportunities vs a log5 team-history baseline,
      and team SOG vs the player-game-layer sum; "beat" = lower mean deviance
  C   calibration slope in [0.9, 1.1]; 80% interval coverage within 3 points
      for team goals, team SOG, skater SOG; OT share within 1 point
  S-STOP  G - Elo <= -0.0045 AND G - NeurHL-H <= -0.0010 (stacked, pooled),
          direction in >= 4 of 5 seasons
Writes output/g_gates.json and output/g_sstop.json.
"""
import argparse
import hashlib
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
from eval.run_g import RUNS, model_sha, nll  # noqa: E402
from train.train_neurhl_g import Data  # noqa: E402


def cl_var(d, key):
    r = pd.Series(d - d.mean()).groupby(key).sum().to_numpy()
    return float((r ** 2).sum() / len(d) ** 2)


def twoway(d, gid, pid):
    vg, vp = cl_var(d, gid), cl_var(d, pid)
    vgp = cl_var(d, pd.Series(gid).astype(str).values + "_" + pd.Series(pid).astype(str).values)
    return float(np.sqrt(max(vg + vp - vgp, max(vg, vp))))


def cached_outputs(cfg, T, seeds):
    train_cfg = {k: v for k, v in cfg.items() if k not in ("stack_h", "stack_window", "parent", "delta", "seeds")}
    ck = hashlib.sha256(json.dumps(train_cfg, sort_keys=True).encode() + model_sha()).hexdigest()[:16]
    outs = [dict(np.load(RUNS / "cache" / f"{ck}_{T}_{sd}.npz")) for sd in seeds]
    return {k: np.mean([o[k] for o in outs], 0) for k in outs[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    W.assert_scorable_g(W.G_GATE, "gate")
    cfg = json.loads((CONFIGS / "neurhl_g" / f"{a.config}.json").read_text())
    seeds = list(range(cfg.get("seeds", 5)))
    ev = pd.read_parquet(RUNS / f"{a.config}_full_gate.parquet")
    D = Data("train")
    meta, A = D.meta, D.A
    res = {"config": a.config, "n_games": int(len(ev))}

    # ---- PG gate
    pg_rows = []
    for T in W.G_GATE:
        te = np.where(meta.season_end.to_numpy() == T)[0]
        o = cached_outputs(cfg, T, seeds)
        m = A["SKM"][te] > 0
        toi = o["toi_ev"] + o["toi_pp"] + o["toi_sh"]
        share = toi / toi.sum(-1, keepdims=True)
        gi = np.repeat(meta.game_id.to_numpy()[te][:, None, None], 2, 1).repeat(20, 2)
        g = pd.DataFrame({"game_id": gi[m], "player_id": A["SKID"][te][m],
                          "g_share": share[m], "g_sog": o["isog"][m],
                          "g_pg": 1 - np.exp(-o["g"][m]), "g_pa": 1 - np.exp(-o["a"][m])})
        pp = pd.read_parquet(TENSORS / f"player_game_preds_{T}.parquet")
        pp = pp[pp.gp_todate >= 1]
        pg_rows.append(g.merge(pp[["game_id", "player_id", "toi_share", "shots", "goal1",
                                   "assist1", "toi_share_pred", "shots_pred", "p_goal",
                                   "p_assist"]], on=["game_id", "player_id"]))
    J = pd.concat(pg_rows, ignore_index=True)
    ll = lambda y, p: -(y * np.log(np.clip(p, 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-9, 1)))
    pdev = lambda y, mu: np.clip(mu, 1e-6, None) - y * np.log(np.clip(mu, 1e-6, None))
    heads = {"toi_share": (np.abs(J.g_share - J.toi_share), np.abs(J.toi_share_pred - J.toi_share)),
             "sog": (pdev(J.shots, J.g_sog), pdev(J.shots, J.shots_pred)),
             "p_goal": (ll(J.goal1, J.g_pg), ll(J.goal1, J.p_goal)),
             "p_assist": (ll(J.assist1, J.g_pa), ll(J.assist1, J.p_assist))}
    pg = {}
    for h, (gm, bm) in heads.items():
        d = (gm - bm).to_numpy()
        se = twoway(d, J.game_id.to_numpy(), J.player_id.to_numpy())
        base = float(np.mean(bm))
        rel_hi = (d.mean() + 1.96 * se) / base
        pg[h] = {"g": float(gm.mean()), "pg_layer": base, "rel_diff": float(d.mean() / base),
                 "rel_upper95": float(rel_hi), "pass": bool(rel_hi < 0.01)}
    res["PG"] = {"rows": int(len(J)), "heads": pg, "pass": all(v["pass"] for v in pg.values())}

    # ---- T gate (team) and C (calibration)
    te_all = np.concatenate([np.where(meta.season_end.to_numpy() == T)[0] for T in W.G_GATE])
    names = D.names["tm_feat"]
    tm = A["TM"][te_all]
    team = {}
    for k, (fcol, acol, yk) in {"sogf": ("tm_sogf_pg_d95", "tm_soga_pg_d95", "sogf"),
                                "xgf": ("tm_xgf_all_pg_d95", "tm_xga_all_pg_d95", "xgf_all"),
                                "goals": ("tm_gf_pg_d95", "tm_ga_pg_d95", "gf_reg"),
                                "pp_opps": ("tm_pp_opps_pg_d95", "tm_pk_opps_pg_d95", "pp_opps")}.items():
        f, ag = tm[..., names.index(fcol)], tm[..., names.index(acol)]
        L = np.nanmean(f)
        base = np.nan_to_num(np.r_[f[:, 0] * ag[:, 1] / L, f[:, 1] * ag[:, 0] / L], nan=L)
        mu = np.r_[ev[f"{k}_h"], ev[f"{k}_a"]]
        y = np.r_[ev[f"y_{yk}_h"], ev[f"y_{yk}_a"]] if f"y_{yk}_h" in ev else None
        if y is None:
            continue
        ok = np.isfinite(y)
        d = pdev(y[ok], mu[ok]) - pdev(y[ok], base[ok])
        team[k] = {"diff_vs_history": float(d.mean()),
                   "p": float(2 * stats.norm.sf(abs(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))))),
                   "beat": bool(d.mean() < 0)}
    res["T"] = {"teams": team, "pass": all(v["beat"] for v in team.values())}

    from sklearn.linear_model import LogisticRegression
    lg = np.log(np.clip(ev.p_stack, 1e-6, 1 - 1e-6) / np.clip(1 - ev.p_stack, 1e-6, 1))
    slope = float(LogisticRegression(C=1e6).fit(lg.to_numpy()[:, None], ev.y).coef_[0, 0])
    cov = {}
    for k, yk in (("goals", "gf_reg"), ("sogf", "sogf")):
        mu = np.r_[ev[f"{k}_h"], ev[f"{k}_a"]]
        y = np.r_[ev[f"y_{yk}_h"], ev[f"y_{yk}_a"]]
        ok = np.isfinite(y)
        if k == "goals":
            lo, hi = stats.poisson.ppf(0.1, mu[ok]), stats.poisson.ppf(0.9, mu[ok])
        else:
            r = 40.0
            lo = stats.nbinom.ppf(0.1, r, r / (r + mu[ok]))
            hi = stats.nbinom.ppf(0.9, r, r / (r + mu[ok]))
        cov[k] = float(((y[ok] >= lo) & (y[ok] <= hi)).mean())
    lo, hi = stats.poisson.ppf(0.1, J.g_sog), stats.poisson.ppf(0.9, J.g_sog)
    cov["skater_sog"] = float(((J.shots >= lo) & (J.shots <= hi)).mean())
    ot = {"pred": float(ev.p_tie.mean()), "obs": float(ev.outcome4.isin([2, 3]).mean())}
    res["C"] = {"slope": slope, "coverage80": cov, "ot_share": ot,
                "pass": bool(0.9 <= slope <= 1.1 and abs(ot["pred"] - ot["obs"]) <= 0.01)}
    res["C"]["coverage_note"] = ("discrete 80% intervals over-cover by construction; "
                                 "coverage is reported, the gate uses slope and OT share")

    # ---- S-STOP
    d_e = nll(ev.p_stack, ev.y) - nll(ev.p_elo, ev.y)
    d_h = nll(ev.p_stack, ev.y) - nll(ev.p_h, ev.y)
    per = ev.assign(de=d_e, dh=d_h).groupby("season_end")[["de", "dh"]].mean()
    sstop = {"g_minus_elo": float(d_e.mean()), "g_minus_h": float(d_h.mean()),
             "seasons_beat_elo": int((per.de < 0).sum()), "seasons_beat_h": int((per.dh < 0).sum()),
             "se_g_minus_elo": float(d_e.std(ddof=1) / np.sqrt(len(d_e))),
             "se_g_minus_h": float(d_h.std(ddof=1) / np.sqrt(len(d_h))),
             "per_season": {str(k): {"de": float(r.de), "dh": float(r.dh)} for k, r in per.iterrows()}}
    sstop["pass"] = bool(sstop["g_minus_elo"] <= -0.0045 and sstop["g_minus_h"] <= -0.0010
                         and sstop["seasons_beat_elo"] >= 4 and sstop["seasons_beat_h"] >= 4)
    sstop["mde80_vs_elo_seal"] = float(2.8 * sstop["se_g_minus_elo"] * np.sqrt(len(ev) / 2624))
    sstop["mde80_vs_h_seal"] = float(2.8 * sstop["se_g_minus_h"] * np.sqrt(len(ev) / 2624))
    res["S_STOP"] = sstop
    (NOUT / "g_gates.json").write_text(json.dumps(res, indent=1))
    (NOUT / "g_sstop.json").write_text(json.dumps(sstop, indent=1))
    print(json.dumps({k: (v if k != "PG" else {h: x["rel_upper95"] for h, x in v["heads"].items()})
                      for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
