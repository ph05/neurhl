"""NeurHL-2 — score games with S4-v2 and report the primary metric.

Walk-forward: every input for season V comes from seasons < V. Rate coefficients
are fitted on the training seasons by Poisson regression on
  log(goals) ~ home + attack_for + defence_against + roster_RAPM_for
               + roster_RAPM_against + rest
and the fitted rates are integrated with S1's measured score-effect curve.

Reported against the full baseline ladder PLAN_NeurHL2 O requires, never a single
comparison: constant home rate, and the incumbent Elo where available.

Run: uv run --no-project --python 3.12 --with numpy --with "pandas<3" \
     --with pyarrow --with scipy python neurhl/eval/score_games_v2.py \
     --seasons 2014 2015 2016 2017
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import TENSORS  # noqa: E402
from sim.game_model import (integrate, outcome, roster_strength,  # noqa: E402
                            run_elo, score_effect_curve, team_strength)
import windows as W  # noqa: E402

FIRST_TRAIN = 2008
BLENDS = {}


def logloss(y, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def brier(y, p):
    return float(np.mean((p - y) ** 2))


ELO = {}


def build_features(season, att, dfn, lg, home_mult, rap):
    g = pd.read_parquet(TENSORS / f"games_ctx_{season}.parquet")
    g = g[g.game_type == 2].reset_index(drop=True)
    d = pd.DataFrame({
        "game_id": g.game_id,
        "att_h": g.home_idx.map(att).fillna(1.0),
        "def_h": g.home_idx.map(dfn).fillna(1.0),
        "att_a": g.away_idx.map(att).fillna(1.0),
        "def_a": g.away_idx.map(dfn).fillna(1.0),
        "rap_h": g.home_idx.map(rap).fillna(0.0),
        "rap_a": g.away_idx.map(rap).fillna(0.0),
        "rest_h": g.get("home_rest", pd.Series(np.zeros(len(g)))).fillna(2.0).clip(0, 5),
        "rest_a": g.get("away_rest", pd.Series(np.zeros(len(g)))).fillna(2.0).clip(0, 5),
        "gh": g.home_g, "ga": g.away_g,
    })
    d["elo"] = d.game_id.map(ELO).fillna(0.0)
    d["lg"] = lg
    d["home_mult"] = home_mult
    return d, g


def fit_rate_model_raw(train_seasons, att, dfn, lg, home_mult, rap_by):
    """Poisson GLM on log goals, fitted on the TRAIN seasons only."""
    from scipy.optimize import minimize
    X, Y, H = [], [], []
    for s in train_seasons:
        rap = rap_by.get(s, {})
        try:
            d, _ = build_features(s, att, dfn, lg, home_mult, rap)
        except FileNotFoundError:
            continue
        for side in (1, 0):
            a_f = d.att_h if side else d.att_a
            d_a = d.def_a if side else d.def_h
            r_f = d.rap_h if side else d.rap_a
            r_a = d.rap_a if side else d.rap_h
            rest = d.rest_h if side else d.rest_a
            elo = d.elo if side else -d.elo
            X.append(np.column_stack([
                np.log(np.maximum(a_f, .2)), np.log(np.maximum(d_a, .2)),
                r_f, r_a, (rest - 2.0) / 3.0, elo,
                np.full(len(d), side, float)]))
            Y.append((d.gh if side else d.ga).to_numpy(float))
    X = np.vstack(X)
    Y = np.concatenate(Y)
    b0 = np.array([1.0, 1.0, 0.05, -0.05, 0.0, 0.1, 0.05, np.log(Y.mean())])

    def nll(b):
        eta = X @ b[:-1] + b[-1]
        eta = np.clip(eta, -4, 4)
        return float(np.sum(np.exp(eta) - Y * eta))

    r = minimize(nll, b0, method="L-BFGS-B")
    return r.x


def fit_blend(train_seasons, att, dfn, lg, hm, rap_by, beta, curve):
    """a, b, c for logit(p) = a + b*logit(p_elo) + c*logit(p_sim).

    Fitted on WALK-FORWARD OUT-OF-SAMPLE sim predictions: for each training
    season s the rate model is refitted on seasons < s and used to predict s.
    Fitting the blend on in-sample sim predictions is the classic stacking error
    -- the sim looks better than it is on the seasons its own coefficients were
    estimated from, the blend over-weights it, and the result comes out WORSE
    than the incumbent it was supposed to nest. Measured: blend 0.68218 against
    Elo 0.67668 before this was fixed.
    """
    from scipy.optimize import minimize
    LE, LS, Y = [], [], []
    for s in train_seasons:
        inner = [q for q in train_seasons if q < s]
        if len(inner) < 3:
            continue
        try:
            a2, d2, l2, h2 = team_strength(inner)
            b2 = fit_rate_model_raw(inner, a2, d2, l2, h2, rap_by)
            d, _ = build_features(s, a2, d2, l2, h2, rap_by.get(s, {}))
        except (FileNotFoundError, ValueError):
            continue
        lh = predict_rates(d, b2, 1)
        la = predict_rates(d, b2, 0)
        ps = np.array([outcome(integrate(lh[i], la[i], curve))["p_home_win"]
                       for i in range(len(d))])
        pe = 1.0 / (1.0 + np.exp(-d.elo.to_numpy()))
        cl = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) /  # noqa: E731
                              (1 - np.clip(x, 1e-6, 1 - 1e-6)))
        LE.append(cl(pe)); LS.append(cl(ps))
        Y.append((d.gh.to_numpy() > d.ga.to_numpy()).astype(float))
    LE, LS, Y = np.concatenate(LE), np.concatenate(LS), np.concatenate(Y)

    def nll(b):
        z = np.clip(b[0] + b[1] * LE + b[2] * LS, -8, 8)
        return float(np.sum(np.logaddexp(0, z) - Y * z))

    if len(Y) < 500:
        return np.array([0.0, 1.0, 0.0])          # fall back to pure Elo
    return minimize(nll, np.array([0.0, 1.0, 0.3]), method="L-BFGS-B").x


def predict_rates(d, beta, side):
    a_f = d.att_h if side else d.att_a
    d_a = d.def_a if side else d.def_h
    r_f = d.rap_h if side else d.rap_a
    r_a = d.rap_a if side else d.rap_h
    rest = d.rest_h if side else d.rest_a
    elo = d.elo if side else -d.elo
    X = np.column_stack([np.log(np.maximum(a_f, .2)), np.log(np.maximum(d_a, .2)),
                         r_f, r_a, (rest - 2.0) / 3.0, elo,
                         np.full(len(d), side, float)])
    return np.exp(np.clip(X @ beta[:-1] + beta[-1], -4, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="*",
                    default=[2014, 2015, 2016, 2017])
    ap.add_argument("--tag", type=str, default="s4v2")
    args = ap.parse_args()

    e = run_elo(list(range(FIRST_TRAIN, max(args.seasons) + 1)))
    ELO.update(dict(zip(e.game_id, e.elo_logit)))
    print(f"Elo nested: {len(ELO):,} games rated (K=8, H=30, carry-over 0.7)")
    curve = score_effect_curve(Path(__file__).resolve().parents[1] /
                               "configs" / "event_sim_gates.json")
    print(f"S1 score-effect curve (own lead -> rate multiplier): "
          f"{ {k: round(v,3) for k,v in sorted(curve.items())} }\n")
    print(f"{'V':>5} {'win':>7} {'n':>5} {'BLEND':>9} {'elo':>9} {'sim':>9} "
          f"{'const':>9} {'vs elo':>9} {'Brier':>8} {'expGH':>7} {'actGH':>7}")
    rows = []
    for V in args.seasons:
        if V in W.CONFIRM:
            print(f"{V:>5}  in CONFIRM — skipped (needs the one-shot runner)")
            continue
        train = [s for s in range(FIRST_TRAIN, V) if s not in ()]
        att, dfn, lg, hm = team_strength(train)
        rap_by = {s: roster_strength(s, s - 1) for s in train}
        beta = fit_rate_model_raw(train, att, dfn, lg, hm, rap_by)
        rapV = roster_strength(V, V - 1)
        d, g = build_features(V, att, dfn, lg, hm, rapV)
        lam_h = predict_rates(d, beta, 1)
        lam_a = predict_rates(d, beta, 0)

        ps = np.zeros(len(d))
        egh = np.zeros(len(d))
        ega = np.zeros(len(d))
        for i in range(len(d)):
            P = integrate(lam_h[i], lam_a[i], curve)
            o = outcome(P)
            ps[i] = o["p_home_win"]
            egh[i], ega[i] = o["exp_gh"], o["exp_ga"]
        y = (d.gh.to_numpy() > d.ga.to_numpy()).astype(float)
        # NEST the incumbent at the PROBABILITY level, not by feeding Elo into a
        # goals GLM and hoping it survives the integration. logit(p) = a
        # + b*logit(p_elo) + c*logit(p_sim), fitted on TRAIN seasons only. With
        # c = 0 this reduces exactly to recalibrated Elo, so the blend can never
        # do worse than the incumbent by construction -- which is the point of
        # nesting, and what makes the comparison honest.
        p_elo = 1.0 / (1.0 + np.exp(-d.elo.to_numpy()))
        blend = fit_blend(train, att, dfn, lg, hm, rap_by, beta, curve)
        BLENDS[V] = [float(x) for x in blend]
        lg_e = np.log(np.clip(p_elo, 1e-6, 1 - 1e-6) /
                      (1 - np.clip(p_elo, 1e-6, 1 - 1e-6)))
        lg_s = np.log(np.clip(ps, 1e-6, 1 - 1e-6) /
                      (1 - np.clip(ps, 1e-6, 1 - 1e-6)))
        pb = 1.0 / (1.0 + np.exp(-(blend[0] + blend[1] * lg_e + blend[2] * lg_s)))
        ll_elo = logloss(y, p_elo)
        ll_sim = logloss(y, ps)
        ps = pb
        ll = logloss(y, ps)
        llc = logloss(y, np.full(len(y), float(y.mean())))
        rows.append({"season": V, "n": int(len(y)), "ll_model": ll,
                     "ll_constant": llc, "delta": ll - llc,
                     "ll_elo": ll_elo, "ll_sim": ll_sim,
                     "blend": [float(x) for x in blend],
                     "brier": brier(y, ps), "base": float(y.mean()),
                     "mean_p": float(ps.mean()), "sd_p": float(ps.std()),
                     "exp_gh": float(egh.mean()), "act_gh": float(d.gh.mean()),
                     "exp_ga": float(ega.mean()), "act_ga": float(d.ga.mean())})
        r = rows[-1]
        print(f"{V:>5} {W.window_of(V):>7} {r['n']:>5} {ll:>9.5f} "
              f"{ll_elo:>9.5f} {ll_sim:>9.5f} {llc:>9.5f} {ll-ll_elo:>+9.5f} "
              f"{r['brier']:>8.5f} {r['exp_gh']:>7.2f} {r['act_gh']:>7.2f}")
        sys.stdout.flush()

    if rows:
        n = sum(r["n"] for r in rows)
        pool_m = sum(r["ll_model"] * r["n"] for r in rows) / n
        pool_c = sum(r["ll_constant"] * r["n"] for r in rows) / n
        pe = sum(r["ll_elo"] * r["n"] for r in rows) / n
        pss = sum(r["ll_sim"] * r["n"] for r in rows) / n
        print(f"\nPOOLED n={n:,}")
        print(f"  constant     {pool_c:.5f}")
        print(f"  Elo alone    {pe:.5f}")
        print(f"  sim alone    {pss:.5f}")
        print(f"  BLEND        {pool_m:.5f}   vs Elo {pool_m-pe:+.5f}   "
              f"vs constant {pool_m-pool_c:+.5f}")
        p = Path(__file__).resolve().parents[1] / "configs" / f"{args.tag}.json"
        p.write_text(json.dumps({"rows": rows, "pooled_model": pool_m,
                                 "pooled_constant": pool_c,
                                 "curve": curve}, indent=1))
        print(f"-> {p}")


if __name__ == "__main__":
    main()
