"""NeurHL — mechanical gate evaluation against PLAN_NeurHL.md (game-level).

Reads the committed per-season prediction artifacts (preds/game_preds_<T>.csv,
preds/game_val_<T>.csv), fits temperature ONLY on each T's val season (P1),
and evaluates G1 (vs v1), G3 (calibration), G4 (vs Tier-0) exactly as locked.
Season-level gates G2/G5 are appended by eval/backtest_season.py after the
Tier-3 sim exists. Writes/updates neurhl/output/params_neurhl.json with full
per-season records; every decision carries rule_eval so the battery can
re-assert rule == outcome mechanically.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import NOUT  # noqa: E402

TUNE_T = [2012, 2013, 2014, 2015, 2016, 2017]
G4_T = [2014, 2015, 2016, 2017]
P4 = ["p_home_reg", "p_away_reg", "p_home_extra", "p_away_extra"]


def fit_temperature(p4: np.ndarray, y: np.ndarray) -> float:
    """1-param temperature on log-probs, grid + golden refinement."""
    logp = np.log(np.clip(p4, 1e-12, 1))

    def nll(tau):
        z = logp / tau
        z = z - z.max(axis=1, keepdims=True)
        q = np.exp(z)
        q = q / q.sum(axis=1, keepdims=True)
        return -np.log(np.clip(q[np.arange(len(y)), y], 1e-12, 1)).mean()

    taus = np.linspace(0.5, 2.5, 41)
    best = min(taus, key=nll)
    for _ in range(30):
        step = 0.01
        cands = [best - step, best, best + step]
        best = min(cands, key=nll)
    return float(best)


def apply_temperature(p4: np.ndarray, tau: float) -> np.ndarray:
    z = np.log(np.clip(p4, 1e-12, 1)) / tau
    z = z - z.max(axis=1, keepdims=True)
    q = np.exp(z)
    return q / q.sum(axis=1, keepdims=True)


def home_ll(p4: np.ndarray, y: np.ndarray) -> float:
    ph = np.clip(p4[:, 0] + p4[:, 2], 1e-9, 1 - 1e-9)
    yh = ((y == 0) | (y == 2)).astype(float)
    return float(-(yh * np.log(ph) + (1 - yh) * np.log(1 - ph)).mean())


def reliability(p4: np.ndarray, y: np.ndarray, bins: int = 10):
    ph = p4[:, 0] + p4[:, 2]
    yh = ((y == 0) | (y == 2)).astype(float)
    edges = np.quantile(ph, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, ph, "right") - 1, 0, bins - 1)
    xs, ys, ns = [], [], []
    for b in range(bins):
        m = idx == b
        if m.sum() > 5:
            xs.append(ph[m].mean())
            ys.append(yh[m].mean())
            ns.append(m.sum())
    xs, ys, ns = np.array(xs), np.array(ys), np.array(ns)
    slope = float(np.polyfit(xs, ys, 1, w=ns)[0])
    ece = float(np.average(np.abs(xs - ys), weights=ns))
    return slope, ece


def v1_per_game() -> pd.DataFrame:
    """Pre-game v1 Elo home-win probability keyed by game_id."""
    import backtest as B
    import engine as E
    from common import PROJ, TENSORS
    v1 = json.loads((PROJ / "output" / "params.json").read_text())
    preds, _, _ = E.run_elo(B.g, K=v1["K"], H=v1["H"], phi_s=v1["phi_s"])
    preds = preds.copy()
    preds["date"] = pd.to_datetime(B.g.date).dt.strftime("%Y-%m-%d").values
    gs = pd.read_parquet(TENSORS / "_edacache" / "game_summary.parquet")
    gs = gs[gs.game_type == 2].copy()
    remap = {"PHX": "UTA", "ARI": "UTA", "ATL": "WPG"}
    gs["home_m"] = gs.home.replace(remap)
    gs["away_m"] = gs.away.replace(remap)
    return gs[["game_id", "date", "home_m", "away_m"]].merge(
        preds[["date", "home", "away", "e_home"]].rename(
            columns={"home": "home_m", "away": "away_m"}),
        on=["date", "home_m", "away_m"], how="inner")[["game_id", "e_home"]]


def significance_vs_v1(per_game: pd.DataFrame) -> dict:
    """Paired test of NeurHL vs v1 log loss (the standard the weak 0.002
    threshold does NOT meet: 0.002 is only ~0.75 SE over this many games)."""
    from scipy import stats
    v1 = v1_per_game()
    m = per_game.merge(v1, on="game_id", how="inner")
    y = m.y.to_numpy()

    def nll(p):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    d = nll(m.p_nn.to_numpy()) - nll(m.e_home.to_numpy())
    n = len(d)
    se = float(d.std(ddof=1) / np.sqrt(n))
    t = float(d.mean() / se)
    # per-season means must be computed from the aligned difference vector,
    # not by re-running nll inside groupby (y there is the full-length array)
    m = m.assign(_d=d)
    per_season = m.groupby("season")._d.mean()
    cl_se = float(per_season.std(ddof=1) / np.sqrt(len(per_season)))
    return {"n_games": int(n), "mean_diff_nn_minus_v1": float(d.mean()),
            "se": se, "t": t, "p_two_sided": float(stats.norm.sf(abs(t)) * 2),
            "nn_better": bool(d.mean() < 0),
            "significant_at_05": bool(d.mean() < 0 and abs(t) > 1.96),
            "clustered_t_by_season": float(per_season.mean() / cl_se)
            if cl_se > 0 else float("nan"),
            "per_season_diff": {str(k): round(float(v), 5)
                                for k, v in per_season.items()},
            "threshold_needed_for_p05": float(nll(m.e_home.to_numpy()).mean()
                                              - 1.96 * se)}


def main():
    base = json.loads((NOUT / "baselines_tune.json").read_text())
    v1_ll = base["game_logloss"]
    tier0 = base["tier0"]["logloss_pooled_2014_2017"]

    rows, taus = {}, {}
    all_p, all_y = [], []
    g4_p, g4_y = [], []
    sig_rows = []
    for T in TUNE_T:
        pred = pd.read_csv(NOUT / "preds" / f"game_preds_{T}.csv")
        val = pd.read_csv(NOUT / "preds" / f"game_val_{T}.csv")
        tau = fit_temperature(val[P4].to_numpy(), val.outcome4.to_numpy())
        taus[str(T)] = tau
        p4 = apply_temperature(pred[P4].to_numpy(), tau)
        y = pred.outcome4.to_numpy()
        sig_rows.append(pd.DataFrame({
            "game_id": pred.game_id, "season": T,
            "p_nn": p4[:, 0] + p4[:, 2],
            "y": ((y == 0) | (y == 2)).astype(int)}))
        rows[str(T)] = {"n": len(y), "tau": tau,
                        "nn_home_ll": home_ll(p4, y),
                        "v1_home_ll": v1_ll[str(T)]}
        all_p.append(p4)
        all_y.append(y)
        if T in G4_T:
            g4_p.append(p4)
            g4_y.append(y)
    P = np.concatenate(all_p)
    Y = np.concatenate(all_y)
    pooled = home_ll(P, Y)
    pooled_g4 = home_ll(np.concatenate(g4_p), np.concatenate(g4_y))
    wins = sum(rows[str(T)]["nn_home_ll"] < rows[str(T)]["v1_home_ll"]
               for T in TUNE_T)
    slope, ece = reliability(P, Y)

    gates = {
        "G1": {"rule": "pooled 2012-2017 home ll <= 0.67385 AND wins >= 4/6",
               "nn_pooled": pooled, "v1_pooled": v1_ll["pooled_2012_2017"],
               "wins": int(wins),
               "rule_eval": bool(pooled <= 0.67385 and wins >= 4),
               "pass": bool(pooled <= 0.67385 and wins >= 4)},
        "G3": {"rule": "slope in [0.90,1.10] AND ECE <= 0.015",
               "slope": slope, "ece": ece,
               "rule_eval": bool(0.90 <= slope <= 1.10 and ece <= 0.015),
               "pass": bool(0.90 <= slope <= 1.10 and ece <= 0.015)},
        "G4": {"rule": "pooled 2014-2017 home ll <= 0.67392",
               "nn_pooled_2014_2017": pooled_g4, "tier0": tier0,
               "rule_eval": bool(pooled_g4 <= 0.67392),
               "pass": bool(pooled_g4 <= 0.67392)},
    }
    sig = significance_vs_v1(pd.concat(sig_rows, ignore_index=True))
    gates["S1"] = {
        "rule": "paired log-loss vs v1 significantly better at p<0.05",
        **sig,
        "rule_eval": sig["significant_at_05"],
        "pass": sig["significant_at_05"]}

    out_p = NOUT / "params_neurhl.json"
    blob = json.loads(out_p.read_text()) if out_p.exists() else {}
    blob.setdefault("prereg", "PLAN_NeurHL.md @ commit 8c8e6d8")
    blob["temperatures"] = taus
    blob["game_per_season"] = rows
    blob.setdefault("gates", {}).update(gates)
    out_p.write_text(json.dumps(blob, indent=1))
    print(json.dumps({"pooled": pooled, "pooled_2014_2017": pooled_g4,
                      "wins_vs_v1": wins, "slope": slope, "ece": ece,
                      "G1": gates["G1"]["pass"], "G3": gates["G3"]["pass"],
                      "G4": gates["G4"]["pass"],
                      "SIGNIFICANCE_vs_v1": {
                          "diff": round(sig["mean_diff_nn_minus_v1"], 5),
                          "t": round(sig["t"], 2),
                          "p": sig["p_two_sided"],
                          "significant_better": sig["significant_at_05"],
                          "need_logloss_below": round(
                              sig["threshold_needed_for_p05"], 5)}}, indent=1))


if __name__ == "__main__":
    main()
