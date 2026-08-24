"""NeurHL-3 — M1 gates: score player-game predictions against the recorded
own-EWMA baselines with the PG1-PG4 instruments (PLAN_NeurHL3).

Baselines, exactly as declared:
  toi_share  — the player's own EWMA(alpha=0.1) toi_share renormalised over
               the dressed skaters (the recorded backtest_player construction);
  shots      — own EWMA(alpha=0.1) shots;
  goal1/assist1 — SHRUNK own EWMA: p = (n*ewma + k*base)/(n+k), base = the
               train-window positional rate, k chosen per vantage on season
               V-1 ONLY from {5, 10, 20, 40, 80}.

Instruments: per-row paired differences (model loss − baseline loss) with
game-clustered robust SE; season-clustered t (df = S−1) over per-season means;
Holm across the four heads for PG-ALL. Gate population = rows where the
baseline is defined (gp_todate >= 1); coverage reported. Writes
configs/player_game_eval.json.

Run: uv run --no-project --python 3.12 --with numpy --with "pandas<3" \
     --with pyarrow --with scipy python neurhl/eval/backtest_player_game.py \
     --window eval
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import TENSORS  # noqa: E402
import windows as W  # noqa: E402

CFG = Path(__file__).resolve().parents[1] / "configs"
K_GRID = (5, 10, 20, 40, 80)


def cluster_p(d: np.ndarray, gid: np.ndarray) -> tuple:
    """Mean of d with game-cluster-robust SE -> (mean, z, p)."""
    df = pd.DataFrame({"d": d, "g": gid})
    n = len(df)
    mean = float(df.d.mean())
    cl = df.groupby("g").d.agg(["sum", "count"])
    resid = cl["sum"] - cl["count"] * mean
    se = float(np.sqrt((resid ** 2).sum()) / n)
    z = mean / se if se > 0 else 0.0
    return mean, z, float(2 * stats.norm.sf(abs(z)))


def season_t(per_season: pd.Series) -> tuple:
    s = per_season.to_numpy(float)
    if len(s) < 2:
        return float(np.mean(s)), 1.0
    t = np.mean(s) / (np.std(s, ddof=1) / np.sqrt(len(s)))
    return float(np.mean(s)), float(2 * stats.t.sf(abs(t), df=len(s) - 1))


def logloss(y, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def pois_dev(y, mu, eps=1e-12):
    mu = np.clip(mu, eps, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(np.clip(y, eps, None) / mu), 0.0)
    return 2.0 * (term - (y - mu))


def shrunk_baseline(d: pd.DataFrame, tgt: str, ew_col: str, base_rate: float,
                    k: float) -> np.ndarray:
    n = d.gp_todate.to_numpy(float)
    ew = d[ew_col].fillna(base_rate).to_numpy(float)
    return (n * ew + k * base_rate) / np.maximum(n + k, 1e-9)


def pick_k(prev: pd.DataFrame, tgt: str, ew_col: str, base: float) -> int:
    best, bk = np.inf, K_GRID[0]
    y = prev[tgt].to_numpy(float)
    for k in K_GRID:
        p = shrunk_baseline(prev, tgt, ew_col, base, k)
        ll = float(logloss(y, p).mean())
        if ll < best:
            best, bk = ll, k
    return bk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", choices=["dev", "tune", "eval", "confirm"],
                    required=True)
    args = ap.parse_args()
    seasons = {"dev": W.PLAYER_DEV, "tune": W.PLAYER_TUNE,
               "eval": W.PLAYER_EVAL, "confirm": W.PLAYER_CONFIRM}[args.window]
    W.assert_scorable_player(seasons, args.window)

    heads = {"toi_share": [], "shots": [], "goal1": [], "assist1": []}
    per_season = {h: {} for h in heads}
    rows_out = []
    for v in seasons:
        p = TENSORS / f"player_game_preds_{v}.parquet"
        if not p.exists():
            raise SystemExit(f"missing predictions for vantage {v}")
        d = pd.read_parquet(p)
        gate = d[d.gp_todate >= 1].copy()
        cov = len(gate) / len(d)
        # baseline toi: own EWMA renormalised over rows with a defined EWMA
        tot = gate.groupby(["game_id", "team"]).toi_share_ewa1.transform("sum")
        gate["toi_base"] = gate.toi_share_ewa1 / tot.clip(lower=1e-9)
        gate["shots_base"] = gate.shots_ewa1
        # shrunk binaries, k picked on V-1 predictions file if present
        prev_p = TENSORS / f"player_game_preds_{v - 1}.parquet"
        prev = pd.read_parquet(prev_p) if prev_p.exists() else gate
        prev = prev[prev.gp_todate >= 1]
        for tgt, ew in (("goal1", "goal_rate_ewa1"),
                        ("assist1", "assist_rate_ewa1")):
            base = float(prev[tgt].mean())
            k = pick_k(prev, tgt, ew, base)
            gate[f"{tgt}_base"] = shrunk_baseline(gate, tgt, ew, base, k)

        gid = gate.game_id.to_numpy()
        diffs = {
            "toi_share": (np.abs(gate.toi_share_pred - gate.toi_share)
                          - np.abs(gate.toi_base - gate.toi_share)).to_numpy(),
            "shots": (pois_dev(gate.shots.to_numpy(), gate.shots_pred.to_numpy())
                      - pois_dev(gate.shots.to_numpy(),
                                 gate.shots_base.to_numpy())),
            "goal1": (logloss(gate.goal1.to_numpy(), gate.p_goal.to_numpy())
                      - logloss(gate.goal1.to_numpy(),
                                gate.goal1_base.to_numpy())),
            "assist1": (logloss(gate.assist1.to_numpy(),
                                gate.p_assist.to_numpy())
                        - logloss(gate.assist1.to_numpy(),
                                  gate.assist1_base.to_numpy())),
        }
        for h, darr in diffs.items():
            heads[h].append(pd.DataFrame({"d": darr, "g": gid}))
            per_season[h][v] = float(np.mean(darr))
        rows_out.append({
            "season": v, "n_gate": int(len(gate)), "coverage": round(cov, 4),
            "toi_mae_model": round(float(np.abs(gate.toi_share_pred
                                                - gate.toi_share).mean()), 6),
            "toi_mae_base": round(float(np.abs(gate.toi_base
                                               - gate.toi_share).mean()), 6),
            "shots_dev_model": round(float(pois_dev(
                gate.shots.to_numpy(), gate.shots_pred.to_numpy()).mean()), 5),
            "shots_dev_base": round(float(pois_dev(
                gate.shots.to_numpy(), gate.shots_base.to_numpy()).mean()), 5),
            "goal_ll_model": round(float(logloss(
                gate.goal1.to_numpy(), gate.p_goal.to_numpy()).mean()), 5),
            "goal_ll_base": round(float(logloss(
                gate.goal1.to_numpy(), gate.goal1_base.to_numpy()).mean()), 5),
            "assist_ll_model": round(float(logloss(
                gate.assist1.to_numpy(), gate.p_assist.to_numpy()).mean()), 5),
            "assist_ll_base": round(float(logloss(
                gate.assist1.to_numpy(), gate.assist1_base.to_numpy()).mean()), 5),
        })
        r = rows_out[-1]
        print(f"{v}: n={r['n_gate']:,} cov={cov:.3f}  "
              f"toi {r['toi_mae_model']:.5f} vs {r['toi_mae_base']:.5f}  "
              f"shots {r['shots_dev_model']:.4f} vs {r['shots_dev_base']:.4f}  "
              f"goal {r['goal_ll_model']:.4f} vs {r['goal_ll_base']:.4f}  "
              f"assist {r['assist_ll_model']:.4f} vs {r['assist_ll_base']:.4f}")
        sys.stdout.flush()

    gates, raw_ps = {}, {}
    for h in heads:
        alld = pd.concat(heads[h], ignore_index=True)
        mean, z, p = cluster_p(alld.d.to_numpy(), alld.g.to_numpy())
        smean, sp = season_t(pd.Series(per_season[h]))
        better = mean < 0
        raw_ps[h] = p
        gates[h] = {"mean_diff": round(mean, 6), "z": round(z, 3),
                    "p_game_clustered": round(p, 6),
                    "per_season_mean": round(smean, 6),
                    "p_season_clustered": round(sp, 4),
                    "direction_agrees": bool(smean < 0),
                    "better": bool(better)}
    # Holm across the four heads
    order = sorted(raw_ps, key=lambda h: raw_ps[h])
    m = len(order)
    holm_ok = True
    for i, h in enumerate(order):
        adj = raw_ps[h] * (m - i)
        gates[h]["p_holm"] = round(min(adj, 1.0), 6)
        passed = (gates[h]["better"] and adj < 0.05
                  and gates[h]["direction_agrees"] and holm_ok)
        gates[h]["pass"] = bool(passed)
        if not passed:
            holm_ok = False   # Holm is sequential
    pg_all = all(gates[h]["pass"] for h in gates)

    out = {"window": args.window, "seasons": seasons, "per_season": rows_out,
           "gates": gates, "PG_ALL": bool(pg_all)}
    (CFG / f"player_game_eval_{args.window}.json").write_text(
        json.dumps(out, indent=1))
    print(f"\nPG-ALL ({args.window}): {'PASS' if pg_all else 'FAIL'}")
    for h in ("toi_share", "shots", "goal1", "assist1"):
        g = gates[h]
        print(f"  {h:<10} mean_diff {g['mean_diff']:+.6f}  p_holm "
              f"{g['p_holm']:.5f}  season-dir "
              f"{'ok' if g['direction_agrees'] else 'NO'}  "
              f"-> {'PASS' if g['pass'] else 'FAIL'}")
    return 0 if pg_all else 1


if __name__ == "__main__":
    sys.exit(main())
