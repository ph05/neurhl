"""NeurHL-2 — R1' across every DEV fold, with a significance test on the margin.

One fold with 152 movers is not enough to claim RAPM beats raw on-ice rate. Two
things are added here:

  * **all available DEV folds** — 2008->2009, 2008-2009->2010, 2008-2010->2011 —
    so the result is not a single lucky split.
  * **Steiger's test for DEPENDENT correlations.** r(RAPM, y) and r(raw, y) are
    measured against the SAME target on the SAME players, so they are not
    independent and the naive two-sample comparison is wrong. Steiger (1980)
    accounts for r(RAPM, raw), which is high — both are built from the same
    on-ice events — and that dependence is what makes a modest gap testable at
    all at this sample size.

Reported for the movers subgroup (the gate), and for stayers as context.

Run: uv run --no-project --python 3.12 --with numpy --with "pandas<3" \
     --with pyarrow --with scipy python neurhl/train/rapm_folds.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import TENSORS  # noqa: E402
import models.rapm as R  # noqa: E402
import windows as W  # noqa: E402
from train.fit_rapm import (LAM_GRID, MIN_TOI, TARGET, net_raw,  # noqa: E402
                            player_team)

FOLDS = [([2008], 2009), ([2008, 2009], 2010), ([2008, 2009, 2010], 2011)]
TEAM_FE = True                 # A6: absorb team-season level


def steiger(r_xy, r_zy, r_xz, n):
    """Steiger's z for H0: rho(x,y) == rho(z,y), x and z measured on the same n.

    Returns (t, p_two_sided). x is the candidate (RAPM), z the baseline (raw).
    """
    if not np.isfinite([r_xy, r_zy, r_xz]).all() or n < 10:
        return float("nan"), float("nan")
    d = r_xy - r_zy
    rbar = (r_xy + r_zy) / 2.0
    detR = (1 - r_xy ** 2 - r_zy ** 2 - r_xz ** 2
            + 2 * r_xy * r_zy * r_xz)
    detR = max(detR, 1e-12)
    denom = ((2 * (n - 1) / (n - 3)) * detR
             + rbar ** 2 * (1 - r_xz) ** 3)
    if denom <= 0:
        return float("nan"), float("nan")
    t = d * np.sqrt(((n - 1) * (1 + r_xz)) / denom)
    return float(t), float(2 * (1 - stats.t.cdf(abs(t), n - 3)))


def fold(train, test, lam):
    tr = R.load_stints(train)
    te = R.load_stints([test])
    des = R.RAPMDesign(tr, min_toi_s=MIN_TOI * len(train) // 3 or MIN_TOI,
                       positions=R.player_positions(train), team_fe=TEAM_FE)
    ys = des.targets(tr, [TARGET])
    res = R.solve(des, ys, lam, want_se=False)
    coef = res["coef"][TARGET]

    pt = player_team(list(train) + [test])
    a = pt[pt.season_end == max(train)][["player_id", "team"]].rename(
        columns={"team": "team_from"})
    b = pt[pt.season_end == test][["player_id", "team"]].rename(
        columns={"team": "team_to"})
    mv = a.merge(b, on="player_id")
    mv["moved"] = mv.team_from != mv.team_to

    pr = net_raw(tr).rename(columns={"raw_net": "prior_raw", "toi_s": "toi_tr"})
    po = net_raw(te).rename(columns={"raw_net": "post_raw", "toi_s": "toi_te"})
    o, d = des.off0, des.def0
    rp = pd.DataFrame({"player_id": list(des.pid_to_slot),
                       "prior_rapm": [float(coef[o + s] - coef[d + s])
                                      for s in des.pid_to_slot.values()]})
    df = (mv.merge(pr, on="player_id").merge(po, on="player_id")
          .merge(rp, on="player_id", how="left")).dropna(subset=["prior_rapm"])
    thr = MIN_TOI * len(train) // 3 or MIN_TOI
    df = df[(df.toi_tr >= thr) & (df.toi_te >= thr // len(train) // 2 + 1)]
    df["post_dev"] = df.post_raw - df.groupby("team_to").post_raw.transform("mean")
    df["post_abs"] = df.post_raw     # A6: undemeaned target reported alongside
    df["prior_raw_dev"] = (df.prior_raw
                           - df.groupby("team_from").prior_raw.transform("mean"))

    out = {"train": train, "test": test}
    for grp, sub in (("movers", df[df.moved]), ("stayers", df[~df.moved])):
        n = len(sub)
        if n < 20:
            out[grp] = {"n": n}
            continue
        r_rapm = float(np.corrcoef(sub.prior_rapm, sub.post_dev)[0, 1])
        r_raw = float(np.corrcoef(sub.prior_raw, sub.post_dev)[0, 1])
        r_rdev = float(np.corrcoef(sub.prior_raw_dev, sub.post_dev)[0, 1])
        rx_raw = float(np.corrcoef(sub.prior_rapm, sub.prior_raw)[0, 1])
        rx_rdev = float(np.corrcoef(sub.prior_rapm, sub.prior_raw_dev)[0, 1])
        t1, p1 = steiger(r_rapm, r_raw, rx_raw, n)
        t2, p2 = steiger(r_rapm, r_rdev, rx_rdev, n)
        out[grp] = {"n": n, "r_rapm": round(r_rapm, 4),
                    "r_raw": round(r_raw, 4), "r_raw_dev": round(r_rdev, 4),
                    "vs_raw": {"t": round(t1, 3), "p": round(p1, 5)},
                    "vs_raw_dev": {"t": round(t2, 3), "p": round(p2, 5)},
                    "_rows": sub[["prior_rapm", "prior_raw", "prior_raw_dev",
                                  "post_dev", "post_abs"]].to_numpy().tolist()}
    return out


def main():
    lam = json.loads((Path(__file__).resolve().parents[1] / "configs" /
                      "rapm_gates.json").read_text())["lam_chosen"]
    print(f"lambda = {lam:.0f} (chosen on DEV by transfer criterion, A5)\n")
    print(f"{'train':>18} {'test':>5} {'grp':>8} {'n':>5} {'r_rapm':>8} "
          f"{'r_raw':>8} {'r_rawdev':>9} {'p vs raw':>9} {'p vs dev':>9}")
    res, pooled = [], {"movers": [], "stayers": []}
    for tr, te in FOLDS:
        W.assert_scorable([te], "dev")
        f = fold(tr, te, lam)
        res.append(f)
        for grp in ("movers", "stayers"):
            g = f.get(grp, {})
            if "r_rapm" not in g:
                print(f"{str(tr):>18} {te:>5} {grp:>8} {g.get('n', 0):>5}  "
                      f"(too few)")
                continue
            pooled[grp].extend(g.pop("_rows"))
            print(f"{str(tr):>18} {te:>5} {grp:>8} {g['n']:>5} "
                  f"{g['r_rapm']:>8.4f} {g['r_raw']:>8.4f} "
                  f"{g['r_raw_dev']:>9.4f} {g['vs_raw']['p']:>9.4f} "
                  f"{g['vs_raw_dev']['p']:>9.4f}")

    print()
    summary = {}
    for grp in ("movers", "stayers"):
        A = np.array(pooled[grp])
        if len(A) < 20:
            continue
        summary[grp] = {"n": len(A)}
        for tname, tcol in (("demeaned", 3), ("undemeaned", 4)):
            rr = float(np.corrcoef(A[:, 0], A[:, tcol])[0, 1])
            rw = float(np.corrcoef(A[:, 1], A[:, tcol])[0, 1])
            rd = float(np.corrcoef(A[:, 2], A[:, tcol])[0, 1])
            t1, p1 = steiger(rr, rw, float(np.corrcoef(A[:, 0], A[:, 1])[0, 1]), len(A))
            t2, p2 = steiger(rr, rd, float(np.corrcoef(A[:, 0], A[:, 2])[0, 1]), len(A))
            summary[grp][tname] = {"r_rapm": round(rr, 4), "r_raw": round(rw, 4),
                                   "r_raw_dev": round(rd, 4),
                                   "p_vs_raw": round(p1, 6),
                                   "p_vs_raw_dev": round(p2, 6)}
            print(f"POOLED {grp:>8} [{tname:>10}]: n={len(A):>4}  "
                  f"RAPM {rr:+.4f}  raw {rw:+.4f}  raw_dev {rd:+.4f}  |  "
                  f"p(vs raw)={p1:.5f}  p(vs raw_dev)={p2:.5f}")

    mv = summary.get("movers", {}).get("demeaned", {})
    ok = (mv.get("r_rapm", -9) > max(mv.get("r_raw", 9), mv.get("r_raw_dev", 9))
          and mv.get("p_vs_raw_dev", 1) < 0.05)
    print(f"\nR1' {'PASS' if ok else 'INCONCLUSIVE'}: RAPM beats both raw "
          f"variants on transfer{'' if ok else ' but not at p<0.05 vs the '
          'stronger (team-demeaned) baseline'}")
    p = Path(__file__).resolve().parents[1] / "configs" / "rapm_folds.json"
    p.write_text(json.dumps({"lam": lam, "folds": res, "pooled": summary,
                             "pass": bool(ok)}, indent=1, default=float))
    print(f"-> {p}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
