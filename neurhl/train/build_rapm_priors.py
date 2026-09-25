"""NeurHL-2 — walk-forward RAPM priors, one table per vantage season.

The simulator's player effects are ANCHORED on RAPM rather than learned free
(PLAN_NeurHL2 S2): `effect = rapm + s*tanh(delta)` with the residual head
zero-initialised, so at step 0 the engine reproduces the validated ridge solution
and can only depart from it by earning that on the event likelihood. This is the
identity-init pattern that rescued v1's Layer 1, where a freely-learned player
model regressed everyone to the league mean (sd 0.006 against an EWMA baseline's
0.042).

Vantage discipline is the whole point of building these as a table per season:
to model events in season V, the player priors must come from a fit on seasons
< V only. A single RAPM fit over all history, applied backwards, would leak
future performance into every historical event and is the same class of error as
using MoneyPuck's full-sample xGoals.

Carries forward the honest caveat from A6: RAPM is retained as the anchor for
STRUCTURAL reasons — it maps an arbitrary on-ice set to a rate, separates offence
from defence, adjusts for opponents and yields posterior SEs — and NOT because it
beat a team-demeaned raw on-ice rate, which it did not (R1' INCONCLUSIVE,
p=0.597). `raw_net` is emitted alongside so the alternative anchor stays testable
downstream without refitting anything.

Window used per vantage: the most recent `WINDOW` seasons before V, which keeps
the design well-conditioned (players move between teams within the window, which
is what identifies them against the team fixed effects) without letting a
decade-old season set a current player's prior.

Run: uv run --no-project --python 3.12 --with numpy --with "pandas<3" \
     --with pyarrow --with scipy python neurhl/train/build_rapm_priors.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import TENSORS  # noqa: E402
import models.rapm as R  # noqa: E402
import manifest as MAN  # noqa: E402

WINDOW = 3                      # seasons of history per fit
LAM = 51200.0                   # chosen on DEV by the A5 transfer criterion
MIN_TOI = 6000                  # 100 minutes inside the window
FIRST = 2008
CFG = {"window": WINDOW, "lam": LAM, "min_toi": MIN_TOI, "team_fe": True,
       "version": 1}


def raw_net_table(st: pd.DataFrame) -> pd.DataFrame:
    """Raw on-ice net rate, UNdemeaned. Emitted so the A6 alternative anchor
    (team-demeaned raw) stays derivable downstream — demeaning is a
    consumer-side step given a team assignment, which stints alone don't fix.
    An earlier docstring called this "team-demeaned"; the code never was."""
    h = st[R.H_COLS].to_numpy(np.int64)
    a = st[R.A_COLS].to_numpy(np.int64)
    dur = st.dur_s.to_numpy(np.float64)
    cf_h = st.cf_h.to_numpy(np.float64)
    cf_a = st.cf_a.to_numpy(np.float64)
    n = int(max(h.max(initial=0), a.max(initial=0))) + 2
    toi = np.zeros(n)
    fo = np.zeros(n)
    ag = np.zeros(n)
    for side, f, g in ((h, cf_h, cf_a), (a, cf_a, cf_h)):
        for c in range(side.shape[1]):
            pid = side[:, c]
            m = pid > 0
            np.add.at(toi, pid[m], dur[m])
            np.add.at(fo, pid[m], f[m])
            np.add.at(ag, pid[m], g[m])
    idx = np.flatnonzero(toi > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        net = 3600 * (fo[idx] - ag[idx]) / toi[idx]
    return pd.DataFrame({"player_id": idx, "toi_s": toi[idx], "raw_net": net})


def build(v: int) -> pd.DataFrame:
    seasons = [s for s in range(max(FIRST, v - WINDOW), v)]
    if not seasons:
        return pd.DataFrame()
    st = R.load_stints(seasons)
    des = R.RAPMDesign(st, min_toi_s=MIN_TOI,
                       positions=R.player_positions(seasons), team_fe=True)
    ys = des.targets(st, ["cf", "g"])
    res = R.solve(des, ys, LAM, want_se=True)

    o, d = des.off0, des.def0
    se = res["se_unit"]
    sig_cf = np.sqrt(res.get("sigma2_cf", 1.0))
    b_cf, b_g = res["coef"]["cf"], res["coef"]["g"]
    rows = []
    for pid, s in des.pid_to_slot.items():
        rows.append((pid, float(b_cf[o + s]), float(b_cf[d + s]),
                     float(b_g[o + s]), float(b_g[d + s]),
                     float(se[o + s] * sig_cf), float(se[d + s] * sig_cf),
                     float(des.toi.get(pid, 0.0)), 0))
    for pg, s in des.repl_slot.items():
        rows.append((-1 - pg, float(b_cf[o + s]), float(b_cf[d + s]),
                     float(b_g[o + s]), float(b_g[d + s]),
                     float(se[o + s] * sig_cf), float(se[d + s] * sig_cf),
                     0.0, 1))
    out = pd.DataFrame(rows, columns=[
        "player_id", "cf_off", "cf_def", "g_off", "g_def",
        "cf_off_se", "cf_def_se", "toi_s", "is_replacement"])

    rn = raw_net_table(st)
    out = out.merge(rn[["player_id", "raw_net"]], on="player_id", how="left")
    out["raw_net"] = out.raw_net.fillna(0.0)
    out["vantage"] = v
    out["intercept_cf"] = float(b_cf[0])
    out["home_cf"] = float(b_cf[1])
    out["n_players"] = len(des.pid_to_slot)
    out["fit_seasons"] = ",".join(str(s) for s in seasons)
    out.attrs["sha_cf"] = R.fingerprint(res, "cf")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vantages", type=int, nargs="*",
                    default=list(range(2009, 2028)))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    print(f"{'V':>5} {'fit on':>16} {'players':>8} {'repl':>5} {'int':>7} "
          f"{'home':>6} {'sd(off)':>8} {'sd(def)':>8} {'sha':>17}")
    for v in args.vantages:
        seasons = [s for s in range(max(FIRST, v - WINDOW), v)]
        src = [TENSORS / f"stints_{s}.parquet" for s in seasons
               if (TENSORS / f"stints_{s}.parquet").exists()]
        if not src:
            print(f"{v:>5}  no stint shards")
            continue
        out = TENSORS / f"rapm_prior_{v}.parquet"
        if not args.force and MAN.is_fresh(out, src, CFG):
            print(f"{v:>5}  fresh, skipping")
            continue
        d = build(v)
        if not len(d):
            continue
        d.to_parquet(out, index=False)
        MAN.write_manifest(out, src, CFG,
                           {"n_players": int((~d.is_replacement.astype(bool)).sum()),
                            "sha_cf": d.attrs.get("sha_cf"),
                            "fit_seasons": seasons})
        real = d[~d.is_replacement.astype(bool)]
        print(f"{v:>5} {str(seasons):>16} {len(real):>8} "
              f"{int(d.is_replacement.sum()):>5} {d.intercept_cf.iloc[0]:>7.2f} "
              f"{d.home_cf.iloc[0]:>6.2f} {real.cf_off.std():>8.3f} "
              f"{real.cf_def.std():>8.3f} {d.attrs.get('sha_cf',''):>17}")
        sys.stdout.flush()
    print("done")


if __name__ == "__main__":
    main()
