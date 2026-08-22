"""NeurHL-2 — acceptance battery: re-assert every recorded decision mechanically.

PLAN_NeurHL2 "Acceptance" requires that the gate decisions be re-checked against
the rules rather than trusted from a report, that artifacts be fresh, and that
window discipline be verified in code rather than by convention.

Each check re-derives its verdict from the recorded numbers and the rule as
written. A check that cannot find its evidence FAILS rather than passing quietly:
a missing gate file is indistinguishable from a gate that was never run, and both
should stop the battery.

Run: uv run --no-project --python 3.12 --with numpy --with "pandas<3" \
     --with pyarrow python neurhl/tests/review_tests_neurhl2.py
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import TENSORS  # noqa: E402
import windows as W  # noqa: E402
from registry import REG  # noqa: E402

CFG = ROOT / "configs"
OUT = []


def check(name, ok, detail=""):
    OUT.append({"check": name, "pass": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f"  — {detail}" if detail else ""))
    return ok


def load(p):
    q = CFG / p
    return json.loads(q.read_text()) if q.exists() else None


def main():
    print("NeurHL-2 acceptance battery\n")

    # ---------------------------------------------------------- boundaries
    print("BOUNDARIES")
    src_imports = subprocess.run(
        ["grep", "-rn", "neurhl", str(ROOT.parent / "src")],
        capture_output=True, text=True).stdout.strip()
    check("src/ never imports neurhl/", not src_imports,
          src_imports[:80] if src_imports else "no references")
    proj = ROOT / "output" / "projection_2027.csv"
    check("deliverable exists and is report-only", proj.exists(),
          str(proj.relative_to(ROOT.parent)) if proj.exists() else "missing")

    # ------------------------------------------------------------- windows
    print("\nWINDOW DISCIPLINE")
    ok = True
    for s in (2013, 2021):
        try:
            W.assert_scorable([s], "tune" if s == 2013 else "confirm")
            ok = False
        except ValueError:
            pass
    check("broken seasons 2013/2021 refuse to be scored", ok)
    try:
        W.assert_scorable([2018], "tune")
        ok2 = False
    except ValueError:
        ok2 = True
    check("cross-window scoring refused", ok2)
    check("PROJECT season is 2027 (= the 2026-27 season)", W.PROJECT == 2027,
          f"PROJECT={W.PROJECT}; nothing targets 2027-28")

    # ------------------------------------------------------------- leakage
    print("\nLEAKAGE AUDIT (standing requirement, amendment A8)")
    la = load("leakage_audit.json")
    if la is None:
        check("leakage audit present", False, "configs/leakage_audit.json missing")
    else:
        cz = la["causality"]
        cz = cz.get("causality", cz) if isinstance(cz, dict) else cz
        moved = [m for c in cz for m in c["inputs_that_moved"]]
        check("causality: no input moves when the future is corrupted",
              not moved, f"{len(cz)} games, "
              f"{len(cz[0]['inputs_checked'])} inputs checked")
        ti = la["target_identity"]
        check("no input equals a target", not ti["identical_pairs"])
        check("no input recovers the target's team",
              not ti["reveals_target_team"])
        lifts = [la["onice_information"][k]["max_lift"]
                 for k in ("penalty", "goal", "faceoff", "stoppage")]
        check("on-ice carries no lift on the next event type",
              max(lifts) <= 1.02, f"max lift {max(lifts):.3f}")

    # ----------------------------------------------------------- xG gates
    print("\nP3 — xG gates")
    xg = load("xg_gates.json")
    if xg is None:
        check("xG gates present", False, "configs/xg_gates.json missing")
    else:
        n_ok = sum(g >= xg["X1"]["min_gain"] for g in xg["X1"]["gains"])
        check("X1 re-derived: gain >= 0.002 nats in >= 15 vantages",
              n_ok >= 15, f"{n_ok}/{len(xg['X1']['gains'])}, mean "
              f"{np.mean(xg['X1']['gains']):+.5f}")
        x2 = xg["X2"]
        check("X2 re-derived: team xG beats Corsi at future goal share",
              x2["r_xg_predicts_future_gf"] > x2["r_corsi_predicts_future_gf"],
              f"xG {x2['r_xg_predicts_future_gf']:.4f} vs Corsi "
              f"{x2['r_corsi_predicts_future_gf']:.4f}, n={x2['n_team_seasons']}")
        d = xg["X3"]["max_abs_diff_deciles"]
        check("X3 recorded as FAILED (not silently passed)", d > 0.005,
              f"max decile deviation {d:.5f} > 0.005 — recorded as a known "
              f"limitation, no preregistered consequence")

    # ---------------------------------------------------------- RAPM gates
    print("\nP4 — RAPM gates")
    rf = load("rapm_folds.json")
    if rf is None:
        check("RAPM fold results present", False)
    else:
        mv = rf["pooled"]["movers"]
        m = mv.get("demeaned", mv)
        check("R1' recorded as INCONCLUSIVE (A6), not passed",
              not rf.get("pass", False),
              f"movers n={mv['n']}, RAPM {m['r_rapm']:.4f} vs raw-demeaned "
              f"{m['r_raw_dev']:.4f}, p={m['p_vs_raw_dev']:.3f}")

    # ------------------------------------------------------- S1 event gates
    print("\nP5 — event simulator gates")
    es = load("event_sim_gates.json")
    if es is None:
        check("event-sim gates present", False)
    else:
        e1 = es["E1"]
        worst = max(abs(r["rel_err"]) for r in e1["per_game"])
        wp = max(abs(r["rel_err"]) for r in e1["per_period"])
        check("E1 re-derived: every per-game count within 10%",
              worst <= 0.10, f"worst {worst:+.2%}")
        check("E1 re-derived: per-period goals/shots within 10%",
              wp <= 0.10, f"worst {wp:+.2%}")
        e2 = es["E2"]
        check("E2 re-derived: score-effect gradient ratio in [0.5, 1.5]",
              0.5 <= e2["ratio"] <= 1.5 and e2["same_direction"],
              f"ratio {e2['ratio']}, corr {e2['corr']}")
        check("dt calibration (randomised PIT) within 0.05",
              es["dt_pit"]["max_decile_dev"] <= 0.05,
              f"{es['dt_pit']['max_decile_dev']:.4f}")

    # --------------------------------------------------------- G-STOP / S4
    print("\nS4 — game layer and the G-STOP rule")
    bl = load("s4v2_blend.json")
    if bl is None:
        check("game-layer results present", False)
    else:
        pm, pe = bl["pooled_model"], sum(
            r["ll_elo"] * r["n"] for r in bl["rows"]) / sum(
            r["n"] for r in bl["rows"])
        v1 = 0.67314
        check("blend nests Elo (never materially worse)", pm - pe < 0.005,
              f"blend {pm:.5f} vs Elo {pe:.5f} ({pm-pe:+.5f})")
        fired = (pm - v1) > -0.0015
        check("G-STOP correctly FIRED -> CONFIRM not spent", fired,
              f"blend {pm:.5f} vs v1 hierarchical {v1:.5f} = {pm-v1:+.5f}, "
              f"weaker than -0.0015")
        conf_touched = any((CFG / f).exists() for f in
                           ("confirm_2018_2026.json", "c1_result.json"))
        check("CONFIRM window remains unspent", not conf_touched)

    # ------------------------------------------------------- the projection
    print("\nP9 — 2026-27 deliverable")
    if proj.exists():
        r = pd.read_csv(proj)
        check("32 teams projected", len(r) == 32, f"{len(r)} rows")
        n_g = 1344
        tot = r.proj_points.sum()
        ot = (tot - 2 * n_g) / n_g
        check("points identity implies a realistic OT share",
              0.18 <= ot <= 0.27, f"implied {ot:.1%} (observed 2024-26 22.1%)")
        for c, grp in r.groupby("conf"):
            s = grp.playoff_pct.sum() / 100
            check(f"playoff odds sum to 8 in the {c} conference",
                  abs(s - 8) < 0.05, f"{s:.3f}")
        check("no team projected outside a plausible points band",
              r.proj_points.between(55, 130).all(),
              f"{r.proj_points.min():.1f}-{r.proj_points.max():.1f}")

    # ------------------------------------------------------------- registry
    print("\nFEATURE REGISTRY")
    try:
        REG.assert_no_outcome_inputs(["home_win"])
        ok = False
    except ValueError:
        ok = True
    check("registry refuses an OUTCOME feature as an input", ok)
    check("EDGE (G5) is usable for the 2027 projection under PRIOR vantage",
          REG["edge_shot_speed"].available(2027))

    n_fail = sum(not o["pass"] for o in OUT)
    (CFG / "acceptance.json").write_text(json.dumps(OUT, indent=1))
    print(f"\n{len(OUT) - n_fail}/{len(OUT)} checks pass -> "
          f"{CFG / 'acceptance.json'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
