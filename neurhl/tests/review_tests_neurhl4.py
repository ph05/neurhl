"""NeurHL-4 (NeurHL-G) acceptance battery (PLAN_NeurHL4, section Q).

  windows     the SEALED guard refuses sealed seasons; unseal() refuses callers
              other than eval/seal_g.py; the master loader drops SEALED seasons
  sealed      every sealed input still matches configs/sealed_inputs.sha256
  loader      no NeurHL-G model/training module reads the tensor directory
              directly (AST scan)
  ledger      G_GATE runs within the cap of 2; G_ITER full runs within 14
  audit       configs/leakage_audit_g.json records ALL CLEAN
  seal        the seal ran at most once, after a FREEZE commit that is an
              ancestor of the result
  live        every published live forecast was committed before its game's
              scheduled start (GitHub commit time), where forecasts exist
Run: uv run --no-project --python 3.12 --with numpy --with "pandas<3" \
     --with pyarrow python neurhl/tests/review_tests_neurhl4.py
"""
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJ = ROOT.parent
sys.path.insert(0, str(ROOT))
from common import CONFIGS, NOUT, TENSORS  # noqa: E402
import windows as W  # noqa: E402

RES = []


def check(name, ok, detail=""):
    RES.append({"check": name, "pass": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


def git(*a):
    return subprocess.run(["git", "-C", str(PROJ), *a], capture_output=True, text=True)


def main():
    print("NeurHL-4 acceptance battery\n\nWINDOWS")
    try:
        W.assert_scorable_g([2025], "sealed")
        refused = False
    except PermissionError:
        refused = True
    check("sealed seasons refused before unseal", refused)
    try:
        W.unseal("neurhl/eval/run_g.py")
        ok = False
    except PermissionError:
        ok = True
    check("unseal() refuses callers other than eval/seal_g.py", ok)
    from data.g_loader import load_master
    _, meta, _ = load_master("train")
    spent = (ROOT / "output" / "g_seal_result.json").exists()
    if not spent:
        check("master loader drops SEALED seasons", not meta.season_end.isin(W.SEALED).any(),
              f"max season {int(meta.season_end.max())}")
    else:   # PLAN_NeurHL4 A5/A6: the seal is spent; the one-shot result file keeps seal_g from running again
        check("seal spent: loader includes 2025-2026 and the one-shot seal result exists",
              meta.season_end.isin(W.SEALED).any() and (ROOT / "output" / "g_seal_result.json").exists(),
              f"max season {int(meta.season_end.max())}")

    print("\nSEALED INPUTS")
    bad = []
    for line in (CONFIGS / "sealed_inputs.sha256").read_text().splitlines():
        h, f = line.split()
        if hashlib.sha256((TENSORS / f).read_bytes()).hexdigest() != h:
            bad.append(f)
    check("all 37 sealed inputs unchanged", not bad, ", ".join(bad[:5]))

    print("\nLOADER DISCIPLINE")
    offenders = []
    for f in (ROOT / "models" / "neurhl_g.py", ROOT / "train" / "train_neurhl_g.py"):
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in ("read_parquet", "load"):
                offenders.append(f"{f.name}:{node.lineno}")
            if isinstance(node, ast.Name) and node.id == "TENSORS":
                offenders.append(f"{f.name}:{node.lineno}")
    check("model and trainer read data only through g_loader", not offenders, ", ".join(offenders))

    print("\nLEDGER")
    led = pd.read_csv(CONFIGS / "search_ledger_g.csv")
    gate = led[led.run_id.str.endswith(":gate")]
    it_full = led[led.run_id.str.contains(":full:iter")]
    # runs declared under PLAN_NeurHL_1_1 (A16-A20): candidates g1rk, g1x, and g1 rescored from its
    # cached snapshots for the A17 comparison; counted apart from the NeurHL-G ladder (A21)
    v11 = it_full.run_id.str.startswith(("g1rk:", "g1x:", "g1:"))
    check("G_GATE runs within cap (2)", len(gate) <= 2, f"{len(gate)}/2")
    check("G_ITER full runs within cap (14), NeurHL-G ladder", int((~v11).sum()) <= 14, f"{int((~v11).sum())}/14")
    check("PLAN_NeurHL_1_1 iteration runs as declared (g1rk, g1x, g1 rescoring)", int(v11.sum()) <= 3,
          ", ".join(it_full.run_id[v11]))

    print("\nAUDIT")
    a = json.loads((CONFIGS / "leakage_audit_g.json").read_text())
    check("state-builder causality audit clean", a.get("pass") is True,
          f"{len(a.get('checks', {}))} checks")

    print("\nSEAL")
    res = NOUT / "g_seal_result.json"
    if not res.exists():
        check("seal not spent (allowed)", True, "no result yet")
    else:
        adds = git("log", "--diff-filter=A", "--format=%H", "--", "neurhl/output/g_seal_result.json").stdout.split()
        check("seal result added exactly once", len(adds) == 1)
        fz = json.loads(res.read_text()).get("freeze_commit", "")
        anc = git("merge-base", "--is-ancestor", fz, adds[0]).returncode == 0 if adds and fz else False
        check("FREEZE commit precedes the seal result", anc)

    print("\nLIVE")
    base = NOUT / "live" / "2027"
    files = sorted(base.glob("*/pregame_*.csv")) + sorted(base.glob("*/morning.csv")) if base.exists() else []
    files = [f for f in files if not f.name.endswith("_players.csv")]
    late = []
    for f in files:
        rel = str(f.relative_to(PROJ))
        c = git("log", "--diff-filter=A", "--format=%cI", "--", rel).stdout.split()
        if not c:
            continue
        committed = pd.Timestamp(c[-1])
        starts = pd.to_datetime(pd.read_csv(f).start_utc, utc=True)
        if (committed >= starts).any():
            late.append(rel)
    check("every live forecast committed before its game's start", not late,
          f"{len(files)} files" + (f"; late: {late[:3]}" if late else ""))

    n, k = len(RES), sum(r["pass"] for r in RES)
    print(f"\n{k}/{n} checks pass")
    (CONFIGS / "acceptance_neurhl4.json").write_text(json.dumps(RES, indent=1))
    sys.exit(0 if k == n else 1)


if __name__ == "__main__":
    main()
