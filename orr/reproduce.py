"""Re-run a published ORR daily forecast from its recorded code and inputs (ORR 2.0).

    python3 -m orr.reproduce --date 2026-10-02 [--lineup-dir DIR] [--keep]

Every daily run writes orr/output/live/<date>/run_<date>.json, which names the
code commit, the model, the seed, the number of simulations, and the
SHA-256 of the results file and of each lineup file it read. This command:

  1. checks out that code commit in a temporary git worktree, with the
     ignored working inputs (orr/cache, data/raw/fastrhockey) linked in;
  2. rebuilds the inputs: the results file from git history, chosen by its
     SHA-256, and exactly the recorded lineup files, each checked by its
     SHA-256 (taken from the recorded paths, --lineup-dir, or NeurHL's
     origin/main);
  3. re-runs the day with the recorded model, seed and simulations;
  4. compares every output file with the published one.

The verdict is "identical" (byte for byte, apart from the created_utc
time stamp), "numerically equal" (every number within 1e-9), or "differs". Any input that could not be matched is
named, and the verdict then says so. Report: orr/output/reproduce/reproduce_<date>.json.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from orr import config as C

ROOT = C.ROOT
LINKED = ("orr/cache", "data/raw/fastrhockey")
RESULTS = "orr/output/live/results_2027.csv"
OUT = C.OUT / "reproduce"
STAMPS = ("created_utc",)          # when the file was made: expected to differ


def git(*args, cwd=ROOT, binary=False):
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, check=True)
    return r.stdout if binary else r.stdout.decode().strip()


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def results_bytes(want: str, data_commit: str) -> tuple[bytes | None, str]:
    """The results file whose SHA-256 is ``want``: first as committed with the
    run, then any earlier committed version."""
    for c in [data_commit] + git("log", "--format=%H", data_commit, "--", RESULTS).split():
        try:
            b = git("show", f"{c}:{RESULTS}", binary=True)
        except subprocess.CalledProcessError:
            continue
        if sha(b) == want:
            return b, c
    return None, ""


def lineup_sources(recorded: list[dict], override: str | None) -> list[Path]:
    """Folders that may hold NeurHL's <date>/... lineup files."""
    out = [Path(override)] if override else []
    for x in recorded:
        p = str(x["path"])
        if "/2027/" in p:
            out.append(Path(p.split("/2027/")[0] + "/2027"))
    out.append(ROOT / "neurhl" / "output" / "live" / "2027")
    return list(dict.fromkeys(out))


def upstream_lineups(tmp: Path) -> Path | None:
    """NeurHL's committed lineup files from origin/main (git archive), or None."""
    try:
        b = git("archive", "origin/main", "neurhl/output/live/2027", binary=True)
    except subprocess.CalledProcessError:
        return None
    d = tmp / "upstream"
    with tarfile.open(fileobj=io.BytesIO(b)) as t:
        t.extractall(d)
    return d / "neurhl" / "output" / "live" / "2027"


def gather_lineups(recorded: list[dict], override: str | None, dest: Path, tmp: Path) -> list[dict]:
    """Copy exactly the recorded lineup files into dest/<date>/<file>, each
    matched by SHA-256. Returns one row per recorded file with its status."""
    sources = lineup_sources(recorded, override)
    rows, fetched = [], False
    for x in recorded:
        rel = Path(*Path(x["path"]).parts[-2:])
        found = None
        while True:
            for s in sources:
                f = s / rel
                if f.exists() and sha(f.read_bytes()) == x["sha256"]:
                    found = f
                    break
            if found or fetched:
                break
            up = upstream_lineups(tmp)
            fetched = True
            if up is None:
                break
            sources.append(up)
        if found:
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(found, dest / rel)
        rows.append({"file": str(rel), "matched": bool(found), "source": str(found) if found else None})
    return rows


def compare(a: bytes, b: bytes) -> dict:
    if a == b:
        return {"identical": True, "max_abs_diff": 0.0}
    x, y = pd.read_csv(io.BytesIO(a)), pd.read_csv(io.BytesIO(b))
    x, y = x.drop(columns=[c for c in STAMPS if c in x]), y.drop(columns=[c for c in STAMPS if c in y])
    if x.equals(y):
        return {"identical": True, "max_abs_diff": 0.0, "ignored": list(STAMPS)}
    if list(x.columns) != list(y.columns) or len(x) != len(y):
        return {"identical": False, "shape": [list(x.shape), list(y.shape)], "max_abs_diff": None}
    num = x.select_dtypes("number").columns
    d = float(np.nanmax(np.abs(x[num].to_numpy(float) - y[num].to_numpy(float)))) if len(num) else 0.0
    other = [c for c in x.columns if c not in num and not x[c].astype(str).equals(y[c].astype(str))]
    return {"identical": False, "max_abs_diff": d, "text_columns_differ": other}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--date", required=True)
    ap.add_argument("--lineup-dir", default=None)
    ap.add_argument("--keep", action="store_true", help="keep the temporary worktree")
    a = ap.parse_args()
    D = a.date
    run_rel = f"orr/output/live/{D}/run_{D}.json"
    data_commit = git("log", "-1", "--format=%H", "--", run_rel)
    if not data_commit:
        sys.exit(f"{run_rel} is not committed")
    run = json.loads(git("show", f"{data_commit}:{run_rel}"))
    code = run["code"]
    model = str(run.get("model", "")).replace("ORR ", "")
    tmp = Path(tempfile.mkdtemp(prefix=f"orr_reproduce_{D}_"))
    wt = tmp / "worktree"
    report = {"date": D, "code": code, "data_commit": data_commit, "model": model,
              "sims": run["sims"], "seed": run["seed"]}
    git("worktree", "add", "--detach", str(wt), code)
    try:
        for p in LINKED:
            (wt / p).parent.mkdir(parents=True, exist_ok=True)
            if (ROOT / p).exists() and not (wt / p).exists():
                (wt / p).symlink_to(ROOT / p)
        res_b, res_c = results_bytes(run["inputs"]["results"]["sha256"], data_commit)
        report["results"] = {"matched": res_b is not None, "from_commit": res_c}
        res_path = tmp / "results.csv"
        res_path.write_bytes(res_b if res_b is not None else git("show", f"{data_commit}:{RESULTS}", binary=True))
        lu_dir = tmp / "lineups" / "2027"
        lu_dir.mkdir(parents=True)
        report["lineups"] = gather_lineups(run["inputs"].get("lineups", []), a.lineup_dir, lu_dir, tmp)
        cmd = [sys.executable, "-m", "orr.inseason", "--date", D, "--results", str(res_path),
               "--sims", str(run["sims"]), "--seed", str(run["seed"]), "--lineup-dir", str(lu_dir)]
        if model:
            cmd += ["--model", model]
        r = subprocess.run(cmd, cwd=wt, capture_output=True, text=True)
        report["exit_code"] = r.returncode
        if r.returncode:
            report["stderr_tail"] = r.stderr[-2000:]
        files = {}
        pub = git("ls-tree", "--name-only", f"{data_commit}:orr/output/live/{D}").split()
        for f in pub:
            if not f.endswith(".csv"):
                continue
            new = wt / "orr" / "output" / "live" / D / f
            if not new.exists():
                files[f] = {"identical": False, "missing": True}
                continue
            files[f] = compare(git("show", f"{data_commit}:orr/output/live/{D}/{f}", binary=True), new.read_bytes())
        report["files"] = files
        new_run = wt / "orr" / "output" / "live" / D / f"run_{D}.json"
        if new_run.exists():
            nr = json.loads(new_run.read_text())
            skip = {"created_utc", "inputs", "code"}
            report["run_fields_differ"] = sorted(k for k in set(run) | set(nr)
                                                 if k not in skip and run.get(k) != nr.get(k))
    finally:
        if not a.keep:
            git("worktree", "remove", "--force", str(wt))
            shutil.rmtree(tmp, ignore_errors=True)
    inputs_ok = report["results"]["matched"] and all(x["matched"] for x in report["lineups"])
    fs = report.get("files", {})
    if report.get("exit_code") or not fs:
        v = "failed to run"
    elif all(x.get("identical") for x in fs.values()):
        v = "identical"
    elif all(x.get("max_abs_diff") is not None and x["max_abs_diff"] <= 1e-9 and not x.get("text_columns_differ")
             for x in fs.values()):
        v = "numerically equal"
    else:
        v = "differs"
    report["verdict"] = v if inputs_ok else f"{v} (inputs not all matched)"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"reproduce_{D}.json").write_text(json.dumps(report, indent=1))
    print(f"{D}: code {code}, model {model}: {report['verdict']}")
    for f, x in fs.items():
        print(f"  {f}: {'identical' if x.get('identical') else x}")
    print(f"  results matched: {report['results']['matched']}; lineup files matched: "
          f"{sum(x['matched'] for x in report['lineups'])}/{len(report['lineups'])}")


if __name__ == "__main__":
    main()
