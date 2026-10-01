"""Materialise the pre-cutoff data snapshot and record its provenance.

The preseason freeze must only see information that existed before the first
puck drop (config.CUTOFF_UTC). Rather than trusting file modification times,
ORR reads every repository input from the git tree of
config.CUTOFF_COMMIT, extracted into orr/cache/snapshot/. The manifest
written next to the freeze lists each file's SHA-256 and the commit time at
which that exact blob was last changed, so anyone can check that no input is
younger than the cutoff.

Run: python -m orr.snapshot
"""
from __future__ import annotations

import hashlib
import io
import json
import subprocess
import tarfile
from datetime import datetime
from pathlib import Path

from orr import config as C

SNAP = C.CACHE / "snapshot"

# Repository paths the freeze is allowed to read (directories or files).
SNAPSHOT_PATHS = [
    "data/processed",
    "data/raw/mp_skaters_{y}.csv",
    "data/raw/mp_goalies_{y}.csv",
    "data/raw/mp_teams_{y}.csv",
    "data/raw/nhl_schedule_20262027.csv",
    "data/raw/rosters",
    "data/raw/arenas.csv",
    "data/raw/nhl_drafts.csv",
    "data/market",
    "data/manual",
    "neurhl/configs/injury_overrides_2027.csv",
    # raw injury fields (MoneyPuck list, DailyFaceoff status) recorded 2026-09-28
    "neurhl/output/neurhl_1_0/availability_2027.csv",
    # the published NeurHL releases that existed before the cutoff (comparison only)
    "neurhl/output/neurhl_1_0",
    "neurhl/output/neurhl_1_1/season",
    "neurhl/output/live/2027/2026-09-29/morning_lineups.json",
    "neurhl/output/live/2027/2026-09-29/morning.csv",
]


def _expand(paths):
    out = []
    for p in paths:
        if "{y}" in p:
            out += [p.format(y=y) for y in range(2007, 2026)]
        else:
            out.append(p)
    return out


def _git(*args, binary=False):
    r = subprocess.run(["git", "-C", str(C.ROOT), *args], capture_output=True,
                       check=True)
    return r.stdout if binary else r.stdout.decode()


def materialise(force: bool = False) -> Path:
    """Extract the snapshot (idempotent). Returns the snapshot root."""
    stamp = SNAP / ".commit"
    if stamp.exists() and stamp.read_text().strip() == C.CUTOFF_COMMIT and not force:
        return SNAP
    commit_time = datetime.fromisoformat(
        _git("show", "-s", "--format=%cI", C.CUTOFF_COMMIT).strip())
    if commit_time >= C.CUTOFF_UTC:
        raise RuntimeError(f"cutoff commit {C.CUTOFF_COMMIT} is not before the cutoff")
    existing = set(_git("ls-tree", "-r", "--name-only", C.CUTOFF_COMMIT).split())
    want = [p for p in _expand(SNAPSHOT_PATHS)
            if p in existing or any(e.startswith(p + "/") for e in existing)]
    blob = _git("archive", "--format=tar", C.CUTOFF_COMMIT, *want, binary=True)
    SNAP.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(blob)) as tf:
        tf.extractall(SNAP, filter="data")
    stamp.write_text(C.CUTOFF_COMMIT + "\n")
    return SNAP


def path(rel: str) -> Path:
    """Path of a repository file as it stood at the cutoff."""
    p = materialise() / rel
    if not p.exists():
        raise FileNotFoundError(f"{rel} is not in the pre-cutoff snapshot")
    return p


def manifest() -> dict:
    """SHA-256 and last-change commit time for every snapshot file."""
    root = materialise()
    files = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != ".commit":
            rel = str(p.relative_to(root))
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            when = _git("log", "-1", "--format=%cI", C.CUTOFF_COMMIT, "--", rel).strip()
            files[rel] = {"sha256": h, "last_changed": when}
    latest = max((datetime.fromisoformat(v["last_changed"]) for v in files.values())).isoformat()
    return {"cutoff_utc": C.CUTOFF_UTC.isoformat(), "commit": C.CUTOFF_COMMIT,
            "latest_input_change": latest, "n_files": len(files), "files": files}


if __name__ == "__main__":
    m = manifest()
    print(f"{m['n_files']} files; newest input changed {m['latest_input_change']}; "
          f"cutoff {m['cutoff_utc']}")
