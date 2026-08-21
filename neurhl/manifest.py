"""NeurHL-2 — artifact lineage: every derived file knows what produced it.

Two real bugs in the v1 build came from artifacts with no provenance:
  * `proj_team_*.parquet` files silently mixed two model generations, because
    nothing recorded which player model had written them;
  * playoff rows contaminated Layer-1 training for every vantage, and the
    resulting projections looked identical to the clean ones from the outside.

Both are the same failure: a consumer cannot tell a stale input from a fresh one.
So every derived artifact here is written with a sidecar manifest recording the
fingerprints of its sources, the code version, and the config that produced it —
and consumers call `require_fresh()`, which REFUSES stale inputs rather than
silently blending generations.

Fingerprints are (size, mtime_ns), the standard build-system compromise: a
regenerated-but-identical file reads as stale (safe — you rebuild), while
changed content never reads as fresh (the direction that matters). Files under
CONTENT_HASH_MAX also carry a sha256 so committed artifacts can be verified
byte-exactly.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

CONTENT_HASH_MAX = 64 * 1024 * 1024      # sha256 files up to 64 MB
MANIFEST_SUFFIX = ".manifest.json"


def code_version() -> dict:
    """Git SHA plus a dirty flag; unknown outside a repo rather than fatal."""
    try:
        root = Path(__file__).resolve().parents[1]
        sha = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                             capture_output=True, text=True,
                             timeout=10).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10).stdout.strip())
        return {"git_sha": sha or "unknown", "dirty": dirty}
    except Exception:
        return {"git_sha": "unknown", "dirty": None}


def fingerprint(path) -> dict:
    """Cheap identity for one file: size + mtime, plus sha256 when small."""
    p = Path(path)
    st = p.stat()
    fp = {"path": str(p), "size": st.st_size, "mtime_ns": st.st_mtime_ns}
    if st.st_size <= CONTENT_HASH_MAX:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        fp["sha256"] = h.hexdigest()
    return fp


def config_hash(config: dict) -> str:
    """Stable hash of a config dict (sorted keys, compact separators)."""
    blob = json.dumps(config or {}, sort_keys=True, separators=(",", ":"),
                      default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def manifest_path(artifact) -> Path:
    return Path(str(artifact) + MANIFEST_SUFFIX)


def write_manifest(artifact, sources=(), config=None, extra=None) -> Path:
    """Record what produced `artifact`. Call immediately after writing it."""
    import datetime as _dt
    man = {
        "artifact": str(Path(artifact).name),
        "artifact_fp": fingerprint(artifact),
        "sources": [fingerprint(s) for s in sources if Path(s).exists()],
        "config_hash": config_hash(config),
        "config": config or {},
        "code": code_version(),
        "built_by": Path(sys.argv[0]).name if sys.argv else "?",
        "built_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    p = manifest_path(artifact)
    p.write_text(json.dumps(man, indent=1, default=str))
    return p


def check(artifact, sources=(), config=None, strict_code=False) -> tuple:
    """-> (ok: bool, reason: str). Pure inspection; never raises on staleness."""
    a = Path(artifact)
    if not a.exists():
        return False, "artifact missing"
    mp = manifest_path(a)
    if not mp.exists():
        return False, "no manifest (provenance unknown — treat as stale)"
    try:
        man = json.loads(mp.read_text())
    except Exception as e:
        return False, f"unreadable manifest: {e}"

    cur = fingerprint(a)
    old = man.get("artifact_fp", {})
    if "sha256" in cur and "sha256" in old:
        if cur["sha256"] != old["sha256"]:
            return False, "artifact modified since manifest was written"
    elif cur["size"] != old.get("size"):
        return False, "artifact size changed since manifest was written"

    want = config_hash(config)
    if man.get("config_hash") != want:
        return False, (f"config changed ({man.get('config_hash')} -> {want})")

    have = {s["path"]: s for s in man.get("sources", [])}
    for s in sources:
        sp = str(Path(s))
        if not Path(s).exists():
            return False, f"source missing: {sp}"
        rec = have.get(sp)
        if rec is None:
            return False, f"source not in manifest: {sp}"
        now = fingerprint(s)
        if "sha256" in rec and "sha256" in now:
            if rec["sha256"] != now["sha256"]:
                return False, f"source changed: {sp}"
        elif (rec.get("size"), rec.get("mtime_ns")) != (now["size"],
                                                        now["mtime_ns"]):
            return False, f"source changed: {sp}"

    if strict_code:
        cv = code_version()
        if cv["git_sha"] != man.get("code", {}).get("git_sha"):
            return False, "code version changed"
    return True, "fresh"


def require_fresh(artifact, sources=(), config=None, strict_code=False) -> None:
    """Refuse to consume a stale artifact. This is the whole point."""
    ok, why = check(artifact, sources, config, strict_code)
    if not ok:
        raise RuntimeError(
            f"STALE ARTIFACT {Path(artifact).name}: {why}. "
            f"Rebuild it rather than mixing generations "
            f"(see neurhl/manifest.py for why this check exists).")


def is_fresh(artifact, sources=(), config=None) -> bool:
    """Non-raising variant, for 'skip if already built' loops."""
    return check(artifact, sources, config)[0]
