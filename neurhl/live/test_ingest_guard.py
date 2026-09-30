"""Write guard of neurhl/live/ingest_2027.py in the REAL tensors mode.

The parity tests run in sandboxes, where the season-name rule is off, so the
real-mode guard is checked here without building anything:
  allowed   <tensors>/events_2027.parquet, <tensors>/_ingest_2027/roster.parquet
  refused   <tensors>/events_2026.parquet, <tensors>/maps.json,
            a file outside the tensors dir, a symlink
Usage: python neurhl/live/test_ingest_guard.py
"""
import importlib.util
import sys
import tempfile
from pathlib import Path

NRL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NRL))
FAIL = []


def check(name, ok):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        FAIL.append(name)


def refused(ctx, p):
    try:
        ctx.guard(p)
        return False
    except RuntimeError:
        return True


def main():
    spec = importlib.util.spec_from_file_location("ingest_2027", NRL / "live" / "ingest_2027.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    with tempfile.TemporaryDirectory() as t:
        out = Path(t).resolve() / "tensors"
        (out / "_ingest_2027").mkdir(parents=True)
        ctx = object.__new__(m.Ctx)
        ctx.S, ctx.out, ctx.real, ctx.cache = 2027, out, True, out / "_ingest_2027"
        check("season table allowed", not refused(ctx, out / "events_2027.parquet"))
        check("cache roster allowed", not refused(ctx, out / "_ingest_2027" / "roster.parquet"))
        check("cache games allowed", not refused(ctx, out / "_ingest_2027" / "games.parquet"))
        check("other season refused", refused(ctx, out / "events_2026.parquet"))
        check("unscoped file refused", refused(ctx, out / "maps.json"))
        check("outside the tensors dir refused", refused(ctx, Path(t) / "x_2027.parquet"))
        (out / "link_2027.parquet").symlink_to(Path(t) / "target")
        check("symlink refused", refused(ctx, out / "link_2027.parquet"))
    print(f"{7 - len(FAIL)}/7 passed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
