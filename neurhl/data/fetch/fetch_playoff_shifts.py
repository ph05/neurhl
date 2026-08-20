"""NeurHL A7 — playoff shift charts gap-fill.

Phase-1 integrity finding: data/raw/shifts/ covers regular season only (the v6
fetcher used gameType=2), so playoff events tensorize with zero on-ice slots
and playoff player-games have zero TOI. The shiftcharts endpoint serves
playoffs identically; this fetcher fills every game present in data/raw/pbp_po/
that lacks a shift file, into the same data/raw/shifts/<season_end>/ layout.
Reuses src/fetch_shifts.py fetch_one (same storage contract). Resumable.
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import RAW  # noqa: E402  (also puts src/ on sys.path)
from fetch_shifts import fetch_one  # noqa: E402  (src/ module)


def main():
    t0 = time.time()
    for sdir in sorted((RAW / "pbp_po").iterdir()):
        se = sdir.name
        dest_dir = RAW / "shifts" / se
        dest_dir.mkdir(parents=True, exist_ok=True)
        jobs = []
        for p in sorted(sdir.glob("*.json.gz")):
            dest = dest_dir / p.name
            if not (dest.exists() and dest.stat().st_size > 200):
                jobs.append((int(p.name.split(".")[0]), dest))
        if not jobs:
            print(f"{se}: complete")
            continue
        counts: dict = {}
        with ThreadPoolExecutor(max_workers=4) as ex:
            for res in ex.map(fetch_one, jobs):
                counts[res] = counts.get(res, 0) + 1
        print(f"{se}: {len(jobs)} playoff games -> {counts} "
              f"[{time.time() - t0:.0f}s]")
        sys.stdout.flush()
    print("done")


if __name__ == "__main__":
    main()
