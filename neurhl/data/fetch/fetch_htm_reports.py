"""NeurHL A1 — NHL HTM report backfill (PLAN_NeurHL acquisition A1).

Fetches the official HTM game reports from www.nhl.com/scores/htmlreports:
  PL = play-by-play, ES = event summary, TH/TV = shift/TOI (home/visitor)

Two jobs:
  1. Backfill season_end 2008-2012 (all four reports, regular season) — extends
     the event corpus back to 2008 and provides shift/TOI for 2008-2010 where the
     shiftcharts API is empty (2009-10) or absent.
  2. Gap-fill TH/TV for the 2024-25 games missing from data/raw/shifts/2025/
     (the shiftcharts API returned empty for the final ~2 weeks of that season).

Storage: data/raw/htm_reports/<season_end>/<RPT><code>.htm.gz (gitignored).
Resumable (existing files skipped), 3 worker threads + per-request sleep to stay
polite against the static report host, retries with backoff, per-season summary.
"""
import gzip
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import RAW, UA  # noqa: E402

DEST = RAW / "htm_reports"
BASE = "https://www.nhl.com/scores/htmlreports"
BACKFILL_ENDS = [2008, 2009, 2010, 2011, 2012]
# RO (roster: official scratches, coaches, officials) and GS (game summary:
# officials) added by NeurHL-3 D3 — the G4 source for the pre-2012 hole.
# Existing PL/ES/TH/TV files are cached, so re-running only fetches the new
# report types.
REPORTS_BACKFILL = ["PL", "ES", "TH", "TV", "RO", "GS"]
SLEEP = 0.2    # per request, per worker; observed bottleneck is server latency


FULL_SEASON_GAMES = 1230   # 30 teams x 82 / 2, true for all backfill seasons


def game_ids(season: str) -> list[int]:
    url = (f"https://api.nhle.com/stats/rest/en/game?limit=-1&"
           f"cayenneExp=season={season}%20and%20gameType=2")
    for attempt in range(2):
        try:
            r = requests.get(url, headers=UA, timeout=60)
            if r.status_code == 200:
                return sorted(g["id"] for g in r.json()["data"])
        except requests.RequestException:
            pass
        time.sleep(5.0 * (attempt + 1))
    # stats-rest index down (observed 503 storms): backfill seasons are full
    # 1230-game seasons with contiguous ids — derive locally, tolerate 404s.
    print(f"  index unavailable for {season}; using contiguous 1..{FULL_SEASON_GAMES}")
    start = int(season[:4])
    return [start * 1_000_000 + 20_000 + n
            for n in range(1, FULL_SEASON_GAMES + 1)]


def fetch_one(args) -> str:
    season, rpt, gid, dest = args
    if dest.exists() and dest.stat().st_size > 500:
        return "cached"
    code = f"{gid % 1_000_000:06d}"
    url = f"{BASE}/{season}/{rpt}{code}.HTM"
    for attempt in range(4):
        try:
            time.sleep(SLEEP)
            r = requests.get(url, headers=UA, timeout=60)
            if r.status_code == 200 and len(r.content) > 1000:
                tmp = dest.with_suffix(".part")
                with gzip.open(tmp, "wb") as f:
                    f.write(r.content)
                tmp.rename(dest)
                return "ok"
            if r.status_code == 404:
                return "404"
        except requests.RequestException:
            pass
        time.sleep(2.0 * (attempt + 1))
    return "fail"


def missing_2025_shift_gids() -> list[int]:
    """Games with PBP but no shift chart in season_end 2025."""
    pbp = {int(p.name.split(".")[0]) for p in (RAW / "pbp" / "2025").glob("*.json.gz")}
    shf = {int(p.name.split(".")[0]) for p in (RAW / "shifts" / "2025").glob("*.json.gz")}
    return sorted(pbp - shf)


def run_jobs(jobs: list, label: str, t0: float) -> None:
    counts: dict = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        for res in ex.map(fetch_one, jobs):
            counts[res] = counts.get(res, 0) + 1
    print(f"{label}: {len(jobs)} reports -> {counts}  [{time.time() - t0:.0f}s elapsed]")
    sys.stdout.flush()
    if counts.get("fail"):
        print(f"  WARNING {label}: {counts['fail']} failures (rerun to resume)")


def main():
    t0 = time.time()
    for end in BACKFILL_ENDS:
        season = f"{end - 1}{end}"
        sdir = DEST / str(end)
        sdir.mkdir(parents=True, exist_ok=True)
        ids = game_ids(season)
        jobs = [(season, rpt, gid, sdir / f"{rpt}{gid % 1_000_000:06d}.htm.gz")
                for gid in ids for rpt in REPORTS_BACKFILL]
        run_jobs(jobs, f"backfill {end}", t0)

    gap = missing_2025_shift_gids()
    if gap:
        sdir = DEST / "2025"
        sdir.mkdir(parents=True, exist_ok=True)
        jobs = [("20242025", rpt, gid, sdir / f"{rpt}{gid % 1_000_000:06d}.htm.gz")
                for gid in gap for rpt in ("TH", "TV")]
        run_jobs(jobs, f"2025 shift gap-fill ({len(gap)} games)", t0)
    print("done")


if __name__ == "__main__":
    main()
