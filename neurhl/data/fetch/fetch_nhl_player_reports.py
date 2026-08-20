"""NeurHL A3 — official NHL stats-rest player season reports (PLAN_NeurHL A3).

Fetches api.nhle.com/stats/rest player-season aggregate reports, seasons
2005-06 .. 2025-26, regular season (gameTypeId=2) and playoffs (gameTypeId=3):
  skater: summary, realtime, faceoffwins, shooting, timeonice
  goalie: summary, advanced

These are the only official player aggregates that cover the pre-2012 era where
no PBP corpus exists, and the independent cross-check for panel tables.
Storage: data/raw/nhl_player_reports/<pos>_<report>_<season_end>[_po].json.gz.
Resumable; paginates if limit=-1 is ever capped; tolerates empty reports in old
seasons (recorded as "empty", not an error).
"""
import gzip
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import RAW, UA  # noqa: E402

DEST = RAW / "nhl_player_reports"
ENDS = list(range(2006, 2027))
REPORTS = [("skater", r) for r in
           ("summary", "realtime", "faceoffwins", "summaryshooting",
            "shottype", "timeonice")] + \
          [("goalie", r) for r in ("summary", "advanced")]


def fetch_report(pos: str, report: str, season: str, gtype: int) -> list | None:
    rows, start = [], 0
    while True:
        url = (f"https://api.nhle.com/stats/rest/en/{pos}/{report}?limit=-1&start={start}"
               f"&cayenneExp=seasonId={season}%20and%20gameTypeId={gtype}")
        for attempt in range(4):
            try:
                r = requests.get(url, headers=UA, timeout=60)
                if r.status_code == 200:
                    blob = r.json()
                    break
                if r.status_code == 404:
                    return None
            except requests.RequestException:
                pass
            time.sleep(1.5 * (attempt + 1))
        else:
            return None
        rows.extend(blob.get("data", []))
        total = blob.get("total", len(rows))
        if len(rows) >= total or not blob.get("data"):
            return rows
        start = len(rows)


def fetch_one(args) -> str:
    pos, report, end, gtype = args
    tag = "" if gtype == 2 else "_po"
    dest = DEST / f"{pos}_{report}_{end}{tag}.json.gz"
    if dest.exists() and dest.stat().st_size > 100:
        return "cached"
    time.sleep(0.25)
    rows = fetch_report(pos, report, f"{end - 1}{end}", gtype)
    if rows is None:
        return "fail"
    if not rows:
        return "empty"
    tmp = dest.with_suffix(".part")
    with gzip.open(tmp, "wt") as f:
        json.dump(rows, f, separators=(",", ":"))
    tmp.rename(dest)
    return "ok"


def main():
    t0 = time.time()
    DEST.mkdir(parents=True, exist_ok=True)
    jobs = [(pos, report, end, gtype)
            for end in ENDS for pos, report in REPORTS for gtype in (2, 3)]
    counts: dict = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        for res in ex.map(fetch_one, jobs):
            counts[res] = counts.get(res, 0) + 1
    print(f"{len(jobs)} report-seasons -> {counts}  [{time.time() - t0:.0f}s]")
    if counts.get("fail"):
        print(f"  WARNING: {counts['fail']} failures (rerun to resume)")
    print("done")


if __name__ == "__main__":
    main()
