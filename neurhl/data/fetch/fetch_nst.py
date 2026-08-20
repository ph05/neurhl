"""NeurHL A2 — Natural Stat Trick season tables (PLAN_NeurHL acquisition A2).

Archival fetch of NST season aggregate tables, seasons 2007-08 .. 2025-26:
  team tables:   sit in {5v5, sva (score-and-venue adj), pp, pk, all}
  skater std:    sit in {5v5, all}
  skater on-ice: sit in {5v5, sva}
  goalies:       sit in {5v5, all}

Raw HTML is stored (parsing happens in the EDA/build layer, mirroring the
hr_html convention): data/raw/nst/<season_end>/<kind>_<sit>.html.gz (gitignored).

Politeness: single-threaded, >=12 s between requests (NST asks for slow access);
resumable; aborts after 3 consecutive non-200s so a block never turns into a
hammering loop.
"""
import gzip
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import RAW  # noqa: E402

DEST = RAW / "nst"
SLEEP = 12.0
ENDS = list(range(2008, 2027))   # season_end 2008..2026
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.naturalstattrick.com/",
}


def urls_for(end: int) -> dict[str, str]:
    s = f"{end - 1}{end}"
    team = ("https://www.naturalstattrick.com/teamtable.php?"
            f"fromseason={s}&thruseason={s}&stype=2&sit={{sit}}&score=all&rate=n"
            "&team=all&loc=B&gpf=410&fd=&td=")
    player = ("https://www.naturalstattrick.com/playerteams.php?"
              f"fromseason={s}&thruseason={s}&stype=2&sit={{sit}}&score=all"
              "&stdoi={stdoi}&rate=n&team=ALL&pos=S&loc=B&toi=0&gpfilt=none"
              "&fd=&td=&tgp=410&lines=single&draftteam=ALL")
    out = {}
    for sit in ("5v5", "sva", "pp", "pk", "all"):
        out[f"team_{sit}"] = team.format(sit=sit)
    for sit in ("5v5", "all"):
        out[f"skater_std_{sit}"] = player.format(sit=sit, stdoi="std")
    for sit in ("5v5", "sva"):
        out[f"skater_oi_{sit}"] = player.format(sit=sit, stdoi="oi")
    for sit in ("5v5", "all"):
        out[f"goalie_{sit}"] = player.format(sit=sit, stdoi="g")
    return out


def main():
    t0, bad_streak = time.time(), 0
    ses = requests.Session()
    ses.headers.update(HEADERS)
    for end in ENDS:
        sdir = DEST / str(end)
        sdir.mkdir(parents=True, exist_ok=True)
        counts: dict = {}
        for kind, url in urls_for(end).items():
            dest = sdir / f"{kind}.html.gz"
            if dest.exists() and dest.stat().st_size > 500:
                counts["cached"] = counts.get("cached", 0) + 1
                continue
            time.sleep(SLEEP)
            try:
                r = ses.get(url, timeout=90)
            except requests.RequestException:
                counts["fail"] = counts.get("fail", 0) + 1
                bad_streak += 1
                continue
            if r.status_code == 200 and b"<table" in r.content:
                tmp = dest.with_suffix(".part")
                with gzip.open(tmp, "wb") as f:
                    f.write(r.content)
                tmp.rename(dest)
                counts["ok"] = counts.get("ok", 0) + 1
                bad_streak = 0
            else:
                counts[str(r.status_code)] = counts.get(str(r.status_code), 0) + 1
                bad_streak += 1
            if bad_streak >= 3:
                print(f"ABORT: 3 consecutive failures at {end}/{kind} "
                      f"(last status {r.status_code if 'r' in dir() else 'n/a'}) — "
                      "likely blocked; rerun later to resume.")
                sys.exit(1)
        print(f"{end}: {counts}  [{time.time() - t0:.0f}s elapsed]")
        sys.stdout.flush()
    print("done")


if __name__ == "__main__":
    main()
