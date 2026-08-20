"""NeurHL A5 — daily lineup/injury snapshot logger (PLAN_NeurHL acquisition A5).

Forward-looking archival infrastructure ONLY (the v6 D7 odds-logger precedent):
no history exists to backfill, so value accrues by running this daily from now.
Nothing here can feed any backtest — 2026-27 context for future report-only use.

Snapshots, stored raw (parsing deferred to whenever a consumer exists):
  data/raw/lineup_snapshots/<ISO date>/df_lines_<slug>.html.gz   (DailyFaceoff line combos, 32 teams)
  data/raw/lineup_snapshots/<ISO date>/df_injuries.html.gz       (DailyFaceoff injury report)
  data/raw/lineup_snapshots/<ISO date>/puckpedia.html.gz         (PuckPedia cap landing page)

Idempotent per day (existing files skipped); single-threaded, 2 s between
requests; a failed page is logged and skipped, never retried in a loop.
"""
import gzip
import sys
import time
from datetime import date
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import RAW  # noqa: E402

HEADERS = {"User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/605.1.15 (KHTML, like Gecko) "
                          "Version/17.4 Safari/605.1.15")}
SLUGS = [
    "anaheim-ducks", "boston-bruins", "buffalo-sabres", "calgary-flames",
    "carolina-hurricanes", "chicago-blackhawks", "colorado-avalanche",
    "columbus-blue-jackets", "dallas-stars", "detroit-red-wings",
    "edmonton-oilers", "florida-panthers", "los-angeles-kings",
    "minnesota-wild", "montreal-canadiens", "nashville-predators",
    "new-jersey-devils", "new-york-islanders", "new-york-rangers",
    "ottawa-senators", "philadelphia-flyers", "pittsburgh-penguins",
    "san-jose-sharks", "seattle-kraken", "st-louis-blues",
    "tampa-bay-lightning", "toronto-maple-leafs", "utah-mammoth",
    "vancouver-canucks", "vegas-golden-knights", "washington-capitals",
    "winnipeg-jets",
]
FALLBACK = {"utah-mammoth": "utah-hockey-club"}


def grab(ses: requests.Session, url: str, dest: Path) -> str:
    if dest.exists() and dest.stat().st_size > 500:
        return "cached"
    time.sleep(2.0)
    try:
        r = ses.get(url, timeout=60, allow_redirects=True)
    except requests.RequestException:
        return "fail"
    if r.status_code != 200 or len(r.content) < 5000:
        return f"{r.status_code}"
    tmp = dest.with_suffix(".part")
    with gzip.open(tmp, "wb") as f:
        f.write(r.content)
    tmp.rename(dest)
    return "ok"


def main():
    day = date.today().isoformat()
    ddir = RAW / "lineup_snapshots" / day
    ddir.mkdir(parents=True, exist_ok=True)
    ses = requests.Session()
    ses.headers.update(HEADERS)
    counts: dict = {}

    for slug in SLUGS:
        url = f"https://www.dailyfaceoff.com/teams/{slug}/line-combinations"
        res = grab(ses, url, ddir / f"df_lines_{slug}.html.gz")
        if res not in ("ok", "cached") and slug in FALLBACK:
            alt = FALLBACK[slug]
            res = grab(ses, f"https://www.dailyfaceoff.com/teams/{alt}/line-combinations",
                       ddir / f"df_lines_{slug}.html.gz")
        counts[res] = counts.get(res, 0) + 1

    counts["injuries:" + grab(ses, "https://www.dailyfaceoff.com/hockey-player-news/injuries",
                              ddir / "df_injuries.html.gz")] = 1
    counts["puckpedia:" + grab(ses, "https://puckpedia.com/",
                               ddir / "puckpedia.html.gz")] = 1
    print(f"{day}: {counts}")


if __name__ == "__main__":
    main()
