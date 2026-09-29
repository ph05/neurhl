"""NeurHL A5 — daily lineup/injury snapshot logger (PLAN_NeurHL acquisition A5).

Forward-looking archival infrastructure ONLY (the v6 D7 odds-logger precedent):
no history exists to backfill, so value accrues by running this daily from now.
Nothing here can feed any backtest — 2026-27 context for future report-only use.

Snapshots, stored raw (a consumer now exists: NeurHL-3 G6 parsing):
  data/raw/lineup_snapshots/<ISO date>/df_lines_<slug>.html.gz   (DailyFaceoff line combos, 32 teams)
  data/raw/lineup_snapshots/<ISO date>/df_injuries.html.gz       (DailyFaceoff injury report)
  data/raw/lineup_snapshots/<ISO date>/df_goalies.html.gz        (DailyFaceoff starting goalies — D4)
  data/raw/lineup_snapshots/<ISO date>/puckpedia.html.gz         (PuckPedia cap landing page)
  data/raw/lineup_snapshots/<ISO date>/mp_injuries.csv.gz        (MoneyPuck injury list: status,
                                                                  expected return, games still to miss)

Scheduling: this must run DAILY through 2026-27 (launchd plist
com.neurhl.snapshot — see neurhl/data/fetch/README_scheduler.md).

Idempotent per day (existing files skipped); single-threaded, 2 s between
requests; a failed page is logged and skipped, never retried in a loop.

--intraday (PLAN_NeurHL4 LIVE; launchd com.neurhl.intraday, every 30 min on
game days): writes data/raw/lineup_snapshots/<ISO date>/<HHMM>/ (local time)
with df_goalies.html.gz plus df_lines_<slug>.html.gz for ONLY the teams playing
that date (teams from https://api-web.nhle.com/v1/score/<date>, falling back to
the local 2026-27 schedule files). Never skips: every run fetches afresh.
Days without games write nothing. The default (daily) mode is unchanged.

CLI: python neurhl/data/fetch/snapshot_lineups.py [--intraday]
"""
import argparse
import glob
import gzip
import json
import sys
import time
from datetime import date, datetime
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


def grab(ses: requests.Session, url: str, dest: Path, skip_existing: bool = True) -> str:
    if skip_existing and dest.exists() and dest.stat().st_size > 500:
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


MP_INJ = "https://moneypuck.com/moneypuck/playerData/playerNews/current_injuries.csv"
MP_INJ_TIME = "https://moneypuck.com/moneypuck/playerData/playerNews/injury_update_time.txt"


def grab_mp_injuries(ses: requests.Session, ddir: Path) -> str:
    """MoneyPuck's current injury list (player id, team, status, expected return
    date, games still to miss) and its update time, stored gzipped as fetched.
    A small file is fine here; the header is checked instead of the size."""
    time.sleep(2.0)
    try:
        r = ses.get(MP_INJ, timeout=60)
        t = ses.get(MP_INJ_TIME, timeout=60)
    except requests.RequestException:
        return "fail"
    if r.status_code != 200 or not r.text.startswith("playerId,"):
        return f"{r.status_code}"
    for name, content in (("mp_injuries.csv.gz", r.content),
                          ("mp_injury_update_time.txt.gz", t.content if t.status_code == 200 else b"")):
        tmp = (ddir / name).with_suffix(".part")
        with gzip.open(tmp, "wb") as f:
            f.write(content)
        tmp.rename(ddir / name)
    return "ok"


def main_daily():
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
    counts["goalies:" + grab(ses, "https://www.dailyfaceoff.com/starting-goalies",
                             ddir / "df_goalies.html.gz")] = 1
    counts["puckpedia:" + grab(ses, "https://puckpedia.com/",
                               ddir / "puckpedia.html.gz")] = 1
    counts["mp_injuries:" + grab_mp_injuries(ses, ddir)] = 1
    print(f"{day}: {counts}")


# ------------------------------------------------------------------ intraday
NHL_TO_SLUG = {  # inverse of neurhl/live/parse_dailyfaceoff.SLUG_TO_NHL (canonical slugs)
    "ANA": "anaheim-ducks", "BOS": "boston-bruins", "BUF": "buffalo-sabres",
    "CGY": "calgary-flames", "CAR": "carolina-hurricanes", "CHI": "chicago-blackhawks",
    "COL": "colorado-avalanche", "CBJ": "columbus-blue-jackets", "DAL": "dallas-stars",
    "DET": "detroit-red-wings", "EDM": "edmonton-oilers", "FLA": "florida-panthers",
    "LAK": "los-angeles-kings", "MIN": "minnesota-wild", "MTL": "montreal-canadiens",
    "NSH": "nashville-predators", "NJD": "new-jersey-devils", "NYI": "new-york-islanders",
    "NYR": "new-york-rangers", "OTT": "ottawa-senators", "PHI": "philadelphia-flyers",
    "PIT": "pittsburgh-penguins", "SJS": "san-jose-sharks", "SEA": "seattle-kraken",
    "STL": "st-louis-blues", "TBL": "tampa-bay-lightning", "TOR": "toronto-maple-leafs",
    "UTA": "utah-mammoth", "VAN": "vancouver-canucks", "VGK": "vegas-golden-knights",
    "WSH": "washington-capitals", "WPG": "winnipeg-jets",
}
assert sorted(NHL_TO_SLUG.values()) == sorted(SLUGS)


def teams_playing(ses: requests.Session, day: str) -> tuple[list[str], str]:
    """NHL abbreviations playing on `day` (regular season or playoffs) and the source used."""
    try:
        r = ses.get(f"https://api-web.nhle.com/v1/score/{day}", timeout=30)
        r.raise_for_status()
        games = [g for g in r.json().get("games", [])
                 if g.get("gameDate", day) == day and int(g.get("gameType", 2)) in (2, 3)]
        return sorted({g[s]["abbrev"] for g in games for s in ("homeTeam", "awayTeam")}), "nhl_api"
    except (requests.RequestException, ValueError, KeyError, TypeError) as e:
        print(f"  score endpoint failed ({e}); using local schedule files", file=sys.stderr)
    teams = set()
    for f in glob.glob(str(RAW / "nhl_sched_*_20262027.json")):
        try:
            for g in json.loads(Path(f).read_text()).get("games", []):
                if g.get("gameDate") == day and int(g.get("gameType", 2)) in (2, 3):
                    teams |= {g["homeTeam"]["abbrev"], g["awayTeam"]["abbrev"]}
        except (ValueError, KeyError, TypeError):
            continue
    return sorted(teams), "schedule"


def main_intraday():
    now = datetime.now()
    day = now.date().isoformat()
    ses = requests.Session()
    ses.headers.update(HEADERS)
    teams, src = teams_playing(ses, day)
    if not teams:
        print(f"{now:%Y-%m-%d %H:%M} intraday: no games on {day} ({src}); nothing fetched")
        return
    ddir = RAW / "lineup_snapshots" / day / f"{now:%H%M}"
    ddir.mkdir(parents=True, exist_ok=True)
    counts: dict = {"goalies:" + grab(ses, "https://www.dailyfaceoff.com/starting-goalies",
                                      ddir / "df_goalies.html.gz", skip_existing=False): 1}
    counts["mp_injuries:" + grab_mp_injuries(ses, ddir)] = 1
    unknown = [t for t in teams if t not in NHL_TO_SLUG]
    for team in teams:
        slug = NHL_TO_SLUG.get(team)
        if slug is None:
            continue
        dest = ddir / f"df_lines_{slug}.html.gz"
        res = grab(ses, f"https://www.dailyfaceoff.com/teams/{slug}/line-combinations",
                   dest, skip_existing=False)
        if res != "ok" and slug in FALLBACK:
            res = grab(ses, f"https://www.dailyfaceoff.com/teams/{FALLBACK[slug]}/line-combinations",
                       dest, skip_existing=False)
        counts[res] = counts.get(res, 0) + 1
    print(f"{now:%Y-%m-%d %H:%M} intraday -> {ddir.relative_to(RAW)}: {len(teams)} teams ({src}) {counts}"
          + (f"; unknown abbrevs {unknown}" if unknown else ""))


def main():
    ap = argparse.ArgumentParser(description="DailyFaceoff/PuckPedia lineup snapshots.")
    ap.add_argument("--intraday", action="store_true",
                    help="goalies + lines of teams playing today into <date>/<HHMM>/, never skipping")
    a = ap.parse_args()
    if a.intraday:
        main_intraday()
    else:
        main_daily()


if __name__ == "__main__":
    main()
