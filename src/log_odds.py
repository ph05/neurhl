"""Cross-book Stanley Cup futures snapshot logger (PLAN_V6 D7).

Scrapes the vegasinsider NHL futures table (the same source as the 2026-08-18
board in data/market/nhl_cup_2027_best_price.csv) and appends one dated snapshot
to data/market/odds_log/cup_<date>.csv (long form: date, team, book, american).
Forward-looking infrastructure for CLV/market-weight calibration; nothing here
feeds any backtest. Intended to be run daily-ish; safe to rerun (overwrites the
same day's file).
"""
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import requests

PROJ = Path(__file__).resolve().parents[1]
DEST = PROJ / "data" / "market" / "odds_log"
URL = "https://www.vegasinsider.com/nhl/odds/futures/"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
FRAN = {"ARI": "UTA", "TB": "TBL", "LA": "LAK", "SJ": "SJS", "NJ": "NJD",
        "MON": "MTL", "CLB": "CBJ", "WAS": "WSH", "VGS": "VGK", "WIN": "WPG",
        "CAL": "CGY", "NSH": "NSH", "UTAH": "UTA"}


def parse(html: str) -> pd.DataFrame:
    books = re.findall(r'<th class="book-logo">.*?alt="([a-z0-9]+)"', html,
                       re.DOTALL)
    rows = []
    for m in re.finditer(r'data-abbr="([A-Z]{2,3})"(.*?)</tr>', html, re.DOTALL):
        abbr, block = m.group(1), m.group(2)
        tds = re.findall(r'<td class="game-odds"[^>]*>(.*?)</td>', block,
                         re.DOTALL)
        for book, td in zip(books, tds):
            v = re.search(r">\s*([+-]\d{3,6})\s*<", td)
            if v:
                rows.append({"team": FRAN.get(abbr, abbr), "book": book,
                             "american": int(v.group(1))})
    return pd.DataFrame(rows).drop_duplicates(["team", "book"])


def main():
    r = requests.get(URL, headers=UA, timeout=60)
    r.raise_for_status()
    df = parse(r.text)
    if len(df) < 60:
        print(f"WARNING: only {len(df)} rows parsed — page layout may have "
              f"changed; snapshot NOT written")
        sys.exit(1)
    today = date.today().isoformat()
    df.insert(0, "date", today)
    DEST.mkdir(parents=True, exist_ok=True)
    out = DEST / f"cup_{today}.csv"
    df.to_csv(out, index=False)
    best = df.loc[df.groupby("team").american.idxmax()]
    print(f"{out.name}: {df.team.nunique()} teams x {df.book.nunique()} books, "
          f"{len(df)} quotes")
    print(best.sort_values("american").head(8)[["team", "book", "american"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
