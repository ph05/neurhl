"""Fetch NHL game results (regular season + playoffs) from Hockey-Reference season pages.

One page per season, cached to data/raw/hr_games_<endyear>.csv. Polite: 4s between requests,
browser UA, 3 retries with backoff. Personal, non-commercial use with attribution.
"""
import sys
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

PROJ = Path(__file__).resolve().parents[1]
RAW = PROJ / "data" / "raw"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

FIRST_END_YEAR = 2006   # 2005-06 season (post-lockout), for Elo burn-in
LAST_END_YEAR = 2026    # 2025-26 season


def fetch_page(url: str) -> str:
    last_err = None
    for attempt in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=30)
            if r.status_code == 200:
                return r.text
            last_err = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            last_err = repr(e)
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last_err}")


def normalize(df: pd.DataFrame, game_type: str, end_year: int) -> pd.DataFrame:
    cols = list(df.columns)
    # Goals columns are the ones named 'G' (first = visitor, second = home)
    import re
    g_idx = [i for i, c in enumerate(cols) if re.fullmatch(r"G(\.\d+)?", str(c).strip())]
    if len(g_idx) < 2:
        raise ValueError(f"{end_year} {game_type}: goal columns not found in {cols}")
    ot_col = None
    for i, c in enumerate(cols):
        if str(c).startswith("Unnamed") and i > g_idx[1]:
            ot_col = c
            break
    out = pd.DataFrame({
        "date": df["Date"],
        "away": df["Visitor"],
        "away_g": pd.to_numeric(df.iloc[:, g_idx[0]], errors="coerce"),
        "home": df["Home"],
        "home_g": pd.to_numeric(df.iloc[:, g_idx[1]], errors="coerce"),
        "ot": df[ot_col].fillna("") if ot_col is not None else "",
        "game_type": game_type,
        "season_end": end_year,
    })
    out = out.dropna(subset=["away_g", "home_g"])  # drop unplayed/header rows
    return out


def fetch_season(end_year: int) -> pd.DataFrame:
    url = f"https://www.hockey-reference.com/leagues/NHL_{end_year}_games.html"
    html = fetch_page(url)
    frames = []
    for gid, gtype in (("games", "R"), ("games_playoffs", "P")):
        try:
            tables = pd.read_html(StringIO(html), attrs={"id": gid})
        except ValueError:
            continue  # table absent
        if tables:
            frames.append(normalize(tables[0], gtype, end_year))
    if not frames:
        raise RuntimeError(f"no game tables parsed for {end_year}")
    return pd.concat(frames, ignore_index=True)


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for end_year in range(FIRST_END_YEAR, LAST_END_YEAR + 1):
        if end_year == 2005:
            continue  # lockout, no season
        dest = RAW / f"hr_games_{end_year}.csv"
        if dest.exists() and dest.stat().st_size > 1000:
            print(f"{end_year}: cached ({dest.stat().st_size} bytes)")
            continue
        df = fetch_season(end_year)
        df.to_csv(dest, index=False)
        n_r = (df.game_type == "R").sum()
        n_p = (df.game_type == "P").sum()
        print(f"{end_year}: {n_r} regular, {n_p} playoff games saved")
        sys.stdout.flush()
        time.sleep(4)
    print("done")


if __name__ == "__main__":
    main()
