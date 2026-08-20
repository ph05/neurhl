"""NeurHL — games_ctx tables for the HTM backfill seasons (2008-2011).

The JSON-era games_ctx came from the api-web envelopes; for HTM seasons the
game meta comes from the PL report header (date + Visitor/Home + on-ice column
abbrevs), joined to games.csv for labels (100%-verified spine), travel_games
for rest/travel, and the same walk-forward era vector as tensorize_games.
sat5 comes from the HTM event shards' reconstructed strength codes.
Output: neurhl/data/tensors/games_ctx_<2008..2011>.parquet (same columns).
"""
import gzip
import json
import re
import sys
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PROC, RAW, TENSORS  # noqa: E402
from data.build_htm import norm_team  # noqa: E402
from data.tensorize_games import ERA_COLS, era_table  # noqa: E402

SEASONS = [2008, 2009, 2010, 2011]
REMAP = {"PHX": "UTA", "ARI": "UTA", "ATL": "WPG"}
_DATE = re.compile(r"[A-Z][a-z]+day, ([A-Z][a-z]+ \d+, \d{4})")
_ONICE = re.compile(r">\s*([A-Z]\.?[A-Z]\.?[A-Z]?)\s+On Ice\s*<")


def header_one(args):
    se, p = args
    gid = (se - 1) * 1_000_000 + 20_000 + int(p.name[4:8])
    with gzip.open(p, "rt", encoding="utf-8", errors="replace") as f:
        head = f.read(80_000)
    m = _DATE.search(head)
    abbrevs = _ONICE.findall(head)
    if not m or len(abbrevs) < 2:
        return None
    date = datetime.strptime(m.group(1), "%B %d, %Y").strftime("%Y-%m-%d")
    return gid, date, norm_team(abbrevs[0]), norm_team(abbrevs[1])


def main():
    maps = json.loads((TENSORS / "maps.json").read_text())
    games = pd.read_csv(PROC / "games.csv")
    travel = pd.read_csv(PROC / "travel_games.csv")
    era = era_table()
    et = maps["event_type"]
    shotfam = [et[c] for c in ("shot-on-goal", "goal", "missed-shot",
                               "blocked-shot")]
    for se in SEASONS:
        out = TENSORS / f"games_ctx_{se}.parquet"
        if out.exists():
            print(f"{se}: exists, skipping")
            continue
        pls = sorted((RAW / "htm_reports" / str(se)).glob("PL*.htm.gz"))
        with Pool(8) as pool:
            heads = [h for h in pool.imap(header_one,
                                          [(se, p) for p in pls], chunksize=32)
                     if h]
        hd = pd.DataFrame(heads, columns=["game_id", "date", "away", "home"])
        hd["home_m"] = hd.home.replace(REMAP)
        hd["away_m"] = hd.away.replace(REMAP)
        lab = games[(games.season_end == se) & (games.game_type == "R")]
        m = hd.merge(lab, left_on=["date", "home_m", "away_m"],
                     right_on=["date", "home", "away"], suffixes=("", "_hr"))
        m = m.merge(travel, left_on=["date", "home_m", "away_m"],
                    right_on=["date", "home", "away"], how="left",
                    suffixes=("", "_tv"))
        ev = pd.read_parquet(TENSORS / f"events_{se}.parquet",
                             columns=["game_id", "event_type", "strength",
                                      "home_event"])
        sat = (ev[(ev.strength == 1551) & ev.event_type.isin(shotfam)]
               .groupby(["game_id", "home_event"]).size().unstack(fill_value=0))
        extra = m.went_ot.astype(bool) | m.went_so.astype(bool)
        home_won = m.home_g > m.away_g
        m["outcome4"] = np.select(
            [home_won & ~extra, ~home_won & ~extra, home_won & extra],
            [0, 1, 2], default=3).astype("uint8")
        m["game_type"] = 2
        first_date = pd.to_datetime(m.date).min()
        m["days_in"] = (pd.to_datetime(m.date) - first_date).dt.days
        m["home_idx"] = m.home.map(maps["team"]).fillna(0).astype("uint8")
        m["away_idx"] = m.away.map(maps["team"]).fillna(0).astype("uint8")
        m["sat5_h"] = m.game_id.map(sat.get(1, pd.Series(dtype=int))).fillna(0)
        m["sat5_a"] = m.game_id.map(sat.get(0, pd.Series(dtype=int))).fillna(0)
        for c in ERA_COLS:
            m[c] = era.loc[se, c] if se in era.index else np.nan
        keep = (["game_id", "game_type", "date", "home_idx", "away_idx",
                 "home_g", "away_g", "outcome4", "sat5_h", "sat5_a", "days_in",
                 "home_rest", "away_rest", "home_km3d", "away_km3d",
                 "home_dtz", "away_dtz"] + ERA_COLS)
        m[keep].to_parquet(out, index=False)
        print(f"{se}: {len(m)}/{len(pls)} games joined "
              f"(header-parsed {len(hd)})")
    print("done")


if __name__ == "__main__":
    main()
