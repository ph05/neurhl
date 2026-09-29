"""Pre-NHL and non-NHL season records for every player with a landing file
(NeurHL rookie priors, PLAN_NeurHL_1_1 A15).

Reads data/raw/player_landing/*.json.gz (the NHL's api-web player landing:
season totals in every league a player has played: AHL, OHL, WHL, QMJHL,
NCAA, USHL, KHL, SHL, Liiga, NL, DEL and others, plus international events).
Regular seasons only (gameTypeId 2). Writes data/tensors/prenhl_seasons.parquet:
  player_id, season_end, league, gp, g, a, p, age, pos_group, draft_overall, birth_year
"""
import gzip
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RAW, TENSORS  # noqa: E402

POS = {"C": 0, "L": 0, "R": 0, "D": 1, "G": 2}


def parse(path: Path):
    with gzip.open(path, "rt") as f:
        j = json.load(f)
    pid = int(path.name.split(".")[0])
    birth = int((j.get("birthDate") or "1900-01-01")[:4])
    pos = POS.get(j.get("position"), 0)
    dd = (j.get("draftDetails") or {}).get("overallPick") or 0
    rows = []
    for s in j.get("seasonTotals", []):
        if s.get("gameTypeId") != 2 or not s.get("gamesPlayed"):
            continue
        se = int(str(s["season"])[4:])
        rows.append((pid, se, s.get("leagueAbbrev") or "?", s["gamesPlayed"], s.get("goals") or 0,
                     s.get("assists") or 0, s.get("points") or 0, se - birth, pos, dd, birth))
    return rows


def main():
    files = sorted((RAW / "player_landing").glob("*.json.gz"))
    with Pool(8) as pool:
        rows = [r for rs in pool.imap_unordered(parse, files, chunksize=64) for r in rs]
    df = pd.DataFrame(rows, columns=["player_id", "season_end", "league", "gp", "g", "a", "p", "age",
                                     "pos_group", "draft_overall", "birth_year"])
    # a player traded within a league has one row per team: sum them
    df = df.groupby(["player_id", "season_end", "league"], as_index=False).agg(
        gp=("gp", "sum"), g=("g", "sum"), a=("a", "sum"), p=("p", "sum"), age=("age", "first"),
        pos_group=("pos_group", "first"), draft_overall=("draft_overall", "first"),
        birth_year=("birth_year", "first"))
    df.to_parquet(TENSORS / "prenhl_seasons.parquet", index=False)
    print(f"{len(files)} players, {len(df):,} player-league-seasons")
    print(df.league.value_counts().head(30).to_string())


if __name__ == "__main__":
    main()
