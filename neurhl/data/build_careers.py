"""NeurHL — career table from player_landing files (career-encoder input).

Parses every data/raw/player_landing/<playerId>.json.gz (all-league season
totals back to 1999, 100% of dressed players after the A6 gap-fill) into
neurhl/data/tensors/careers.parquet:
  player_id, season_end, league_idx, game_type, gp, g, a, p, age
plus neurhl/data/tensors/career_bios.parquet:
  player_id, pos_group, height_in, weight_lb, shoots_left, draft_round,
  draft_overall, birth_year
League vocabulary (top leagues by row count + OTHER) frozen to
tensors/league_map.json on first build. Regular-season rows only (gameTypeId 2).
"""
import gzip
import json
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RAW, TENSORS  # noqa: E402

POS_GROUP = {"C": 0, "L": 0, "R": 0, "D": 1, "G": 2}


def parse_one(path: Path):
    with gzip.open(path, "rt") as f:
        j = json.load(f)
    pid = int(path.name.split(".")[0])
    birth = int(j.get("birthDate", "1900-01-01")[:4])
    dd = j.get("draftDetails") or {}
    bio = (pid, POS_GROUP.get(j.get("position"), 0),
           j.get("heightInInches") or 0, j.get("weightInPounds") or 0,
           int(j.get("shootsCatches") == "L"), dd.get("round", 0),
           dd.get("overallPick", 0), birth)
    rows = []
    for s in j.get("seasonTotals", []):
        if s.get("gameTypeId") != 2 or not s.get("gamesPlayed"):
            continue
        se = int(str(s["season"])[4:])
        rows.append((pid, se, s.get("leagueAbbrev") or "?", s["gamesPlayed"],
                     s.get("goals") or 0, s.get("assists") or 0,
                     s.get("points") or 0, se - birth))
    return bio, rows


def main():
    files = sorted((RAW / "player_landing").glob("*.json.gz"))
    print(f"parsing {len(files)} landing files ...")
    bios, rows = [], []
    with Pool(8) as pool:
        for bio, r in pool.imap_unordered(parse_one, files, chunksize=64):
            bios.append(bio)
            rows.extend(r)
    df = pd.DataFrame(rows, columns=["player_id", "season_end", "league",
                                     "gp", "g", "a", "p", "age"])
    lm_path = TENSORS / "league_map.json"
    if lm_path.exists():
        lmap = json.loads(lm_path.read_text())
    else:
        top = [l for l, _ in Counter(df.league).most_common(28)]
        lmap = {l: i + 1 for i, l in enumerate(sorted(top))}   # 0 = OTHER
        lm_path.write_text(json.dumps(lmap, indent=1, sort_keys=True))
    df["league_idx"] = df.league.map(lmap).fillna(0).astype("int8")
    df.drop(columns=["league"]).to_parquet(TENSORS / "careers.parquet",
                                           index=False)
    pd.DataFrame(bios, columns=["player_id", "pos_group", "height_in",
                                "weight_lb", "shoots_left", "draft_round",
                                "draft_overall", "birth_year"]
                 ).to_parquet(TENSORS / "career_bios.parquet", index=False)
    print(f"careers: {len(df):,} season rows, {len(bios):,} players, "
          f"{df.league_idx.nunique()} league ids")


if __name__ == "__main__":
    main()
