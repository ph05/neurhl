"""NeurHL — per-vantage player vocabularies (PLAN_NeurHL P1/P2).

vocab_<V>.json maps playerId -> contiguous index (1-based; 0 = PAD/unseen) for
every player DRESSED in any corpus game with season_end <= V-1: the JSON era
(scan_pbp player_appearances cache, 2012+) UNION the HTM backfill era
(player_games_2008..2011 shards, once built). A player unseen at vantage V
routes through the career encoder / position mean (cold start). Deterministic
(sorted ids). Vantages 2009..2027 (2027 = the 2026-27 projection vantage).
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import TENSORS  # noqa: E402

CACHE = TENSORS / "_edacache"


def main():
    app = pd.read_parquet(CACHE / "player_appearances.parquet")
    app = app[app.dressed > 0][["player_id", "season_end"]]
    extra = []
    for se in range(2008, 2012):
        p = TENSORS / f"player_games_{se}.parquet"
        if p.exists():
            pg = pd.read_parquet(p, columns=["player_id"])
            pg["season_end"] = se
            extra.append(pg.drop_duplicates())
    if extra:
        app = pd.concat([app] + extra, ignore_index=True)
    first = app.groupby("player_id").season_end.min()
    start = int(first.min()) + 1
    for v in range(start, 2028):
        ids = sorted(first[first <= v - 1].index.tolist())
        vocab = {int(pid): i + 1 for i, pid in enumerate(ids)}
        out = TENSORS / f"vocab_{v}.json"
        out.write_text(json.dumps(vocab, sort_keys=True))
        print(f"vocab_{v}: {len(vocab)} players")
    print("done")


if __name__ == "__main__":
    main()
