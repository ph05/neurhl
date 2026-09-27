"""Parse right-rail records into per-game official stats and officials.

Output data/tensors/rr_{s}.parquet, one row per regular-season game:
  game_id, pp_opps_h, pp_opps_a, ppg_h, ppg_a, pim_h, pim_a, hits_h, hits_a,
  blk_h, blk_a, give_h, give_a, take_h, take_a, sog_h, sog_a,
  ref1, ref2 (full names), coach_h, coach_a, n_scratch_h, n_scratch_a

Then updates tgx_{s}.parquet: pp_opps becomes the OFFICIAL count where a
right-rail record exists (the stint-derived count, which matches official
counts in only ~60% of games, is kept as pp_opps_stint).

Usage: ... python neurhl/data/build_right_rail.py [--seasons 2012-2026]
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RAW, TENSORS  # noqa: E402

CATS = {"powerPlay": "pp", "pim": "pim", "hits": "hits", "blockedShots": "blk",
        "giveaways": "give", "takeaways": "take", "sog": "sog"}


def parse(p: Path) -> dict:
    d = json.load(gzip.open(p, "rt"))
    row = {"game_id": int(p.name.split(".")[0])}
    for x in d.get("teamGameStats", []):
        k = CATS.get(x.get("category"))
        if not k:
            continue
        for side, key in (("h", "homeValue"), ("a", "awayValue")):
            v = x.get(key)
            if k == "pp" and isinstance(v, str) and "/" in v:
                g, o = v.split("/")
                row[f"ppg_{side}"], row[f"pp_opps_{side}"] = float(g), float(o)
            elif k != "pp":
                try:
                    row[f"{k}_{side}"] = float(v)
                except (TypeError, ValueError):
                    pass
    gi = d.get("gameInfo", {})
    refs = [r.get("fullName", {}).get("default") or r.get("default")
            for r in gi.get("referees", [])]
    row["ref1"], row["ref2"] = (refs + [None, None])[:2]
    for side, key in (("h", "homeTeam"), ("a", "awayTeam")):
        t = gi.get(key, {})
        row[f"coach_{side}"] = (t.get("headCoach") or {}).get("default")
        row[f"n_scratch_{side}"] = len(t.get("scratches", []) or [])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2012-2026")
    a, b = map(int, ap.parse_args().seasons.split("-"))
    for s in range(a, b + 1):
        files = sorted((RAW / "right_rail" / str(s)).glob("*.json.gz"))
        if not files:
            print(f"{s}: no right-rail files")
            continue
        rr = pd.DataFrame([parse(f) for f in files])
        rr.to_parquet(TENSORS / f"rr_{s}.parquet", index=False)
        t = pd.read_parquet(TENSORS / f"tgx_{s}.parquet")
        if "pp_opps_stint" not in t:
            t["pp_opps_stint"] = t.pp_opps
        off = pd.concat([rr[["game_id", "pp_opps_h"]].assign(is_home=1).rename(columns={"pp_opps_h": "o"}),
                         rr[["game_id", "pp_opps_a"]].assign(is_home=0).rename(columns={"pp_opps_a": "o"})])
        t = t.drop(columns=["pp_opps"]).merge(off, on=["game_id", "is_home"], how="left")
        t["pp_opps"] = t.o.fillna(t.pp_opps_stint).astype("float32")
        t = t.drop(columns=["o"])
        t.to_parquet(TENSORS / f"tgx_{s}.parquet", index=False)
        cov = rr.pp_opps_h.notna().mean()
        agree = (t.pp_opps == t.pp_opps_stint).mean()
        print(f"{s}: {len(rr):,} games, official PP coverage {cov:.3f}, "
              f"official/team-game {t.pp_opps.mean():.2f} (stint {t.pp_opps_stint.mean():.2f}, "
              f"exact agreement {agree:.2f}), refs found {rr.ref1.notna().mean():.3f}")


if __name__ == "__main__":
    main()
