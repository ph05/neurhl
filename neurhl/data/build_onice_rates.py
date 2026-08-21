"""NeurHL — per-player-per-game on-ice rates (foundation of the hierarchical model).

For every skater-game: shot attempts FOR and AGAINST while that player was on
the ice, split all-situations and 5v5, plus close-range (<25 ft) attempts.
This is the supervision target for Layer 1 (player performance), and it is
~100x more abundant than team-game outcomes: ~44k skater-games per season
against ~1.2k team-games.

Uses the on-ice slots already reconstructed in the event shards, so no new
data is required. Output: neurhl/data/tensors/onice_rates_<season>.parquet with
one row per (game_id, player_id):
  cf, ca            attempts for/against while on ice (all situations)
  cf5, ca5          same, 5v5 only
  clf, cla          close-range (<25 ft) attempts for/against
  is_home           side
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import TENSORS  # noqa: E402

ON_H = [f"h_on{i}" for i in range(7)]
ON_A = [f"a_on{i}" for i in range(7)]


def main():
    maps = json.loads((TENSORS / "maps.json").read_text())
    et = maps["event_type"]
    shot_types = [et[c] for c in ("shot-on-goal", "goal", "missed-shot",
                                  "blocked-shot") if c in et]
    seasons = sorted(int(p.stem.split("_")[1]) for p in
                     TENSORS.glob("events_2*.parquet")
                     if p.stem.split("_")[1].isdigit())
    for se in seasons:
        out_p = TENSORS / f"onice_rates_{se}.parquet"
        if out_p.exists():
            print(f"{se}: exists, skipping")
            continue
        ev = pd.read_parquet(TENSORS / f"events_{se}.parquet",
                             columns=["game_id", "game_type", "event_type",
                                      "home_event", "strength", "xn", "yn",
                                      "has_coord"] + ON_H + ON_A)
        ev = ev[(ev.game_type == 2) & (ev.home_event >= 0)
                & ev.event_type.isin(shot_types)]
        dist = np.hypot(89 - ev.xn.abs().clip(upper=100), ev.yn)
        is_close = (dist < 25) & ev.has_coord.astype(bool)
        is_5v5 = ev.strength == 1551
        home_shot = ev.home_event.to_numpy() == 1
        gid = ev.game_id.to_numpy()

        recs = {}
        for cols, side_is_home in ((ON_H, True), (ON_A, False)):
            slots = ev[cols].to_numpy()                       # E x 7
            # a shot is FOR this side when the shooting team is this side
            forr = home_shot if side_is_home else ~home_shot
            for c in range(slots.shape[1]):
                pid = slots[:, c]
                keep = pid > 0
                if not keep.any():
                    continue
                df = pd.DataFrame({
                    "game_id": gid[keep], "player_id": pid[keep],
                    "is_home": side_is_home,
                    "cf": forr[keep].astype(np.int32),
                    "ca": (~forr[keep]).astype(np.int32),
                    "cf5": (forr[keep] & is_5v5.to_numpy()[keep]).astype(np.int32),
                    "ca5": (~forr[keep] & is_5v5.to_numpy()[keep]).astype(np.int32),
                    "clf": (forr[keep] & is_close.to_numpy()[keep]).astype(np.int32),
                    "cla": (~forr[keep] & is_close.to_numpy()[keep]).astype(np.int32),
                })
                key = "h" if side_is_home else "a"
                recs.setdefault(key, []).append(df)
        allr = pd.concat([d for v in recs.values() for d in v],
                         ignore_index=True)
        agg = (allr.groupby(["game_id", "player_id", "is_home"],
                            as_index=False)
               [["cf", "ca", "cf5", "ca5", "clf", "cla"]].sum())
        agg.to_parquet(out_p, index=False)
        print(f"{se}: {len(agg):,} player-games -> {out_p.name}")
        sys.stdout.flush()
    print("done")


if __name__ == "__main__":
    main()
