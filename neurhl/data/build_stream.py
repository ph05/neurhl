"""NeurHL-2 — S0: the unified event stream the simulator trains on.

One row per event, per game, in order, carrying everything the marked temporal
point process conditions on: time since the last event, event type, location in
the ATTACKING frame, the on-ice sets for both sides, strength, score state, and
expected goals for shot events.

Three things this adds over the raw event shards:

  1. **Coordinates for 2008-2011.** The NHL feed carries none before 2012
     (`events_2011.has_coord` is 0.000). MoneyPuck's distance and angle are
     flip-invariant and complete back to 2008, so `(xa, ya)` is reconstructed as
     `89 - d*cos(theta), d*sin(theta)`. Verified against 2016 where both sources
     exist: 97.3% of x and 97.6% of y within one foot, median error 0.00.
     This covers SHOTS only — MoneyPuck has no rows for hits, faceoffs or
     giveaways — so non-shot events genuinely have no location before 2012 and
     carry `has_xy = 0` rather than a fabricated zero.
  2. **Expected goals** joined per shot from the walk-forward L0 model, so the
     simulator never has to re-derive shot quality and never sees an xG fitted
     on its own future.
  3. **Derived strength and score state**, computed from the on-ice slots rather
     than trusted from `situationCode` — which is corrupt in ~10% of 2020 games
     (see neurhl/eval/validate_stints.py).

Everything is oriented so that "for" means the team that owns the event. A model
fed raw home/away fields learns "the home team shoots more" instead of "a team on
the power play shoots more".

Shootouts (period 5+) are excluded: they are a skills competition with different
physics, and the plan scores regular-season regulation + OT.

Run: uv run --no-project --python 3.12 --with numpy --with "pandas<3" \
     --with pyarrow python neurhl/data/build_stream.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import TENSORS  # noqa: E402
import manifest as MAN  # noqa: E402

CFG = {"version": 1, "max_period": 4}
H_ON = [f"h_on{i}" for i in range(7)]
A_ON = [f"a_on{i}" for i in range(7)]
SHOT_NAMES = ("shot-on-goal", "goal", "missed-shot")
MP_MAP = {"SHOT": "shot-on-goal", "MISS": "missed-shot", "GOAL": "goal"}


def strength_key(n_f, n_a, g_f, g_a) -> np.ndarray:
    """Event-team-relative strength class. Mirrors build_stints.strength_enum
    so the two tables agree, but from the event owner's point of view."""
    k = np.full(len(n_f), 9, np.int8)
    k[(n_f == 5) & (n_a == 5)] = 0
    k[(n_f == 4) & (n_a == 4)] = 5
    k[(n_f == 3) & (n_a == 3)] = 6
    d = n_f - n_a
    k[d == 1] = 1
    k[d >= 2] = 2
    k[d == -1] = 3
    k[d <= -2] = 4
    k[g_a == 0] = 7          # shooting at an empty net
    k[g_f == 0] = 8          # own net empty
    return k


def build_season(se: int, maps: dict) -> pd.DataFrame:
    et = maps["event_type"]
    ev = pd.read_parquet(TENSORS / f"events_{se}.parquet")
    ev = ev[(ev.period >= 1) & (ev.period <= CFG["max_period"])].copy()
    ev = ev.sort_values(["game_id", "period", "t", "event_idx"],
                        kind="stable").reset_index(drop=True)

    # ---------------------------------------------------------- coordinates
    ev["xa"] = ev.xn.astype("float32")
    ev["ya"] = ev.yn.astype("float32")
    ev["has_xy"] = ev.has_coord.astype("uint8")

    mp_p = TENSORS / f"mp_shots_{se}.parquet"
    if mp_p.exists():
        mp = pd.read_parquet(mp_p, columns=["game_id", "time", "event",
                                            "shooterPlayerId", "shotDistance",
                                            "shotAngle"])
        mp["event_type"] = mp.event.map(lambda v: et[MP_MAP[v]]).astype("int16")
        th = np.deg2rad(mp.shotAngle.to_numpy(np.float64))
        mp["xr"] = (89.0 - mp.shotDistance.to_numpy() * np.cos(th)).astype("float32")
        mp["yr"] = (mp.shotDistance.to_numpy() * np.sin(th)).astype("float32")
        mp = mp.drop_duplicates(["game_id", "time", "event_type",
                                 "shooterPlayerId"])
        ev = ev.merge(
            mp[["game_id", "time", "event_type", "shooterPlayerId", "xr", "yr"]],
            left_on=["game_id", "t", "event_type", "p1"],
            right_on=["game_id", "time", "event_type", "shooterPlayerId"],
            how="left")
        fill = (ev.has_xy == 0) & ev.xr.notna()
        ev.loc[fill, "xa"] = ev.loc[fill, "xr"]
        ev.loc[fill, "ya"] = ev.loc[fill, "yr"]
        ev.loc[fill, "has_xy"] = 1
        ev = ev.drop(columns=["time", "shooterPlayerId", "xr", "yr"])

    # ---------------------------------------------------------------- xG
    ev["xg"] = np.nan
    xg_p = TENSORS / f"xg_shots_{se}.parquet"
    if xg_p.exists():
        xg = pd.read_parquet(xg_p, columns=["game_id", "time", "shooter", "xg"])
        xg = xg.drop_duplicates(["game_id", "time", "shooter"])
        ev = ev.merge(xg.rename(columns={"xg": "_xg"}),
                      left_on=["game_id", "t", "p1"],
                      right_on=["game_id", "time", "shooter"], how="left")
        ev["xg"] = ev._xg
        ev = ev.drop(columns=["time", "shooter", "_xg"])
    ev["has_xg"] = ev.xg.notna().astype("uint8")
    ev["xg"] = ev.xg.fillna(0.0).astype("float32")

    # -------------------------------------------------- on-ice, strength, score
    h = ev[H_ON].to_numpy(np.int64)
    a = ev[A_ON].to_numpy(np.int64)
    n_h = (h[:, 1:] > 0).sum(1).astype(np.int8)
    n_a = (a[:, 1:] > 0).sum(1).astype(np.int8)
    g_h = (h[:, 0] > 0).astype(np.int8)
    g_a = (a[:, 0] > 0).astype(np.int8)
    home = ev.home_event.to_numpy() == 1
    ev["n_for"] = np.where(home, n_h, n_a).astype("int8")
    ev["n_against"] = np.where(home, n_a, n_h).astype("int8")
    ev["g_for"] = np.where(home, g_h, g_a).astype("int8")
    ev["g_against"] = np.where(home, g_a, g_h).astype("int8")
    ev["strength_key"] = strength_key(ev.n_for.to_numpy(), ev.n_against.to_numpy(),
                                      ev.g_for.to_numpy(), ev.g_against.to_numpy())
    sh, sa = ev.score_h.to_numpy(np.int16), ev.score_a.to_numpy(np.int16)
    ev["score_diff"] = np.clip(np.where(home, sh - sa, sa - sh), -4, 4).astype("int8")

    # ------------------------------------------------------------ sequencing
    ev["seq"] = ev.groupby("game_id").cumcount().astype("int32")
    ev["dt"] = ev.dt.clip(0, 600).astype("float32")
    ev["t_rem"] = (3600 - ev.t.astype(np.int32)).clip(-1200, 3600).astype("int16")
    ev["season_end"] = np.int16(se)

    keep = ["game_id", "season_end", "game_type", "seq", "period", "t", "t_rem",
            "dt", "event_type", "zone", "shot_type", "home_event",
            "xa", "ya", "has_xy", "xg", "has_xg",
            "n_for", "n_against", "g_for", "g_against", "strength_key",
            "score_diff", "score_h", "score_a", "p1", "p2", "goalie",
            "venue"] + H_ON + A_ON
    return ev[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    maps = json.loads((TENSORS / "maps.json").read_text())

    seasons = args.seasons or list(range(2008, 2027))
    print(f"{'yr':>5} {'events':>9} {'games':>6} {'xy':>7} {'xy shots':>9} "
          f"{'xg':>7} {'dt med':>7}")
    et = maps["event_type"]
    shot_ids = [et[c] for c in SHOT_NAMES if c in et]
    for se in seasons:
        src = [TENSORS / f"events_{se}.parquet"]
        for extra in (f"mp_shots_{se}.parquet", f"xg_shots_{se}.parquet"):
            if (TENSORS / extra).exists():
                src.append(TENSORS / extra)
        out = TENSORS / f"stream_{se}.parquet"
        if not args.force and MAN.is_fresh(out, src, CFG):
            print(f"{se:>5}  fresh, skipping")
            continue
        d = build_season(se, maps)
        d.to_parquet(out, index=False)
        MAN.write_manifest(out, src, CFG, {"n_events": len(d),
                                           "n_games": int(d.game_id.nunique())})
        sm = d.event_type.isin(shot_ids)
        print(f"{se:>5} {len(d):>9,} {d.game_id.nunique():>6} "
              f"{d.has_xy.mean():>7.3f} {d.loc[sm, 'has_xy'].mean():>9.3f} "
              f"{d.has_xg.mean():>7.3f} {d.dt.median():>7.0f}")
        sys.stdout.flush()
    print("done")


if __name__ == "__main__":
    main()
