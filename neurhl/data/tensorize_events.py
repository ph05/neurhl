"""NeurHL — event tensorization: raw PBP+shifts -> per-season parquet shards.

One row per play, seasons 2012-2026 (reg + playoffs), written to
neurhl/data/tensors/events_<season_end>.parquet (gitignored). Encodes:
  - event/zone/shot-type/team ids via maps.json (built once from the EDA cache
    and frozen — indices are stable across reruns);
  - attacking-direction normalization inferred per (game, side, period) from
    offensive-zone shot medians (EDA-02: defendingSide only exists 2020+;
    inference agrees 99.97% where ground truth exists);
  - exact on-ice slots [goalie, sk1..sk6] per team from the shifts join
    (EDA-03 conventions: faceoff = A, others = B);
  - raw NHL playerIds (vocab indexing happens per-vantage at load time, P1).

Excludes the degenerate game 2012020660 (EDA-01). Deterministic output.
"""
import gzip
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RAW, TENSORS  # noqa: E402
from data.build_onice import load_shifts, slots_at  # noqa: E402

CACHE = TENSORS / "_edacache"
MAPS = TENSORS / "maps.json"
EXCLUDE_GAMES = {2012020660}
SHOT_FAMILY = {"shot-on-goal", "missed-shot", "blocked-shot", "goal",
               "failed-shot-attempt"}
ACTORS = {
    "faceoff": ("winningPlayerId", "losingPlayerId", None),
    "goal": ("scoringPlayerId", "assist1PlayerId", "assist2PlayerId"),
    "shot-on-goal": ("shootingPlayerId", None, None),
    "missed-shot": ("shootingPlayerId", None, None),
    "failed-shot-attempt": ("shootingPlayerId", None, None),
    "blocked-shot": ("shootingPlayerId", "blockingPlayerId", None),
    "hit": ("hittingPlayerId", "hitteePlayerId", None),
    "giveaway": ("playerId", None, None),
    "takeaway": ("playerId", None, None),
    "penalty": ("committedByPlayerId", "drawnByPlayerId", None),
}
COLS = (["game_id", "game_type", "event_idx", "period", "t", "dt", "event_type",
         "zone", "shot_type", "strength", "home_event", "x", "y", "dir",
         "has_coord", "xn", "yn", "score_h", "score_a", "p1", "p2", "p3",
         "goalie", "venue"]
        + [f"h_on{i}" for i in range(7)] + [f"a_on{i}" for i in range(7)])


def build_maps() -> dict:
    if MAPS.exists():
        return json.loads(MAPS.read_text())
    ev = pd.read_parquet(CACHE / "events_summary.parquet")
    sh = pd.read_parquet(CACHE / "shot_events.parquet")
    gs = pd.read_parquet(CACHE / "game_summary.parquet")
    maps = {
        "event_type": {t: i + 1 for i, t in enumerate(sorted(ev.event_type.unique()))},
        "shot_type": {t: i + 1 for i, t in
                      enumerate(sorted(t for t in sh.shot_type.unique() if t))},
        "team": {t: i + 1 for i, t in
                 enumerate(sorted(set(gs.home) | set(gs.away)))},
        "zone": {"O": 1, "D": 2, "N": 3},
    }
    MAPS.write_text(json.dumps(maps, indent=1, sort_keys=True))
    return maps


_M = None   # per-process maps


def _init(maps):
    global _M
    _M = maps


def mmss(s: str) -> int:
    m, ss = s.split(":")
    return int(m) * 60 + int(ss)


def infer_dirs(plays, home_id) -> dict:
    """(is_home_event, period) -> +1/-1 attacking-direction multiplier."""
    acc: dict = {}
    for p in plays:
        if p.get("typeDescKey") not in SHOT_FAMILY:
            continue
        d = p.get("details", {}) or {}
        if d.get("zoneCode") != "O" or "xCoord" not in d:
            continue
        key = (d.get("eventOwnerTeamId") == home_id,
               p.get("periodDescriptor", {}).get("number", 0))
        acc.setdefault(key, []).append(d["xCoord"])
    dirs = {k: (1 if np.median(v) > 0 else -1) for k, v in acc.items() if v}
    # fill gaps: sides attack opposite ends; ends swap each period
    out = dict(dirs)
    periods = {k[1] for k in dirs} | {k[1] for k in acc}
    for per in range(1, max(periods | {3}) + 2):
        for is_home in (True, False):
            if (is_home, per) in out:
                continue
            if (not is_home, per) in out:
                out[(is_home, per)] = -out[(not is_home, per)]
            elif (is_home, per - 1) in out:
                out[(is_home, per)] = -out[(is_home, per - 1)]
            elif (is_home, per + 1) in dirs:
                out[(is_home, per)] = -out[(is_home, per + 1)]
    return out


def tensorize_game(args):
    pbp_path, shift_path = args
    with gzip.open(pbp_path, "rt") as f:
        g = json.load(f)
    if g["id"] in EXCLUDE_GAMES:
        return None
    home_id = g["homeTeam"]["id"]
    goalie_ids = {r["playerId"] for r in g.get("rosterSpots", [])
                  if r.get("positionCode") == "G"}
    shifts = load_shifts(shift_path, goalie_ids) if shift_path.exists() else None
    dirs = infer_dirs(g.get("plays", []), home_id)
    venue = _M["team"].get(g["homeTeam"]["abbrev"], 0)
    zeros7 = np.zeros(7, dtype=np.int64)
    rows, prev_t, sh, sa = [], 0, 0, 0
    for idx, p in enumerate(g.get("plays", [])):
        et = p.get("typeDescKey", "?")
        d = p.get("details", {}) or {}
        per = p.get("periodDescriptor", {}).get("number", 0)
        t = (per - 1) * 1200 + mmss(p.get("timeInPeriod", "0:00"))
        owner = d.get("eventOwnerTeamId")
        home_event = 1 if owner == home_id else (0 if owner is not None else -1)
        if et == "goal":
            sh = d.get("homeScore", sh)
            sa = d.get("awayScore", sa)
        has_xy = int("xCoord" in d and "yCoord" in d)
        x = int(d.get("xCoord", 0))
        y = int(d.get("yCoord", 0))
        di = dirs.get((home_event == 1, per), 0) if owner is not None else 0
        k1, k2, k3 = ACTORS.get(et, (None, None, None))
        if per >= 5 and g["gameType"] == 2:
            h_on, a_on = zeros7, zeros7          # shootout
        elif shifts is not None:
            h_on, a_on = slots_at(shifts, home_id, t, et)
        else:
            h_on, a_on = zeros7, zeros7
        rows.append((
            g["id"], g["gameType"], idx, per, t, max(0, t - prev_t),
            _M["event_type"].get(et, 0), _M["zone"].get(d.get("zoneCode"), 0),
            _M["shot_type"].get(d.get("shotType"), 0),
            int(p.get("situationCode") or 0), home_event, x, y, di, has_xy,
            x * di, y * di, sh, sa,
            d.get(k1) or 0 if k1 else 0, d.get(k2) or 0 if k2 else 0,
            d.get(k3) or 0 if k3 else 0, d.get("goalieInNetId") or 0, venue,
            *h_on.tolist(), *a_on.tolist(),
        ))
        prev_t = t
    return rows


def main():
    maps = build_maps()
    for sdir in sorted((RAW / "pbp").iterdir()) :
        se = sdir.name
        out = TENSORS / f"events_{se}.parquet"
        if out.exists():
            print(f"{se}: exists, skipping")
            continue
        jobs = []
        for src in ("pbp", "pbp_po"):
            d = RAW / src / se
            if d.exists():
                jobs += [(p, RAW / "shifts" / se / p.name)
                         for p in sorted(d.glob("*.json.gz"))]
        all_rows = []
        with Pool(8, initializer=_init, initargs=(maps,)) as pool:
            for rows in pool.imap(tensorize_game, jobs, chunksize=32):
                if rows:
                    all_rows.extend(rows)
        df = pd.DataFrame(all_rows, columns=COLS)
        for c, dt in [("game_id", "uint32"), ("game_type", "uint8"),
                      ("event_idx", "uint16"), ("period", "uint8"),
                      ("t", "uint16"), ("dt", "uint16"), ("event_type", "uint8"),
                      ("zone", "uint8"), ("shot_type", "uint8"),
                      ("strength", "uint16"), ("home_event", "int8"),
                      ("x", "int16"), ("y", "int16"), ("dir", "int8"),
                      ("has_coord", "uint8"), ("xn", "int16"), ("yn", "int16"),
                      ("score_h", "uint8"), ("score_a", "uint8"),
                      ("p1", "uint32"), ("p2", "uint32"), ("p3", "uint32"),
                      ("goalie", "uint32"), ("venue", "uint8")]:
            df[c] = df[c].astype(dt)
        for i in range(7):
            df[f"h_on{i}"] = df[f"h_on{i}"].astype("uint32")
            df[f"a_on{i}"] = df[f"a_on{i}"].astype("uint32")
        df.to_parquet(out, index=False)
        print(f"{se}: {len(df):,} events -> {out.name}")
        sys.stdout.flush()
    print("done")


if __name__ == "__main__":
    main()
