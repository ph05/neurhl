"""NeurHL — on-ice reconstruction library (PLAN_NeurHL Phase 1).

Joins shift charts to PBP events to produce the exact on-ice players per event.
Conventions validated in eda/eda_03_shifts_join.md (98.9% either-convention
agreement with situationCode): faceoffs occur at shift boundaries where the
NEW players are already on (convention A: start <= t < end); all other events
belong to the shift ending at t if it ends exactly there (convention B:
start < t <= end).

Slot layout per team: [goalie, skater1..skater6] (7 slots, 0-padded, skaters
sorted ascending for determinism; 6 skater slots cover pulled-goalie 6F).
"""
import gzip
import json
from pathlib import Path

import numpy as np

FACEOFF = "faceoff"
N_SLOTS = 7


def mmss(s: str) -> int:
    m, ss = s.split(":")
    return int(m) * 60 + int(ss)


def load_shifts(path: Path, goalie_ids: set):
    """-> (team_ids, player_ids, is_goalie, t0, t1) numpy arrays, or None."""
    with gzip.open(path, "rt") as f:
        recs = json.load(f)
    if isinstance(recs, dict):
        recs = recs.get("data", [])
    rows = []
    for r in recs:
        if r.get("typeCode") != 517 or not r.get("startTime") or not r.get("endTime"):
            continue
        per = r["period"]
        if per >= 5:            # shootout
            continue
        t0 = (per - 1) * 1200 + mmss(r["startTime"])
        t1 = (per - 1) * 1200 + mmss(r["endTime"])
        if t1 <= t0:
            continue
        rows.append((r["teamId"], r["playerId"], r["playerId"] in goalie_ids, t0, t1))
    if not rows:
        return None
    team = np.array([r[0] for r in rows], dtype=np.int64)
    pid = np.array([r[1] for r in rows], dtype=np.int64)
    is_g = np.array([r[2] for r in rows], dtype=bool)
    t0 = np.array([r[3] for r in rows], dtype=np.int32)
    t1 = np.array([r[4] for r in rows], dtype=np.int32)
    return team, pid, is_g, t0, t1


def slots_at(shifts, home_team_id: int, t: int, event_type: str):
    """On-ice slots at time t -> (home[7], away[7]) int64 arrays."""
    team, pid, is_g, t0, t1 = shifts
    if event_type == FACEOFF:
        on = (t0 <= t) & (t < t1)
    else:
        on = (t0 < t) & (t <= t1)
    out = []
    for is_home in (True, False):
        side = on & ((team == home_team_id) == is_home)
        slots = np.zeros(N_SLOTS, dtype=np.int64)
        g = pid[side & is_g]
        if len(g):
            slots[0] = g[0]
        sk = np.sort(pid[side & ~is_g])[:N_SLOTS - 1]
        slots[1:1 + len(sk)] = sk
        out.append(slots)
    return out[0], out[1]
