"""Aggregate NHL shift charts (PLAN_V6 D1) to pair-TOI and team-season tables.

Per game and team, a per-second on-ice boolean matrix over forwards (positions
C/L/R per panel_bios) gives pair TOI by matrix product. Outputs (committed):
  data/processed/shift_pairs.csv         (season_end, team, p1, p2, toi_sec)
                                         forward pairs with >= 10 min together
  data/processed/shift_team_seasons.csv  toi_hhi_f (Herfindahl of forward TOI
                                         shares), line_cont (share of season-V
                                         pair TOI by pairs with >=30 min together
                                         on the same team in V-1), avg_shift_sec,
                                         team_toi identity check column
All situations included (PP inflates star pairs slightly; acceptable for a
continuity/concentration measure — documented choice).
"""
import gzip
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from players import MP_FRAN

PROJ = Path(__file__).resolve().parents[1]
RAW = PROJ / "data" / "raw" / "shifts"
PROC = PROJ / "data" / "processed"
SHIFT_TYPE = 517
MIN_PAIR_SEC = 600           # keep pairs with >= 10 min/season in the pair table
CONT_FLOOR = 1800            # "established pair" in V-1: >= 30 min together


def mmss(t: str) -> int:
    m, s = t.split(":")
    return int(m) * 60 + int(s)


def parse_season(args) -> tuple:
    sdir, fwd_ids = args
    end = int(sdir.name)
    pair: dict = {}
    ptoi: dict = {}
    shift_sum: dict = {}
    shift_n: dict = {}
    for f in sorted(sdir.glob("*.json.gz")):
        with gzip.open(f, "rt") as fh:
            shifts = json.load(fh)
        by_team: dict = {}
        maxsec = 3600
        for s in shifts:
            if s.get("typeCode") != SHIFT_TYPE or not s.get("startTime") \
                    or not s.get("endTime") or not s.get("period"):
                continue
            base = (s["period"] - 1) * 1200
            a, b = base + mmss(s["startTime"]), base + mmss(s["endTime"])
            if b <= a:
                continue
            t = MP_FRAN.get(s["teamAbbrev"], s["teamAbbrev"])
            by_team.setdefault(t, []).append((s["playerId"], a, b))
            maxsec = max(maxsec, b)
            key = (t, s["playerId"])
            shift_sum[key] = shift_sum.get(key, 0) + (b - a)
            shift_n[key] = shift_n.get(key, 0) + 1
        for t, rows in by_team.items():
            fplayers = sorted({p for p, _, _ in rows if p in fwd_ids})
            if len(fplayers) < 2:
                continue
            idx = {p: i for i, p in enumerate(fplayers)}
            M = np.zeros((len(fplayers), maxsec + 1), dtype=np.uint8)
            for p, a, b in rows:
                if p in idx:
                    M[idx[p], a:b] = 1
            P = M @ M.T
            for i in range(len(fplayers)):
                ptoi[(t, fplayers[i])] = ptoi.get((t, fplayers[i]), 0) + int(P[i, i])
                for j in range(i + 1, len(fplayers)):
                    if P[i, j]:
                        k = (t, fplayers[i], fplayers[j])
                        pair[k] = pair.get(k, 0) + int(P[i, j])
    return end, pair, ptoi, shift_sum, shift_n


def main():
    bios = pd.read_csv(PROC / "panel_bios.csv")
    fwd_ids = set(bios[bios.position.isin(["C", "L", "R", "F"])].playerId)
    sdirs = sorted(d for d in RAW.iterdir() if d.is_dir())
    pair_rows, team_rows = [], []
    season_pairs: dict = {}
    with ProcessPoolExecutor(max_workers=6) as ex:
        for end, pair, ptoi, ssum, sn in ex.map(
                parse_season, [(d, fwd_ids) for d in sdirs]):
            season_pairs[end] = pair
            for (t, p1, p2), sec in pair.items():
                if sec >= MIN_PAIR_SEC:
                    pair_rows.append({"season_end": end, "team": t,
                                      "p1": p1, "p2": p2, "toi_sec": sec})
            teams = sorted({t for (t, _p) in ptoi})
            for t in teams:
                toi = np.array([v for (tt, _), v in ptoi.items() if tt == t],
                               dtype=float)
                toi = toi[toi > 0]
                hhi = float(((toi / toi.sum()) ** 2).sum()) if toi.sum() else np.nan
                tot = sum(v for (tt, _), v in ssum.items() if tt == t)
                n = sum(v for (tt, _), v in sn.items() if tt == t)
                team_rows.append({"season_end": end, "team": t, "toi_hhi_f": hhi,
                                  "avg_shift_sec": tot / n if n else np.nan,
                                  "team_toi_hr": tot / 3600.0})
            print(f"{end}: {len(pair)} pairs")
            sys.stdout.flush()

    tdf = pd.DataFrame(team_rows)
    # line continuity: share of season-V pair TOI by pairs established in V-1
    cont = {}
    for end, pair in season_pairs.items():
        prev = season_pairs.get(end - 1)
        if prev is None:
            continue
        est = {k for k, v in prev.items() if v >= CONT_FLOOR}
        num: dict = {}
        den: dict = {}
        for (t, p1, p2), sec in pair.items():
            den[t] = den.get(t, 0) + sec
            if (t, p1, p2) in est:
                num[t] = num.get(t, 0) + sec
        for t in den:
            cont[(end, t)] = num.get(t, 0) / den[t]
    tdf["line_cont"] = [cont.get((r.season_end, r.team), np.nan)
                        for r in tdf.itertuples()]
    pd.DataFrame(pair_rows).to_csv(PROC / "shift_pairs.csv", index=False)
    tdf.round(6).to_csv(PROC / "shift_team_seasons.csv", index=False)
    print(f"shift_pairs: {len(pair_rows)} rows; shift_team_seasons: {len(tdf)} "
          f"rows {tdf.season_end.min()}-{tdf.season_end.max()}")


if __name__ == "__main__":
    main()
