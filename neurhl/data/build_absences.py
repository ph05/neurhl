"""NeurHL-3 — D2: per-game absence table, 2008-2026.

Who is OUT is precisely what a player's own-form EWMA and a team Elo cannot
see (registry: `scratches`, `vacated_toi`, `replacement_delta`). This builder
derives it walk-forward from the dressed lists alone, so it covers the whole
corpus without the (still unfetched) right-rail source:

  * a team-game's REGULARS = players dressed in >= REGULAR_SHARE of the team's
    previous REGULAR_WINDOW games (needs >= MIN_HISTORY prior games, else the
    game emits nothing — early-season cold start is honest, not imputed);
  * an ABSENCE row = a regular not dressed today, carrying his pre-game EWMA
    TOI (his recent workload = the minutes his absence vacates), position, and
    the current missed streak;
  * everything derives from games strictly before, plus TODAY'S dressed list —
    which is INGAME-at-puck-drop information, the same conditioning the
    recorded H4 protocol and Tier-0 skater_form already use.

Season-level cross-check: data/processed/player_absences.csv (2012+, built
from pbp rosterSpots by src/build_absences.py) — reconciliation printed, not
enforced (definitions differ slightly by design).

Run: uv run --no-project --python 3.12 --with numpy --with "pandas<3" \
     --with pyarrow python neurhl/data/build_absences.py
"""
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import TENSORS  # noqa: E402
import manifest as MAN  # noqa: E402

REGULAR_WINDOW = 10          # team games defining "regular"
REGULAR_SHARE = 0.5          # dressed in >= half of them
MIN_HISTORY = 3              # team games before absences are emitted
EWMA_ALPHA = 0.1             # house per-game form constant (train_game.add_form)
SEASONS = list(range(2008, 2027))
CFG = {"regular_window": REGULAR_WINDOW, "regular_share": REGULAR_SHARE,
       "min_history": MIN_HISTORY, "ewma_alpha": EWMA_ALPHA, "version": 1}


def build(season: int) -> pd.DataFrame:
    pg = pd.read_parquet(TENSORS / f"player_games_{season}.parquet",
                         columns=["game_id", "game_type", "player_id",
                                  "is_home", "pos_group", "toi_sec"])
    pg = pg[pg.game_type == 2]
    gc = pd.read_parquet(TENSORS / f"games_ctx_{season}.parquet",
                         columns=["game_id", "game_type", "date", "home_idx",
                                  "away_idx"])
    gc = gc[gc.game_type == 2].copy()
    gc["date"] = pd.to_datetime(gc.date)
    pg = pg.merge(gc[["game_id", "home_idx", "away_idx"]], on="game_id")
    pg["team"] = np.where(pg.is_home, pg.home_idx, pg.away_idx)

    # per (team, game) dressed roster, in team schedule order
    order = pd.concat([
        gc[["game_id", "date", "home_idx"]].rename(columns={"home_idx": "team"}),
        gc[["game_id", "date", "away_idx"]].rename(columns={"away_idx": "team"}),
    ], ignore_index=True).sort_values(["team", "date", "game_id"],
                                      kind="stable")
    dressed = pg.groupby(["team", "game_id"]).apply(
        lambda d: dict(zip(d.player_id, zip(d.pos_group, d.toi_sec))),
        include_groups=False)

    rows = []
    for team, sched in order.groupby("team", sort=False):
        window = deque(maxlen=REGULAR_WINDOW)     # sets of dressed player_ids
        ewma = {}                                  # player -> EWMA toi_sec
        pos_of = {}
        missed = defaultdict(int)                  # consecutive games missed
        for g in sched.itertuples():
            today = dressed.get((team, g.game_id), {})
            if len(window) >= MIN_HISTORY:
                counts = defaultdict(int)
                for s_ in window:
                    for pid in s_:
                        counts[pid] += 1
                need = REGULAR_SHARE * len(window)
                regulars = {p for p, c in counts.items() if c >= need}
                for pid in sorted(regulars):
                    if pid not in today:
                        rows.append((g.game_id, int(team), int(pid),
                                     int(pos_of.get(pid, 0)),
                                     float(ewma.get(pid, 0.0)),
                                     int(missed[pid] + 1)))
            # update state AFTER emitting (today's list is INGAME, its TOI is not)
            for pid, (pos, toi) in today.items():
                pos_of[pid] = pos
                prev = ewma.get(pid)
                ewma[pid] = (toi if prev is None
                             else EWMA_ALPHA * toi + (1 - EWMA_ALPHA) * prev)
            seen_any = set(today)
            for pid in list(missed) + list(seen_any):
                missed[pid] = 0 if pid in seen_any else missed[pid]
            for pid in (set(pos_of) - seen_any):
                missed[pid] += 1
            window.append(frozenset(today))
    out = pd.DataFrame(rows, columns=["game_id", "team", "player_id",
                                      "pos_group", "ewma_toi_sec",
                                      "missed_streak"])
    return out


def main():
    for s in SEASONS:
        src = [TENSORS / f"player_games_{s}.parquet",
               TENSORS / f"games_ctx_{s}.parquet"]
        if not all(p.exists() for p in src):
            print(f"{s}: missing sources, skipping")
            continue
        out = TENSORS / f"absences_{s}.parquet"
        if MAN.is_fresh(out, src, CFG):
            print(f"{s}: fresh, skipping")
            continue
        d = build(s)
        d.to_parquet(out, index=False)
        MAN.write_manifest(out, src, CFG, {"n_rows": len(d)})
        per_game = len(d) / max(d.game_id.nunique(), 1)
        vac = d.ewma_toi_sec.sum() / max(d.game_id.nunique(), 1) / 60
        print(f"{s}: {len(d):,} absence rows, {per_game:.2f}/game with an "
              f"absence, vacated {vac:.0f} EWMA-min/game")
        sys.stdout.flush()
    print("done")


if __name__ == "__main__":
    main()
