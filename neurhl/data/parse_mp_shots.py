"""NeurHL-2 — MoneyPuck shot corpus: raw recorded fields only (P3 allowlist).

~350 MB of shot data for 2007-2025 was already on disk, downloaded and never
parsed. It is the only source that closes the **pre-2012 coordinate hole**: the
NHL JSON feed carries no x/y before 2012 (measured: `events_2011.has_coord` is
0.000 across the whole shard), while MoneyPuck has 100% coordinate coverage back
to 2007. That is ~547,000 shots in season_end 2008-2012 that currently have no
geometry at all, and geometry is the entire basis of an xG model.

P3 DISCIPLINE. MoneyPuck's model outputs are fitted on the full sample, so using
them as inputs leaks the future into every historical vantage. This module uses
an explicit **allowlist** rather than a denylist -- a denylist silently admits
any new modelled column the upstream adds. Excluded here: `xGoal`, `xFroze`,
`xRebound`, `xPlayContinuedInZone`, `xPlayContinuedOutsideZone`, `xPlayStopped`,
`xShotWasOnGoal`, and every `arenaAdjusted*` field (arena adjustment is itself a
full-sample fit over rink-count biases).

What IS admissible is large and genuinely additive: raw coordinates, distance and
angle, shot type, rush/rebound flags, time since last event and since the
faceoff, the previous event's category/coordinates/team, on-ice skater counts,
penalty time remaining, shooter handedness and off-wing, and 38 on-ice
time-on-ice/fatigue columns. Those are recorded facts, not model output.

Caveat carried forward: MoneyPuck EXCLUDES blocked shots, so this table covers
SHOT/MISS/GOAL only. Corsi must still come from the NHL event shards.

Join key: `nhl_gid = season * 1_000_000 + game_id` (game_id already encodes
2xxxx regular / 3xxxx playoff), and `season_end = season + 1`.

Run: uv run --no-project --python 3.12 --with numpy --with "pandas<3" \
     --with pyarrow python neurhl/data/parse_mp_shots.py
"""
import argparse
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RAW, TENSORS  # noqa: E402
import manifest as MAN  # noqa: E402

SRC = RAW / "mp_shots"

# --- P3 ALLOWLIST. Add a column only after confirming it is RECORDED, not fitted.
KEYS = ["shotID", "season", "game_id", "isPlayoffGame", "period", "time",
        "team", "event", "goal", "isHomeTeam", "teamCode",
        "shooterPlayerId", "goalieIdForShot", "playerPositionThatDidEvent"]
GEOMETRY = ["xCord", "yCord", "shotDistance", "shotAngle", "shotAngleAdjusted",
            "shotAnglePlusRebound", "shotAngleReboundRoyalRoad", "shotType",
            "offWing", "shooterLeftRight"]
CONTEXT = ["shotRush", "shotRebound", "shotOnEmptyNet", "shotWasOnGoal",
           "shotGeneratedRebound", "shotGoalieFroze", "shotPlayStopped",
           "shotPlayContinuedInZone", "shotPlayContinuedOutsideZone",
           "timeSinceLastEvent", "timeUntilNextEvent", "timeSinceFaceoff",
           "speedFromLastEvent", "distanceFromLastEvent",
           "lastEventCategory", "lastEventTeam", "lastEventxCord",
           "lastEventyCord", "lastEventShotAngle", "lastEventShotDistance",
           "homeSkatersOnIce", "awaySkatersOnIce", "homeEmptyNet",
           "awayEmptyNet", "homeTeamGoals", "awayTeamGoals",
           "homePenalty1TimeLeft", "homePenalty1Length",
           "awayPenalty1TimeLeft", "awayPenalty1Length",
           "timeDifferenceSinceChange", "averageRestDifference",
           "shootingTeamForwardsOnIce", "shootingTeamDefencemenOnIce",
           "defendingTeamForwardsOnIce", "defendingTeamDefencemenOnIce"]
FATIGUE_PREFIX = ("shooterTimeOnIce", "shootingTeamAverageTimeOnIce",
                  "shootingTeamMaxTimeOnIce", "shootingTeamMinTimeOnIce",
                  "defendingTeamAverageTimeOnIce", "defendingTeamMaxTimeOnIce",
                  "defendingTeamMinTimeOnIce")
# Anything matching these is model output or a full-sample fit -- never admitted.
FORBIDDEN = ("xGoal", "xFroze", "xRebound", "xPlayContinued", "xPlayStopped",
             "xShotWasOnGoal", "arenaAdjusted")
CFG = {"version": 1, "p3_allowlist": True}


def admissible(cols) -> list:
    keep = []
    for c in cols:
        if any(f.lower() in c.lower() for f in FORBIDDEN):
            continue
        if c in KEYS or c in GEOMETRY or c in CONTEXT:
            keep.append(c)
        elif c.startswith(FATIGUE_PREFIX):
            keep.append(c)
    return keep


def parse_season(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        name = next(n for n in z.namelist() if n.endswith(".csv"))
        raw = pd.read_csv(io.BytesIO(z.read(name)), low_memory=False)
    cols = admissible(raw.columns)
    d = raw[cols].copy()
    dropped = [c for c in raw.columns if c not in cols]
    d["game_id"] = (d.season.astype("int64") * 1_000_000
                    + d.game_id.astype("int64"))
    d["season_end"] = d.season.astype("int64") + 1
    d.attrs["dropped"] = dropped
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="*", default=None,
                    help="season_end values")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    zips = sorted(SRC.glob("shots_*.zip"))
    if not zips:
        print(f"no shot zips under {SRC}")
        return 1
    total, first_drop = 0, None
    for zp in zips:
        mp_season = int(zp.stem.split("_")[1])
        se = mp_season + 1
        if args.seasons and se not in args.seasons:
            continue
        out = TENSORS / f"mp_shots_{se}.parquet"
        if not args.force and MAN.is_fresh(out, [zp], CFG):
            print(f"{se}: fresh, skipping")
            continue
        d = parse_season(zp)
        if first_drop is None:
            first_drop = d.attrs["dropped"]
        d.to_parquet(out, index=False)
        MAN.write_manifest(out, [zp], CFG,
                           {"n_shots": len(d), "n_cols": d.shape[1],
                            "dropped_p3": d.attrs["dropped"]})
        total += len(d)
        coord = d.xCord.notna().mean() if "xCord" in d else float("nan")
        print(f"{se}: {len(d):>7,} shots, {d.shape[1]:>3} cols kept | "
              f"coords {coord:.3f} | games {d.game_id.nunique():>4} | "
              f"{d.event.value_counts().to_dict()}")
        sys.stdout.flush()
    if first_drop:
        print(f"\nP3-excluded columns ({len(first_drop)}): {sorted(first_drop)}")
    print(f"\ntotal {total:,} shots -> {TENSORS}/mp_shots_<season_end>.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
