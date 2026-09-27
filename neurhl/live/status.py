"""NeurHL LIVE: manual player availability (PLAN_NeurHL4 section D).

data/manual/player_status_2027.csv records availability that rosters do not
show (suspensions, holdouts, IR/LTIR, waivers). Columns:

  recorded_at, effective_from, player_id, team, status, source_url, note

Entries change inputs (who can dress) only, never outputs, and each must be
committed before the first prediction that uses it.

unavailable(date) returns the player_ids whose LATEST entry with
effective_from <= date has an unavailable status; a later ACTIVE entry clears
a player. Ties on effective_from are broken by recorded_at, then file order.

CLI: python neurhl/live/status.py [YYYY-MM-DD]
"""
import sys
from datetime import date as _date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PROJ  # noqa: E402

STATUS_CSV = PROJ / "data" / "manual" / "player_status_2027.csv"
UNAVAILABLE = frozenset({"SUSPENDED_NOT_REPORTING", "IR", "LTIR", "WAIVED", "HOLDOUT"})
KNOWN = UNAVAILABLE | {"ACTIVE"}
COLS = ["recorded_at", "effective_from", "player_id", "team", "status", "source_url", "note"]


def load(path: Path = STATUS_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"player_id": "int64"}, keep_default_na=False)
    missing = set(COLS) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    df["status"] = df["status"].str.strip().str.upper()
    bad = sorted(set(df["status"]) - KNOWN)
    if bad:
        raise ValueError(f"{path}: unknown status values {bad}; allowed {sorted(KNOWN)}")
    for c in ("recorded_at", "effective_from"):
        df[c] = pd.to_datetime(df[c], format="%Y-%m-%d").dt.date
    df["_row"] = range(len(df))
    return df


def current(date, path: Path = STATUS_CSV) -> pd.DataFrame:
    """Latest entry per player effective on/before `date` (str or date)."""
    d = pd.Timestamp(date).date()
    df = load(path)
    df = df[df["effective_from"] <= d]
    df = df.sort_values(["player_id", "effective_from", "recorded_at", "_row"])
    return df.groupby("player_id").tail(1).drop(columns="_row").reset_index(drop=True)


def unavailable(date, path: Path = STATUS_CSV) -> set[int]:
    cur = current(date, path)
    return set(int(x) for x in cur.loc[cur["status"].isin(UNAVAILABLE), "player_id"])


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else _date.today().isoformat()
    cur = current(d)
    print(f"as of {d}: {len(unavailable(d))} unavailable")
    if len(cur):
        print(cur[["player_id", "team", "status", "effective_from", "note"]].to_string(index=False))
