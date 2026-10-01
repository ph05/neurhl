"""Results for the in-season loop: orr/output/live/results_2027.csv.

    python3 -m orr.ingest                       # NHL API, every finished day
    python3 -m orr.ingest --add "2026020006,2026-09-30,PHI,PIT,4,2,REG"   # by hand
    python3 -m orr.ingest --add "2026020006,2026-09-30,PHI,PIT,4,2,REG,manual,31,27"   # with shots on goal

The NHL score endpoint (api-web.nhle.com/v1/score/<date>) is used where the
network allows it. Where it does not (this project's build container blocks
it), finished games can be appended by hand or merged from any CSV with the
same columns: game_id, date, home, away, home_g, away_g, last_period
(REG/OT/SO; for a shootout the score includes the deciding goal, as on NHL.com).
Each row is checked against the frozen schedule (same game id, date, teams).
"""
from __future__ import annotations

import argparse
import datetime as dt
import time

import pandas as pd

from orr import config as C

PATH = C.OUT / "live" / "results_2027.csv"
COLS = ["game_id", "date", "home", "away", "home_g", "away_g", "last_period", "source"]
OPTIONAL = ["shots_home", "shots_away"]          # used by the in-season filter when present
SCORE_URL = "https://api-web.nhle.com/v1/score/{d}"


def load() -> pd.DataFrame:
    return pd.read_csv(PATH) if PATH.exists() else pd.DataFrame(columns=COLS)


def validate(rows: pd.DataFrame) -> pd.DataFrame:
    sch = pd.read_csv(C.OUT / f"freeze_{C.TARGET_SEASON}" / f"schedule_{C.TARGET_SEASON}.csv")
    m = rows.merge(sch, on="game_id", suffixes=("", "_sch"), how="left")
    bad = m[(m.home != m.home_sch) | (m.away != m.away_sch) | (m.date.astype(str) != m.date_sch.astype(str))]
    if len(bad):
        raise ValueError(f"rows disagree with the schedule:\n{bad[['game_id', 'date', 'home', 'away', 'date_sch', 'home_sch', 'away_sch']]}")
    if not rows.last_period.isin(["REG", "OT", "SO"]).all():
        raise ValueError("last_period must be REG, OT or SO")
    if (rows.home_g == rows.away_g).any():
        raise ValueError("a finished game cannot be tied")
    return rows


def save(new: pd.DataFrame) -> pd.DataFrame:
    new = validate(new.copy())
    keep = COLS + [c for c in OPTIONAL if c in new.columns or c in load().columns]
    out = (pd.concat([load(), new.reindex(columns=keep)]).drop_duplicates("game_id", keep="last")
           .sort_values(["date", "game_id"]))[keep]
    PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(PATH, index=False)
    return out


def fetch_api(through: dt.date) -> pd.DataFrame:
    import requests
    have_ids = set(load().game_id)
    sch = pd.read_csv(C.OUT / f"freeze_{C.TARGET_SEASON}" / f"schedule_{C.TARGET_SEASON}.csv")
    rows = []
    for ds in sorted(set(sch.date.astype(str))):
        # skip a date only when every one of its games is already stored
        if ds > through.isoformat() or set(sch[sch.date.astype(str) == ds].game_id) <= have_ids:
            continue
        js = requests.get(SCORE_URL.format(d=ds), timeout=30).json()
        for g in js.get("games", []):
            if g.get("gameType") == 2 and g.get("gameState") in ("OFF", "FINAL"):
                rows.append({"game_id": g["id"], "date": ds, "home": g["homeTeam"]["abbrev"],
                             "away": g["awayTeam"]["abbrev"], "home_g": g["homeTeam"]["score"],
                             "away_g": g["awayTeam"]["score"],
                             "last_period": (g.get("gameOutcome") or {}).get("lastPeriodType", "REG"),
                             "source": "api-web.nhle.com",
                             "shots_home": g["homeTeam"].get("sog"), "shots_away": g["awayTeam"].get("sog")})
        time.sleep(0.4)
    return pd.DataFrame(rows, columns=COLS + OPTIONAL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", action="append", default=[],
                    help="game_id,date,home,away,home_g,away_g,last_period[,source[,shots_home,shots_away]]")
    ap.add_argument("--merge", help="CSV with the same columns")
    ap.add_argument("--through", default=str(dt.date.today() - dt.timedelta(days=1)))
    a = ap.parse_args()
    if a.add or a.merge:
        rows = [r.split(",") for r in a.add]
        rows = [r + ["manual"] * (8 - len(r)) if len(r) < 8 else r for r in rows]
        rows = [r + [None, None] if len(r) == 8 else r for r in rows]
        new = pd.DataFrame(rows, columns=COLS + OPTIONAL)
        new[OPTIONAL] = new[OPTIONAL].apply(pd.to_numeric, errors="coerce")
        if a.merge:
            m = pd.read_csv(a.merge)
            m["source"] = m.get("source", a.merge)
            new = pd.concat([new, m[COLS]])
        new = new.astype({"game_id": int, "home_g": int, "away_g": int})
    else:
        new = fetch_api(dt.date.fromisoformat(a.through))
    out = save(new)
    print(f"{len(out)} results through {out.date.max()} -> {PATH}")


if __name__ == "__main__":
    main()
