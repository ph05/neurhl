"""NeurHL LIVE: dated NHL roster snapshots (PLAN_NeurHL4 section D).

Pulls all 32 current rosters from the NHL web API and writes

  data/raw/rosters/<YYYY-MM-DD>/nhl_roster_{TEAM}.json   raw API response
  data/raw/rosters/<YYYY-MM-DD>/rosters.csv              team, player_id, first, last,
                                                         pos (C/L/R/D/G), sweater, birthdate
  data/raw/rosters/<YYYY-MM-DD>/moves_vs_<prev>.csv      team, player_id, name, pos,
                                                         change (added/removed)

<prev> is the latest earlier dated snapshot; if none exists, the frozen August
1.0 inputs data/raw/nhl_roster_{TEAM}_20262027.json (label "2026-08"). Those
August files are read only, never written.

Camp rosters are inflated until the NHL roster deadline (2026-09-28 17:00 ET);
the final pre-season pull must come after it.

CLI: python neurhl/live/fetch_rosters.py [--date YYYY-MM-DD] [--offline]
  --offline rebuilds rosters.csv / moves from raw JSON already on disk.
"""
import argparse
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RAW, UA  # noqa: E402

SEASON = "20262027"
API = "https://api-web.nhle.com/v1/roster/{team}/" + SEASON
ROSTERS = RAW / "rosters"
AUGUST_LABEL = "2026-08"
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
COLS = ["team", "player_id", "first", "last", "pos", "sweater", "birthdate"]


def teams() -> list[str]:
    """The 32 abbreviations, taken from the frozen August roster filenames."""
    out = sorted(p.name.split("_")[2] for p in RAW.glob(f"nhl_roster_*_{SEASON}.json"))
    if len(out) != 32:
        raise RuntimeError(f"expected 32 August roster files, found {len(out)}")
    return out


def roster_rows(team: str, js: dict) -> list[dict]:
    rows = []
    for grp in ("forwards", "defensemen", "goalies"):
        for p in js.get(grp) or []:
            rows.append({
                "team": team,
                "player_id": int(p["id"]),
                "first": (p.get("firstName") or {}).get("default", ""),
                "last": (p.get("lastName") or {}).get("default", ""),
                "pos": p.get("positionCode", ""),
                "sweater": p.get("sweaterNumber"),
                "birthdate": p.get("birthDate", ""),
            })
    return rows


def frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLS)
    df["sweater"] = df["sweater"].astype("Int64")
    return df.sort_values(["team", "pos", "last", "first"]).reset_index(drop=True)


def fetch(day: str, sleep: float = 0.5) -> dict[str, str]:
    ddir = ROSTERS / day
    ddir.mkdir(parents=True, exist_ok=True)
    ses = requests.Session()
    ses.headers.update(UA)
    status = {}
    for i, team in enumerate(teams()):
        if i:
            time.sleep(sleep)
        try:
            r = ses.get(API.format(team=team), timeout=30)
            r.raise_for_status()
            js = r.json()
            if not any(js.get(k) for k in ("forwards", "defensemen", "goalies")):
                raise ValueError("empty roster payload")
        except (requests.RequestException, ValueError) as e:
            status[team] = f"FAIL {e}"
            print(f"  {team}: FAIL {e}", file=sys.stderr)
            continue
        tmp = ddir / f"nhl_roster_{team}.json.part"
        tmp.write_text(json.dumps(js, ensure_ascii=False, indent=1))
        tmp.rename(ddir / f"nhl_roster_{team}.json")
        status[team] = "ok"
    return status


def load_dir(ddir: Path) -> pd.DataFrame:
    rows = []
    for team in teams():
        f = ddir / f"nhl_roster_{team}.json"
        if f.exists():
            rows += roster_rows(team, json.loads(f.read_text()))
    return frame(rows)


def load_august() -> pd.DataFrame:
    rows = []
    for team in teams():
        rows += roster_rows(team, json.loads((RAW / f"nhl_roster_{team}_{SEASON}.json").read_text()))
    return frame(rows)


def previous(day: str) -> tuple[str, pd.DataFrame]:
    """Latest dated snapshot strictly before `day` that has a rosters.csv, else August."""
    if ROSTERS.exists():
        prior = sorted(p.name for p in ROSTERS.iterdir()
                       if p.is_dir() and DATE_RE.match(p.name) and p.name < day
                       and (p / "rosters.csv").exists())
        if prior:
            return prior[-1], pd.read_csv(ROSTERS / prior[-1] / "rosters.csv", dtype={"sweater": "Int64"})
    return AUGUST_LABEL, load_august()


def moves(prev: pd.DataFrame, cur: pd.DataFrame) -> pd.DataFrame:
    key = ["team", "player_id"]
    m = prev.merge(cur, on=key, how="outer", suffixes=("_p", "_c"), indicator=True)
    out = []
    for side, change in (("right_only", "added"), ("left_only", "removed")):
        s = "_c" if change == "added" else "_p"
        sub = m[m["_merge"] == side]
        out.append(pd.DataFrame({
            "team": sub["team"], "player_id": sub["player_id"],
            "name": sub["first" + s] + " " + sub["last" + s],
            "pos": sub["pos" + s], "change": change}))
    return (pd.concat(out).sort_values(["team", "change", "name"]).reset_index(drop=True)
            if out else pd.DataFrame(columns=["team", "player_id", "name", "pos", "change"]))


def latest_rosters(on_or_before: str | None = None) -> tuple[str, pd.DataFrame]:
    """(date, rosters frame) of the latest dated snapshot, optionally on/before a date."""
    days = sorted(p.name for p in ROSTERS.iterdir()
                  if p.is_dir() and DATE_RE.match(p.name) and (p / "rosters.csv").exists()
                  and (on_or_before is None or p.name <= on_or_before))
    if not days:
        raise FileNotFoundError(f"no rosters.csv snapshot under {ROSTERS}")
    return days[-1], pd.read_csv(ROSTERS / days[-1] / "rosters.csv", dtype={"sweater": "Int64"})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--date", default=date.today().isoformat())
    ap.add_argument("--offline", action="store_true",
                    help="do not fetch; rebuild csv files from raw JSON on disk")
    a = ap.parse_args()
    if not DATE_RE.match(a.date):
        ap.error("--date must be YYYY-MM-DD")
    ddir = ROSTERS / a.date
    if not a.offline:
        st = fetch(a.date)
        bad = {k: v for k, v in st.items() if v != "ok"}
        print(f"{a.date}: fetched {len(st) - len(bad)}/32" + (f"; failures {bad}" if bad else ""))
    cur = load_dir(ddir)
    if cur.empty:
        sys.exit(f"no roster JSON in {ddir}")
    cur.to_csv(ddir / "rosters.csv", index=False)

    label, prev = previous(a.date)
    for f in ddir.glob("moves_vs_*.csv"):       # one moves file per snapshot
        f.unlink()
    mv = moves(prev, cur)
    mv.to_csv(ddir / f"moves_vs_{label}.csv", index=False)

    # summary
    cnt = cur.pivot_table(index="team", columns="pos", values="player_id",
                          aggfunc="count", fill_value=0)
    cnt = cnt.reindex(columns=["C", "L", "R", "D", "G"], fill_value=0)
    cnt["total"] = cnt.sum(axis=1)
    mc = mv.groupby(["team", "change"]).size().unstack(fill_value=0) if len(mv) else None
    if mc is not None:
        cnt = cnt.join(mc, how="left").fillna(0).astype(int)
    print(cnt.to_string())
    print(f"players: {len(cur)} across {cur['team'].nunique()} teams "
          f"(mean {cnt['total'].mean():.1f}/team); moves vs {label}: "
          f"{(mv['change'] == 'added').sum()} added, {(mv['change'] == 'removed').sum()} removed")
    if a.date < "2026-09-29":
        print("NOTE: camp rosters are inflated until the NHL roster deadline "
              "(2026-09-28 17:00 ET); re-pull after it.")


if __name__ == "__main__":
    main()
