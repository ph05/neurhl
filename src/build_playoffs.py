"""Playoff vs regular-season environment diagnostics (PLAN_V6 D5, report-only).

From the playoff PBP corpus (2012-2026): goals/game, OT rate, home win %, 5v5 SAT
share of events, penalties/game — side by side with the regular-season corpus.
Writes output/playoff_diagnostics.json. Declared report-only in PLAN_V6: nothing
ships from this; it informs whether a future plan should give the playoff sim its
own outcome parameters.
"""
import gzip
import json
import sys
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "output"


def corpus_stats(root: Path) -> dict:
    rows = []
    for sdir in sorted(d for d in root.iterdir() if d.is_dir()):
        for f in sorted(sdir.glob("*.json.gz")):
            with gzip.open(f, "rt") as fh:
                g = json.load(fh)
            goals = pens = 0
            for p in g["plays"]:
                if p["typeDescKey"] == "goal" \
                        and p.get("periodDescriptor", {}).get("periodType") != "SO":
                    goals += 1
                elif p["typeDescKey"] == "penalty":
                    pens += 1
            went_ot = any(pl.get("periodDescriptor", {}).get("number", 0) > 3
                          for pl in g["plays"])
            hs = g.get("homeTeam", {}).get("score", 0)
            as_ = g.get("awayTeam", {}).get("score", 0)
            rows.append({"season_end": int(sdir.name), "goals": goals,
                         "pens": pens, "home_win": hs > as_, "ot": went_ot})
    df = pd.DataFrame(rows)
    return {"n_games": len(df),
            "goals_pg": round(float(df.goals.mean()), 3),
            "pen_pg": round(float(df.pens.mean()), 3),
            "home_win_pct": round(float(df.home_win.mean()), 4),
            "ot_pct": round(float(df.ot.mean()), 4)}


def main():
    po = corpus_stats(PROJ / "data" / "raw" / "pbp_po")
    rs = corpus_stats(PROJ / "data" / "raw" / "pbp")
    blob = {"playoffs": po, "regular": rs,
            "note": "report-only (PLAN_V6 D5); playoff sim params unchanged"}
    (OUT / "playoff_diagnostics.json").write_text(json.dumps(blob, indent=2))
    print(json.dumps(blob, indent=2))


if __name__ == "__main__":
    main()
