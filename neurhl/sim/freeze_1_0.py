"""Append the FREEZE section to PLAN_NeurHL_1_0.md (run once, after
sim/unified_2027.py and tests/check_neurhl_1_0.py pass on the post-deadline
rosters, before the first regular-season game).

Records the SHA-256 of every NeurHL 1.0 output file, the run's settings
(rosters, draws, simulated seasons, season shock, A1 multiplier, engine
bundle, code commit) and the consistency check result. Refuses if the check
has not passed or if a FREEZE section already exists.
"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJ = ROOT.parent
OUT = ROOT / "output" / "neurhl_1_0"
PLAN = PROJ / "PLAN_NeurHL_1_0.md"
FILES = ["games_2027.csv", "teams_2027.csv", "team_points_quantiles_2027.csv", "skaters_2027.csv",
         "goalies_2027.csv", "player_games_2027.csv.gz", "consistency_2027.json", "checks_2027.json",
         "run_2027.json", "player_rates_2027.csv", "availability_2027.csv"]


def main():
    text = PLAN.read_text()
    if "\n## FREEZE" in text:
        sys.exit("PLAN_NeurHL_1_0.md already has a FREEZE section")
    chk = json.loads((OUT / "checks_2027.json").read_text())
    if not chk["pass"]:
        sys.exit("consistency check has not passed")
    run = json.loads((OUT / "run_2027.json").read_text())
    rows = "\n".join(f"| `neurhl/output/neurhl_1_0/{f}` | `{hashlib.sha256((OUT / f).read_bytes()).hexdigest()}` |"
                     for f in FILES)
    t = pd.read_csv(OUT / "teams_2027.csv").sort_values("points", ascending=False)
    top = ", ".join(f"{r.team} {r.points:.1f}" for r in t.head(5).itertuples())
    cup = ", ".join(f"{r.team} {r.cup_pct:.1f}%" for r in t.sort_values("cup_pct", ascending=False).head(5).itertuples())
    text += f"""
## FREEZE ({run['created_utc'][:10]}): the NeurHL 1.0 predictions for 2026-27

Run on the post-deadline rosters of {run['rosters_date']} with {run['draws']} lineup
draws and {run['sims']:,} simulated seasons (seed {run['seed']}); season shock sd
{run['team_sigma']}; stat-sheet goal multiplier {run['goal_mult_m0']:.4f}; engine bundle
`{run['bundle']}` (sha256 {run['bundle_sha']}...); lineups from {run['availability']};
code at commit `{run['code']}`. The consistency check passed {chk['passed']} of {chk['n']}.

| File | SHA-256 |
|---|---|
{rows}

Highest projected points: {top}. Highest Cup odds: {cup}.

These files are never edited. A correction is issued as a new, dated file set
beside them, and both are scored.
"""
    PLAN.write_text(text)
    print(f"FREEZE appended to {PLAN.name}")


if __name__ == "__main__":
    main()
