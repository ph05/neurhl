"""Post-deadline refresh of the 2026-09-25 preseason projections (PLAN_NeurHL4 D).

Run after the NHL roster deadline (2026-09-28, 17:00 ET):
  1. dated roster snapshot (neurhl/live/fetch_rosters.py) unless it exists;
  2. the 2026-09-25 team and player projections rerun on it with the frozen code
     and models (sim/project_2027.py and sim/project_players.py --roster-dir); status
     entries (data/manual/player_status_2027.csv) remove unavailable players;
  3. PLAN_NeurHL_LIVE_U1_<tag>.md: the dated update notice with SHA-256 hashes
     of the new files, the roster snapshot and the status file.
The frozen 2026-09-25 files and PLAN_NeurHL_LIVE.md are never modified; the originals
stay primary and the dated files are scored alongside
(eval/score_live_2027.py --variant <tag>).

Usage: ... python neurhl/live/refresh_preseason.py --date 2026-09-28
"""
import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJ = ROOT.parent


def sh(*cmd):
    r = subprocess.run(cmd, cwd=PROJ, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(f"failed: {' '.join(cmd)}")
    return r.stdout


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    a = ap.parse_args()
    tag = a.date.replace("-", "")
    snap = PROJ / "data" / "raw" / "rosters" / a.date
    py = sys.executable
    if not (snap / "rosters.csv").exists():
        print(sh(py, str(ROOT / "live" / "fetch_rosters.py"), "--date", a.date)[-1500:])
    out_team = sh(py, str(ROOT / "sim" / "project_2027.py"), "--roster-dir", str(snap), "--tag", tag)
    out_pl = sh(py, str(ROOT / "sim" / "project_players.py"), "--roster-dir", str(snap), "--tag", tag)
    files = [ROOT / "output" / f"games_2027_{tag}.csv",
             ROOT / "output" / f"projection_2027_{tag}.csv",
             ROOT / "output" / f"player_proj_2027_{tag}.csv"]
    status = PROJ / "data" / "manual" / "player_status_2027.csv"
    moves = sorted(snap.glob("moves_vs_*.csv"))
    rows = "\n".join(f"| `{f.relative_to(PROJ)}` | `{sha(f)}` |" for f in files)
    note = PROJ / f"PLAN_NeurHL_LIVE_U1_{tag}.md"
    note.write_text(f"""# PLAN NeurHL LIVE — update U1 ({a.date}): post-deadline rosters

STATUS: COMMITTED AFTER THE NHL ROSTER DEADLINE ({a.date} 17:00 ET) AND BEFORE
THE FIRST REGULAR-SEASON GAME (2026-09-29). PLAN_NeurHL_LIVE.md and its frozen
files are unchanged; under its correction rule these dated files are issued
beside the originals and both are scored. The originals remain primary.

## What changed

Only the inputs: the rosters teams filed at the deadline (snapshot
`data/raw/rosters/{a.date}/`, with its moves file against the previous
snapshot) and the availability entries in `data/manual/player_status_2027.csv`
(sha256 `{sha(status)}`), which remove players who cannot dress (for example
Connor Hellebuyck, whose suspension Winnipeg announced on 2026-09-16). The models,
code paths and random seeds are those of the 2026-09-25 freeze; its team model has
no goalie term, so a goalie's absence moves it only through the roster's skaters.
NeurHL 1.0's own predictions are frozen separately (PLAN_NeurHL_1_0.md).

## Files

| File | SHA-256 |
|---|---|
{rows}

Moves file: `{moves[-1].relative_to(PROJ) if moves else 'none'}`.

## Scoring

`neurhl/eval/score_live_2027.py --variant {tag}` scores these files with the
same metrics and the same single end-of-season inference as the originals.
""")
    print(out_team[-1200:])
    print(f"-> {note.relative_to(PROJ)}")


if __name__ == "__main__":
    main()
