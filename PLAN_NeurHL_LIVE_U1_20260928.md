# PLAN NeurHL LIVE — update U1 (2026-09-28): post-deadline rosters

STATUS: COMMITTED AFTER THE NHL ROSTER DEADLINE (2026-09-28 17:00 ET) AND BEFORE
THE FIRST REGULAR-SEASON GAME (2026-09-29). PLAN_NeurHL_LIVE.md and its frozen
files are unchanged; under its correction rule these dated files are issued
beside the originals and both are scored. The originals remain primary.

## What changed

Only the inputs: the rosters teams filed at the deadline (snapshot
`data/raw/rosters/2026-09-28/`, with its moves file against the previous
snapshot) and the availability entries in `data/manual/player_status_2027.csv`
(sha256 `4cc764899a0442cb47e4b37f6c9b75040026f2127870ee3d686d65ba75b844e4`), which remove players who cannot dress (for example
Connor Hellebuyck, whose suspension Winnipeg announced on 2026-09-16). The models,
code paths and random seeds are those of the 2026-09-25 freeze; its team model has
no goalie term, so a goalie's absence moves it only through the roster's skaters.
NeurHL 1.0's own predictions are frozen separately (PLAN_NeurHL_1_0.md).

## Files

| File | SHA-256 |
|---|---|
| `neurhl/output/games_2027_20260928.csv` | `b82ddd9ccebc4dc732259dbda726568205b363573ad2d3c4ca1a8237c8d9803d` |
| `neurhl/output/projection_2027_20260928.csv` | `527456d11b049dcb400b013d1a4c4f126e3f924d0fdf7f3d93d1ea388e48c4c6` |
| `neurhl/output/player_proj_2027_20260928.csv` | `7ee929c90567e9f53a06bb3b2f854ae21df7220692e8552aa8b696922fbd8175` |

Moves file: `data/raw/rosters/2026-09-28/moves_vs_2026-09-27.csv`.

## Scoring

`neurhl/eval/score_live_2027.py --variant 20260928` scores these files with the
same metrics and the same single end-of-season inference as the originals.
