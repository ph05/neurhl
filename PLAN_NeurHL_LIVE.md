# PLAN NeurHL LIVE — prospective scoring of the 2026-27 season

STATUS: COMMITTED 2026-09-25, before the first regular-season game
(2026-09-29). Nothing in this file or in the files it freezes changes after
commit. A correction is issued as a new, dated file beside the original, and
both are scored.

## 1. Frozen predictions

| File | What it predicts | SHA-256 |
|---|---|---|
| `neurhl/output/games_2027.csv` | Home-win probability for all 1,344 regular-season games: NeurHL (`p_home_win`) and the preseason Elo reference (`p_home_win_elo`) | `85565efd773b1bd5c8df9eaf64235fb247874e9bcff634eac6a3e0b17369f039` |
| `neurhl/output/projection_2027.csv` | Team points (mean, 10th and 90th percentiles) and playoff probability, 20,000 simulated seasons | `4c60f50f525e751a79d4038aa65d6354b52e4500cfc5e5bc46a01b95e8320c51` |
| `neurhl/output/player_proj_2027.csv` | Skater goals, assists and points for 711 skaters on announced rosters (the PS1 blend, plus its two paths) | `cc4cc26de835d875a159603d9d42ddf6078fb85230ba0937f1d536a8953642de` |
| `output/projections_2026_27_howe.csv` | HOWE team points (comparator; committed in August, before NeurHL existed) | `a4a693f4929bcae8cd7a4453291ae1863a235c8cab7f641255f43c6530788b0f` |

All four are preseason and static: nothing updates in-season. That keeps
every comparison on the same footing. The house's in-season models (v1, v4,
HOWE, scored by `src/live.py`) are a separate track and are unaffected.

NeurHL-H, the game model under one-shot confirmation (PLAN_NeurHL A5),
needs each game's dressed lineup and so cannot be frozen preseason. It is not
part of this file.

## 2. What is scored

**L1 — games.** Per-game log loss of `p_home_win` against `p_home_win_elo`,
with the outcome defined as a home win in regulation, overtime or shootout.
Statistic: the mean paired difference (NeurHL minus Elo) over all 1,344 games.

**L2 — standings.** Final team points against NeurHL `proj_points` and HOWE
`xPts`: mean absolute error over 32 teams, and the share of teams whose final
points fall inside NeurHL's 10th-90th percentile interval (nominal 0.80).

**L3 — players.** Final points for skaters with at least 40 games played:
mean absolute error of the shipped blend (`proj_p`), Path A
(`proj_p_path_a`) and Path B (`proj_p_path_b`). PS1 predicts the blend is
lowest.

## 3. When inference happens

Once, after the last regular-season game (2027-04-10). Interim scorecards
(`neurhl/output/live/scorecard_2027.json`) are descriptive and carry no
p-values, because repeated looks at a fixed-horizon test inflate its error
rate. The final L1 interval is a bootstrap over weeks of play (9,999 draws,
seed 711), which respects the dependence between games played close
together.

## 4. Expectations, stated before any game

- **L1:** the engine behind `p_home_win` scored 0.6842 on the tune window,
  against Elo's 0.6767 (PLAN_NeurHL2, final status). Its probabilities are also
  about twice as dispersed as Elo's (SD 0.115 vs 0.057). We expect Elo to win
  L1. With one season the standard error of the difference is roughly 0.002
  to 0.003, so only a gap of about 0.006 or more is likely to be resolved.
- **L2:** on matched backtest seasons the NeurHL season layer trails v1
  (MAE 9.99 vs 9.17, not significant; `neurhl/configs/season_matched_comparison.json`).
  We expect HOWE to be at least as accurate. Coverage should be near 0.80
  (0.794 in backtest).
- **L3:** the blend should beat both paths, as in PS1 (9.07 vs 9.41 and 9.77
  per 82 games, pooled over 11 backtest seasons).

A season is 32 teams and 1,344 games. No single season can confirm a small
effect at any of these levels; the purpose here is prospective accountability.
All three results are published whatever they show.

## 5. Procedure

`neurhl/eval/score_live_2027.py` fetches completed games from the public NHL
API, caches them in `neurhl/output/live/results_2027.csv`, and writes the
scorecard. Cached results are committed with each update, so any scorecard can
be rebuilt offline (`--offline`).
