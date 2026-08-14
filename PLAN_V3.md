# v3 — Accuracy Core + Living Model (commissioned 2026-08-13)

Selected directions: Tier-1 accuracy stack + in-season living model.
Discipline unchanged: every addition tunes on train (<=2017) only; 2018-2026 preseason windows
are SPENT and are not re-litigated; new evaluation axes (in-season replay calibration) use
2012-2017 for tuning and 2022-2026 for a single reported validation; the live 2026-27 season is
the pre-registered out-of-sample test for everything.

## A. Accuracy core
A1. Finishing skill (skaters): EB posterior on (goals - ixG)/shot from panels
    (I_F_goals, I_F_xGoals) — same machinery as goalie GSAx. Feeds player value and a
    team finishing feature.
A2. Special-teams split: team 5on4 xGF/60 and 4on5 xGA/60 (mp_teams situational rows,
    already cached), EB-shrunk by TOI, as 2 features with own reliability.
A3. Defensive/two-way value: on-ice vs off-ice xG% isolation (rel-xG%), EB-shrunk;
    upgrades player valuation beyond points; team aggregate as feature candidate.
A4. Injuries/availability: per-player beta-binomial GP model (age, prior GP) fit on panels;
    in sims, draw availability for each team's top-9 value players; team strength hit =
    missing value x replacement gap; adds mean adjustment (expected man-games lost) and
    variance (star fragility fattens tails).
A5. Goalie-start rotation: real schedule has dates -> back-to-back detection; starter/backup
    assignment per game (share model + b2b backup rule); per-game strength adjust by
    assigned goalie GSAx delta.
A6. Prospect arrival curve: name-join drafts 2009-2019 to panels (name+position), fit
    value-realized-by-year-since-draft ramp; replaces flat draft_cap decay. (If join quality
    <80%, keep declared curve, document.)
Gate: A1-A3, A6 enter the ridge only if train LOSO MAE improves or is neutral with stable
signs (ledger rows regardless). A4-A5 are simulator upgrades gated by train-replay
calibration (coverage/Brier), not MAE.

## B. Living model
B1. Prior-fade posterior: in-season team strength = w(n)*preseason_ridge + (1-w(n))*in-season
    Elo, w(n) = n0/(n0+games_played); n0 tuned on 2012-2017 in-season replays (log loss of
    game predictions + rest-of-season points MAE at checkpoints g in {10,20,41,60}).
B2. Replay engine: reconstruct any past date's state (ratings, standings, remaining schedule),
    sim rest-of-season -> playoff odds path. Validation (single run, reported): 2022-2026
    replays, checkpoint calibration of playoff odds vs eventual outcomes.
B3. live.py: fetch new 2026-27 results from NHL API score endpoints -> update state -> rest-of-
    season sims -> live_odds.csv + dated snapshots. Runnable nightly from Sept 29, 2026.
B4. Cap/contract: age+service UFA/RFA heuristic flags on roster projections (no clean public
    contract source since CapFriendly shut; documented limitation).

## Deliverables
- v3 spreadsheet: revised Projections_2026_27 (passed features + injury/goalie sim), new
  Availability sheet, Finishing sheet, updated ledger; in-season calibration report sheet.
- live.py + LIVE_README.md (nightly operation).
- All gates/results appended to params_v3.json + NOTES.md; git-tagged v3.
