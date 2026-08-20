# EDA-07 — HTM backfill integrity (A1)

## 2011

{
 "games": 1230,
 "gpg_htm": 5.5854,
 "sog_pg_htm": 30.3915,
 "fac_pg_htm": 57.9138,
 "pen_pg_htm": 9.4325,
 "gpg_official": 5.5854,
 "sog_pg_official": 30.3898,
 "gpg_rel_err": 0.0,
 "sog_rel_err": 0.0001,
 "pass_98pct": true
}

## 2012

{
 "games": 1230,
 "gpg_htm": 5.4683,
 "sog_pg_htm": 29.75,
 "fac_pg_htm": 57.6358,
 "pen_pg_htm": 8.7919,
 "gpg_official": 5.4683,
 "sog_pg_official": 29.7476,
 "gpg_rel_err": 0.0,
 "sog_rel_err": 0.0001,
 "pass_98pct": true
}

## 2012 HTM-vs-JSON cross-validation (1230 games)

Per-type totals and per-game exact-count agreement:

| event        |   exact_game_rate |   rel_err |
|:-------------|------------------:|----------:|
| blocked-shot |                 1 |         0 |
| faceoff      |                 1 |         0 |
| giveaway     |                 1 |         0 |
| goal         |                 1 |         0 |
| hit          |                 1 |         0 |
| missed-shot  |                 1 |         0 |
| penalty      |                 1 |         0 |
| shot-on-goal |                 1 |         0 |
| takeaway     |                 1 |         0 |

Goals: 6545/6545 matched on exact (game, second); scorer playerId agreement 1.0000

Faceoffs matched on (game, second): 70962/70892; winner playerId agreement 0.9311

On-ice exact-set agreement at matched goals: 0.9719 (6545 goals)
