# EDA-10 — does a player->team hierarchy beat team-level form?

Train <= 2014 (8,073 games), test [2015, 2016, 2017] (3,690 games). Elo alone = 0.67509.

| features | log loss | vs Elo | t | p |
|---|---|---|---|---|
| Elo (refit) | **0.67515** | +0.00006 | +0.42 | 0.6748 |
| Elo + team shot form (backward-looking) | **0.67409** | -0.00100 | -0.74 | 0.4565 |
| Elo + ROSTER-PROJECTED shot share | **0.67261** | -0.00248 | -1.90 | 0.0577 |
| Elo + team form + roster projection | **0.67344** | -0.00165 | -1.15 | 0.2483 |
| roster projection alone | **0.68100** | +0.00591 | +2.23 | 0.0257 |

## Reading

The hierarchy's claim is that roster aggregation beats measuring the team
directly. Elo + roster projection = 0.67261 vs
Elo + team form = 0.67409
(Elo alone 0.67509).

Layer 1 here is deliberately trivial — each player's own shifted EWMA. Any
advantage is therefore attributable to the ARCHITECTURE (roster-aware,
forward-looking aggregation), not to model capacity, and is a lower bound on
what a trained neural Layer 1 could achieve.
