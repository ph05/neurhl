# EDA-09 — where the game-outcome signal lives

Trained on seasons <= 2013 (6,862 games), tested on 2015 (1,230 games). Logistic regression, best of C in (0.003, 0.01, 0.03, 0.3). Report-only.

| feature set | test home-win log loss |
|---|---|
| A. roster embeddings (mean-pooled) + context | **0.6888** (C=0.003) |
| B. team rolling form only (6 features) | **0.6787** (C=0.3) |
| C. team form + context | **0.6765** (C=0.3) |
| D. team form + context + embeddings | **0.6840** (C=0.003) |
| — constant home rate (0.549) | 0.6898 |
| — v1 (Elo), same season | 0.6709 |
| — NeurHL config #1 network, same season | 0.6851 |

## Reading

Roster embeddings alone (0.6888) are indistinguishable from knowing
nothing (0.6898), while **six team-history features (0.6787) beat
the entire 1.2M-parameter network (0.6851)**, and adding embeddings on
top of form makes it *worse* (0.6840 vs 0.6765) — they enter as
noise.

A logistic regression cannot be accused of failing to optimize. The signal is
therefore not present-but-hard-to-extract in pooled roster embeddings; it is
largely absent from that representation. Combined with EDA-08 (position
linearly decodable at 0.952, points/60 R^2 0.54), the interpretation is that
the event-LM learned what KIND of player someone is much better than how GOOD
their team is — and since every NHL roster carries a similar distribution of
roles, pooling over one barely separates teams.

Consequence for the project: capacity/regularization tuning cannot rescue
config #1, because even an optimally-regularized linear model on these features
tops out at 0.6888. PLAN_NeurHL AMENDMENT A1 therefore adds team form
to the context vector rather than spending budget on capacity sweeps, and
restates the scientific claim accordingly.
