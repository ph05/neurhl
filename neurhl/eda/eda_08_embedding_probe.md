# EDA-08 — embedding probe (vantage 2017, train window only)

Embedding matrix: 5181 players x 64 dims; 2213 with NHL games, 2968 career-encoder-only (cold start).

## P1 — position separation (F/D/G)

5-fold linear accuracy **0.952** vs majority-class baseline 0.597 (n=1664).


## P2 — skill/usage regression from embeddings

- points per 60: 5-fold R^2 **0.540** (n=968)

- TOI per game: 5-fold R^2 **0.592** (n=968)


## P3 — nearest neighbours (cosine)

- **Ryan Suter** (D): Mark Streit (D, 0.48), Matt Cullen (F, 0.38), Marian Gaborik (F, 0.37), Noah Welch (D, 0.36), Paul Martin (D, 0.35)

- **Drew Doughty** (D): John Curry (G, 0.47), 8456531 (D, 0.40), Miroslav Satan (F, 0.40), Alexandre Picard (D, 0.38), Niclas Havelid (D, 0.37)

- **Shea Weber** (D): Dylan Reese (D, 0.36), Brett Festerling (D, 0.36), Steve Wagner (D, 0.36), Justin Schultz (D, 0.35), Scott Hannan (D, 0.35)


## P4 — cold-start (career-encoder-only) players

2968 cold-start players; mean cosine to their nearest experienced player **0.458** (median 0.451). Higher = the career encoder places rookies inside the learned manifold rather than off in a corner.


## Verdict

**Event-LM embeddings carry real signal.** Position is linearly decodable at 0.952 (chance 0.597), and a linear probe recovers a majority of the variance in scoring rate and ice time even though neither was ever a training target — the model saw only next-event prediction. Nearest-neighbour structure is noisier (cosines ~0.35-0.50, with occasional cross-position neighbours), which is consistent with the embedding encoding role and usage volume more sharply than fine-grained quality within a position.


**DOCUMENTED NULL — the career encoder did not learn.** Across vantages its validation MSE is 0.991-1.004x the predict-the-mean baseline, i.e. indistinguishable from (and at two vantages slightly worse than) simply emitting the centroid. Pre-NHL junior/AHL/European production does not linearly explain where a player lands in event-LM space. Consequence is benign but must be stated plainly: cold-start players receive approximately the mean embedding — the same thing the position-mean fallback would give them — so NeurHL has NO working rookie-projection mechanism, and any 2026-27 result must not be attributed to one. The rookie-blend weight w = gp/(gp+40) still functions as sensible shrinkage of thin-sample players toward the centroid.
