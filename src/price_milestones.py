"""Price X+ point milestone props off the exact HOWE 2026-27 sim distribution.

Rebuilds the report4 h1 HOWE simulation deterministically (howe.rebuild_sim, same
seeds/closures), writes output/point_threshold_probs_howe.csv — P(pts >= k) for
every team, k in 75..125 — and prices any board in
data/market/nhl_point_milestones_2027.csv: model probability, implied probability,
edge, EV per dollar, full and quarter Kelly.

One-sided milestone boards hide the hold (no Under quoted): a bet needs a cushion,
not just a positive sign. Report-only; nothing feeds back into the model.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from howe import rebuild_sim
from scoring import american_to_decimal, american_to_prob

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "output"
THRESHOLDS = list(range(75, 126))


def main():
    sim = rebuild_sim()
    pts = sim["pts"].astype(float)
    rows = []
    for i, t in enumerate(sim["teams"]):
        row = {"team": t, "xPts": round(float(pts[:, i].mean()), 2)}
        for k in THRESHOLDS:
            row[f"p_ge_{k}"] = float((pts[:, i] >= k).mean())
        rows.append(row)
    tab = pd.DataFrame(rows).set_index("team")
    tab.round(4).to_csv(OUT / "point_threshold_probs_howe.csv")

    board = pd.read_csv(PROJ / "data/market/nhl_point_milestones_2027.csv")
    board["p_model"] = [float(tab.loc[r.team, f"p_ge_{int(r.threshold)}"])
                        for r in board.itertuples()]
    board["p_implied"] = american_to_prob(board.american.to_numpy())
    board["dec"] = american_to_decimal(board.american.to_numpy())
    board["edge"] = board.p_model - board.p_implied
    board["ev_per_dollar"] = board.p_model * board.dec - 1
    board["kelly_full"] = ((board.p_model * board.dec - 1) / (board.dec - 1)).clip(lower=0)
    board["stake_quarter_kelly_100"] = (25.0 * board.kelly_full).round(2)
    board["model_xPts"] = board.team.map(tab.xPts)
    cols = ["team", "threshold", "american", "model_xPts", "p_model", "p_implied",
            "edge", "ev_per_dollar", "kelly_full", "stake_quarter_kelly_100"]
    out = board[cols].sort_values("ev_per_dollar", ascending=False)
    print(out.round(3).to_string(index=False))
    out.round(4).to_csv(OUT / "milestones_vs_model.csv", index=False)
    print("\nwrote point_threshold_probs_howe.csv + milestones_vs_model.csv")


if __name__ == "__main__":
    main()
