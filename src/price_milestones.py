"""Price X+ point milestone props off the exact ENS 2026-27 sim distribution.

Rebuilds the report4 h1 ENS simulation deterministically (same seeds/closures), writes
output/point_threshold_probs_ens.csv — P(pts >= k) for every team, k in 75..125 —
and prices any board in data/market/nhl_point_milestones_2027.csv: model probability,
implied probability, edge, EV per dollar, full and quarter Kelly.

One-sided milestone boards hide the hold (no Under quoted): a bet needs a cushion,
not just a positive sign. Report-only; nothing feeds back into the model.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as E
import goalie_game as GG
import players as P
from features import FeatureBuilder
from report4 import prod_noise, real_schedule_flags
from scoring import american_to_decimal, american_to_prob

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "output"
THRESHOLDS = list(range(75, 126))


def rebuild_ens_sim(n_sims=10_000):
    p2 = json.loads((OUT / "params_v2.json").read_text())
    p4 = json.loads((OUT / "params_v4.json").read_text())
    v1 = json.loads((OUT / "params.json").read_text())
    ship = p4["shipped"]
    g, ts = E.load()
    preds, end_r, _ = E.run_elo(g, K=v1["K"], H=v1["H"], phi_s=v1["phi_s"])
    fb = FeatureBuilder(end_r, ts, goalie_hp=p2["goalie_hp"],
                        skater_delta=p2["skater_delta"])
    om = E.fit_outcome(preds, list(range(2006, 2027)))
    sk, go, skt, got, bios = P.load_panels()
    ratings = dict(pd.read_csv(OUT / "v4_prior_ratings.csv", index_col=0)["rating_ens"])
    noise_fn, nteams, *_ , tand, frag = prod_noise(fb, sk, got, bios, ts, p2["k"], 1,
                                                   ship["n0_g"])
    sched = real_schedule_flags()
    gn = GG.make_game_noise(sched, tand, p2["k"]) if ship["goalie_layer"] else None
    sim = E.simulate_season(ratings, ship["sigma_c4"], sched[["home", "away", "d_adj"]],
                            om, E.DIVISIONS_CURRENT, n_sims,
                            np.random.default_rng(411 + 1),  # report4 ENS h1 seed
                            playoffs=False, extra_noise=noise_fn, game_noise=gn)
    return sim


def main():
    sim = rebuild_ens_sim()
    pts = sim["pts"].astype(float)
    rows = []
    for i, t in enumerate(sim["teams"]):
        row = {"team": t, "xPts": round(float(pts[:, i].mean()), 2)}
        for k in THRESHOLDS:
            row[f"p_ge_{k}"] = float((pts[:, i] >= k).mean())
        rows.append(row)
    tab = pd.DataFrame(rows).set_index("team")
    tab.round(4).to_csv(OUT / "point_threshold_probs_ens.csv")

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
    print("\nwrote point_threshold_probs_ens.csv + milestones_vs_model.csv")


if __name__ == "__main__":
    main()
