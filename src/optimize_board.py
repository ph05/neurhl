"""Portfolio allocation over a full totals board (PLAN_V4 I2 tooling; report-only).

Method, in the order it matters:
1. Exact joint distribution: the HOWE 2026-27 sim (10k draws x 32 teams) prices every
   instrument AND their correlations (same-team ladders are nested; cross-team totals
   are weakly coupled through head-to-head games).
2. Market ensemble shrink: betting probability = 0.5*model + 0.5*devigged market.
   The same logic that ensembles v1 with v4 applies to the market itself: this book's
   totals sum to 2998 (feasible league total ~2997), i.e. the market is a coherent
   forecaster and gets a 50% vote. Two-sided O/U pairs devig exactly; milestone lines
   use the market curve Normal(devigged median, team sim SD).
3. Joint Kelly: maximize E[log wealth] across the 10k scenarios (SLSQP, analytic
   gradient). To make the optimizer size on SHRUNK edges while keeping the model's
   correlation structure, each instrument's decimal odds are rescaled by
   p_bet/p_model so that model-frequency x effective-odds reproduces the shrunk EV.
4. Discipline: final stakes = quarter of the log-optimal fractions, $10/line cap,
   min $0.50, candidates need shrunk EV > 2% (one-sided milestone vig is hidden).

Outputs: output/board_allocation.csv + printed table. Nothing feeds the model.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from howe import rebuild_sim
from scoring import american_to_decimal, american_to_prob

PROJ = Path(__file__).resolve().parents[1]
MARKET = PROJ / "data" / "market"
OUT = PROJ / "output"
BANKROLL = 100.0
KELLY_SCALE = 0.25
CAP_PER_LINE = 10.0
MIN_STAKE = 0.50
MIN_EDGE = 0.02
MKT_WEIGHT = 0.5


def main():
    sim = rebuild_sim()
    teams = sim["teams"]
    draws = sim["pts"].astype(float)          # (n_sims, 32), integer-valued
    tidx = {t: i for i, t in enumerate(teams)}
    sd = draws.std(axis=0, ddof=1)

    ou = pd.read_csv(MARKET / "nhl_totals_ou_2027.csv")
    ms = pd.read_csv(MARKET / "nhl_point_milestones_2027.csv")

    # market curve per team from the two-sided O/U
    mkt_median, holds = {}, []
    for r in ou.itertuples():
        po = american_to_prob(np.array([r.over_american]))[0]
        pu = american_to_prob(np.array([r.under_american]))[0]
        holds.append(po + pu - 1)
        p_over = po / (po + pu)
        j = tidx[r.team]
        mkt_median[r.team] = r.line + sd[j] * norm.ppf(p_over)
    print(f"book: O/U hold mean {np.mean(holds):.3%}; totals sum "
          f"{ou.line.sum():.0f} vs model xPts sum {draws.mean(0).sum():.0f} "
          f"(feasible ~2997) -> coherent board, ordering disagreements only")

    def mkt_prob_ge(team, k):
        j = tidx[team]
        return float(1 - norm.cdf((k - 0.5 - mkt_median[team]) / sd[j]))

    instruments = []  # (label, win_bool_vector, dec, p_model, p_mkt)
    for r in ou.itertuples():
        j = tidx[r.team]
        po = american_to_prob(np.array([r.over_american]))[0]
        pu = american_to_prob(np.array([r.under_american]))[0]
        k = int(np.ceil(r.line))
        win_over = draws[:, j] >= k
        instruments.append((f"{r.team} Over {r.line}", win_over,
                            american_to_decimal(np.array([r.over_american]))[0],
                            float(win_over.mean()), po / (po + pu)))
        instruments.append((f"{r.team} Under {r.line}", ~win_over,
                            american_to_decimal(np.array([r.under_american]))[0],
                            float((~win_over).mean()), pu / (po + pu)))
    for r in ms.itertuples():
        j = tidx[r.team]
        win = draws[:, j] >= r.threshold
        instruments.append((f"{r.team} {r.threshold}+ (milestone)", win,
                            american_to_decimal(np.array([r.american]))[0],
                            float(win.mean()), mkt_prob_ge(r.team, r.threshold)))

    rows, keep = [], []
    for label, win, dec, p_m, p_k in instruments:
        p_bet = (1 - MKT_WEIGHT) * p_m + MKT_WEIGHT * p_k
        ev = p_bet * dec - 1
        rows.append({"line": label, "dec": dec, "p_model": p_m, "p_market": p_k,
                     "p_blend": p_bet, "ev_blend": ev, "ev_model_pure": p_m * dec - 1})
        if ev > MIN_EDGE and p_m > 1e-4:
            keep.append((label, win, dec, p_m, p_bet))
    table = pd.DataFrame(rows)

    # joint Kelly on shrunk edges with model correlations (odds rescaled p_bet/p_model)
    n = len(keep)
    R = np.zeros((draws.shape[0], n))
    for i, (label, win, dec, p_m, p_bet) in enumerate(keep):
        dec_eff = dec * p_bet / p_m
        R[:, i] = np.where(win, dec_eff - 1.0, -1.0)

    def neg_elog(f):
        w = 1.0 + R @ f
        return -np.mean(np.log(np.maximum(w, 1e-9)))

    def grad(f):
        w = np.maximum(1.0 + R @ f, 1e-9)
        return -(R / w[:, None]).mean(axis=0)

    cons = [{"type": "ineq", "fun": lambda f: 0.9 - f.sum(),
             "jac": lambda f: -np.ones(n)}]
    res = minimize(neg_elog, np.full(n, 0.005), jac=grad, method="SLSQP",
                   bounds=[(0.0, 0.5)] * n, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-12})
    f_opt = res.x

    stakes = np.minimum(f_opt * KELLY_SCALE * BANKROLL, CAP_PER_LINE)
    stakes = np.round(stakes * 2) / 2
    picks = []
    for i, (label, win, dec, p_m, p_bet) in enumerate(keep):
        if stakes[i] >= MIN_STAKE:
            picks.append({"line": label, "stake": stakes[i], "dec_odds": round(dec, 2),
                          "p_model": round(p_m, 3), "p_market": round(p_bet * 2 - p_m, 3),
                          "p_blend": round(p_bet, 3),
                          "ev_blend": round(p_bet * dec - 1, 3),
                          "kelly_full_frac": round(float(f_opt[i]), 4)})
    picks = pd.DataFrame(picks).sort_values("stake", ascending=False)
    print(f"\noptimizer: {n} candidates, {len(picks)} funded, "
          f"total ${picks.stake.sum():.2f} of ${BANKROLL:.0f} "
          f"(full-Kelly would deploy {f_opt.sum():.1%})")
    print(picks.to_string(index=False))
    picks.to_csv(OUT / "board_allocation.csv", index=False)
    table.round(4).to_csv(OUT / "board_all_lines_priced.csv", index=False)

    # portfolio risk profile at the recommended stakes (under model scenarios, real odds)
    Rr = np.zeros((draws.shape[0], len(keep)))
    for i, (label, win, dec, p_m, p_bet) in enumerate(keep):
        Rr[:, i] = np.where(win, dec - 1.0, -1.0)
    pnl = Rr @ stakes
    print(f"\nportfolio P&L under MODEL scenarios: mean ${pnl.mean():+.2f}, "
          f"P5 ${np.percentile(pnl, 5):+.2f}, P50 ${np.percentile(pnl, 50):+.2f}, "
          f"P95 ${np.percentile(pnl, 95):+.2f}, P(loss) {np.mean(pnl < 0):.1%}")
    print("wrote board_allocation.csv + board_all_lines_priced.csv")


if __name__ == "__main__":
    main()
