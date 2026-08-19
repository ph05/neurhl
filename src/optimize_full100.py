"""Complete $100 test-budget allocation (ring-fenced; Σ stakes = $100 exactly).

Differences from optimize_board.py: the budget is DEDICATED — the objective is to
deploy all of it as well as possible, so we maximize E[log(floor + W)] over the joint
scenarios subject to sum(f) = 1 (not <= a Kelly fraction of a larger bankroll). The
floor (2% of budget) encodes that a wipeout of the test budget is survivable; log
still forces genuine diversification and starves negative-tail allocations.

Instruments (all priced on ONE joint 10k-scenario HOWE sim with playoffs, so Cup bets
and totals bets are coherent):
  - O/U totals sides (recorded board, two-sided devig exact)
  - point milestones (recorded board; OTT 110+/120+ excluded per the Tkachuk rho-stress)
  - Cup futures at BEST cross-book price (vegasinsider table 2026-08-18, devigged)
Probabilities: 50% model / 50% devigged market (the market earned its vote: totals
sum to the feasible league total). Model correlations kept via odds rescaling.
Caps: 15% per line; min ticket $1; rounding to $0.50 preserving the $100 total.
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
BUDGET = 100.0
CAP = 0.15
FLOOR = 0.02
MIN_TICKET = 1.0
MKT_W = 0.5
EXCLUDE = {("OTT", 110), ("OTT", 120)}   # Tkachuk rho-stress casualties


def main():
    sim = rebuild_sim(playoffs=True)
    teams = sim["teams"]
    tidx = {t: i for i, t in enumerate(teams)}
    draws = sim["pts"].astype(float)
    cup = sim["won_cup"]
    sd = draws.std(axis=0, ddof=1)

    ou = pd.read_csv(MARKET / "nhl_totals_ou_2027.csv")
    ms = pd.read_csv(MARKET / "nhl_point_milestones_2027.csv")
    cb = pd.read_csv(MARKET / "nhl_cup_2027_best_price.csv")

    mkt_median = {}
    for r in ou.itertuples():
        po = american_to_prob(np.array([r.over_american]))[0]
        pu = american_to_prob(np.array([r.under_american]))[0]
        mkt_median[r.team] = r.line + sd[tidx[r.team]] * norm.ppf(po / (po + pu))

    def mkt_ge(team, k):
        return float(1 - norm.cdf((k - 0.5 - mkt_median[team]) / sd[tidx[team]]))

    cb["p_imp"] = american_to_prob(cb.american.to_numpy())
    cup_devig = dict(zip(cb.team, cb.p_imp / cb.p_imp.sum()))
    cup_book = dict(zip(cb.team, cb.book))

    instruments = []   # (label, win_vec, dec, p_model, p_market, venue)
    for r in ou.itertuples():
        j = tidx[r.team]
        po = american_to_prob(np.array([r.over_american]))[0]
        pu = american_to_prob(np.array([r.under_american]))[0]
        w = draws[:, j] >= int(np.ceil(r.line))
        instruments.append((f"{r.team} Over {r.line}", w,
                            american_to_decimal(np.array([r.over_american]))[0],
                            float(w.mean()), po / (po + pu), "your book"))
        instruments.append((f"{r.team} Under {r.line}", ~w,
                            american_to_decimal(np.array([r.under_american]))[0],
                            float((~w).mean()), pu / (po + pu), "your book"))
    for r in ms.itertuples():
        if (r.team, int(r.threshold)) in EXCLUDE:
            continue
        j = tidx[r.team]
        w = draws[:, j] >= r.threshold
        instruments.append((f"{r.team} {r.threshold}+ pts", w,
                            american_to_decimal(np.array([r.american]))[0],
                            float(w.mean()), mkt_ge(r.team, r.threshold), "your book"))
    for r in cb.itertuples():
        j = tidx[r.team]
        w = cup[:, j].astype(bool)
        p_m = float(w.mean())
        if p_m < 0.002:
            continue
        instruments.append((f"{r.team} Stanley Cup", w,
                            american_to_decimal(np.array([r.american]))[0],
                            p_m, cup_devig[r.team], cup_book[r.team]))

    keep, meta = [], []
    for label, w, dec, p_m, p_k, venue in instruments:
        p_bet = (1 - MKT_W) * p_m + MKT_W * p_k
        if p_bet * dec - 1 > 0.005:
            keep.append((label, w, dec, p_m, p_bet, venue))
    n = len(keep)
    R = np.zeros((draws.shape[0], n))
    Rr = np.zeros_like(R)
    for i, (label, w, dec, p_m, p_bet, venue) in enumerate(keep):
        R[:, i] = np.where(w, dec * p_bet / p_m - 1.0, -1.0)   # shrunk-edge odds
        Rr[:, i] = np.where(w, dec - 1.0, -1.0)                # real odds

    def neg_elog_clean(f):
        return -np.mean(np.log(np.maximum(FLOOR + 1.0 + R @ f, FLOOR)))

    def grad(f):
        wlt = np.maximum(FLOOR + 1.0 + R @ f, FLOOR)
        return -(R / wlt[:, None]).mean(axis=0)

    cons = [{"type": "eq", "fun": lambda f: f.sum() - 1.0,
             "jac": lambda f: np.ones(n)}]
    res = minimize(neg_elog_clean, np.full(n, 1.0 / n), jac=grad, method="SLSQP",
                   bounds=[(0.0, CAP)] * n, constraints=cons,
                   options={"maxiter": 800, "ftol": 1e-13})
    stakes = res.x * BUDGET
    stakes[stakes < MIN_TICKET] = 0.0
    # second pass: re-optimize over the surviving lines only, so the dust from
    # dropped sub-$1 tickets is reallocated optimally rather than heuristically
    live = np.where(stakes > 0)[0]
    if len(live) and len(live) < n:
        Rl = R[:, live]

        def neg_l(fl):
            return -np.mean(np.log(np.maximum(FLOOR + 1.0 + Rl @ fl, FLOOR)))

        def grad_l(fl):
            wl = np.maximum(FLOOR + 1.0 + Rl @ fl, FLOOR)
            return -(Rl / wl[:, None]).mean(axis=0)

        res2 = minimize(neg_l, stakes[live] / stakes[live].sum(), jac=grad_l,
                        method="SLSQP", bounds=[(0.01, CAP)] * len(live),
                        constraints=[{"type": "eq", "fun": lambda f: f.sum() - 1.0,
                                      "jac": lambda f: np.ones(len(live))}],
                        options={"maxiter": 800, "ftol": 1e-13})
        stakes = np.zeros(n)
        stakes[live] = res2.x * BUDGET
    stakes = np.round(stakes * 2) / 2
    # settle the rounding residual in $0.50 steps onto the largest lines still
    # under the per-line cap (never breach CAP)
    cap_d = CAP * BUDGET
    guard = 0
    while stakes.sum() < BUDGET - 0.01 and guard < 1000:
        room = np.where(stakes < cap_d, stakes, -np.inf)
        i = int(np.argmax(np.where(room > 0, room, -np.inf)))
        stakes[i] += 0.5
        guard += 1
    while stakes.sum() > BUDGET + 0.01:
        i = int(np.argmax(stakes))
        stakes[i] -= 0.5

    rows = []
    for i in np.argsort(-stakes):
        if stakes[i] <= 0:
            continue
        label, w, dec, p_m, p_bet, venue = keep[i]
        rows.append({"line": label, "venue": venue, "stake": stakes[i],
                     "dec_odds": round(dec, 2), "p_model": round(p_m, 3),
                     "p_blend": round(p_bet, 3),
                     "ev_blend_per_$": round(p_bet * dec - 1, 3),
                     "win_pays": round(stakes[i] * dec, 2)})
    alloc = pd.DataFrame(rows)
    print(alloc.to_string(index=False))
    print(f"\ntotal staked: ${alloc.stake.sum():.2f} across {len(alloc)} tickets "
          f"({n} candidates)")
    f_final = stakes / BUDGET
    pnl = (Rr @ stakes)
    print(f"P&L under MODEL scenarios:  mean ${pnl.mean():+7.2f} | P5 ${np.percentile(pnl,5):+7.2f} | "
          f"P50 ${np.percentile(pnl,50):+7.2f} | P95 ${np.percentile(pnl,95):+7.2f} | "
          f"P(profit) {np.mean(pnl>0):.1%} | worst ${pnl.min():+.2f}")
    ev_blend = sum(f_final[i] * BUDGET * (keep[i][4] * keep[i][2] - 1) for i in range(n))
    print(f"expected P&L under BLENDED probabilities: ${ev_blend:+.2f}")
    alloc.to_csv(OUT / "final_100_allocation.csv", index=False)
    print("wrote output/final_100_allocation.csv")


if __name__ == "__main__":
    main()
