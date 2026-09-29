"""NeurHL 1.1 C2b live: tonight's projection of the 2026-27 final standings
(PLAN_NeurHL_1_1 A2; run nightly only if C2b was adopted).

Inputs:
  results   neurhl/output/live/results_2027.csv (completed regular-season games)
  games     the frozen neurhl/output/neurhl_1_0/games_2027.csv: every game's
            stacked probability, outcome4 and rate sensitivity, unchanged
  params    sigma0 and sw from neurhl/configs/live_standings_1_1.json (the
            backtest's selected "update" variant, published as exploratory:
            PLAN_NeurHL_1_1 A9)

Team strength is updated from the games played (sim/live_standings_1_1.posterior)
and the remaining games are simulated with drift (project). Playoff odds use the
NHL format (top three per division plus two wild cards per conference), ranked
by points, then regulation wins... simplified here to points with random
tiebreaks (documented, not a claim about tiebreakers).

Writes neurhl/output/live/standings_1_1_2027.csv.
"""
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sim.unified_2027 as U  # noqa: E402
from sim.live_standings_1_1 import posterior, project  # noqa: E402

LIVE = ROOT / "output" / "live"
UNI = ROOT / "output" / "neurhl_1_0"
PARAMS = ROOT / "configs" / "live_standings_1_1.json"
SIMS, SEED = 20000, 711


def main():
    prm = json.loads(PARAMS.read_text())
    sigma0, sw = prm["sigma0"], prm["sw"]
    g = pd.read_csv(UNI / "games_2027.csv")
    res_p = LIVE / "results_2027.csv"
    res = pd.read_csv(res_p) if res_p.exists() else pd.DataFrame(columns=["game_id"])
    res = res[res.game_id.isin(g.game_id)].drop_duplicates("game_id")
    teams = sorted(set(g.home))
    ti = {t: i for i, t in enumerate(teams)}
    T = len(teams)
    hi, ai = g.home.map(ti).to_numpy(), g.away.map(ti).to_numpy()
    z = U.logit(g.p_home_win.to_numpy())
    k = g.rate_sensitivity.to_numpy()
    o4 = g[["p_home_reg", "p_away_reg", "p_home_ot", "p_away_ot"]].to_numpy()
    wk = ((pd.to_datetime(g.date) - pd.to_datetime(g.date).min()).dt.days // 7).to_numpy()
    played = g.game_id.isin(res.game_id).to_numpy()
    r = res.set_index("game_id").loc[g.game_id[played]]
    hw = (r.home_g > r.away_g).to_numpy()
    extra = r.last_period.isin(["OT", "SO"]).to_numpy()
    pts = np.zeros(T)
    np.add.at(pts, hi[played], np.where(hw, 2, np.where(extra, 1, 0)))
    np.add.at(pts, ai[played], np.where(~hw, 2, np.where(extra, 1, 0)))
    m, C = posterior(z[played], k[played], hi[played], ai[played], hw.astype(float), T, sigma0)
    now = int(wk[played].max()) + 1 if played.any() else 0
    rem = ~played
    sim = project(z[rem], k[rem], hi[rem], ai[rem], o4[rem], wk[rem], now, pts, m, C, sw, SIMS, SEED)
    div, conf = U.divisions()
    key = sim + np.random.default_rng(SEED + 7).random(sim.shape)
    po = np.zeros(T)
    for cf in ("E", "W"):
        ids = np.array([ti[t] for t in teams if conf[t] == cf])
        for s_i in range(SIMS):
            ks = key[s_i]
            q = []
            rest = []
            for d_ in sorted({div[t] for t in teams if conf[t] == cf}):
                di = np.array([ti[t] for t in teams if div[t] == d_])
                o = di[np.argsort(-ks[di])]
                q.extend(o[:3])
                rest.extend(o[3:])
            rest = np.array(rest)
            q.extend(rest[np.argsort(-ks[rest])][:2])
            po[q] += 1
    gp = np.zeros(T)
    np.add.at(gp, hi[played], 1)
    np.add.at(gp, ai[played], 1)
    out = pd.DataFrame({"team": teams, "gp": gp, "points_now": pts, "points": sim.mean(0),
                        "points_p10": np.percentile(sim, 10, 0), "points_p50": np.percentile(sim, 50, 0),
                        "points_p90": np.percentile(sim, 90, 0), "playoff_pct": 100 * po / SIMS,
                        "strength_shift": m, "strength_sd": np.sqrt(np.diag(C))})
    out = out.sort_values("points", ascending=False)
    out.insert(0, "as_of", dt.date.today().isoformat())
    out.insert(1, "status", "exploratory")
    LIVE.mkdir(parents=True, exist_ok=True)
    if not played.any():
        print("[live_standings] no completed games yet; nothing written")
        return 0
    out.to_csv(LIVE / "standings_1_1_2027.csv", index=False, float_format="%.3f")
    hist = LIVE / "standings_1_1"                       # one dated copy per night, for season-end scoring
    hist.mkdir(parents=True, exist_ok=True)
    q = pd.DataFrame(np.percentile(sim, np.arange(1, 100), axis=0).T, columns=[f"q{i:02d}" for i in range(1, 100)])
    q.insert(0, "team", teams)                            # full distribution, for season-end CRPS
    out.merge(q, on="team").to_csv(hist / f"{dt.date.today().isoformat()}.csv", index=False, float_format="%.3f")
    print(out.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
