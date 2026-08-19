"""Independent review test battery (2026-08-16 model review).

Read-only: verifies data integrity, engine correctness, and output consistency.
Does NOT retune anything; spent validation windows are only re-checked against
already-published numbers (pure reproduction, no decisions).

Run: uv run --no-project --with numpy --with "pandas<3" --with openpyxl python review_tests.py
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ / "src"))

import engine as E  # noqa: E402

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    status = "PASS" if cond else "FAIL"
    if cond:
        PASS += 1
    else:
        FAIL += 1
    print(f"[{status}] {name}" + (f"  -- {detail}" if detail else ""))


print("=" * 78)
print("SECTION 1: data integrity")
print("=" * 78)
g = pd.read_csv(PROJ / "data/processed/games.csv", keep_default_na=False, parse_dates=["date"])
ts = pd.read_csv(PROJ / "data/processed/team_seasons.csv")

check("games row count matches README claim (27,166)", len(g) == 27166, f"{len(g)}")
check("season span 2006-2026", (g.season_end.min(), g.season_end.max()) == (2006, 2026))
check("no zero-margin games", (g.home_g != g.away_g).all())
check("no duplicate games", g.duplicated(["date", "home", "away"]).sum() == 0)

# team-season structural checks (mirrors build_dataset.validate)
ok_n = True
for season, grp in ts.groupby("season_end"):
    exp_n = 30 if season <= 2017 else (31 if season <= 2021 else 32)
    ok_n &= len(grp) == exp_n
check("teams per season 30/31/32 by era", ok_n)
r = g[g.game_type == "R"]
pts_ok = True
for season, grp in r.groupby("season_end"):
    lp = ts[ts.season_end == season].pts.sum()
    pts_ok &= lp == 2 * len(grp) + (grp.went_ot | grp.went_so).sum()
check("league points identity per season (2/game + OT games)", pts_ok)
check("xG coverage complete 2008+", ts[(ts.season_end >= 2008)].xg_pct_all.notna().all())

# player panels
sk = pd.read_csv(PROJ / "data/processed/panel_skaters.csv")
go = pd.read_csv(PROJ / "data/processed/panel_goalies.csv")
check("skater panel unique (player, season)", sk.duplicated(["playerId", "season_end"]).sum() == 0)
check("goalie panel unique (player, season)", go.duplicated(["playerId", "season_end"]).sum() == 0)
check("panels span 2009-2026", (sk.season_end.min(), sk.season_end.max()) == (2009, 2026))

print()
print("=" * 78)
print("SECTION 2: engine unit tests")
print("=" * 78)

# --- 2.1 synthetic 84-game schedule structure, multiple seeds
def schedule_structure_ok(seed):
    rng = np.random.default_rng(seed)
    sched = E.synthetic_schedule_84(E.DIVISIONS_CURRENT, rng)
    div_of = {t: dv for dv, ts_ in E.DIVISIONS_CURRENT.items() for t in ts_}
    conf_of = {t: c for c, dvs in E.CONFS.items() for dv in dvs
               for t in E.DIVISIONS_CURRENT[dv]}
    cnt = {}
    for h, a in sched.itertuples(index=False):
        key = tuple(sorted([h, a]))
        cnt[key] = cnt.get(key, 0) + 1
    for (t1, t2), n in cnt.items():
        same_div = div_of[t1] == div_of[t2]
        same_conf = conf_of[t1] == conf_of[t2]
        want = 4 if same_div else (3 if same_conf else 2)
        if n != want:
            return False, f"{t1}-{t2}: {n} games, want {want}"
    home = sched.home.value_counts()
    away = sched.away.value_counts()
    if not ((home == 42).all() and (away == 42).all()):
        return False, "home/away != 42"
    return True, ""

ok_all = True
for seed in (0, 1, 7, 42, 999):
    ok, msg = schedule_structure_ok(seed)
    ok_all &= ok
check("synthetic schedule: 4/3/2 matrix + 42H/42A over 5 seeds", ok_all)

# --- 2.2 series win prob DP vs brute force (full 7-game sequences, winner = first to 4)
def brute_series(p_games):
    win = 0.0
    for outcome in itertools.product([1, 0], repeat=7):
        pr = 1.0
        for gm, res in enumerate(outcome):
            pr *= p_games[gm] if res else 1 - p_games[gm]
        w = 0
        l = 0
        for res in outcome:
            w += res
            l += 1 - res
            if w == 4:
                win += pr
                break
            if l == 4:
                break
    return win

rng = np.random.default_rng(3)
ok_all = True
for _ in range(200):
    p = rng.uniform(0.2, 0.8, 7)
    if abs(E._series_win_prob(p) - brute_series(p)) > 1e-12:
        ok_all = False
        break
check("best-of-7 DP == brute-force enumeration (200 random cases)", ok_all)
check("series prob: p=0.5 flat -> 0.5", abs(E._series_win_prob(np.full(7, 0.5)) - 0.5) < 1e-12)

# --- 2.3 IRLS logistic vs closed-form check on synthetic data
rng = np.random.default_rng(11)
x = rng.normal(0, 1, 60000)
true_b = (0.25, 0.9)
py = 1 / (1 + np.exp(-(true_b[0] + true_b[1] * x)))
y = (rng.random(60000) < py).astype(float)
b = E.fit_logistic(x[:, None], y)
check("IRLS logistic recovers known coefficients",
      abs(b[0] - true_b[0]) < 0.03 and abs(b[1] - true_b[1]) < 0.03,
      f"fit {b.round(3)} vs true {true_b}")

# --- 2.4 Elo zero-sum within season + carryover math
preds, end_r, pre_r = E.run_elo(g, K=6, H=35, phi_s=0.7)
r06 = end_r[2006]
check("Elo zero-sum: mean end-2006 rating == 1505",
      abs(np.mean(list(r06.values())) - 1505.0) < 1e-9,
      f"{np.mean(list(r06.values())):.6f}")
# carryover: pre_2007 = 1505 + 0.7*(end_2006-1505) for continuing teams
ok_all = all(abs(pre_r[2007][t] - (1505 + 0.7 * (r06[t] - 1505))) < 1e-9 for t in r06)
check("season carryover formula applied exactly", ok_all)
# expansion entries
check("VGK enters 2018 at 1470", abs(pre_r[2018].get("VGK", end_r[2018]["VGK"] if "VGK" in end_r[2018] else 0) - 1470) < 200,
      "(entry rating checked below in preds)")
vgk_first = preds[(preds.home == "VGK") | (preds.away == "VGK")].iloc[0]
vgk_r = vgk_first.rh if vgk_first.home == "VGK" else vgk_first.ra
check("VGK first-game pregame rating == 1470", abs(vgk_r - 1470) < 1e-9, f"{vgk_r}")
sea_first = preds[(preds.home == "SEA") | (preds.away == "SEA")].iloc[0]
sea_r = sea_first.rh if sea_first.home == "SEA" else sea_first.ra
check("SEA first-game pregame rating == 1470", abs(sea_r - 1470) < 1e-9, f"{sea_r}")

# --- 2.5 the pre-v4 float32 key formulation drops ROW (documents why B1 was fixed;
#         the fixed engine path is asserted in section 7.1)
pts32 = np.float32(100.0)
rw32 = np.float32(30.0)
key_a = pts32 * 1e8 + rw32 * 1e4 + np.float32(45.0)   # ROW 45
key_b = pts32 * 1e8 + rw32 * 1e4 + np.float32(44.0)   # ROW 44
check("old float32 key formulation loses ROW (the bug v4's B1 fixed)",
      key_a == key_b,
      f"key(ROW=45)-key(ROW=44) = {float(key_a - key_b):.1f} (float32 ULP at 1e10 ~ 1024)")

# --- 2.6 analytic xpts == simulated mean (sigma=0)
om = E.fit_outcome(preds, list(range(2006, 2027)))
rngs = np.random.default_rng(5)
sched = E.synthetic_schedule_84(E.DIVISIONS_CURRENT, rngs)
ratings = {t: 1505.0 + rngs.normal(0, 50) for dv in E.DIVISIONS_CURRENT.values() for t in dv}
xp = E.analytic_xpts(ratings, sched, om)
sim = E.simulate_season(ratings, 0.0, sched, om, E.DIVISIONS_CURRENT, 20000, np.random.default_rng(6), playoffs=False)
sim_mean = pd.Series(sim["pts"].mean(0), index=sim["teams"])
gap = (xp - sim_mean).abs().max()
check("analytic xpts == simulated mean (20k sims, sigma=0), max gap < 0.7 pts",
      gap < 0.7, f"max abs gap {gap:.3f}")

# --- 2.7 game model calibration at d=0 vs empirical HFA
p_ot0, p_reg0, p_otw0 = E.game_probs(np.array([0.0]), om)
sub = preds[(preds.game_type == "R")]
emp_reg = sub[~(sub.went_ot | sub.went_so)].home_win.mean()
check("P(home reg win | even teams) within 2pts of league home reg win rate",
      abs(p_reg0[0] - emp_reg) < 0.02, f"model {p_reg0[0]:.3f} vs empirical {emp_reg:.3f}")

print()
print("=" * 78)
print("SECTION 3: frozen-parameter reproduction (NO retuning)")
print("=" * 78)
params = json.loads((PROJ / "output/params.json").read_text())
check("params.json frozen flag", params.get("frozen") is True, str(params))

import backtest as B  # noqa: E402  (imports run E.load again)
bt_pub = pd.read_csv(PROJ / "output/backtest_seasons.csv")
preds_f, end_rf, _ = E.run_elo(g, K=params["K"], H=params["H"], phi_s=params["phi_s"])
rows = []
for h, phi in ((1, params["phi1"]), (2, params["phi2"])):
    for T in range(2018, 2027):
        _, xp, _, _ = B.project_season(end_rf, preds_f, T, h, params["w"], phi)
        rows.append({"horizon": h, "season": T, "mae_model": B.norm_mae(xp, T)})
rep = pd.DataFrame(rows)
merged = rep.merge(bt_pub[["horizon", "season", "mae_model"]], on=["horizon", "season"],
                   suffixes=("_rep", "_pub"))
max_diff = (merged.mae_model_rep - merged.mae_model_pub).abs().max()
check("validation MAEs reproduce exactly from frozen params",
      max_diff < 1e-6, f"max |diff| = {max_diff:.2e} across {len(merged)} season-horizons")
h1 = merged[merged.horizon == 1]
h2 = merged[merged.horizon == 2]
print(f"    h1 mean MAE {h1.mae_model_rep.mean():.2f} (published claim 10.54); "
      f"h2 {h2.mae_model_rep.mean():.2f} (claim 11.57)")

print()
print("=" * 78)
print("SECTION 4: production outputs consistency")
print("=" * 78)
v3 = pd.read_csv(PROJ / "output/projections_2026_27_v3.csv")
check("v3 2026-27: 32 teams", len(v3) == 32)
check("v3 Playoff% sums to 16", abs(v3["Playoff%"].sum() - 16) < 0.05, f"{v3['Playoff%'].sum():.3f}")
check("v3 Division% sums to 4", abs(v3["Division%"].sum() - 4) < 0.03, f"{v3['Division%'].sum():.3f}")
check("v3 Conference% sums to 2", abs(v3["Conference%"].sum() - 2) < 0.03, f"{v3['Conference%'].sum():.3f}")
check("v3 Cup% sums to 1", abs(v3["Cup%"].sum() - 1) < 0.02, f"{v3['Cup%'].sum():.3f}")
mean_pts = (v3.xPts * 1).mean()
check("v3 league mean pts in 84-game range [92, 97]", 92 <= mean_pts <= 97, f"{mean_pts:.2f}")
check("v3 percentiles monotone P5<=P50<=P95",
      ((v3.P5 <= v3.P50) & (v3.P50 <= v3.P95)).all())
sd_ok = v3.SD.between(9, 16).all()
check("v3 team pts SD plausible (9-16 for 84g)", sd_ok, f"range {v3.SD.min():.1f}-{v3.SD.max():.1f}")

ov = pd.read_csv(PROJ / "output/overlay_team_deltas.csv", index_col=0)
check("overlay dElo zero-sum (within stored rounding)", abs(ov.dElo.sum()) < 0.05,
      f"sum {ov.dElo.sum():.3f} Elo")
check("overlay dElo capped at +/-25", ov.dElo.abs().max() <= 25.0 + 1e-9, f"max {ov.dElo.abs().max():.1f}")

pr = pd.read_csv(PROJ / "output/v3_prior_ratings.csv", index_col=0)["rating"]
check("v3 prior ratings: 32 teams centered near 1505",
      len(pr) == 32 and abs(pr.mean() - 1505) < 1.0, f"mean {pr.mean():.2f}")

# real 2026-27 schedule + b2b adjustments
sch = pd.read_csv(PROJ / "data/raw/nhl_schedule_20262027.csv", parse_dates=["date"])
counts = pd.concat([sch.home, sch.away]).value_counts()
check("real 2026-27 schedule: every team 84 games", (counts == 84).all(),
      f"n games {len(sch)} (expect 1344)")
home_counts = sch.home.value_counts()
check("real schedule 42 home each", (home_counts == 42).all())
sys.path.insert(0, str(PROJ / "src"))
from report3 import real_schedule_b2b  # noqa: E402
s_b2b = real_schedule_b2b()
n_b2b_games = (s_b2b.d_adj != 0).sum()
frac = n_b2b_games / len(s_b2b)
# team-game b2b rate ~12-14% -> net-asymmetric game rate ~2p(1-p) ~= 20-24%
check("b2b games present at plausible rate (14-26% of games have net adj)",
      0.14 <= frac <= 0.26, f"{n_b2b_games} games ({frac:.1%}) carry a net b2b adjustment")
check("b2b adjustment magnitude is +/-38 Elo",
      set(np.unique(np.abs(s_b2b.d_adj[s_b2b.d_adj != 0]))) == {38.0})

# v1 vs v2 vs v3 2026-27 spread sanity
v2 = pd.read_csv(PROJ / "output/projections_2026_27_v2.csv")
j = v3.merge(v2, on="Abbr", suffixes=("_v3", "_v2"))
corr = j.xPts_v3.corr(j.xPts_v2)
check("v2/v3 2026-27 xPts strongly correlated (>0.9)", corr > 0.9, f"r={corr:.3f}")
top_v3 = v3.iloc[0]
print(f"    v3 top team: {top_v3.Team} {top_v3.xPts:.1f} pts, Cup {top_v3['Cup%']:.1%}; "
      f"NOTES claims CAR 107.6 on top after v3.1")

print()
print("=" * 78)
print("SECTION 5: availability noise & Elo value chain")
print("=" * 78)
import availability as A  # noqa: E402
import players as P  # noqa: E402
skp, gop, skt, got, bios = P.load_panels()
par = A.fit_availability(skp, bios, 2026)
for (lo, hi), p_ in zip(A.BUCKETS, par):
    print(f"    bucket {lo}-{hi}: E[GP share] {p_['mean']:.3f}, a={p_['a']:.2f}, "
          f"b={p_['b']:.2f}, n={p_['n']}")
check("availability: E[GP share] decreasing with age",
      par[0]["mean"] > par[-1]["mean"],
      f"{par[0]['mean']:.3f} -> {par[-1]['mean']:.3f}")
rosters = {"AAA": [(25.0, 20.0), (31.0, 15.0), (37.0, 10.0)]}
fn = A.make_extra_noise(["AAA"], rosters, par, k_elo=6.0)
draws = fn(200000, np.random.default_rng(1))
check("availability noise is zero-mean (|mean| < 0.05 Elo at 200k draws)",
      abs(draws.mean()) < 0.05, f"mean {draws.mean():.4f}, sd {draws.std():.2f} Elo")

p2 = json.loads((PROJ / "output/params_v2.json").read_text())
check("value chain k*c/82 in sane band (documented ~0.24)",
      0.20 <= p2["k"] * p2["c"] / 82 <= 0.50, f"{p2['k']*p2['c']/82:.3f}")
check("v2 confirm/final single-use blocks present",
      "confirm_result" in p2 and "final_result" in p2 and "prereg" in p2)

print()
print("=" * 78)
print("SECTION 6: deeper diagnostics")
print("=" * 78)

# --- 6.1 materiality of the float32 ROW-tiebreak bug: how often does it bind?
rng6 = np.random.default_rng(1234)
sched6 = E.synthetic_schedule_84(E.DIVISIONS_CURRENT, rng6)
ratings6 = {t: 1505.0 for dv in E.DIVISIONS_CURRENT.values() for t in dv}
sim6 = E.simulate_season(ratings6, 30.0, sched6, om, E.DIVISIONS_CURRENT, 3000,
                         np.random.default_rng(77), playoffs=False)
pts6, rw6, row6 = sim6["pts"], sim6["rw"], sim6["row"]
n_binding = 0
for kk in range(3000):
    seen = {}
    for i in range(32):
        seen.setdefault((pts6[kk, i], rw6[kk, i]), []).append(row6[kk, i])
    if any(len(set(v)) > 1 for v in seen.values() if len(v) > 1):
        n_binding += 1
print(f"    sims containing >=1 (pts,RW)-tied pair with differing ROW: "
      f"{n_binding/3000:.1%} of seasons (bug replaces ROW order with coin flip there)")

# --- 6.2 availability survivorship: regulars with NO next-season NHL row are excluded
cur = skp[skp.toi_min >= 1200][["playerId", "season_end"]]
cur = cur[cur.season_end.between(2010, 2025) & ~cur.season_end.isin([2012, 2019, 2020])]
nxt_ids = set(zip(skp.playerId, skp.season_end))
gone = [(pid, s) for pid, s in cur.itertuples(index=False) if (pid, s + 1) not in nxt_ids]
print(f"    regulars (>=1200 min) with ZERO games next season (excluded from availability "
      f"fit): {len(gone)}/{len(cur)} = {len(gone)/len(cur):.1%}")

# --- 6.3 age curve peak for forwards
import players as P2
fe_f = P2.age_curve_fe(skp, bios, 2026, "F")
ages = np.arange(18, 40, 0.25)
vals = P2.curve_value(fe_f, ages)
peak_age = float(ages[np.argmax(vals)])
check("forward age-curve peak in [24, 28] (NOTES claims ~27)", 24 <= peak_age <= 28,
      f"peak at {peak_age:.1f}")

# --- 6.4 empirical b2b effect on historical games (sanity for B2B_ELO=38)
gg = g[(g.game_type == "R") & (g.season_end >= 2010)].sort_values("date").copy()
last_date = {}
hb, ab = [], []
for gm in gg.itertuples():
    hb.append(1 if (gm.home in last_date and (gm.date - last_date[gm.home]).days <= 1) else 0)
    ab.append(1 if (gm.away in last_date and (gm.date - last_date[gm.away]).days <= 1) else 0)
    last_date[gm.home] = gm.date
    last_date[gm.away] = gm.date
gg["hb"], gg["ab"] = hb, ab
hw = gg.home_g > gg.away_g
base = hw[(gg.hb == 0) & (gg.ab == 0)].mean()
h_pen = hw[(gg.hb == 1) & (gg.ab == 0)].mean() - base
a_pen = hw[(gg.hb == 0) & (gg.ab == 1)].mean() - base
print(f"    raw home win%: rested-vs-rested {base:.3f}; home-on-b2b delta {h_pen:+.3f}; "
      f"away-on-b2b delta {a_pen:+.3f} (claim ~ -/+6%)")
check("b2b effect sign & rough size (both sides, 3-9%)",
      -0.09 <= h_pen <= -0.02 and 0.02 <= a_pen <= 0.09)

# --- 6.5 v3 xlsx sheet agrees with CSV
x3 = pd.read_excel(PROJ / "output/nhl_2026_27_projections_v3.xlsx",
                   sheet_name="Projections_2026_27")
jj = x3.merge(v3, on="Abbr", suffixes=("_x", "_c"))
check("v3 xlsx == v3 csv (xPts, Cup%)",
      (jj.xPts_x - jj.xPts_c).abs().max() < 1e-6 and
      (jj["Cup%_x"] - jj["Cup%_c"]).abs().max() < 1e-9)

# --- 6.6 reproduce published v2 final-window table (frozen hyperparams, walk-forward)
try:
    import backtest2 as B2
    from features import FeatureBuilder as FB2
    fb2 = FB2(end_rf, ts, goalie_hp=p2["goalie_hp"], skater_delta=p2["skater_delta"])
    pub = pd.read_csv(PROJ / "output/backtest_v2.csv")
    lam1 = p2["lams"]["h1"]["full"]
    predT = B2.expanding_predictions(fb2, 1, [2022, 2023, 2024, 2025, 2026], list(B2.FEATURES), lam1)
    rows6 = []
    ACT2 = ts.set_index(["season_end", "team"])
    for T in [2022, 2023, 2024, 2025, 2026]:
        act = ACT2.loc[T]
        ydev = ((act.pts_pct - act.pts_pct.mean()) * 164).reindex(predT[T].index)
        rows6.append(float((predT[T] - ydev).abs().mean()))
    rep_v2 = np.array(rows6)
    pub_v2 = pub[(pub.horizon == 1)].mae_v2.to_numpy()
    check("v2 final-window h1 MAEs reproduce from frozen pipeline",
          np.abs(rep_v2 - pub_v2).max() < 1e-6,
          f"max diff {np.abs(rep_v2 - pub_v2).max():.2e}")
except Exception as ex:  # pragma: no cover
    check("v2 final-window h1 MAEs reproduce from frozen pipeline", False, f"ERROR {ex!r}")

print()
print("=" * 78)
print("SECTION 7: v4 components (added 2026-08-17)")
print("=" * 78)
import json as _json

import availability2 as A2v4
import goalie_game as GGv4
import prospects as PRv4
import scoring as SCv4

# 7.1 float64 standings key preserves every level (the bug this battery caught)
kA = E.standings_key(100, 30, 45, 50, 0.0)
kB = E.standings_key(100, 30, 44, 50, 0.999)
kC = E.standings_key(100, 30, 44, 51, 0.0)
check("v4 standings key: ROW and W levels preserved, ordered", kA > kC > kB)

# 7.2 CRPS estimator vs closed-form Gaussian
rng7 = np.random.default_rng(2)
mu_, sd_, y_ = 95.0, 13.0, 101.5
draws = rng7.normal(mu_, sd_, 400_000)
z = (y_ - mu_) / sd_
from math import erf, exp, pi, sqrt
Phi = 0.5 * (1 + erf(z / sqrt(2)))
phi = exp(-z * z / 2) / sqrt(2 * pi)
crps_closed = sd_ * (z * (2 * Phi - 1) + 2 * phi - 1 / sqrt(pi))
crps_emp = SCv4.crps_draws(draws, y_)
check("CRPS estimator matches closed-form Gaussian (<0.5%)",
      abs(crps_emp - crps_closed) / crps_closed < 0.005,
      f"emp {crps_emp:.4f} vs closed {crps_closed:.4f}")

# 7.3 availability2 mixture noise is zero-mean
skp2, gop2, skt2, got2, bios2 = P.load_panels()
par2 = A2v4.fit_availability2(skp2, bios2, 2017)
gpar2 = A2v4.fit_goalie_availability(got2, ts, 2017)
fn2 = A2v4.make_extra_noise2(["X"], {"X": [(1, 0.9, 20.0), (4, 0.8, 12.0)]}, par2,
                             {"X": (0.75, 10.0)}, gpar2, k_elo=60.0)
d7 = fn2(300_000, np.random.default_rng(3))
check("availability2 mixture noise zero-mean (|mean| < 3*SE)",
      abs(d7.mean()) < 3 * d7.std() / np.sqrt(300_000),
      f"mean {d7.mean():+.4f}, sd {d7.std():.2f}")

# 7.4 goalie game noise: zero-mean per game incl b2b nights; b2b raises backup usage
p2j = _json.loads((PROJ / "output/params_v2.json").read_text())
sched7 = pd.DataFrame({"home": ["AAA", "AAA"], "away": ["BBB", "BBB"],
                       "hb2b": [0.0, 1.0], "ab2b": [0.0, 0.0]})
tand7 = pd.DataFrame({"team": ["AAA", "BBB"], "g1": [1, 2], "g2": [3, 4],
                      "theta1": [0.008, 0.002], "theta2": [-0.004, -0.002],
                      "s1": [0.65, 0.6], "gap_gpg": [0.36, 0.12]}).set_index("team")
gn7 = GGv4.make_game_noise(sched7, tand7, k_elo=p2j["k"])
d8 = gn7(400_000, np.random.default_rng(4))
check("goalie game noise zero-mean per game (rested and b2b)",
      abs(d8.mean(0)).max() < 0.05, f"per-game |mean| max {abs(d8.mean(0)).max():.4f}")
# start-mix variance matches the analytic Bernoulli-mix prediction (rested game):
# var = sum over both teams of p1(1-p1) * (a1-a2)^2
var_pred = 0.0
for tm in ("AAA", "BBB"):
    r_ = tand7.loc[tm]
    a1 = p2j["k"] * GGv4.SHOTS_PG * (r_.theta1 - (r_.s1 * r_.theta1 + (1 - r_.s1) * r_.theta2))
    a2 = p2j["k"] * GGv4.SHOTS_PG * (r_.theta2 - (r_.s1 * r_.theta1 + (1 - r_.s1) * r_.theta2))
    var_pred += r_.s1 * (1 - r_.s1) * (a1 - a2) ** 2
check("goalie noise sd matches analytic Bernoulli-mix (<3%)",
      abs(d8[:, 0].std() - np.sqrt(var_pred)) / np.sqrt(var_pred) < 0.03,
      f"sim {d8[:, 0].std():.2f} vs analytic {np.sqrt(var_pred):.2f} "
      f"(b2b night sd {d8[:, 1].std():.2f}: near-equal by design — p(1-p) is flat "
      f"around 0.5, and per-game centering keeps the league-mean b2b penalty in the "
      f"flat 38 Elo constant)")

# 7.5 prospect join + ramp
join7 = PRv4.build_join(skp2)
q7 = PRv4.join_quality(join7)
check("draft join quality bar (>=85% picks 1-15, spot checks)", q7["pass"], str(q7))
ramp7 = PRv4.fit_ramp(join7, skp2)
top_y4 = ramp7[(ramp7.bucket == 0) & (ramp7.y == 4)].mean_pts.iloc[0]
late_y4 = ramp7[(ramp7.bucket == 3) & (ramp7.y == 4)].mean_pts.iloc[0]
check("ramp ordering: top-10 picks >> picks 61+ at year 4", top_y4 > 5 * late_y4,
      f"{top_y4:.1f} vs {late_y4:.1f}")

# 7.6 v4/HOWE output consistency
for tag in ("v4", "howe"):
    dfp = pd.read_csv(PROJ / f"output/projections_2026_27_{tag}.csv")
    check(f"2026-27 {tag}: prob sums (PO 16 / Div 4 / Conf 2 / Cup 1)",
          abs(dfp["Playoff%"].sum() - 16) < 0.05 and abs(dfp["Division%"].sum() - 4) < 0.03
          and abs(dfp["Conference%"].sum() - 2) < 0.03 and abs(dfp["Cup%"].sum() - 1) < 0.02)
howe7 = pd.read_csv(PROJ / "output/projections_2026_27_howe.csv").set_index("Abbr")
v47 = pd.read_csv(PROJ / "output/projections_2026_27_v4.csv").set_index("Abbr")
v17 = pd.read_csv(PROJ / "output/v4_prior_ratings.csv", index_col=0)
mid = 0.5 * (v17.rating_v1 + v17.rating_v4)
check("HOWE prior ratings are the exact 50/50 blend",
      (v17.rating_howe - mid).abs().max() < 0.02,
      f"max dev {(v17.rating_howe - mid).abs().max():.3f}")

# 7.7 market sheet math
mk7 = pd.read_csv(PROJ / "output/market_vs_model_2026_27.csv")
check("market: stakes only where both devig edges and EV positive",
      bool(((mk7.stake > 0) <= ((mk7.edge_prop > 0) & (mk7.edge_pow > 0)
                                & (mk7.ev_per_dollar > 0))).all()))
check("market: overround recorded ~1.255", abs(mk7.p_imp.sum() - 1.255) < 0.01,
      f"{mk7.p_imp.sum():.3f}")
check("market: total quarter-Kelly stake is small (< $6 of $100)",
      mk7.stake.sum() < 6.0, f"${mk7.stake.sum():.2f}")

# 7.8 committed reproductions + overlay v4 invariants
import repro_b2b as RB
res_b2b = RB.main()
check("B2B measurement REPRODUCED from committed code", res_b2b["reproduced"],
      str({kk: res_b2b[kk] for kk in ("dwin_hb2b", "dwin_ab2b", "elo_equiv_mean")}))
ov7 = _json.loads((PROJ / "output/overlay_v4.json").read_text())
check("overlay v3.1 estimate reproduced within band", ov7["reproduced_within_band"],
      f"repro {ov7['repro_toi_share']['rho_partial']} vs 1.011")
d_ov = pd.read_csv(PROJ / "output/overlay_team_deltas_v4.csv", index_col=0)
check("overlay v4 deltas zero-sum and capped",
      abs(d_ov.dElo.sum()) < 1e-6 and d_ov.dElo.abs().max() <= 25 + 1e-9)

# 7.9 live path parity
import live as LV
check("live estimator parity vs replay engine (<0.5 Elo)", LV.selftest(2024, 20))

# 7.10 gates recorded and internally consistent
p4j = _json.loads((PROJ / "output/params_v4.json").read_text())
check("gate G passed with true mean shift < 0.4",
      p4j["gate_G"]["pass"] and p4j["gate_G"]["max_mean_shift"] < 0.4)
check("persistence dropped (EB lost to nested baseline)",
      not p4j["screens"]["persistence_kept"])
check("gate F2 h2 gain recorded (dMAE <= -0.10)",
      p4j["gate_F2"]["h2"]["pass"] and p4j["gate_F2"]["h2"]["dmae"] <= -0.10,
      f"{p4j['gate_F2']['h2']['dmae']}")

print()
print("=" * 78)
print(f"RESULT: {PASS} passed, {FAIL} failed")
print("=" * 78)
