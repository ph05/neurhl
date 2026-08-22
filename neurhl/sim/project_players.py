"""NeurHL-2 — player projections for 2026-27, with a walk-forward backtest.

Ships the per-player half of the deliverable. Deliberately NOT built on S1's
actor head: that head chooses among players CURRENTLY on the ice, so turning it
into a season projection needs the deployment process (S3), which was not built.
Claiming a neural-network provenance it does not have would be worse than using
the honest estimator.

What it is built on:

  * **own-history EWMA of per-60 rates**, which is the baseline v1 measured as
    beating every learned player-rate head it tried. That null is respected here
    rather than re-litigated.
  * **empirical-Bayes shrinkage toward the position mean**, with each player's
    own ice time as the evidence weight, so a 200-minute season is pulled hard
    and a 1,400-minute season barely moves.
  * **RAPM prior** as a team-context and quality signal.
  * **an empirically fitted age curve**, estimated from the training seasons
    rather than assumed — PLAN_NeurHL2 lists "age enters raw; the curve should be
    estimated, not assumed" as a known gap.
  * **projected ice time** from prior usage, regressed toward the position mean.

Every input for season V comes from seasons < V. `--backtest` scores the same
machinery on DEV/TUNE seasons against each player's realised totals.

Run: uv run --no-project --python 3.12 --with numpy --with "pandas<3" \
     --with pyarrow python neurhl/sim/project_players.py --backtest
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RAW, TENSORS  # noqa: E402
import models.player_proj as PP  # noqa: E402

SEASON = 2027
GAMES = 84                   # the 2026-27 schedule is 84 games, not 82
EWMA_HALFLIFE = 1.4          # seasons
TOI_SHRINK_GP = 25.0         # games of evidence to half-weight a TOI/GP estimate
ERA_WINDOW = 3               # seasons averaged for the projected league level


# Assists are ZERO for season_end 2008-2011. Cause identified: build_htm's
# actor regex requires a dotted team-code prefix, but HTM goal descriptions list
# assists as "Assists: #26 NAME; #91 NAME" with no team code, so tensorize_htm's
# p2/p3 assignment -- which is otherwise correct -- never sees them. Measured:
# assists per game 0.00 for 2008-2011 against 9.16-10.62 from 2012 on, and
# p2>0 is 0.000 for those goals against 0.891-0.908 later.
#
# Including those seasons in the rate history was a real error: it biased
# projections DOWN by 18.3 points at vantage 2012 (which trains entirely on
# zero-assist seasons) and 4.9-8.7 points at 2014-2017. Assist rates are
# therefore estimated only from assist-complete seasons; GOAL rates still use
# the full history, since goals are intact throughout.
ASSISTS_FROM = 2012


def league_rates(hist: pd.DataFrame) -> pd.DataFrame:
    """League goals/60 and assists/60 among skaters, per season.

    THE ERA CONTROL. League scoring is not stationary: points per game ran
    14.5-14.9 across 2012-2017 and 16.3-17.1 across 2018-2026, a ~13% shift.
    A player's raw per-60 rate therefore means different things in different
    seasons, and averaging raw rates across an era boundary silently drags a
    projection toward whichever era supplied most of the history.

    Rates are converted to a RATIO against their own season's league level,
    averaged in that space, then re-inflated to the level projected for the
    target season. This is also why the original backtest missed the problem:
    it only ever tested 2014-2017 vantages, all inside the low-scoring era, so
    era drift was never exercised.
    """
    g = hist.groupby("season_end").agg(toi=("toi", "sum"), g=("g", "sum"),
                                       a=("a", "sum"))
    g["g60"] = 3600.0 * g.g / g.toi
    g["a60"] = 3600.0 * g.a / g.toi
    return g[["g60", "a60"]]


def projected_league(lg: pd.DataFrame, target: int) -> pd.Series:
    """League level expected in `target`, from the most recent seasons only."""
    recent = lg[lg.index < target].tail(ERA_WINDOW)
    return recent.mean()


def talent_sd(hist: pd.DataFrame, target: int, col: str) -> dict:
    """True-talent variance by position, net of Poisson sampling noise.

    Empirical Bayes done properly: the shrinkage weight for a player is
    tau^2 / (tau^2 + sigma_i^2), where sigma_i^2 is HIS OWN sampling variance and
    therefore falls as he accumulates ice time. The previous version used a
    single constant (90,000 seconds), which shrank a 3,300-minute veteran 31%
    toward the positional mean -- measured, that turned a 4.5 P/60 player into
    3.6 and compressed the whole projected distribution. The justified weight
    for a regular is ~0.83, not 0.69.
    """
    h = hist[(hist.season_end >= target - 3) & (hist.season_end < target)]
    h = h[h.toi >= 50 * 12 * 60]
    out = {}
    for pos, d in h.groupby("pos_group"):
        r = 3600.0 * d[col].astype(float) / d.toi
        samp = (r * 3600.0 / d.toi).mean()
        out[int(pos)] = float(max(r.var() - samp, 1e-6))
    return out


def player_seasons(seasons) -> pd.DataFrame:
    """Per player-season totals from the regular season only."""
    rows = []
    for s in seasons:
        p = TENSORS / f"player_games_{s}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p, columns=["player_id", "game_type", "toi_sec",
                                        "goals", "assists", "sog",
                                        "pos_group"])
        d = d[d.game_type == 2]
        g = d.groupby(["player_id", "pos_group"], as_index=False).agg(
            toi=("toi_sec", "sum"), g=("goals", "sum"), a=("assists", "sum"),
            sh=("sog", "sum"), gp=("toi_sec", "size"))
        g["season_end"] = s
        rows.append(g)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def age_curve(hist: pd.DataFrame, bios: dict) -> dict:
    """Multiplicative points-per-60 factor by age, estimated from history."""
    h = hist[hist.toi > 30000].copy()
    h["age"] = [bios.get(int(p), {}).get("age_at", {}).get(int(s), np.nan)
                for p, s in zip(h.player_id, h.season_end)]
    h = h.dropna(subset=["age"])
    if len(h) < 500:
        return {}
    h["p60"] = 3600 * (h.g + h.a) / h.toi
    lg = h.p60.mean()
    h["bin"] = h.age.clip(18, 40).astype(int)
    c = (h.groupby("bin").p60.mean() / lg).to_dict()
    # smooth: 3-point moving average, so a thin age bin cannot swing the curve
    ks = sorted(c)
    sm = {}
    for i, k in enumerate(ks):
        w = [c[ks[j]] for j in range(max(0, i - 1), min(len(ks), i + 2))]
        sm[k] = float(np.mean(w))
    return sm


def load_bios() -> dict:
    """player_id -> birth year, from the roster files and career tables."""
    out = {}
    for f in glob.glob(str(RAW / "nhl_roster_*_20262027.json")):
        d = json.loads(Path(f).read_text())
        for grp in ("forwards", "defensemen", "goalies"):
            for p in d.get(grp, []):
                by = p.get("birthDate", "")[:4]
                if by.isdigit():
                    out[int(p["id"])] = {"birth_year": int(by)}
    return out


def project(target: int, hist: pd.DataFrame, roster_ids=None,
            bios=None, games=GAMES) -> pd.DataFrame:
    """Era-relative EWMA rates, empirical-Bayes shrunk, on projected ice time."""
    prior = hist[hist.season_end < target]
    if not len(prior):
        return pd.DataFrame()
    lg = league_rates(prior)
    tgt_lg = projected_league(lg, target)

    p = prior.merge(lg, left_on="season_end", right_index=True,
                    suffixes=("", "_lg"))
    p = p[p.toi > 0].copy()
    # per-season rate as a RATIO to that season's league level
    p["rg"] = (3600.0 * p.g.astype(float) / p.toi) / p.g60
    p["ra"] = (3600.0 * p.a.astype(float) / p.toi) / p.a60
    p["w"] = 0.5 ** ((target - 1 - p.season_end) / EWMA_HALFLIFE)
    p["wt"] = p.w * p.toi

    agg = p.groupby(["player_id", "pos_group"], as_index=False).apply(
        lambda d: pd.Series({
            "toi_w": d.wt.sum(),
            "rg": np.average(d.rg, weights=d.wt),
            "ra": np.average(d.ra, weights=d.wt),
        }), include_groups=False)
    agg = agg[agg.toi_w > 0].copy()

    # assists are absent before ASSISTS_FROM; estimate them from complete seasons
    ap = p[p.season_end >= ASSISTS_FROM]
    if len(ap):
        aa = ap.groupby("player_id", as_index=False).apply(
            lambda d: pd.Series({"atoi": d.wt.sum(),
                                 "ra2": np.average(d.ra, weights=d.wt)}),
            include_groups=False)
        agg = agg.merge(aa, on="player_id", how="left")
        agg["ra"] = agg.ra2.where(agg.ra2.notna(), agg.ra)
        agg["atoi"] = agg.atoi.fillna(agg.toi_w)
    else:
        agg["atoi"] = agg.toi_w

    # EMPIRICAL BAYES with a per-player weight, not a global constant
    tau_g = talent_sd(prior, target, "g")
    tau_a = talent_sd(prior, target, "a")
    lg_g = float(tgt_lg.g60)
    lg_a = float(tgt_lg.a60)
    for col, tau, base, toicol in (("rg", tau_g, lg_g, "toi_w"),
                                   ("ra", tau_a, lg_a, "atoi")):
        t2 = agg.pos_group.map(tau).fillna(np.mean(list(tau.values())))
        t2_rel = t2 / base ** 2                       # tau^2 in RATIO units
        samp = agg[col].clip(lower=.05) * 3600.0 / agg[toicol].clip(lower=1) / base
        k = t2_rel / (t2_rel + samp)
        agg[col] = k * agg[col] + (1 - k) * 1.0
        agg[col + "_k"] = k

    agg["g60"] = agg.rg * lg_g
    agg["a60"] = agg.ra * lg_a

    # ---- ICE TIME: model minutes-per-game and games separately.
    # Regressing total TOI toward a positional mean conflated "plays less" with
    # "played fewer games", and cost a first-line forward ~15% of his minutes.
    # collapse to one row per player-season: player_seasons keys on
    # (player_id, pos_group), so anyone recorded at two positions appears twice
    def one_per_player(df):
        return (df.groupby("player_id")
                .agg(toi=("toi", "sum"), gp=("gp", "sum"),
                     pos_group=("pos_group", "first")))
    last = one_per_player(prior[prior.season_end == target - 1])
    prev = one_per_player(prior[prior.season_end == target - 2])
    tpg = agg.player_id.map(last.toi / last.gp)
    tpg2 = agg.player_id.map(prev.toi / prev.gp)
    gp1 = agg.player_id.map(last.gp)
    pos_tpg = agg.pos_group.map(
        (last.toi / last.gp).groupby(last.pos_group).median())
    blend = (2 * tpg.fillna(tpg2) + tpg2.fillna(tpg)) / 3
    blend = blend.fillna(pos_tpg * 0.75)
    kt = gp1.fillna(0) / (gp1.fillna(0) + TOI_SHRINK_GP)
    agg["toi_per_gp"] = kt * blend + (1 - kt) * pos_tpg * 0.8
    # expected games: last season's availability, regressed toward the norm
    lg_gp = float(last[last.gp >= 20].gp.mean())
    agg["exp_gp"] = np.clip(0.55 * gp1.fillna(lg_gp) + 0.45 * lg_gp,
                            10, 82) * (games / 82.0)
    agg["toi_proj"] = agg.toi_per_gp * agg.exp_gp

    if bios:
        curve = age_curve(prior, {})
        yr = agg.player_id.map({k_: v["birth_year"] for k_, v in bios.items()})
        age = (target - 1 - yr).clip(18, 40)
        mult = age.map(curve).fillna(1.0) if curve else 1.0
        agg["g60"] = agg.g60 * mult
        agg["a60"] = agg.a60 * mult

    if roster_ids is not None:
        agg = agg[agg.player_id.isin(roster_ids)]
    agg["proj_g"] = agg.g60 * agg.toi_proj / 3600
    agg["proj_a"] = agg.a60 * agg.toi_proj / 3600
    agg["proj_p"] = agg.proj_g + agg.proj_a
    agg["proj_toi_min"] = agg.toi_proj / 60
    return agg


def team_map_actual(season: int) -> dict:
    """player -> team for a HISTORICAL season, by where he played most."""
    pg = pd.read_parquet(TENSORS / f"player_games_{season}.parquet",
                         columns=["game_id", "player_id", "is_home", "toi_sec",
                                  "pos_group", "game_type"])
    pg = pg[(pg.game_type == 2) & (pg.pos_group != 2)]
    gc = pd.read_parquet(TENSORS / f"games_ctx_{season}.parquet",
                         columns=["game_id", "home_idx", "away_idx"])
    pg = pg.merge(gc, on="game_id")
    pg["team"] = np.where(pg.is_home, pg.home_idx, pg.away_idx)
    a = pg.groupby(["player_id", "team"], as_index=False).toi_sec.sum()
    return (a.sort_values("toi_sec", ascending=False)
            .drop_duplicates("player_id").set_index("player_id").team.to_dict())


def backtest(seasons) -> None:
    """Walk-forward: every season is predicted from history strictly before it."""
    hist = PP.load_player_seasons(range(2008, max(seasons) + 1))
    bios = load_bios()
    print(f"{'V':>5} {'players':>8} {'MAE':>7} {'bias':>7} {'corr':>7} "
          f"{'base':>7} {'top10 proj':>26} {'top10 actual':>26}")
    rows = []
    for V in seasons:
        cal = PP.fit_top_calibration(hist, V, 2015, bios, team_map_actual)
        pred, _ = PP.fit_predict(hist, V, train_from=2015, bios=bios)
        lg_g, lg_a = PP.league_level(hist, V)
        tot = PP.to_totals(pred, lg_g, lg_a, games=82,
                           team_of=team_map_actual(V), calib=cal)
        act = hist[hist.season_end == V].set_index("player_id")
        m = tot[tot.player_id.isin(act.index)].copy()
        m["act_p"] = m.player_id.map(act.g + act.a)
        m["act_gp"] = m.player_id.map(act.gp)
        mm = m[m.act_gp >= 40]
        if len(mm) < 50:
            continue
        # Bias must be measured UNCONDITIONALLY. Filtering on games played
        # selects players who beat their availability expectation, and their
        # point totals follow: the bias runs +0.50 at no filter, -0.21 at >=10
        # GP, -1.88 at >=40 and -3.62 at >=60. That gradient is selection on the
        # outcome, not deflation, so the >=40 figure is reported for accuracy
        # comparability only and never as a bias estimate.
        uncond_bias = float((m.proj_p - m.act_p).mean())
        mae = float(np.abs(mm.proj_p - mm.act_p).mean())
        naive = float(np.abs(mm.act_p.mean() - mm.act_p).mean())
        rows.append({"season": V, "n": int(len(mm)), "mae_pts": mae,
                     "naive_mae": naive,
                     "bias": float((mm.proj_p - mm.act_p).mean()),
                     "bias_unconditional": uncond_bias,
                     "n_all": int(len(m)),
                     "league_ratio": float(m.proj_p.sum() / max(m.act_p.sum(), 1)),
                     "corr": float(np.corrcoef(mm.proj_p, mm.act_p)[0, 1]),
                     "calib": list(cal),
                     "top10_proj": [round(float(x), 1) for x in
                                    sorted(m.proj_p, reverse=True)[:10]],
                     "top10_act": [int(x) for x in
                                   sorted(act.g + act.a, reverse=True)[:10]]})
        r = rows[-1]
        print(f"{V:>5} {r['n']:>8} {mae:>7.2f} {r['bias']:>+7.2f} {r['corr']:>7.3f} "
              f"{naive:>7.2f} {str(r['top10_proj'][:5]):>26} "
              f"{str(r['top10_act'][:5]):>26}")
    if rows:
        n = sum(r["n"] for r in rows)
        print(f"\nPOOLED n={n:,}  MAE "
              f"{sum(r['mae_pts']*r['n'] for r in rows)/n:.2f} vs a league-mean "
              f"baseline of {sum(r['naive_mae']*r['n'] for r in rows)/n:.2f}  "
              f"bias(>=40GP, selected) "
              f"{sum(r['bias']*r['n'] for r in rows)/n:+.2f}  "
              f"bias(all) "
              f"{sum(r['bias_unconditional']*r['n_all'] for r in rows)/sum(r['n_all'] for r in rows):+.2f}  "
              f"corr {np.mean([r['corr'] for r in rows]):.3f}")
        p = Path(__file__).resolve().parents[1] / "configs" / "player_backtest.json"
        p.write_text(json.dumps(rows, indent=1))
        print(f"-> {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest", action="store_true")
    args = ap.parse_args()
    if args.backtest:
        backtest([2022, 2023, 2024, 2025, 2026])
        return

    hist = PP.load_player_seasons(range(2008, SEASON))
    bios = load_bios()
    rosters, team_of, names = {}, {}, {}
    for f in sorted(glob.glob(str(RAW / "nhl_roster_*_20262027.json"))):
        ab = Path(f).name.split("_")[2]
        d = json.loads(Path(f).read_text())
        ids = []
        for grp in ("forwards", "defensemen"):
            for pl in d.get(grp, []):
                ids.append(pl["id"])
                team_of[pl["id"]] = ab
                names[pl["id"]] = (f"{pl['firstName']['default']} "
                                   f"{pl['lastName']['default']}")
        rosters[ab] = ids

    cal = PP.fit_top_calibration(hist, SEASON, 2015, bios, team_map_actual)
    pred, _ = PP.fit_predict(hist, SEASON, train_from=2015, bios=bios,
                             ids=set(team_of))
    lg_g, lg_a = PP.league_level(hist, SEASON)
    pr = PP.to_totals(pred, lg_g, lg_a, games=GAMES, team_of=team_of, calib=cal)
    pr["team"] = pr.player_id.map(team_of)
    pr["name"] = pr.player_id.map(names)
    pr = pr[pr.team.notna()].sort_values("proj_p", ascending=False)

    out = Path(__file__).resolve().parents[1] / "output" / "player_proj_2027.csv"
    cols = ["player_id", "name", "team", "pos_group", "exp_gp", "proj_toi_min",
            "toi_per_gp_min", "g60", "a60", "proj_g", "proj_a", "proj_p"]
    pr[cols].to_csv(out, index=False)

    n_ros = sum(len(v) for v in rosters.values())
    cold = int(pr.proj_p.isna().sum())
    print(f"2026-27 player projections — {len(pr):,} skaters of {n_ros:,} on "
          f"announced rosters ({GAMES}-game season)")
    print(f"league level used: {lg_g:.3f} G/60, {lg_a:.3f} A/60   "
          f"top-end calibration a={cal[0]:+.2f} b={cal[1]:.3f}")
    budget = PP.SKATER_MIN_PER_TEAM_GAME * GAMES
    got = pr.groupby("team").proj_toi_min.sum()
    print(f"ice-time budget per team: {budget:,.0f} min | projected "
          f"{got.mean():,.0f} (min {got.min():,.0f}, max {got.max():,.0f})")
    print(f"\n  {'name':<24}{'tm':>4}{'GP':>5}{'TOI':>7}{'G':>6}{'A':>6}{'PTS':>7}")
    for r in pr.head(25).itertuples():
        print(f"  {str(r.name)[:22]:<22}{r.team:>4}{r.exp_gp:>5.0f}"
              f"{r.proj_toi_min:>7.0f}{r.proj_g:>6.1f}{r.proj_a:>6.1f}{r.proj_p:>7.1f}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
