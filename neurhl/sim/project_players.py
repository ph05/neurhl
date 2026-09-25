"""NeurHL-2 — player projections for 2026-27, with a walk-forward backtest.

Ships the per-player half of the deliverable. Deliberately NOT built on S1's
actor head: that head chooses among players CURRENTLY on the ice, so turning it
into a season projection needs the deployment process (S3), which was not built.
Claiming a neural-network provenance it does not have would be worse than using
the honest estimator.

What it is built on (models/player_proj.py, the live model):

  * **gradient boosting on walk-forward features**: own-history EWMAs of per-60
    rates as RATIOS to each season's league level (the era control), usage and
    availability as SHARES of fixed budgets, RAPM prior, age, career shape.
  * **conservation-law totals** in to_totals(): positional ice-time budgets and
    the dressed-roster games identity, enforced rather than hoped for.
  * **walk-forward top-end calibration** fitted on the season before the target.
  An explicit age-curve multiplier was removed as dead code — the raw profile
  it would fit is survivorship (only good players last to 34) — so age enters
  as a feature the trees condition on, not a cohort multiplier.

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
# Earliest training vantage for the GBMs. Was 2015 when assists only existed
# from 2012 (a training vantage needs assist-complete rel_a60 targets plus
# history before it). A10 recovered assists at source for 2008-2011
# (configs/assist_recovery.json: 2012 ground truth p2/p3 agreement
# 0.9994/0.9989, zero invented assists), so vantage 2010 -- features from
# 2008-2009, targets 2010 -- is now the earliest supportable one.
TRAIN_FROM = 2010


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
        cal = PP.fit_top_calibration(hist, V, TRAIN_FROM, bios, team_map_actual)
        pred, _ = PP.fit_predict(hist, V, train_from=TRAIN_FROM, bios=bios)
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
        def pooled(rs, label):
            n = sum(r["n"] for r in rs)
            na = sum(r["n_all"] for r in rs)
            print(f"POOLED {label:<34} n={n:,}  MAE "
                  f"{sum(r['mae_pts']*r['n'] for r in rs)/n:.2f} vs league-mean "
                  f"{sum(r['naive_mae']*r['n'] for r in rs)/n:.2f}  "
                  f"bias(>=40GP, selected) "
                  f"{sum(r['bias']*r['n'] for r in rs)/n:+.2f}  bias(all) "
                  f"{sum(r['bias_unconditional']*r['n_all'] for r in rs)/na:+.2f}  "
                  f"corr {np.mean([r['corr'] for r in rs]):.3f}")
        # A10: report BOTH windows, whatever they show -- the widened
        # era-diverse pool and the previous 2022-2026 window for continuity.
        print()
        pooled(rows, "era-diverse (A10)")
        prev = [r for r in rows if r["season"] >= 2022]
        if prev and len(prev) < len(rows):
            pooled(prev, "previous window 2022-2026")
        p = Path(__file__).resolve().parents[1] / "configs" / "player_backtest.json"
        p.write_text(json.dumps(rows, indent=1))
        print(f"-> {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest", action="store_true")
    args = ap.parse_args()
    if args.backtest:
        # A10 era-diverse vantages: DEV/TUNE-era 2011-2017 (NO_SCORE 2013
        # excluded) plus the previously declared 2022-2026 spend. 2018-2021
        # remain untouched CONFIRM (2021 is also NO_SCORE).
        backtest([2011, 2012, 2014, 2015, 2016, 2017,
                  2022, 2023, 2024, 2025, 2026])
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

    cal = PP.fit_top_calibration(hist, SEASON, TRAIN_FROM, bios, team_map_actual)
    pred, _ = PP.fit_predict(hist, SEASON, train_from=TRAIN_FROM, bios=bios,
                             ids=set(team_of))
    lg_g, lg_a = PP.league_level(hist, SEASON)
    pr = PP.to_totals(pred, lg_g, lg_a, games=GAMES, team_of=team_of, calib=cal)
    pr["team"] = pr.player_id.map(team_of)
    pr["name"] = pr.player_id.map(names)
    pr = pr[pr.team.notna()]

    # ---- PS1 (PLAN_NeurHL3) shipped the 50/50 blend of Path A (above) and
    # Path B (the player-game chain aggregated over the schedule). The blend
    # averages goals and assists exactly as the backtest averaged points;
    # games and ice time stay Path A's, because those carry the conservation
    # laws. A skater with no NHL feature row (no games before 2026-27) has no
    # Path B estimate and keeps Path A.
    import models.player_game as PGM
    from sim.player_season_v2 import project_b
    from sim.project_2027 import load_schedule
    from sim.schedule_context import build as schedule_ctx
    team_idx = json.loads((TENSORS / "maps.json").read_text())["team"]
    b = project_b(PGM.build_frames(), SEASON, games=GAMES,
                  team_of={p: team_idx[t] for p, t in team_of.items()},
                  gc=schedule_ctx(load_schedule(), SEASON))
    pr = pr.merge(b[["player_id", "proj_g", "proj_a"]].rename(
        columns={"proj_g": "b_g", "proj_a": "b_a"}), on="player_id", how="left")
    pr["proj_g_a"], pr["proj_a_a"] = pr.proj_g, pr.proj_a
    has_b = pr.b_g.notna()
    pr.loc[has_b, "proj_g"] = 0.5 * (pr.proj_g_a + pr.b_g)[has_b]
    pr.loc[has_b, "proj_a"] = 0.5 * (pr.proj_a_a + pr.b_a)[has_b]
    pr["proj_p"] = pr.proj_g + pr.proj_a
    pr["proj_p_path_a"] = pr.proj_g_a + pr.proj_a_a
    pr["proj_p_path_b"] = pr.b_g + pr.b_a
    pr = pr.sort_values("proj_p", ascending=False)

    out = Path(__file__).resolve().parents[1] / "output" / "player_proj_2027.csv"
    cols = ["player_id", "name", "team", "pos_group", "exp_gp", "proj_toi_min",
            "toi_per_gp_min", "g60", "a60", "proj_g", "proj_a", "proj_p",
            "proj_p_path_a", "proj_p_path_b"]
    pr[cols].round(3).to_csv(out, index=False)
    print(f"PS1 blend: {int(has_b.sum())} skaters blended, "
          f"{int((~has_b).sum())} Path A only (no NHL feature history)")

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
