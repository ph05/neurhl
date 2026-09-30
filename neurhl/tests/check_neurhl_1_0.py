"""NeurHL 1.0 consistency check (PLAN_NeurHL_1_0.md): every level must agree.

Reads only the files sim/unified_2027.py wrote to neurhl/output/neurhl_1_0/
and re-derives, independently of the simulator's code:

  games       probabilities in (0, 1); outcome4 sums to 1 and its home mass
              equals the home-win probability; league scoring, overtime share
              and home-win rate inside the historical bands
  teams       wins + losses + overtime losses = games; league points equal
              2 x games + overtime games; playoff, division, Presidents'
              Trophy and bracket probabilities sum to their slot counts
  players     each team's skater goals, shots and extra statistics sum to the
              team's totals; skater games = 18 x team games; goalie starts =
              team games; goalie goals against and wins sum to the team's
  player-games  18 dressed skaters per team-game on average; per team-game
              skater goals equal the game file's team goals; ice time inside
              the physical budget; faceoffs balanced between the two teams;
              season totals equal the sums of the player-game rows
  across paths  when the opening-night preview exists, the season model's
              opening-night probabilities agree with the game-day path
Writes output/neurhl_1_0/checks_2027.json (checks_2027_rerun.json once
PLAN_NeurHL_1_0.md has its FREEZE section, so the hashed record is never
overwritten); exits 1 if any check fails.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "neurhl_1_0"
RES = []


def check(name, ok, detail=""):
    RES.append({"check": name, "pass": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def main():
    global OUT
    if "--dir" in sys.argv:                  # check a dated file set instead of the frozen files
        OUT = Path(sys.argv[sys.argv.index("--dir") + 1]).resolve()
    g = pd.read_csv(OUT / "games_2027.csv")
    t = pd.read_csv(OUT / "teams_2027.csv")
    sk = pd.read_csv(OUT / "skaters_2027.csv")
    gl = pd.read_csv(OUT / "goalies_2027.csv")
    pg = pd.read_csv(OUT / "player_games_2027.csv.gz")
    N = len(g)
    print("NeurHL 1.0 consistency check\n\nGAMES")
    check("1,344 regular-season games", N == 1344, str(N))
    check("home-win probabilities strictly inside (0, 1)", g.p_home_win.between(0, 1, inclusive="neither").all())
    o4 = g[["p_home_reg", "p_away_reg", "p_home_ot", "p_away_ot"]].to_numpy()
    check("outcome4 sums to 1", np.abs(o4.sum(1) - 1).max() < 1e-4, f"max err {np.abs(o4.sum(1) - 1).max():.1e}")
    check("outcome4 home mass equals the home-win probability",
          np.abs(o4[:, 0] + o4[:, 2] - g.p_home_win).max() < 1e-4)
    gpg = g[["goals_home", "goals_away"]].to_numpy().mean()
    check("league goals per team-game in [2.7, 3.3]", 2.7 <= gpg <= 3.3, f"{gpg:.3f}")
    check("share of games past regulation in [0.18, 0.27]", 0.18 <= g.p_ot.mean() <= 0.27, f"{g.p_ot.mean():.3f}")
    check("home-win rate in [0.50, 0.57]", 0.50 <= g.p_home_win.mean() <= 0.57, f"{g.p_home_win.mean():.3f}")

    print("\nTEAMS")
    gp_team = g.home.value_counts().add(g.away.value_counts(), fill_value=0)
    t = t.set_index("team")
    check("32 teams", len(t) == 32)
    check("each team plays 84 games", (t.index.map(gp_team) == 84).all())
    # the file stores four decimals, so the three rounded columns can miss by ~2e-4
    check("wins + losses + OT losses = games", (t.w + t.l + t.otl - t.gp).abs().max() < 1e-3,
          f"max |diff| {(t.w + t.l + t.otl - t.gp).abs().max():.1e}")
    lp = t.points.sum()
    ident = 2 * N + t.otl.sum()
    check("league points = 2 x games + overtime games", abs(lp - ident) < 0.05, f"{lp:.2f} vs {ident:.2f}")
    for col, slots in (("playoff_pct", 16), ("division_pct", 4), ("presidents_pct", 1),
                       ("round2_pct", 8), ("conf_final_pct", 4), ("cup_final_pct", 2), ("cup_pct", 1)):
        s = t[col].sum() / 100
        check(f"{col} sums to {slots} slots", abs(s - slots) < 1e-6, f"{s:.6f}")
    check("expected wins from the MC match the games file within 1 win",
          (t.w - t.exp_wins_from_games).abs().max() < 1.0,
          f"max |diff| {(t.w - t.exp_wins_from_games).abs().max():.3f}")

    print("\nPLAYERS")
    tol = 0.02
    xt = "so_goals_for" in t          # NeurHL 1.2 R2: totals include goals past regulation
    if xt:
        d = (t.goals_for - t.goals_for_reg - t.ot_goals_for - t.so_goals_for).abs().max()
        check("team goals_for = regulation + overtime + shootout-deciding goals", d < tol, f"max |diff| {d:.4f}")
        d = (sk.groupby("team").goals.sum() - (t.goals_for - t.so_goals_for)).abs().max()
        check("team goals_for less shootout-deciding goals = sum of its skaters' goals", d < tol,
              f"max |diff| {d:.4f}")
    for s_col, t_col in ((() if xt else (("goals", "goals_for"),)) + (("sog", "sog_for"),)):
        d = (sk.groupby("team")[s_col].sum() - t[t_col]).abs().max()
        check(f"team {t_col} = sum of its skaters' {s_col}", d < tol, f"max |diff| {d:.4f}")
    for c in ("hits", "blocks", "giveaways", "takeaways", "pim", "fo_won"):
        if c in sk and c in t:
            d = (sk.groupby("team")[c].sum() - t[c]).abs().max()
            check(f"team {c} = sum of its skaters", d < tol, f"max |diff| {d:.4f}")
    d = (sk.groupby("team").gp.sum() - 18 * t.index.map(gp_team).to_series(index=t.index)).abs().max()
    check("skater games = 18 x team games", d < tol, f"max |diff| {d:.4f}")
    d = (gl.groupby("team").starts.sum() - t.index.map(gp_team).to_series(index=t.index)).abs().max()
    check("goalie starts = team games", d < tol, f"max |diff| {d:.4f}")
    d = (gl.groupby("team").ga.sum() - (t.goals_against - (t.so_goals_against if xt else 0))).abs().max()
    check("goalie goals against sum to the team's" + (" (less shootout-deciding goals)" if xt else ""),
          d < tol, f"max |diff| {d:.4f}")
    d = (gl.groupby("team").wins.sum() - t.exp_wins_from_games).abs().max()
    check("goalie wins sum to the team's expected wins", d < tol, f"max |diff| {d:.4f}")

    unnamed = int((sk.name.isna() & (sk.gp >= 1)).sum() + (gl.name.isna() & (gl.starts >= 1)).sum())
    check("every skater with a game and every goalie with a start has a name", unnamed == 0, f"{unnamed} unnamed")

    print("\nPLAYER-GAMES")
    tg = pg.groupby(["game_id", "side"])
    dress = tg.dress.sum()
    check("18 skaters dressed per team-game", (dress - 18).abs().max() < 1e-3, f"max |diff| {(dress - 18).abs().max():.1e}")
    goals_pg = tg.g.sum().unstack()
    gi = g.set_index("game_id")
    so_h, so_a = (gi.so_goals_home, gi.so_goals_away) if "so_goals_home" in gi else (0, 0)
    d = max((goals_pg[0] - (gi.goals_home - so_h)).abs().max(), (goals_pg[1] - (gi.goals_away - so_a)).abs().max())
    check("per team-game skater goals = the game file's team goals", d < 1e-3, f"max |diff| {d:.1e}")
    toi = tg.toi.sum()
    check("skater ice time per team-game within the budget [280, 300] minutes",
          toi.between(280, 300.01).all(), f"range {toi.min():.1f}-{toi.max():.1f}")
    if "fo_won" in pg:
        fo = tg.fo_won.sum().unstack().sum(1)
        taken = tg.fo_taken.sum().unstack()
        check("faceoff wins of the two teams sum to the game's faceoffs",
              (fo - taken[0]).abs().max() < 1e-3 and (taken[0] - taken[1]).abs().max() < 1e-3)
    for c in ("goals", "assists", "gp", "sog", "toi"):
        src = {"goals": "g", "assists": "a", "gp": "dress"}.get(c, c)
        d = (pg.groupby("player_id")[src].sum() - sk.groupby("player_id")[c].sum()).abs().max()
        check(f"season {c} = sum of the player-game rows", d < 1e-2, f"max |diff| {d:.1e}")

    run = json.loads((OUT / "run_2027.json").read_text()) if (OUT / "run_2027.json").exists() else {}
    if "level_ratios" in run:               # NeurHL 1.2 checks (PLAN_NeurHL_1_2 R1, R4)
        print("\nNEURHL 1.2")
        ppo = t.pp_opps_for.sum() / (2 * N)
        check("league power-play opportunities per team-game in [2.5, 3.3]", 2.5 <= ppo <= 3.3, f"{ppo:.3f}")
        share = t.pp_goals_for.sum() / t.goals_for.sum()
        check("power-play goals share of all goals in [0.15, 0.25]", 0.15 <= share <= 0.25, f"{share:.3f}")
        if run.get("rookie_priors"):
            sys.path.insert(0, str(ROOT / "eval"))
            import rookie_priors as RP
            d_ = RP.load()
            nhl = d_[(d_.lg == "NHL") & (d_.season_end < 2027)].groupby("player_id").gp.sum()
            recent = set(d_[(d_.lg != "NHL") & d_.season_end.isin([2025, 2026]) & (d_.gp > 0)].player_id)
            table = set(pd.read_csv(ROOT.parent / run["rookie_priors"]).player_id) \
                if (ROOT.parent / run["rookie_priors"]).exists() else set(pd.read_csv(run["rookie_priors"]).player_id)
            dressed = set(pg[pg.dress > 0].player_id)
            miss = sorted(p_ for p_ in dressed if nhl.get(p_, 0) < 20 and p_ in recent and p_ not in table)
            check("every dressed rookie with a recent pre-NHL record is in the rookie table", not miss,
                  f"{len(miss)} missing {miss[:5]}")

    prev = ROOT / "output" / "live" / "2027" / "2026-09-29" / "preview.csv"
    run_b = json.loads((OUT / "run_2027.json").read_text()).get("bundle") if (OUT / "run_2027.json").exists() else None
    same_engine = prev.exists() and run_b in set(pd.read_csv(prev).get("bundle", pd.Series(dtype=str)))
    if prev.exists() and not same_engine:
        print(f"\nACROSS PATHS: skipped (the preview was made by a different engine than {run_b})")
    if same_engine:
        print("\nACROSS PATHS")
        pv = pd.read_csv(prev).set_index("game_id")
        d = (gi.loc[pv.index, "p_home_win"] - pv.p_home_win_neurhl_g).abs()
        check("opening night agrees with the game-day preview within 0.06", d.max() < 0.06,
              f"max |diff| {d.max():.3f} over {len(d)} games")

    ok = all(r["pass"] for r in RES)
    # after the freeze, checks_2027.json is a hashed record: a rerun writes beside it
    frozen = "\n## FREEZE" in (ROOT.parent / "PLAN_NeurHL_1_0.md").read_text() \
        and OUT == (ROOT / "output" / "neurhl_1_0").resolve()
    dest = OUT / ("checks_2027_rerun.json" if frozen else "checks_2027.json")
    dest.write_text(json.dumps({"pass": ok, "n": len(RES),
                                                      "passed": sum(r["pass"] for r in RES),
                                                      "checks": RES}, indent=1))
    print(f"\n{sum(r['pass'] for r in RES)}/{len(RES)} checks pass")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
