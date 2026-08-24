"""NeurHL-3 — M4 Path B: season projections by aggregating the player-game
chain over the real schedule.

Decomposition per player-season: E[points] = G * gp_share * mean_g(lambda_g),
where gp_share comes from the INCUMBENT's validated availability head
(models/player_proj fit_predict — reusing it isolates the A-vs-B comparison to
the RATES, which is what the player-game layer improved), and lambda_g =
-ln(1 - p) maps the chain's per-game P(goal>=1)/P(assist>=1) to Poisson means,
evaluated on FROZEN pre-season feature rows (a player's last observed state
before V) with per-game schedule context (home, rest, travel, opponent Elo and
defensive form frozen at end of V-1). Opposing starters are unknown
pre-season -> NaN under the availability mask, exactly as live 2027 predicts
before lineups post. Preseason health is assumed (absence features neutral),
matching the incumbent's assumption.

Backtest (`--backtest`): vantages {2011,2012,2014-2017} + {2022-2026}, >=40GP
rows restated per-82, against Path A run in the same process — the PS1 ship
rule (PLAN_NeurHL3) is then applied mechanically. Writes
configs/player_season_v2.json.

Run: uv run --no-project --python 3.12 --with numpy --with "pandas<3" \
     --with pyarrow --with scikit-learn python neurhl/sim/player_season_v2.py \
     --backtest
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import TENSORS  # noqa: E402
import models.player_game as PGM  # noqa: E402
import models.player_proj as PP  # noqa: E402
import windows as W  # noqa: E402
from train.train_player_game import fit_chain, predict_rows  # noqa: E402
from sim.project_players import team_map_actual, load_bios  # noqa: E402

CFG = ROOT / "configs"
BACKTEST_V = [2011, 2012, 2014, 2015, 2016, 2017,
              2022, 2023, 2024, 2025, 2026]


def frozen_rows(frame: pd.DataFrame, v: int, team_of: dict) -> pd.DataFrame:
    """Per (player, scheduled team-game of season v): the player's last
    pre-v feature row with per-game schedule context substituted."""
    hist = frame[frame.season_end < v]
    last = hist.sort_values(["player_id", "date", "game_id"]) \
        .groupby("player_id").tail(1).set_index("player_id")
    gc = pd.read_parquet(TENSORS / f"games_ctx_{v}.parquet")
    gc = gc[gc.game_type == 2].copy()
    gc["date"] = pd.to_datetime(gc.date)

    # frozen end-of-(v-1) team state: elo (0.7 carry) and GA form
    prev_rows = hist[hist.season_end == v - 1]
    if not len(prev_rows):
        prev_rows = hist[hist.season_end == hist.season_end.max()]
    elo_end = {}
    for side, tcol in (("own_elo", "team"), ):
        e = prev_rows.sort_values("date").groupby(tcol).own_elo.last()
        elo_end = (1500.0 + 0.7 * (e - 1500.0)).to_dict()
    ga_end = prev_rows.sort_values("date").groupby("opp").opp_ga_ew.last() \
        .to_dict()
    era = prev_rows.sort_values("date").tail(1)[
        ["prior_gpg", "prior_ot_share", "prior_margin_abs", "prior_parity",
         "season_scaled"]].iloc[0]

    players = [p for p in team_of if p in last.index]
    rows = []
    for pid in players:
        team = team_of[pid]
        base = last.loc[pid]
        sched = gc[(gc.home_idx == team) | (gc.away_idx == team)]
        r = pd.DataFrame(index=range(len(sched)))
        for c in PGM.FEATURES:
            r[c] = base.get(c, np.nan)
        is_home = (sched.home_idx == team).to_numpy()
        r["is_home"] = is_home.astype(float)
        r["rest"] = np.where(is_home, sched.home_rest, sched.away_rest)
        r["b2b"] = (r.rest <= 1).astype(float)
        r["km3d"] = np.where(is_home, sched.home_km3d, sched.away_km3d)
        r["dtz"] = np.where(is_home, sched.home_dtz, sched.away_dtz)
        r["days_in"] = sched.days_in.to_numpy()
        opp = np.where(is_home, sched.away_idx, sched.home_idx)
        r["opp_elo"] = pd.Series(opp).map(elo_end).fillna(1500.0).to_numpy()
        r["own_elo"] = elo_end.get(team, 1500.0)
        r["opp_ga_ew"] = pd.Series(opp).map(ga_end).to_numpy()
        for c in ("opp_gq", "opp_gsax60", "opp_g_starts7"):
            r[c] = np.nan
        for c in ("vacated", "n_absent", "above_me_out", "ret_gap"):
            r[c] = 0.0
        for c in era.index:
            r[c] = float(era[c])
        r["player_id"] = pid
        r["team"] = team
        r["game_id"] = sched.game_id.to_numpy()
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


def project_b(frame: pd.DataFrame, v: int, games: int = 82) -> pd.DataFrame:
    """Path B season totals for vantage v."""
    team_of = team_map_actual(v)
    ch = fit_chain(frame, v)
    rows = frozen_rows(frame, v, team_of)
    pred = predict_rows(ch, rows)
    lam_g = -np.log(np.clip(1 - pred.p_goal, 1e-6, 1.0))
    lam_a = -np.log(np.clip(1 - pred.p_assist, 1e-6, 1.0))
    per = pred.assign(lam_g=lam_g, lam_a=lam_a).groupby(
        "player_id", as_index=False).agg(lam_g=("lam_g", "mean"),
                                         lam_a=("lam_a", "mean"),
                                         n_sched=("game_id", "size"))
    # availability from the incumbent's validated head
    hist = PP.load_player_seasons(range(2008, v))
    bios = load_bios()
    pa, _ = PP.fit_predict(hist, v, train_from=2010, bios=bios)
    per = per.merge(pa[["player_id", "gp_share"]], on="player_id", how="left")
    per["gp_share"] = per.gp_share.clip(0.02, 1.0).fillna(0.5)
    per["exp_gp"] = per.gp_share * games
    per["proj_g"] = per.exp_gp * per.lam_g
    per["proj_a"] = per.exp_gp * per.lam_a
    per["proj_p"] = per.proj_g + per.proj_a
    return per


def backtest():
    frame = PGM.build_frames()
    hist_all = PP.load_player_seasons(range(2008, 2027))
    bios = load_bios()
    out = []
    for v in BACKTEST_V:
        b = project_b(frame, v)
        # Path A in the same process, same population
        hist = hist_all[hist_all.season_end < v]
        cal = PP.fit_top_calibration(hist_all, v, 2010, bios, team_map_actual)
        pa, _ = PP.fit_predict(hist_all, v, train_from=2010, bios=bios)
        lg_g, lg_a = PP.league_level(hist_all, v)
        ta = PP.to_totals(pa, lg_g, lg_a, games=82,
                          team_of=team_map_actual(v), calib=cal)
        act = hist_all[hist_all.season_end == v].set_index("player_id")
        sched = act.gp.max()
        m = b.merge(ta[["player_id", "proj_p"]], on="player_id",
                    suffixes=("_b", "_a"))
        m["act_p"] = m.player_id.map((act.g + act.a) * 82.0 / max(sched, 40))
        m["act_gp"] = m.player_id.map(act.gp)
        mm = m[(m.act_gp >= 40 * sched / 82)].dropna(subset=["act_p"])
        mae_b = float(np.abs(mm.proj_p_b - mm.act_p).mean())
        mae_a = float(np.abs(mm.proj_p_a - mm.act_p).mean())
        mae_blend = float(np.abs(0.5 * (mm.proj_p_b + mm.proj_p_a)
                                 - mm.act_p).mean())
        out.append({"season": v, "n": int(len(mm)), "mae_B": round(mae_b, 3),
                    "mae_A": round(mae_a, 3), "mae_blend": round(mae_blend, 3),
                    "B_wins": bool(mae_b < mae_a)})
        print(f"{v}: n={len(mm):>4}  B {mae_b:6.2f}  A {mae_a:6.2f}  "
              f"blend {mae_blend:6.2f}  {'B' if mae_b < mae_a else 'A'}")
        sys.stdout.flush()
    n = sum(r["n"] for r in out)
    pool = {k: round(sum(r[f"mae_{k}"] * r["n"] for r in out) / n, 3)
            for k in ("B", "A", "blend")}
    wins_b = sum(r["B_wins"] for r in out)
    # PS1 ship rule, mechanical
    if pool["blend"] < pool["A"] and pool["blend"] < pool["B"]:
        ship = "blend"
    elif pool["B"] < pool["A"] and wins_b >= 6:
        ship = "B"
    else:
        ship = "A"
    res = {"per_season": out, "pooled": pool, "B_wins": wins_b,
           "n_vantages": len(out), "ship": ship,
           "rule": "PS1: B iff pooled<B beats A AND wins>=6/11; blend only if "
                   "beats both; else A"}
    (CFG / "player_season_v2.json").write_text(json.dumps(res, indent=1))
    print(f"\nPOOLED  B {pool['B']}  A {pool['A']}  blend {pool['blend']}  "
          f"B wins {wins_b}/{len(out)}  -> SHIP {ship}")
    print("-> configs/player_season_v2.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest", action="store_true")
    args = ap.parse_args()
    if args.backtest:
        backtest()
        return


if __name__ == "__main__":
    main()
