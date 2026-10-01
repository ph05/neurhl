"""Backtest of ORR 1.1's in-season loop (pre-registered item X3, orr/PLAN_1_1.md).

ORR 1.1 is the shipped loop plus every ACCEPTED ORR 1.1 item. Of the nine
pre-registered items only X1 (lineup-aware game forecasts) was accepted, so
the combination is: dressed-skater lineup offsets (orr/lineups.py, the fixed
X1 configuration) plus the known starting goalies through the existing goalie
layer, both in the filter's prediction and in its update. The core code path
is used throughout (ratings.run_filter(use_goalie=..., lineup=...)), not the
experiment copies.

Protocol, as orr/backtest/inseason_bt.py: for each clean market season (2019,
2020, 2022-24) the preseason pipeline's ratings are rebuilt exactly as
gamefile_bt.py does (views mkt+td, convex weights fitted leave-one-season-
out, ratings solved on the actual schedule), and the walk-forward filter
starts from them (pre_override). Each arm takes its OT/SO parameters from its
own team-history-start run. Every prediction uses only games before its date;
the dressed lineups and starters of the game being forecast are the box
score's (known before puck drop; live they come from pregame lineup files).
Games with no box score (2020 partial, 2024 after 2023-11-07) get no lineup.

Variants: goals+shots (frozen ratings_hp.json) and goals-only (use_shots=False,
what the live loop runs while results carry no shots).
Arms per variant: shipped (ORR 1.0: no starters, no lineups) and orr_1_1.
Scored on NeurHL's gate games 2019-24 (n=6289) against the shipped loop of
the same variant, NeurHL-G and NeurHL Elo, with paired bootstrap CIs
(games_bt.paired), pooled and by season.

Pre-registered decision (orr/experiments/X3/prereg.json): X3 is accepted if
ORR 1.1 beats the shipped loop with a 95% CI below 0 in BOTH variants;
otherwise X1 still ships if its point estimate is not worse in either variant.

Run ONCE: python3 -m orr.backtest.inseason_bt_1_1   (refuses to overwrite)
Writes orr/output/backtest/inseason_bt_1_1.json and inseason_bt_1_1_preds.csv.gz.
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time
from datetime import datetime, timezone

import pandas as pd

from orr import config as C
from orr import lineups as LU
from orr import ratings as R
from orr.backtest import gamefile_bt as GB
from orr.backtest.games_bt import PREDS, paired

OUT = C.OUT / "backtest"
OUTF = OUT / "inseason_bt_1_1.json"
PREDF = OUT / "inseason_bt_1_1_preds.csv.gz"
VIEWS = ["mkt_rel82", "td_rel82"]
ACCEPTED = ["X1"]


def preseason_ratings(hp: R.HP) -> tuple[dict, dict]:
    over, logs = {}, {}
    hist = GB.hist_frame()
    for V in GB.SEASONS:
        P_V = R.fit_gamemodel_params(V, hp, write=False)
        _, lg, r = GB.game_file_season(V, hist, VIEWS, P_V, with_ratings=True)
        over[V] = (r, lg["mu"])
        logs[V] = {k: lg[k] for k in ("weights", "mu", "rating_sd_net_mean")}
    return over, logs


def arm_probs(hp: R.HP, over: dict, use_goalie: bool, lineup) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(team-history start, shipped start) home-win probabilities; the shipped
    run uses the OT/SO parameters of the team-history run."""
    th = R.run_filter(hp, GB.SEASONS, use_goalie=use_goalie, lineup=lineup)
    ot: dict = {}
    pth = R.predict_probs(th[th.season_end >= 2010], hp, ot_params=ot)
    sh = R.run_filter(hp, GB.SEASONS, use_goalie=use_goalie, lineup=lineup, pre_override=over)
    psh = R.predict_probs(sh[sh.season_end >= 2010], hp, ot_params=dict(ot))
    return pth[["gid", "p_home_win"]], psh[["gid", "p_home_win"]]


def score(m: pd.DataFrame, variants: list[str]) -> dict:
    """Log losses and paired CIs on the gate games (pooled, by season, and on
    games with both lineups and starters known), plus the decision."""
    def block(x: pd.DataFrame) -> dict:
        y = x.home_win.to_numpy()
        r = {"n": int(len(x)), "lineup_known_share": float(x.lineup_known.mean()),
             "starters_known_share": float(x.gk_known.mean()),
             "neurhl_g": R.logloss(x.p_stack, y), "neurhl_elo": R.logloss(x.p_elo, y)}
        for vt in variants:
            r[vt] = {"shipped": R.logloss(x[f"p_shipped_{vt}"], y),
                     "orr_1_1": R.logloss(x[f"p_orr_1_1_{vt}"], y),
                     "shipped_teamhist": R.logloss(x[f"th_shipped_{vt}"], y),
                     "orr_1_1_teamhist": R.logloss(x[f"th_orr_1_1_{vt}"], y),
                     "d_orr_1_1_vs_shipped": paired(x[f"p_orr_1_1_{vt}"], x[f"p_shipped_{vt}"], y),
                     "d_orr_1_1_vs_neurhl_g": paired(x[f"p_orr_1_1_{vt}"], x.p_stack, y),
                     "d_orr_1_1_vs_neurhl_elo": paired(x[f"p_orr_1_1_{vt}"], x.p_elo, y),
                     "d_shipped_vs_neurhl_g": paired(x[f"p_shipped_{vt}"], x.p_stack, y)}
        return r

    out = {"all": block(m), "by_season": {int(V): block(x) for V, x in m.groupby("season_gate")},
           "lineup_and_starters_known_games": block(m[m.lineup_known & m.gk_known])}
    a = out["all"]
    ci_ok = {vt: bool(a[vt]["d_orr_1_1_vs_shipped"]["ci95"][1] < 0) for vt in variants}
    no_harm = {vt: bool(a[vt]["d_orr_1_1_vs_shipped"]["diff"] <= 0) for vt in variants}
    out["decision"] = {
        "rule": "Accept if the combined loop improves on the current shipped loop (same variant) "
                "with a 95% CI that excludes 0 (read as: in both variants; prereg.json).",
        "ci_excludes_0_by_variant": ci_ok, "accepted": all(ci_ok.values()),
        "x1_does_not_hurt_by_variant": no_harm, "ship_x1_default_on": all(no_harm.values()),
        "note": "With X1 the only accepted item, 'the combination' and 'the best accepted item "
                "alone' are the same model."}
    # sensitivity only (prereg amendment): the primary CIs with 20,000 resamples,
    # drawn after every standard call so that those are unaffected
    y = m.home_win.to_numpy()
    out["sensitivity_nb20000"] = {vt: paired(m[f"p_orr_1_1_{vt}"], m[f"p_shipped_{vt}"], y, nb=20000)
                                  for vt in variants}
    return out


def main():
    if OUTF.exists() and "--force" not in sys.argv:
        raise SystemExit(f"{OUTF} exists: the ORR 1.1 hindcast has already been run")
    t0 = time.time()
    hp = R.load_hp()
    variants = {"goals_shots": hp, "goals_only": dataclasses.replace(hp, use_shots=False)}
    over, logs = preseason_ratings(hp)
    print(f"preseason ratings rebuilt ({time.time() - t0:.0f}s)", flush=True)
    lo = LU.backtest_offsets(LU.X1)
    arms = {"shipped": (False, None), "orr_1_1": (LU.X1.use_goalie, lo)}

    g = R.S.game_frame()[["gid", "game_id", "season_end", "home_win"]]
    P = g.copy()
    for vt, h in variants.items():
        for arm, (gk, lu) in arms.items():
            pth, psh = arm_probs(h, over, gk, lu)
            P = P.merge(pth.rename(columns={"p_home_win": f"th_{arm}_{vt}"}), on="gid", how="left")
            P = P.merge(psh.rename(columns={"p_home_win": f"p_{arm}_{vt}"}), on="gid", how="left")
            print(f"{vt} {arm} done ({time.time() - t0:.0f}s)", flush=True)
    gt = R.S.goalie_game_talent(*hp.goalie)
    P = P.merge(gt[["gid", "gdiff_h", "gdiff_a"]], on="gid", how="left")
    P["gk_known"] = P.gdiff_h.notna() & P.gdiff_a.notna()
    P["lineup_known"] = P.gid.isin(set(lo.gid))

    gate = pd.read_csv(PREDS / "g_gate_games.csv")
    m = gate[["game_id", "season_end", "y", "p_stack", "p_elo"]].rename(
        columns={"season_end": "season_gate"}).merge(P, on="game_id", how="inner")
    assert len(m) == len(gate), (len(m), len(gate))
    assert (m.y == m.home_win).all()
    keep = ["gid", "game_id", "season_gate", "home_win", "lineup_known", "gk_known", "p_stack", "p_elo"] + \
        [c for c in m.columns if c.startswith(("p_shipped", "p_orr_1_1", "th_"))]
    m[keep].to_csv(PREDF, index=False, float_format="%.6f")      # written before any scoring
    res = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "protocol": __doc__, "accepted_items": ACCEPTED,
           "lineup_settings": LU.settings(LU.X1), "seasons": GB.SEASONS, "season_logs": logs,
           **score(m, list(variants))}
    res["secs"] = round(time.time() - t0, 1)
    OUTF.write_text(json.dumps(res, indent=1, default=float))
    report(res, list(variants))


def report(res: dict, variants: list[str]) -> None:
    for V, r in [("all", res["all"])] + list(res["by_season"].items()):
        for vt in variants:
            d, dg = r[vt]["d_orr_1_1_vs_shipped"], r[vt]["d_orr_1_1_vs_neurhl_g"]
            print(f"{str(V):>4} {vt:<11} n={r['n']:>5} shipped {r[vt]['shipped']:.5f} 1.1 {r[vt]['orr_1_1']:.5f}"
                  f" G {r['neurhl_g']:.5f} Elo {r['neurhl_elo']:.5f}"
                  f"  1.1-ship {d['diff']:+.5f} [{d['ci95'][0]:+.5f},{d['ci95'][1]:+.5f}]"
                  f"  1.1-G {dg['diff']:+.5f} [{dg['ci95'][0]:+.5f},{dg['ci95'][1]:+.5f}]")
    print(json.dumps(res["decision"], indent=1))


if __name__ == "__main__":
    main()
