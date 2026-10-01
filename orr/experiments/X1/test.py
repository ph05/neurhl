"""X1 task 5: the single test run (configuration fixed in ledger.json).

Arms (each: team-history start run -> OT/SO walk-forward -> shipped run from
the preseason pipeline's ratings with those OT/SO parameters, exactly the
protocol of orr/backtest/inseason_bt.py):

  base       no starters, no skater lineups      ("ORR without lineups")
  starters   known starters only (goalie layer)   (inseason_bt 'known_starters')
  x1         known starters + skater lineup offsets (the X1 model)
  x1_sk      skater lineup offsets only
  base_go / x1_go   the same as base / x1 with goals-only updates

Scored on the gate games 2019-24 (primary: x1 vs base, shipped start) and on
NeurHL's restatement games 2018-24 against NeurHL-H (team-history start).

Run once: python3 -m orr.experiments.X1.test   (refuses to overwrite its output)
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from orr import ratings as R
from orr import structural as S
from orr.backtest import gamefile_bt as GB
from orr.backtest.games_bt import PREDS, paired
from orr.experiments.X1 import lfilter as LF
from orr.experiments.X1 import lineups as LU

HERE = Path(__file__).resolve().parent
OUTF = HERE / "test_output.json"
PREDF = HERE / "test_preds.csv.gz"
VIEWS = ["mkt_rel82", "td_rel82"]


DRY = "--dry" in sys.argv          # lineup offsets zeroed: validates the pipeline against
                                   # the published baselines without scoring X1
OVER_CACHE = Path("/tmp/claude-0/-home-user-neurhl/3d07af1c-5d5a-5d32-9e5e-017a2574dcf3/scratchpad/x1_over.pkl")


def main():
    global OUTF, PREDF
    if DRY:
        OUTF, PREDF = HERE / "dry_output.json", HERE / "dry_preds.csv.gz"
    elif OUTF.exists() and "--force" not in sys.argv:
        raise SystemExit(f"{OUTF} exists: the test has already been run")
    t0 = time.time()
    L = json.loads((HERE / "ledger.json").read_text())
    cfg = L["chosen"]["config"]
    hp = R.load_hp()
    hp_go = dataclasses.replace(hp, use_shots=False)
    w = LU.wide(cfg["halflife"])
    lo = LU.offsets(w, cfg["beta_x"], cfg.get("beta_st", 0.0), cfg.get("beta_t", 0.0))
    if DRY:
        lo["lo_h"], lo["lo_a"] = 0.0, 0.0
    sm = cfg.get("shot_mult", 0.0)

    # the shipped loop's starting ratings (as inseason_bt.py)
    import pickle
    if OVER_CACHE.exists():
        over = pickle.loads(OVER_CACHE.read_bytes())
    else:
        hist = GB.hist_frame()
        over = {}
        for V in GB.SEASONS:
            P_V = R.fit_gamemodel_params(V, hp, write=False)
            _, lg, r = GB.game_file_season(V, hist, VIEWS, P_V, with_ratings=True)
            over[V] = (r, lg["mu"])
        OVER_CACHE.write_bytes(pickle.dumps(over))
    print(f"preseason ratings rebuilt ({time.time() - t0:.0f}s)", flush=True)

    arms = {"base": (hp, False, None), "starters": (hp, True, None), "x1": (hp, True, lo),
            "x1_sk": (hp, False, lo), "base_go": (hp_go, False, None), "x1_go": (hp_go, True, lo)}
    g = S.game_frame()[["gid", "game_id", "season_end", "date", "home_win"]]
    P = g.copy()
    for tag, (h, gk, lu) in arms.items():
        th = LF.run_filter_lineup(h, GB.SEASONS, use_goalie=gk, lineup=lu, shot_mult=sm)
        ot = {}
        pth = R.predict_probs(th[th.season_end >= 2010], h, ot_params=ot)
        sh = LF.run_filter_lineup(h, GB.SEASONS, use_goalie=gk, lineup=lu, shot_mult=sm,
                                  pre_override=over)
        psh = R.predict_probs(sh[sh.season_end >= 2010], h, ot_params=dict(ot))
        P = P.merge(pth[["gid", "p_home_win"]].rename(columns={"p_home_win": f"th_{tag}"}), on="gid", how="left")
        P = P.merge(psh[["gid", "p_home_win"]].rename(columns={"p_home_win": f"sh_{tag}"}), on="gid", how="left")
        print(f"arm {tag} done ({time.time() - t0:.0f}s)", flush=True)

    gt = S.goalie_game_talent(*hp.goalie)
    P = P.merge(gt[["gid", "gdiff_h", "gdiff_a"]], on="gid", how="left")
    P["gk_known"] = P.gdiff_h.notna() & P.gdiff_a.notna()
    P["lineup_known"] = P.gid.isin(set(w.gid))
    P = P.merge(lo, on="gid", how="left").fillna({"lo_h": 0.0, "lo_a": 0.0})

    res = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "config": cfg, "protocol": L["protocol"]["test_primary"]}

    # --- gate games 2019-24
    gate = pd.read_csv(PREDS / "g_gate_games.csv")
    m = gate[["game_id", "season_end", "y", "p_stack", "p_elo"]].rename(
        columns={"season_end": "season_gate"}).merge(P, on="game_id", how="inner")
    assert len(m) == len(gate), (len(m), len(gate))
    assert (m.y == m.home_win).all()

    def block(x):
        y = x.home_win.to_numpy()
        r = {"n": int(len(x)), "lineup_known_share": float(x.lineup_known.mean()),
             "gk_known_share": float(x.gk_known.mean()), "neurhl_g": R.logloss(x.p_stack, y),
             "neurhl_elo": R.logloss(x.p_elo, y)}
        for s in ("sh", "th"):
            for tag in arms:
                r[f"{s}_{tag}"] = R.logloss(x[f"{s}_{tag}"], y)
        for s in ("sh", "th"):
            r[f"d_{s}_x1_vs_base"] = paired(x[f"{s}_x1"], x[f"{s}_base"], y)
            r[f"d_{s}_x1_vs_starters"] = paired(x[f"{s}_x1"], x[f"{s}_starters"], y)
            r[f"d_{s}_x1sk_vs_base"] = paired(x[f"{s}_x1_sk"], x[f"{s}_base"], y)
            r[f"d_{s}_starters_vs_base"] = paired(x[f"{s}_starters"], x[f"{s}_base"], y)
            r[f"d_{s}_x1go_vs_basego"] = paired(x[f"{s}_x1_go"], x[f"{s}_base_go"], y)
        r["d_sh_x1_vs_neurhl_g"] = paired(x.sh_x1, x.p_stack, y)
        r["d_sh_x1go_vs_neurhl_g"] = paired(x.sh_x1_go, x.p_stack, y)
        return r

    res["gate"] = {"all": block(m), "by_season": {int(V): block(x) for V, x in m.groupby("season_gate")},
                   "lineup_known_games": block(m[m.lineup_known & m.gk_known])}

    # --- restatement games 2018-24 vs NeurHL-H (team-history start)
    h = pd.read_csv(PREDS / "hier_restatement_games.csv")
    h = h[h.season.between(2018, 2024)]
    mh = h[["game_id", "season", "y", "p_neurhl_h", "p_elo"]].merge(P, on="game_id", how="inner")
    assert len(mh) == len(h), (len(mh), len(h))
    assert (mh.y == mh.home_win).all()

    def hblock(x):
        y = x.home_win.to_numpy()
        return {"n": int(len(x)), "lineup_known_share": float(x.lineup_known.mean()),
                "neurhl_h": R.logloss(x.p_neurhl_h, y), "neurhl_elo": R.logloss(x.p_elo, y),
                "orr_no_lineups": R.logloss(x.th_base, y), "orr_starters": R.logloss(x.th_starters, y),
                "orr_x1": R.logloss(x.th_x1, y),
                "d_x1_vs_neurhl_h": paired(x.th_x1, x.p_neurhl_h, y),
                "d_x1_vs_no_lineups": paired(x.th_x1, x.th_base, y),
                "d_no_lineups_vs_neurhl_h": paired(x.th_base, x.p_neurhl_h, y)}

    res["restatement_2018_24"] = {"all": hblock(mh),
                                  "by_season": {int(V): hblock(x) for V, x in mh.groupby("season")},
                                  "lineup_known_games": hblock(mh[mh.lineup_known & mh.gk_known])}
    res["secs"] = round(time.time() - t0, 1)
    OUTF.write_text(json.dumps(res, indent=1, default=float))
    keep = ["gid", "game_id", "season_end", "home_win", "lineup_known", "gk_known", "lo_h", "lo_a"] + \
        [c for c in P.columns if c.startswith(("sh_", "th_"))]
    P[P.season_end >= 2018][keep].to_csv(PREDF, index=False, float_format="%.5f")
    a = res["gate"]["all"]
    d = a["d_sh_x1_vs_base"]
    print(f"GATE n={a['n']}: base {a['sh_base']:.4f} starters {a['sh_starters']:.4f} x1 {a['sh_x1']:.4f} "
          f"x1_sk {a['sh_x1_sk']:.4f} | NeurHL-G {a['neurhl_g']:.4f}")
    print(f"  x1 - base {d['diff']:+.5f} CI [{d['ci95'][0]:+.5f}, {d['ci95'][1]:+.5f}]")
    print(json.dumps({k: v for k, v in res["restatement_2018_24"]["all"].items()}, default=float))


if __name__ == "__main__":
    main()
