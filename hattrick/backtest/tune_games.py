"""Walk-forward hyperparameter search for the HatTrick ratings filter and Elo.

TUNING WINDOW: seasons 2011-12, 2013-14, 2014-15, 2015-16, 2016-17
(season_end 2012, 2014-2017; the 48-game 2013 season is excluded). Every
prediction in the window is walk-forward (structural fits, preseason
regression and OT fits read only earlier seasons; the filter reads only
earlier game days). Nothing from 2017-18 onward is ever evaluated here.

Every configuration tried is appended to the ledger
hattrick/output/backtest/games_search_ledger.json with its tuning-window
scores. The selected configuration is frozen to
hattrick/output/params/ratings_hp.json (and Elo's to elo_hp.json).

Run: python3 -m hattrick.backtest.tune_games [--quick]
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from hattrick import config as C
from hattrick import ratings as R
from hattrick import structural as S

TUNE = [2012, 2014, 2015, 2016, 2017]
# ledger rows carry the model version; earlier versions stay in the ledger
# (they were scored on the same window) but are never resumed from.
#   v1: Poisson base + layer, GLM ridge 0.2 (compressed preseason targets)
#   v2: COM-Poisson base (kappa) + layer, GLM ridge 1.0, pace/net filter
MODEL_VERSION = "v2"
LEDGER = C.OUT / "backtest" / "games_search_ledger.json"


def _ledger() -> list:
    try:
        return json.loads(LEDGER.read_text())
    except FileNotFoundError:
        return []


def _write_ledger(rows):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(rows, indent=1, default=float))


def evaluate(hp: R.HP, use_goalie: bool = False) -> dict:
    """Tuning-window scores for one configuration."""
    assert max(TUNE) < 2018
    pred = R.run_filter(hp, TUNE, use_goalie=use_goalie)
    pred = pred[pred.season_end >= 2010]
    g = S.game_frame()[["gid", "season_end", "home_win", "reg_h", "reg_a"]]
    pr = R.predict_probs(pred, hp).merge(g, on=["gid", "season_end"])
    pf = R.predict_probs(pred, hp, frozen=True).merge(g, on=["gid", "season_end"])
    pr, pf = pr[pr.season_end.isin(TUNE)], pf[pf.season_end.isin(TUNE)]
    if use_goalie:      # goalie variant is scored on games with both starters known
        known = S.goalie_game_talent(*hp.goalie)
        known = known[known.gdiff_h.notna() & known.gdiff_a.notna()].gid
        pr, pf = pr[pr.gid.isin(known)], pf[pf.gid.isin(known)]

    def pois_dev(p):
        lh, la = p.lam_h.to_numpy(), p.lam_a.to_numpy()
        yh, ya = p.reg_h.to_numpy(), p.reg_a.to_numpy()
        return float(np.mean(lh - yh * np.log(lh) + la - ya * np.log(la)))

    by = {int(V): R.logloss(x.p_home_win, x.home_win) for V, x in pr.groupby("season_end")}
    return {"ll": R.logloss(pr.p_home_win, pr.home_win),
            "ll_frozen": R.logloss(pf.p_home_win, pf.home_win),
            "goal_nll": pois_dev(pr), "goal_nll_frozen": pois_dev(pf),
            "brier": R.brier(pr.p_home_win, pr.home_win), "n": int(len(pr)),
            "by_season": by}


def record(tag: str, hp, res: dict, rows: list, kind: str = "filter",
           goalie_variant: bool = False):
    cfg = asdict(hp) if hasattr(hp, "__dataclass_fields__") else hp
    rows.append({"time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "model_version": MODEL_VERSION,
                 "kind": kind, "tag": tag, "goalie_variant": goalie_variant,
                 "config": cfg, "window": TUNE, **res})
    _write_ledger(rows)
    print(f"{kind:7s} {tag:34s} ll {res['ll']:.5f}"
          + (f"  frozen {res['ll_frozen']:.5f}  gnll {res['goal_nll']:.5f}"
             if "ll_frozen" in res else ""), flush=True)


def coordinate_search(base: R.HP, grid: dict, rows: list, rounds: int = 2,
                      use_goalie: bool = False, objective: str = "ll") -> R.HP:
    """Greedy coordinate descent over ``grid`` (field -> candidate values)."""
    best = base
    best_res = evaluate(best, use_goalie)
    record("base", best, best_res, rows, goalie_variant=use_goalie)
    seen = {best.key(): best_res}
    for rd in range(rounds):
        improved = False
        for f, vals in grid.items():
            for v in vals:
                hp = replace(best, **{f: v})
                if hp.key() in seen:
                    continue
                res = evaluate(hp, use_goalie)
                seen[hp.key()] = res
                record(f"r{rd} {f}={v}", hp, res, rows, goalie_variant=use_goalie)
                if res[objective] < best_res[objective] - 1e-6:
                    best, best_res, improved = hp, res, True
        if not improved:
            break
    return best, best_res


def select_final(rows: list, tol: float = 1e-4) -> dict:
    """Pre-declared final selection rule (fixed before any test-season
    number was computed): among model-version-v2 no-goalie configurations
    whose tuning-window win log loss is within ``tol`` of the best, take the
    one with the best tuning-window goal NLL (regulation goals, Poisson
    deviance form), so the one scoring model is also as good as possible for
    goals. Goalie parameters come from the goalie stage."""
    cand = [r for r in rows if r["kind"] == "filter" and not r.get("goalie_variant")
            and r.get("model_version") == MODEL_VERSION]
    best_ll = min(r["ll"] for r in cand)
    near = [r for r in cand if r["ll"] <= best_ll + tol]
    pick = min(near, key=lambda r: r["goal_nll"])
    cur = json.loads((C.PARAMS / "ratings_hp.json").read_text())
    hp = dict(pick["config"])
    hp["goalie"] = cur["hp"]["goalie"]
    hp["gk_scale"] = cur["hp"]["gk_scale"]
    cur.update({"hp": hp, "tuning_ll": pick["ll"], "tuning_ll_frozen": pick["ll_frozen"],
                "tuning_goal_nll": pick["goal_nll"], "best_tuning_ll": best_ll,
                "selection_rule": select_final.__doc__.strip(),
                "n_candidates_within_tol": len(near)})
    (C.PARAMS / "ratings_hp.json").write_text(json.dumps(cur, indent=1, default=float))
    print(json.dumps(cur, indent=1, default=float))
    return cur


def tune_elo(rows: list) -> R.EloHP:
    g = S.game_frame()[["gid", "season_end", "home_win"]]
    g = g[g.season_end.isin(TUNE)]
    best, best_ll = None, 9.0
    for K in (4, 6, 8, 10, 12):
        for H in (20, 30, 40, 50):
            for carry in (0.5, 0.6, 0.7, 0.8):
                hp = R.EloHP(K=K, H=H, carry=carry)
                e = R.elo_run(hp).merge(g, on="gid")
                ll = R.logloss(e.p_elo, e.home_win)
                llf = R.logloss(e.p_elo_frozen, e.home_win)
                rows.append({"time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "kind": "elo", "tag": f"K={K} H={H} carry={carry}",
                             "config": asdict(hp), "window": TUNE, "ll": ll,
                             "ll_frozen": llf})
                if ll < best_ll:
                    best, best_ll = hp, ll
    _write_ledger(rows)
    print(f"elo best {best} ll {best_ll:.5f}")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--stage", default="all")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    rows = _ledger()
    t0 = time.time()
    if a.stage in ("all", "elo"):
        ehp = tune_elo(rows)
        (C.PARAMS / "elo_hp.json").parent.mkdir(parents=True, exist_ok=True)
        (C.PARAMS / "elo_hp.json").write_text(json.dumps(
            {"hp": asdict(ehp), "tuned_on": TUNE}, indent=1))
    if a.stage in ("all", "filter"):
        base = R.HP()
        prev = [r for r in rows if r["kind"] == "filter" and not r.get("goalie_variant")
                and r.get("model_version") == MODEL_VERSION]
        if a.resume and prev:           # continue from the best configuration so far
            cfg = dict(min(prev, key=lambda r: r["ll"])["config"])
            cfg["goalie"] = tuple(cfg["goalie"])
            base = R.HP(**cfg)
        # starting point: the v1 optimum (any start is fine; all scores are
        # tuning-window scores)
        base = replace(base, prior_scale=2.0, q_s=5e-5, q_f=3e-5, phi_g=1.25, phi_s=0.6,
                       mu_sd=0.06, h_sd_scale=0.5, integrate=False, window=12) \
            if not (a.resume and prev) else base
        grid = {
            "pace_prior": [0.25, 0.5, 2.0],
            "pace_q": [0.1, 0.25, 0.5, 2.0],
            "prior_scale": [0.6, 1.0, 1.5, 3.0],
            "q_s": [1e-5, 2e-5, 1e-4],
            "q_f": [0.0, 1e-5, 6e-5],
            "phi_s": [0.4, 1.0, 1.5],
            "phi_g": [1.0, 1.5],
            "use_shots": [False],
            "mu_sd": [0.015, 0.03],
            "h_sd_scale": [1.0, 2.0],
            "integrate": [True],
            "ridge": [0.0, 0.2],
            "window": [5, 8],
            "h_halflife": [1.5, 6.0],
        }
        if a.quick:
            grid = {k: v[:2] for k, v in list(grid.items())[:4]}
        best, res = coordinate_search(base, grid, rows)
        out = {"hp": asdict(best), "tuned_on": TUNE, "tuning_ll": res["ll"],
               "tuning_ll_frozen": res["ll_frozen"],
               "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        (C.PARAMS / "ratings_hp.json").write_text(json.dumps(out, indent=1, default=float))
        print(json.dumps(out, indent=1, default=float))
    if a.stage in ("all", "goalie"):
        # goalie-variant parameters on top of the frozen no-goalie optimum;
        # scored on tuning-window games with both starters known.
        # n0 is in Marcel-WEIGHTED shots (weights 3/2/1), so n0 = 12000 is
        # about 4000 shots of regression for one season of history.
        cur = json.loads((C.PARAMS / "ratings_hp.json").read_text())
        cfg = dict(cur["hp"])
        cfg["goalie"] = tuple(cfg["goalie"])
        best = R.HP(**cfg)
        ggrid = {"goalie": [(n0, -0.002, w_in, hl) for n0 in (3000.0, 6000.0, 12000.0, 20000.0)
                            for w_in in (1.0, 3.0) for hl in (6.0, 12.0, 30.0)],
                 "gk_scale": [0.5, 0.75, 1.25]}
        gbest, gres = coordinate_search(best, ggrid, rows, rounds=2, use_goalie=True)
        cur["hp"]["goalie"] = list(gbest.goalie)
        cur["hp"]["gk_scale"] = gbest.gk_scale
        cur["goalie_variant_ll"] = gres["ll"]
        cur["goalie_variant_n"] = gres["n"]
        (C.PARAMS / "ratings_hp.json").write_text(json.dumps(cur, indent=1, default=float))
        print(json.dumps(cur, indent=1, default=float))
    if a.stage == "select":
        select_final(rows)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
