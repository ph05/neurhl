"""M1 tuning: the in-season filter's hyperparameters for goals-only updates.

Plan item M1 (orr/PLAN_1_1.md). Tunes, with ``use_shots=False``, the HP fields
q_s, q_f, q_mu (process noise), phi_g (goal observation scale) and
prior_scale (preseason prior scale), starting from the frozen HP
(orr/output/params/ratings_hp.json) with use_shots=False.

TUNING WINDOW: season_end 2012, 2014-2017 (2013 excluded), the window of
orr/backtest/tune_games.py. Filter started from the team-history preseason
table (ratings.run_filter without pre_override: no market lines exist before
2019), scored with ratings.predict_probs (OT/SO walk-forward within the run).
CONFIRM: season_end 2018 only (2019 is a test season for games), once, after
the configuration is fixed; diagnostic only, it does not change the choice.

Identifiability note (stated before any run): without shots the goal
observations depend on o = so + fo and d = sd + fd only, and the process noise
adds q_s + q_f to each of them per day (also along the pace direction), so
only the SUM q_s + q_f matters. The search therefore scales q_s and q_f
together (multiplier m on the shipped 5e-5 / 3e-5, keeping their ratio); one
logged configuration with the split swapped checks the claim numerically.

Every configuration is logged to orr/experiments/M1/ledger.json.
Run: python3 -m orr.experiments.M1.tune [--confirm]
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from orr import ratings as R
from orr import structural as S
from orr.backtest.games_bt import paired

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "ledger.json"
CONFIG = HERE / "config.json"
TUNE = [2012, 2014, 2015, 2016, 2017]
CONFIRM = [2018]
MAX_CONFIGS = 60
Q_S0, Q_F0 = 5e-5, 3e-5            # shipped split (ratio kept)

GRID = {
    "q_mult": [0.25, 0.5, 0.75, 1.5, 2.0, 3.0],     # q_s = 5e-5 m, q_f = 3e-5 m
    "phi_g": [1.0, 1.25, 1.75, 2.0, 2.5],
    "prior_scale": [0.6, 0.8, 1.25, 1.5, 2.0],
    "q_mu": [0.0, 5e-7, 1e-6, 4e-6, 1e-5],
}
# one step beyond a grid end, tried only if the best value sits at that end
EXTEND = {
    "q_mult": (0.125, 4.5),
    "phi_g": (0.75, 3.0),
    "prior_scale": (0.4, 3.0),
    "q_mu": (None, 2e-5),
}

PRE_REGISTRATION = {
    "written_utc": None,
    "item": "M1 (orr/PLAN_1_1.md): re-tune the in-season filter for goals-only updates",
    "hypothesis": "The filter's hyperparameters were tuned with shots in the update. Re-tuning the "
                  "process noise (q_s, q_f, q_mu), the goal observation scale (phi_g) and the prior "
                  "scale for use_shots=False closes at least 30% of the 0.0033 log-loss gap to NeurHL-G.",
    "start": "frozen ratings_hp.json HP with use_shots=False (the current goals-only filter)",
    "fixed_fields": "every other HP field as frozen (mu_sd, h_sd_scale, pace_prior, pace_q, ridge, "
                    "window, h_halflife, pre_sp, integrate=False, goalie, gk_scale)",
    "tuning_window": TUNE, "excluded": [2013],
    "tuning_metric": "pooled home-win log loss on 2012, 2014-2017, filter from the team-history "
                     "table (ratings.run_filter), probabilities from ratings.predict_probs",
    "identifiability": "only q_s + q_f is identified without shots; q_s and q_f are scaled "
                       "together by q_mult (ratio 5:3 kept), one swapped-split check is logged",
    "grid": GRID, "edge_extension": EXTEND,
    "search": "greedy coordinate descent from the start over q_mult, phi_g, prior_scale, q_mu (in "
              "that order), up to 3 rounds, stopping after a round with no improvement (> 1e-6) "
              "and no new extension; after each round, a parameter whose best value sits at a grid "
              "end gets its one pre-declared extension value added to the grid (tried in the next "
              "round, or in a final extension-only pass after round 3); configurations already "
              "logged are never re-run; hard cap 60 configurations in total (start, split check "
              "and extensions included)",
    "selection": "lowest pooled tuning-window log loss among all logged configurations",
    "confirm": "season_end 2018, once, after the configuration is fixed: candidate vs current "
               "goals-only (paired bootstrap CI); diagnostic only, it does not change the choice",
    "test": {"window": "NeurHL-G gate games 2019-24 (n = 6289): the shipped loop of "
                       "orr/backtest/inseason_bt.py (preseason pipeline ratings with the frozen HP "
                       "as pre_override; OT/SO parameters from the same HP's team-history run), "
                       "no starters, use_shots=False",
             "comparisons": ["candidate vs current goals-only filter (primary, paired bootstrap 95% CI, "
                             "games_bt.paired)", "candidate vs NeurHL-G", "candidate vs NeurHL Elo",
                             "team-history start, candidate vs current (secondary)"],
             "runs": "once, after the configuration is fixed"},
    "accept_rule": "Accept if the pooled gate log loss improves on the current goals-only filter "
                   "with a 95% CI that excludes 0 and the gain is at least 0.0010.",
}


def ledger() -> dict:
    if LEDGER.exists():
        return json.loads(LEDGER.read_text())
    L = {"pre_registration": dict(PRE_REGISTRATION), "configs": []}
    L["pre_registration"]["written_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    save(L)
    return L


def save(L: dict) -> None:
    LEDGER.write_text(json.dumps(L, indent=1, default=float))


def start_hp() -> R.HP:
    return replace(R.load_hp(), use_shots=False)


def make_hp(base: R.HP, q_mult: float, phi_g: float, prior_scale: float, q_mu: float) -> R.HP:
    return replace(base, q_s=Q_S0 * q_mult, q_f=Q_F0 * q_mult, phi_g=phi_g,
                   prior_scale=prior_scale, q_mu=q_mu, use_shots=False)


def _pois_dev(p) -> float:
    lh, la = p.lam_h.to_numpy(), p.lam_a.to_numpy()
    yh, ya = p.reg_h.to_numpy(), p.reg_a.to_numpy()
    return float(np.mean(lh - yh * np.log(lh) + la - ya * np.log(la)))


def evaluate(hp: R.HP, seasons=TUNE) -> tuple[dict, "pd.DataFrame"]:
    """Pooled and per-season scores on ``seasons`` (team-history start)."""
    assert not hp.use_shots
    assert max(seasons) <= 2018 and 2013 not in seasons
    pred = R.run_filter(hp, seasons)
    pred = pred[pred.season_end >= 2010]
    g = S.game_frame()[["gid", "season_end", "home_win", "reg_h", "reg_a"]]
    pr = R.predict_probs(pred, hp).merge(g, on=["gid", "season_end"])
    pr = pr[pr.season_end.isin(seasons)]
    by = {int(V): R.logloss(x.p_home_win, x.home_win) for V, x in pr.groupby("season_end")}
    return ({"ll": R.logloss(pr.p_home_win, pr.home_win), "goal_nll": _pois_dev(pr),
             "brier": R.brier(pr.p_home_win, pr.home_win), "n": int(len(pr)), "by_season": by},
            pr)


def coords(hp: R.HP) -> dict:
    return {"q_mult": round((hp.q_s + hp.q_f) / (Q_S0 + Q_F0), 10), "phi_g": hp.phi_g,
            "prior_scale": hp.prior_scale, "q_mu": hp.q_mu}


def run_config(L: dict, tag: str, hp: R.HP, seen: dict) -> dict:
    k = hp.key()
    if k in seen:
        return seen[k]
    t0 = time.time()
    res, _ = evaluate(hp)
    row = {"i": len(L["configs"]) + 1, "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "tag": tag, "coords": coords(hp), "hp": asdict(hp), "window": TUNE,
           "secs": round(time.time() - t0, 1), **res}
    L["configs"].append(row)
    save(L)
    seen[k] = row
    print(f"{row['i']:>3} {tag:28s} ll {res['ll']:.6f}  gnll {res['goal_nll']:.6f}  "
          f"{coords(hp)}", flush=True)
    return row


def search(L: dict) -> dict:
    seen = {}
    for r in L["configs"]:                       # resume: never re-run a logged configuration
        c = dict(r["hp"]); c["goalie"] = tuple(c["goalie"])
        seen[R.HP(**c).key()] = r
    base = start_hp()
    state = {"best": base, "row": run_config(L, "start (current goals-only)", base, seen)}
    # split check: q_s and q_f swapped (same sum); should equal the start's ll
    run_config(L, "check: q_s/q_f swapped", replace(base, q_s=Q_F0, q_f=Q_S0), seen)

    def scan(grid: dict, label: str) -> bool:
        """One coordinate pass; configurations already logged are skipped."""
        improved = False
        for f, vals in grid.items():
            for v in vals:
                c = coords(state["best"])
                c[f] = v
                hp = make_hp(base, **c)
                if hp.key() in seen:
                    continue
                if len(L["configs"]) >= MAX_CONFIGS:
                    print("configuration cap reached")
                    return improved
                row = run_config(L, f"{label} {f}={v:g}", hp, seen)
                if row["ll"] < state["row"]["ll"] - 1e-6:
                    state["best"], state["row"], improved = hp, row, True
        return improved

    def extend(grid: dict) -> dict:
        """Pre-declared one-step extension where the best value sits at a grid end."""
        c, pend = coords(state["best"]), {}
        for f, (lo, hi) in EXTEND.items():
            vals = sorted(set(grid[f]) | {coords(base)[f]})
            for e, end in ((lo, min(vals)), (hi, max(vals))):
                if e is not None and np.isclose(c[f], end) and e not in grid[f]:
                    grid[f].append(e)
                    pend.setdefault(f, []).append(e)
                    print(f"extend {f} -> {e}")
        return pend

    grid = {k: list(v) for k, v in GRID.items()}
    pending: dict = {}
    for rd in range(3):
        improved = scan(grid, f"r{rd}")
        pending = extend(grid)
        if not improved and not pending:
            pending = {}
            break
    else:
        if pending:                              # extensions found after the last round
            scan(pending, "ext")
    return state["row"]


def confirm(L: dict) -> dict:
    cfg = json.loads(CONFIG.read_text())
    if "confirm_2018" in L:
        print("confirm already run once; not re-running")
        return L["confirm_2018"]
    c = dict(cfg["hp"]); c["goalie"] = tuple(c["goalie"])
    cand = R.HP(**c)
    base = start_hp()
    _, pc = evaluate(cand, CONFIRM)
    _, pb = evaluate(base, CONFIRM)
    m = pc[["gid", "p_home_win", "home_win"]].merge(
        pb[["gid", "p_home_win"]].rename(columns={"p_home_win": "p_base"}), on="gid")
    y = m.home_win.to_numpy()
    out = {"season": 2018, "n": int(len(m)), "candidate": R.logloss(m.p_home_win, y),
           "current_goals_only": R.logloss(m.p_base, y),
           "d_candidate_vs_current": paired(m.p_home_win, m.p_base, y),
           "time": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    L["confirm_2018"] = out
    save(L)
    print(json.dumps(out, indent=1, default=float))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true")
    a = ap.parse_args()
    L = ledger()
    if a.confirm:
        confirm(L)
        return
    t0 = time.time()
    best = search(L)
    allc = L["configs"]
    pick = min(allc, key=lambda r: r["ll"])
    start = allc[0]
    cfg = {"selected_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "rule": PRE_REGISTRATION["selection"], "n_configs": len(allc),
           "tag": pick["tag"], "i": pick["i"], "coords": pick["coords"], "hp": pick["hp"],
           "tuning_ll": pick["ll"], "tuning_ll_start": start["ll"],
           "tuning_gain": start["ll"] - pick["ll"], "by_season": pick["by_season"],
           "by_season_start": start["by_season"]}
    CONFIG.write_text(json.dumps(cfg, indent=1, default=float))
    print(json.dumps({k: v for k, v in cfg.items() if k != "hp"}, indent=1, default=float))
    print(f"done in {time.time() - t0:.0f}s ({len(allc)} configurations)")


if __name__ == "__main__":
    main()
