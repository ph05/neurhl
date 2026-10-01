"""L1 tuning (2011-2017 only, 2013 excluded) and the 2018 confirmation.

Stage A  choice model: leave-one-season-out choice log loss over the team-games
         of 2011, 2012, 2014-2017 (2011's first 20 games per team are burn-in).
Stage B  the in-season filter (team-history start, frozen HP, goals + shots):
         walk-forward choice-model fits (seasons < V, 2013 and burn-in excluded),
         game log loss on 2012, 2014-2017 for three ways of using the expected
         starter; the mix and the actual starters are reported as bounds.
Confirm  2018: the chosen configuration vs the mix (game log loss, paired CI)
         and the choice model's 2018 scores (fitted on 2011-2017).

Every configuration is logged to ledger.json. Run: python3 -m orr.experiments.L1.tune
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from orr import ratings as R
from orr import structural as S
from orr.backtest.games_bt import paired
from orr.experiments.L1 import lfilter as LF
from orr.experiments.L1 import starters as ST

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "ledger.json"
TUNE_CHOICE = [2011, 2012, 2014, 2015, 2016, 2017]
TUNE_GAMES = [2012, 2014, 2015, 2016, 2017]
BURN = 20

F0 = ["lsh_l"]
F1 = F0 + ["last", "last_b2b", "lstreak", "app_yday"]
F2 = F1 + ["lgsa", "pulled", "relief"]
F3 = list(ST.CAND_FEATS)

PRE_REGISTRATION = {
    "written_utc": None,
    "hypothesis": "A walk-forward model of who starts improves in-season log loss over the start-share mix.",
    "tuning_windows": {"choice_model": TUNE_CHOICE, "games": TUNE_GAMES, "confirm": [2018],
                       "excluded": [2013], "burn_in": f"2011: each team's first {BURN} games"},
    "stage_a_selection": "lowest leave-one-season-out pooled choice log loss",
    "stage_b_selection": "lowest pooled game log loss on 2012, 2014-2017 among the 'pred' variants",
    "walk_forward": "for every season V the choice model's coefficients and m_other are re-estimated "
                    "on all seasons < V with starters (2013 and the 2011 burn-in excluded); the "
                    "configuration (features, half-lives, penalty, offset use) is fixed by tuning",
    "coverage": "predicted offsets replace the mix on exactly the games where the known-starter "
                "variant uses actual starters (both starters recorded in the box scores, goalie GLM "
                "available); elsewhere every arm uses the mix. 2024 is covered only to 2023-11-07.",
    "test": {"window": "NeurHL-G gate games 2019-24 (n = 6289), shipped loop of orr/backtest/inseason_bt.py "
                       "(preseason pipeline ratings, OT/SO parameters from the same arm's team-history run)",
             "primary": "goals+shots (frozen HP): pred vs mix, paired bootstrap 95% CI (games_bt.paired)",
             "secondary": ["goals-only (use_shots=False): pred vs mix",
                           "pred vs actual starters (upper bound)",
                           "hybrid: predicted offsets for the forecast, actual starters in the update",
                           "NeurHL-G and NeurHL Elo on the same games"],
             "accept_rule": "Accept if the improvement over the mix has a 95% CI that excludes 0 "
                            "(primary variant; the CI's upper end below 0)",
             "runs": "once, after the configuration is fixed"},
}


def ledger() -> dict:
    if LEDGER.exists():
        return json.loads(LEDGER.read_text())
    L = {"pre_registration": dict(PRE_REGISTRATION), "stage_a": [], "stage_b": []}
    L["pre_registration"]["written_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return L


def save(L):
    LEDGER.write_text(json.dumps(L, indent=1, default=float))


def train_mask(sit: pd.DataFrame, seasons) -> np.ndarray:
    m = sit.season_end.isin(seasons) & (sit.season_end != 2013)
    m &= ~((sit.season_end == 2011) & (sit.k_season < BURN))
    return m.to_numpy()


def gdiff_r2(e: pd.DataFrame) -> dict:
    x = e.dropna(subset=["gdiff_act"])
    r = x.gdiff_act - x.egd
    return {"mse_x1e6": float(1e6 * np.mean(r ** 2)),
            "r2": float(1 - np.mean(r ** 2) / np.mean((x.gdiff_act - x.gdiff_act.mean()) ** 2))}


def stage_a(L: dict, goalie_key) -> dict:
    """Leave-one-season-out choice log loss."""
    configs = [("A0_mix", None, (4.0, 30.0), None), ("A1_F0", F0, (4.0, 30.0), 1.0),
               ("A2_F1", F1, (4.0, 30.0), 1.0), ("A3_F2", F2, (4.0, 30.0), 1.0),
               ("A4_F3", F3, (4.0, 30.0), 1.0), ("A5_F3_h3_15", F3, (3.0, 15.0), 1.0),
               ("A6_F3_h6_60", F3, (6.0, 60.0), 1.0)]
    done = {c["id"] for c in L["stage_a"]}
    for cid, feats, (hs, hl), lam in configs:
        if cid in done:
            continue
        L["stage_a"].append(_score_a(cid, feats, hs, hl, lam, goalie_key))
        save(L)
    best = min((c for c in L["stage_a"] if c["feats"] is not None), key=lambda c: c["logloss"])
    # penalty check at the best half-lives
    cid = "A7_best_lam100"
    if cid not in {c["id"] for c in L["stage_a"]}:
        L["stage_a"].append(_score_a(cid, best["feats"], best["hs"], best["hl"], 100.0, goalie_key))
        save(L)
    best = min((c for c in L["stage_a"] if c["feats"] is not None), key=lambda c: c["logloss"])
    return best


def _score_a(cid, feats, hs, hl, lam, goalie_key) -> dict:
    t0 = time.time()
    sit = ST.situations(goalie_key, hs, hl)
    ps, rows = [], []
    for V in TUNE_CHOICE:
        te = (sit.season_end == V).to_numpy() & train_mask(sit, [V])
        sv = sit[te]
        if feats is None:
            p = ST.mix_probs(sv)
            m_other = float(sit[train_mask(sit, [s for s in TUNE_CHOICE if s != V])
                                & (sit.is_other == 1).to_numpy() & (sit.chosen == 1).to_numpy()].t_act.mean())
        else:
            mod = ST.fit_clogit(sit[train_mask(sit, [s for s in TUNE_CHOICE if s != V])], feats, lam)
            p = ST.predict(sv, mod)
            m_other = mod["m_other"]
        ps.append(sv.assign(p=p))
        e = ST.expected_gdiff(sv, p, m_other)
        rows.append({"season": V, **ST.choice_scores(sv, p), **gdiff_r2(e)})
    allp = pd.concat(ps)
    sc = ST.choice_scores(allp, allp.p.to_numpy())
    out = {"id": cid, "feats": feats, "hs": hs, "hl": hl, "lam": lam, **sc,
           "gdiff": {"mse_x1e6_mean": float(np.mean([r["mse_x1e6"] for r in rows])),
                     "r2_mean": float(np.mean([r["r2"] for r in rows]))},
           "by_season": rows, "secs": round(time.time() - t0, 1)}
    if feats is not None:     # coefficients of the fit on all six tuning seasons
        mod = ST.fit_clogit(sit[train_mask(sit, TUNE_CHOICE)], feats, lam)
        out["beta_all"] = dict(zip(feats + ST.OTHER_FEATS, map(float, mod["beta"])))
    print(f"{cid:16s} ll {out['logloss']:.4f} acc {out['acc']:.3f} gdiff r2 {out['gdiff']['r2_mean']:.3f}"
          f"  ({out['secs']}s)", flush=True)
    return out


def walkforward_egd(goalie_key, cfg: dict, seasons, ref_kind: str = "ewma") -> tuple[pd.DataFrame, dict]:
    """Expected starter gdiff per game (egd_h, egd_a) for the given seasons,
    each from a choice model fitted on seasons < V only (``ref_kind``: see
    starters.expected_gdiff)."""
    sit = ST.situations(goalie_key, cfg["hs"], cfg["hl"])
    frames, info = [], {}
    for V in seasons:
        tr = train_mask(sit, range(2011, V))
        if not tr.any():
            continue
        mod = ST.fit_clogit(sit[tr], cfg["feats"], cfg["lam"])
        sv = sit[(sit.season_end == V).to_numpy()]
        p = ST.predict(sv, mod)
        e = ST.expected_gdiff(sv, p, mod["m_other"], ref_kind)
        frames.append(e)
        info[V] = {"n_train": mod["n"], "m_other": mod["m_other"], "converged": mod["converged"],
                   "choice": ST.choice_scores(sv, p)}
    e = pd.concat(frames, ignore_index=True)
    w = e.pivot_table(index="gid", columns="side", values="egd")
    w = w.rename(columns={0: "egd_h", 1: "egd_a"}).reset_index()
    return w, info


def score_games(pred: pd.DataFrame, hp, seasons) -> tuple[pd.DataFrame, dict]:
    p = R.predict_probs(pred, hp)
    g = S.game_frame()[["gid", "home_win"]]
    m = p.merge(g, on="gid")
    m = m[m.season_end.isin(seasons)]
    by = {int(V): R.logloss(x.p_home_win, x.home_win) for V, x in m.groupby("season_end")}
    return m, {"pooled": R.logloss(m.p_home_win, m.home_win), "by_season": by, "n": int(len(m))}


def stage_b(L: dict, hp, cfg: dict) -> dict:
    egd, info = walkforward_egd(hp.goalie, cfg, range(2012, 2018))
    arms = [("B0_mix", dict(pred_src="mix")), ("B1_actual", dict(pred_src="actual")),
            ("B2_pred_gk_lam1", dict(pred_src="pred", ctx_kind="gk", lam=1.0)),
            ("B3_pred_gk_lam0.5", dict(pred_src="pred", ctx_kind="gk", lam=0.5)),
            ("B4_pred_ctx_lam1", dict(pred_src="pred", ctx_kind="ctx", lam=1.0))]
    done = {c["id"]: c for c in L["stage_b"]}
    preds = {}
    for cid, kw in arms:
        pr = LF.run_filter_gk(hp, TUNE_GAMES, egd=egd, **kw)
        m, sc = score_games(pr, hp, TUNE_GAMES)
        preds[cid] = m
        if cid not in done:
            rec = {"id": cid, **kw, **sc}
            L["stage_b"].append(rec)
            save(L)
        print(f"{cid:20s} {sc['pooled']:.5f} " + " ".join(f"{k}:{v:.4f}" for k, v in sc["by_season"].items()),
              flush=True)
    cand = [c for c in L["stage_b"] if c["id"] in ("B2_pred_gk_lam1", "B3_pred_gk_lam0.5", "B4_pred_ctx_lam1")]
    best = min(cand, key=lambda c: c["pooled"])
    # paired diagnostics on the tuning games (best vs mix, actual vs mix)
    y = preds["B0_mix"].home_win.to_numpy()
    L["stage_b_diag"] = {
        "best_vs_mix": paired(preds[best["id"]].p_home_win, preds["B0_mix"].p_home_win, y),
        "actual_vs_mix": paired(preds["B1_actual"].p_home_win, preds["B0_mix"].p_home_win, y),
        "choice_walkforward": info}
    # hybrid diagnostic (reported, not a selection candidate)
    kw = {k: best[k] for k in ("ctx_kind", "lam")}
    pr = LF.run_filter_gk(hp, TUNE_GAMES, egd=egd, pred_src="pred", upd_src="actual", **kw)
    m, sc = score_games(pr, hp, TUNE_GAMES)
    L["stage_b_diag"]["hybrid"] = {**sc, "vs_mix": paired(m.p_home_win, preds["B0_mix"].p_home_win, y)}
    save(L)
    print("best", best["id"], "diag", json.dumps(L["stage_b_diag"]["best_vs_mix"]),
          "hybrid", round(sc["pooled"], 5), flush=True)
    return best


def confirm_2018(L: dict, hp, cfg: dict, best_b: dict):
    egd, info = walkforward_egd(hp.goalie, cfg, range(2012, 2019))
    kw = {k: best_b[k] for k in ("ctx_kind", "lam")}
    pm = LF.run_filter_gk(hp, [2018], pred_src="mix")
    pp = LF.run_filter_gk(hp, [2018], egd=egd, pred_src="pred", **kw)
    pa = LF.run_filter_gk(hp, [2018], pred_src="actual")
    mm, sm = score_games(pm, hp, [2018])
    mp, sp = score_games(pp, hp, [2018])
    ma, sa = score_games(pa, hp, [2018])
    y = mm.home_win.to_numpy()
    L["confirm_2018"] = {"mix": sm["pooled"], "pred": sp["pooled"], "actual": sa["pooled"],
                         "n": sm["n"], "pred_vs_mix": paired(mp.p_home_win, mm.p_home_win, y),
                         "actual_vs_mix": paired(ma.p_home_win, mm.p_home_win, y),
                         "choice_2018": info[2018]["choice"]}
    save(L)
    print("confirm 2018", json.dumps(L["confirm_2018"], default=float), flush=True)


EXT_NOTE = (
    "Stage B extension, added after stage B and the 2018 confirmation of B4. Diagnostic on "
    "2014-2017 (tuning seasons): E[gdiff] is calibrated (OLS slope of actual gdiff on E[gdiff] 0.99) "
    "but most of its variance is in-season talent drift (the candidate's CURRENT talent minus the "
    "EWMA reference of past starters' talents): the plain start-share mix's E[gdiff] already has "
    "slope 0.98 and R2 0.16, and the filter sees that drift through goals against. The live code "
    "(inseason.starter_diffs) defines the starter offset against the start-share mix's talent with "
    "current talents, so B5-B7 use E_model[t] - E_mix[t]. Selection rule unchanged: lowest pooled "
    "2012/2014-17 log loss among ALL pred variants (B2-B7). 2018 has been used once (for B4) and is "
    "NOT used again: if B5-B7 win, the final configuration has no separate 2018 confirmation.")


def stage_b_ext(L: dict, hp, cfg: dict) -> dict:
    L["stage_b_extension_note"] = EXT_NOTE
    save(L)
    egd_m, _ = walkforward_egd(hp.goalie, cfg, range(2012, 2018), ref_kind="mix")
    arms = [("B5_pred_mixref_gk_lam1", dict(ctx_kind="gk", lam=1.0)),
            ("B6_pred_mixref_ctx_lam1", dict(ctx_kind="ctx", lam=1.0))]
    done = {c["id"] for c in L["stage_b"]}
    pm = LF.run_filter_gk(hp, TUNE_GAMES, pred_src="mix")
    mm, _ = score_games(pm, hp, TUNE_GAMES)
    y = mm.home_win.to_numpy()

    def run(cid, kw):
        pr = LF.run_filter_gk(hp, TUNE_GAMES, egd=egd_m, pred_src="pred", **kw)
        m, sc = score_games(pr, hp, TUNE_GAMES)
        if cid not in done:
            L["stage_b"].append({"id": cid, "pred_src": "pred", "ref_kind": "mix", **kw, **sc,
                                 "vs_mix": paired(m.p_home_win, mm.p_home_win, y)})
            save(L)
        print(f"{cid:26s} {sc['pooled']:.5f} " + " ".join(f"{k}:{v:.4f}" for k, v in sc["by_season"].items()),
              flush=True)
        return sc
    sc = {cid: run(cid, kw) for cid, kw in arms}
    kbest = min(sc, key=lambda c: sc[c]["pooled"])
    kind = dict(arms)[kbest]["ctx_kind"]
    run(f"B7_pred_mixref_{kind}_lam0.5", dict(ctx_kind=kind, lam=0.5))
    best = min((c for c in L["stage_b"] if c["id"][:2] in ("B2", "B3", "B4", "B5", "B6", "B7")),
               key=lambda c: c["pooled"])
    L["chosen"] = {"choice_model": L["chosen_choice_model"],
                   "offsets": {"id": best["id"], "ctx_kind": best["ctx_kind"], "lam": best["lam"],
                               "ref_kind": best.get("ref_kind", "ewma")},
                   "note": "selected among B2-B7 on 2012/2014-17; see stage_b_extension_note"}
    save(L)
    print("chosen after extension", best["id"], flush=True)
    return best


def main():
    t0 = time.time()
    L = ledger()
    save(L)
    hp = R.load_hp()
    best_a = stage_a(L, hp.goalie)
    cfg = {k: best_a[k] for k in ("feats", "hs", "hl", "lam")}
    L["chosen_choice_model"] = {"id": best_a["id"], **cfg}
    save(L)
    print("chosen choice model", best_a["id"], flush=True)
    best_b = stage_b(L, hp, cfg)
    L["chosen"] = {"choice_model": L["chosen_choice_model"],
                   "offsets": {k: best_b[k] for k in ("id", "ctx_kind", "lam")}}
    save(L)
    confirm_2018(L, hp, cfg, best_b)
    L["tuning_secs"] = round(time.time() - t0)
    save(L)


if __name__ == "__main__":
    import sys
    if "--ext" in sys.argv:
        L_ = ledger()
        hp_ = R.load_hp()
        cfg_ = {k: L_["chosen_choice_model"][k] for k in ("feats", "hs", "hl", "lam")}
        stage_b_ext(L_, hp_, cfg_)
    else:
        main()
