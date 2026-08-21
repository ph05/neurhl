"""NeurHL-H — the ONE-SHOT confirmatory test on 2018-2026 (P7/P8).

This is the only sanctioned use of the SPENT window. It runs at most once, and
only after the tune-window gates have been recorded as passing. The point is
statistical power that the tune window cannot supply: ~11,000 games against
~6,150, which roughly halves the standard error on a paired log-loss
difference. Because the window is genuinely out-of-sample, a significant result
here is stronger evidence than any tune-window number could be.

Protocol, unchanged from the tune-window evaluation:
  * Layer 1 for predict-season T is trained only on seasons < T
  * the thin Layer-3 head is fit only on seasons < T
  * scored against v1 (Elo) by the S1 paired test. Structurally broken seasons
    (A4: 2013 lockout, 2021 COVID division-only) are TRAINING data but are
    never scored.

Refuses to run twice, and refuses to run before the gates pass.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import NOUT, TENSORS  # noqa: E402
from eval.backtest_hier import COLS_DEFAULT, NO_SCORE, nll  # noqa: E402
from train.train_game import elo_features, load_frames  # noqa: E402

RESTATE = list(range(2018, 2027))
CS = (0.01, 0.03, 0.1, 0.3, 1.0)


def main(force: bool = False):
    from scipy import stats
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    out_p = NOUT / "hier_restatement.json"
    assert force or not out_p.exists(), \
        "restatement already run (single-use, P7)"
    tune = NOUT / "hier_result.json"
    assert tune.exists(), "run backtest_hier.py (tune window) first"
    t0 = json.loads(tune.read_text())
    assert t0.get("G1_pass"), \
        ("tune-window G1 must pass before the SPENT window may be touched "
         f"(current pooled {t0.get('neurhl_h')})")
    # G1 as locked spans 2012-2017; a pass computed on fewer seasons does not
    # unlock the window
    assert t0.get("seasons_evaluated", 0) >= 6, \
        (f"G1 covers 6 tune seasons; only {t0.get('seasons_evaluated')} were "
         "evaluated. Complete the tune window first.")

    seasons = list(range(2008, max(RESTATE) + 1))
    gc, _ = load_frames(seasons)
    elo = elo_features(gc)
    have = sorted(int(p.stem.split("_")[-1])
                  for p in TENSORS.glob("proj_team_*.parquet"))
    missing = [s for s in RESTATE if s not in have]
    assert not missing, f"missing Layer-1 projections for {missing}"
    proj = pd.concat([pd.read_parquet(TENSORS / f"proj_team_{s}.parquet")
                      for s in have])
    d = gc.set_index("game_id").join(elo).join(proj)
    d = d.dropna(subset=["elo_logit", "cfpct_h", "cfpct_a"])
    d["y"] = ((d.outcome4 == 0) | (d.outcome4 == 2)).astype(int)
    d["proj_diff"] = d.cfpct_h - d.cfpct_a
    d["proj_cl_diff"] = (d.p_clf_h / (d.p_clf_h + d.p_cla_h).clip(lower=1e-6)
                         - d.p_clf_a / (d.p_clf_a + d.p_cla_a).clip(lower=1e-6))
    d["rest_diff"] = (d.home_rest.clip(upper=7)
                      - d.away_rest.clip(upper=7)) / 7
    d = d.fillna({"rest_diff": 0.0})

    rows, per_season = [], {}
    for T in RESTATE:
        if T in NO_SCORE:          # trains, never scores (A4)
            continue
        tr, te = d[d.season_end < T], d[d.season_end == T]
        if not len(te) or len(tr) < 900:
            continue
        sc = StandardScaler().fit(tr[COLS_DEFAULT])
        best = (9.0, None)
        for C in CS:
            m = LogisticRegression(C=C, max_iter=4000).fit(
                sc.transform(tr[COLS_DEFAULT]), tr.y)
            l = nll(m.predict_proba(sc.transform(tr[COLS_DEFAULT]))[:, 1],
                    tr.y.to_numpy()).mean()
            if l < best[0]:
                best = (l, m)
        p = best[1].predict_proba(sc.transform(te[COLS_DEFAULT]))[:, 1]
        p_elo = 1 / (1 + np.exp(-te.elo_logit.to_numpy()))
        rows.append(pd.DataFrame({"season": T,
                                  "ll_h": nll(p, te.y.to_numpy()),
                                  "ll_v1": nll(p_elo, te.y.to_numpy())}))
        per_season[str(T)] = {"n": int(len(te)),
                              "neurhl_h": float(nll(p, te.y.to_numpy()).mean()),
                              "v1": float(nll(p_elo, te.y.to_numpy()).mean())}
    R = pd.concat(rows, ignore_index=True)

    def stats_for(Rx, label):
        dx = (Rx.ll_h - Rx.ll_v1).to_numpy()
        se = dx.std(ddof=1) / np.sqrt(len(dx))
        t = dx.mean() / se
        cl = Rx.groupby("season").apply(lambda g: (g.ll_h - g.ll_v1).mean(),
                                        include_groups=False)
        cl_t = float(cl.mean() / (cl.std(ddof=1) / np.sqrt(len(cl))))
        return {"label": label, "n_games": int(len(dx)),
                "neurhl_h": float(Rx.ll_h.mean()), "v1": float(Rx.ll_v1.mean()),
                "diff": float(dx.mean()), "se": float(se), "t": float(t),
                "p_two_sided": float(stats.norm.sf(abs(t)) * 2),
                "clustered_t": cl_t,
                "clustered_p": float(stats.t.sf(abs(cl_t), df=len(cl) - 1) * 2),
                "seasons_won": int((cl < 0).sum()),
                "seasons": int(len(cl)),
                "S1_pass": bool(dx.mean() < 0 and abs(t) > 1.96)}

    res = {"window": "2018-2026 (SPENT window, single use)",
           "scored": stats_for(R, "scored seasons"),
           "not_scored_structurally_broken": sorted(NO_SCORE),
           "seasons_scored": sorted(int(s) for s in per_season),
           "per_season": per_season,
           "tune_window_reference": {k: t0.get(k) for k in
                                     ("neurhl_h", "v1", "diff", "p_two_sided")}}
    out_p.write_text(json.dumps(res, indent=1))
    for k in ("scored",):
        v = res[k]
        print(f"{v['label']:22s} n={v['n_games']:5d}  NeurHL-H {v['neurhl_h']:.5f} "
              f"vs v1 {v['v1']:.5f}  diff {v['diff']:+.5f}  t={v['t']:+.2f} "
              f"p={v['p_two_sided']:.4f}  {v['seasons_won']}/{v['seasons']} seasons"
              f"{'  *** S1 PASS' if v['S1_pass'] else ''}")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
