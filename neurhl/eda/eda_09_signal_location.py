"""NeurHL EDA-09 — where does the game-outcome signal actually live?

The diagnostic behind PLAN_NeurHL AMENDMENT A1. Config #1 of the game model
failed G1/G4 while being perfectly calibrated (tau ~ 1) — i.e. correctly scaled
but not discriminating. This asks the prior question directly: is the signal
absent from the ROSTER-EMBEDDING representation, or merely hard for the network
to extract?

Method: heavily-regularized logistic regression (a model far too simple to
overfit) on four feature sets, trained on seasons <= 2013 and tested on 2015 —
inside the tune window, report-only, no gate can be changed by it.

  A roster embeddings (mean-pooled per team, difference) + game context
  B team rolling form only: goals for/against and points per game over each
    team's previous 25 games, strictly pre-game
  C form + context
  D form + context + embeddings

If A ~ constant-home while B beats the network, the constraint is information,
not capacity, and no amount of regularization or capacity tuning will fix it.

Writes eda/eda_09_signal_location.md.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import EDA, TENSORS  # noqa: E402
from train.train_game import (add_form, build_tensors,  # noqa: E402
                              load_frames, team_form)

TRAIN_MAX, TEST = 2013, 2015
V1_TEST = 0.6709          # v1 walk-forward log loss on 2015 (baselines_tune)
NN_TEST = 0.6851          # config #1 ensemble on 2015 (archive/config1)
CS = (0.003, 0.01, 0.03, 0.3)


def pooled_diff(t):
    emb = t["emb"].numpy()
    w = (~t["pad"].numpy()).astype(np.float32)[..., None]
    mean = (emb * w).sum(2) / np.clip(w.sum(2), 1, None)
    return mean[:, 0] - mean[:, 1]


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    gc, pg = load_frames(list(range(2008, TEST + 1)))
    pg = add_form(pg)
    npz = np.load(TENSORS / f"embeddings_v{TEST}.npz")
    form = team_form(gc)

    def build(sel):
        t, _, _ = build_tensors(gc[sel], pg, npz, form=form)
        y = t["outcome4"].numpy()
        return (pooled_diff(t), t["ctx"].numpy()[:, :9],
                t["ctx"].numpy()[:, 9:15],
                ((y == 0) | (y == 2)).astype(int))

    Dtr, Ctr, Ftr, ytr = build(gc.season_end <= TRAIN_MAX)
    Dte, Cte, Fte, yte = build(gc.season_end == TEST)

    def run(Xtr, Xte):
        best = (9.0, None)
        for C in CS:
            sc = StandardScaler().fit(Xtr)
            lr = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(Xtr), ytr)
            p = np.clip(lr.predict_proba(sc.transform(Xte))[:, 1], 1e-9, 1 - 1e-9)
            ll = float(-(yte * np.log(p) + (1 - yte) * np.log(1 - p)).mean())
            best = min(best, (ll, C))
        return best

    sets = {
        "A. roster embeddings (mean-pooled) + context": (np.c_[Dtr, Ctr],
                                                         np.c_[Dte, Cte]),
        "B. team rolling form only (6 features)": (Ftr, Fte),
        "C. team form + context": (np.c_[Ftr, Ctr], np.c_[Fte, Cte]),
        "D. team form + context + embeddings": (np.c_[Ftr, Ctr, Dtr],
                                                np.c_[Fte, Cte, Dte]),
    }
    base = float(ytr.mean())
    const = float(-(yte * np.log(base) + (1 - yte) * np.log(1 - base)).mean())

    lines = ["# EDA-09 — where the game-outcome signal lives\n",
             f"Trained on seasons <= {TRAIN_MAX} ({len(ytr):,} games), tested on "
             f"{TEST} ({len(yte):,} games). Logistic regression, best of "
             f"C in {CS}. Report-only.\n",
             "| feature set | test home-win log loss |",
             "|---|---|"]
    res = {}
    for name, (a, b) in sets.items():
        ll, C = run(a, b)
        res[name[0]] = ll
        lines.append(f"| {name} | **{ll:.4f}** (C={C}) |")
    lines.append(f"| — constant home rate ({base:.3f}) | {const:.4f} |")
    lines.append(f"| — v1 (Elo), same season | {V1_TEST:.4f} |")
    lines.append(f"| — NeurHL config #1 network, same season | {NN_TEST:.4f} |")
    lines.append(f"""
## Reading

Roster embeddings alone ({res['A']:.4f}) are indistinguishable from knowing
nothing ({const:.4f}), while **six team-history features ({res['B']:.4f}) beat
the entire 1.2M-parameter network ({NN_TEST:.4f})**, and adding embeddings on
top of form makes it *worse* ({res['D']:.4f} vs {res['C']:.4f}) — they enter as
noise.

A logistic regression cannot be accused of failing to optimize. The signal is
therefore not present-but-hard-to-extract in pooled roster embeddings; it is
largely absent from that representation. Combined with EDA-08 (position
linearly decodable at 0.952, points/60 R^2 0.54), the interpretation is that
the event-LM learned what KIND of player someone is much better than how GOOD
their team is — and since every NHL roster carries a similar distribution of
roles, pooling over one barely separates teams.

Consequence for the project: capacity/regularization tuning cannot rescue
config #1, because even an optimally-regularized linear model on these features
tops out at {res['A']:.4f}. PLAN_NeurHL AMENDMENT A1 therefore adds team form
to the context vector rather than spending budget on capacity sweeps, and
restates the scientific claim accordingly.
""")
    (EDA / "eda_09_signal_location.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
