"""Rookie season projections: engine, translated pre-NHL records, or a blend
(PLAN_NeurHL_1_1 A15). Walk-forward, never 2025 or 2026.

For each season V of the preseason backtest (eval/backtest_unified_season.py:
g1 snapshots trained on seasons < V, every game assembled from opening-night
rows), the engine's per-game goals and assists for every skater in an opening
lineup are averaged over his team's games. For the rookies among them
(first NHL season, fewer than 20 NHL games before it), three projections of
per-game goals and assists are compared with what they actually scored,
weighted by games played:
  engine       the preseason engine
  translated   eval/rookie_priors.py (league factors and shrinkage fitted on < V)
  blend(w)     w x translated + (1 - w) x engine, w in {0.25, 0.5, 0.75}
The weight is chosen on 2012 and 2014-2018 and judged on 2019, 2020 and
2022-2024.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
import rookie_priors as RP  # noqa: E402
from backtest_unified_season import ALIAS, preseason_arrays, preseason_elo  # noqa: E402
from finetune_g_c4 import SNAP, build  # noqa: E402
from train.train_neurhl_g import DEFAULT, Data, predict  # noqa: E402
from common import TENSORS  # noqa: E402

FIT = [2012, 2014, 2015, 2016, 2017, 2018]
JUDGE = [2019, 2020, 2022, 2023, 2024]
WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
OUT = ROOT / "output" / "neurhl_1_1" / "rookie_season_test.json"


def main():
    torch.set_num_threads(4)
    cfg = {**DEFAULT, **json.loads((ROOT / "configs" / "neurhl_g" / "g1.json").read_text())}
    D = Data("train")
    s_all = D.meta.season_end.to_numpy()
    tidx = json.loads((TENSORS / "maps.json").read_text())["team"]
    abbr = {v: ALIAS.get(k, k) for k, v in tidx.items()}
    d = RP.load()
    rk = RP.rookies(d)
    rows = []
    for V in FIT + JUDGE:
        idx = np.where(s_all == V)[0]
        idx = idx[np.argsort(D.meta.date.values[idx], kind="stable")]
        meta = D.meta.iloc[idx].reset_index(drop=True)
        pre, H = preseason_elo(V)
        rh = meta.home_idx.map(lambda t: pre[abbr[int(t)]]).to_numpy()
        ra = meta.away_idx.map(lambda t: pre[abbr[int(t)]]).to_numpy()
        elo_pre = (rh + H - ra) * np.log(10) / 400.0
        P = D.prepare((s_all < V) & (s_all >= cfg["train_from"]))
        Q = preseason_arrays(D, P, idx, elo_pre, dict(D.stats))
        acc = []
        for sd in range(5):
            m = build(D, cfg)
            m.load_state_dict(torch.load(SNAP / f"snap_{V}_{sd}.pt"))
            m.eval()
            acc.append(predict(m, Q, np.arange(len(idx))))
        g_ = np.mean([a["g"] for a in acc], 0)
        a_ = np.mean([a["a"] for a in acc], 0)
        # player ids of the substituted (opening-night) rows: first game of each team
        first = {}
        for j, r in enumerate(meta.itertuples()):
            for s_, t in ((0, r.home_idx), (1, r.away_idx)):
                first.setdefault(int(t), (j, s_))
        per = []
        for t, (fj, fs) in first.items():
            ids = D.A["SKID"][idx[fj], fs]
            msk = D.A["SKM"][idx[fj], fs] > 0
            games = [(j, 0) for j, r in enumerate(meta.itertuples()) if r.home_idx == t] + \
                    [(j, 1) for j, r in enumerate(meta.itertuples()) if r.away_idx == t]
            for k in np.where(msk)[0]:
                per.append({"player_id": int(ids[k]),
                            "eng_g": float(np.mean([g_[j, s, k] for j, s in games])),
                            "eng_a": float(np.mean([a_[j, s, k] for j, s in games]))})
        eng = pd.DataFrame(per).groupby("player_id").mean()
        f = RP.factors(d, V)
        tr = rk[rk.season_end < V]
        tr = tr.merge(RP.translated(d, tr[["player_id", "season_end"]], f), on=["player_id", "season_end"])
        te = rk[(rk.season_end == V) & rk.player_id.isin(eng.index)]
        te = te.merge(RP.translated(d, te[["player_id", "season_end"]], f), on=["player_id", "season_end"])
        pr = RP.fit_predict(tr, te)
        pr = pr.join(eng, on="player_id")
        pr["season"] = V
        rows.append(pr)
        print(V, "rookies in opening lineups:", len(pr), flush=True)
    r = pd.concat(rows, ignore_index=True)
    out = {"n_fit": int(r.season.isin(FIT).sum()), "n_judge": int(r.season.isin(JUDGE).sum())}

    def mae(df, w, st):
        pred = w * df[f"pred_{st}"] + (1 - w) * df[f"eng_{st}"]
        return float(np.average(np.abs(pred - df[st] / df.gp), weights=df.gp))

    for st in ("g", "a"):
        fit, jud = r[r.season.isin(FIT)], r[r.season.isin(JUDGE)]
        wbest = min(WEIGHTS, key=lambda w: mae(fit, w, st))
        out[st] = {"fit_mae": {str(w): mae(fit, w, st) for w in WEIGHTS}, "selected_w": wbest,
                   "judge_mae": {"engine": mae(jud, 0.0, st), "translated": mae(jud, 1.0, st),
                                 "selected": mae(jud, wbest, st)}}
    pts = lambda df, w: (w * (df.pred_g + df.pred_a) + (1 - w) * (df.eng_g + df.eng_a)) * df.gp  # noqa: E731
    jud = r[r.season.isin(JUDGE)]
    out["judge_season_points_mae"] = {str(w): float(np.abs(pts(jud, w) - (jud.g + jud.a)).mean()) for w in WEIGHTS}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    r.to_csv(OUT.with_suffix(".csv"), index=False)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
