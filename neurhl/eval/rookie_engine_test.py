"""Rookie priors inside the engine (PLAN_NeurHL_1_1 A15), walk-forward.

For each season V (2014-2020, 2022-2024; never 2025 or 2026), the season's
g1 snapshots (five seeds, trained on seasons < V; saved by
eval/backtest_unified_season.py) predict every game of V twice:
  frozen   the master tensor as built (rookies' baselines shrink toward
           their position's average)
  rookie   the same, except each rookie's scoring baselines shrink toward
           his rookie prior (eval/rookie_priors.py, fitted on seasons < V):
           shots, attempts and individual xG per 60 scale with the prior's
           goals per game; assists per 60 with its assists per game. His own
           NHL rates take over as his games accumulate, exactly as the
           shrinkage already does.
Rookies are players in their first NHL season with fewer than 20 NHL games
before it. Reported: Poisson log loss of rookie skater-games (goals, assists,
shots), and home-win log loss of the engine on games with a rookie.
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
from data.build_g_tensors import SK_BASE  # noqa: E402
from finetune_g_c4 import SNAP, build  # noqa: E402
from train.train_neurhl_g import DEFAULT, Data, predict  # noqa: E402

SEASONS = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2022, 2023, 2024]
MODE = "finishing" if "--finishing" in sys.argv else "relative" if "--relative" in sys.argv else "absolute"
OUT = ROOT / "output" / "neurhl_1_1" / f"rookie_engine_test_{MODE}.json"


def pois_ll(mu, y):
    mu = np.clip(mu, 1e-6, None)
    return mu - y * np.log(mu)


def nll(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main():
    torch.set_num_threads(2)
    cfg = {**DEFAULT, **json.loads((ROOT / "configs" / "neurhl_g" / "g1.json").read_text())}
    D = Data("train")
    A, meta, names = D.A, D.meta, D.names
    pri = names["priors"]
    sf, sb, st = names["sk_feat"], names["sk_base"], names["sk_tgt"]
    s_all = meta.season_end.to_numpy()
    y_all = np.isin(meta.outcome4.to_numpy(), [0, 2]).astype(float)
    d = RP.load()
    rk_all = RP.rookies(d)
    rows, grows = [], []
    for V in SEASONS:
        f = RP.factors(d, V)
        tr = rk_all[rk_all.season_end < V]
        tr = tr.merge(RP.translated(d, tr[["player_id", "season_end"]], f), on=["player_id", "season_end"])
        te = rk_all[rk_all.season_end == V]
        te = te.merge(RP.translated(d, te[["player_id", "season_end"]], f), on=["player_id", "season_end"])
        pr = RP.fit_predict(tr, te).set_index("player_id")
        trp = RP.fit_predict(tr, tr)                    # earlier rookies' own priors, for the average
        mean_pred = {(p_, s_): float(trp.loc[trp.pos == p_, f"pred_{s_}"].mean()) for p_ in (0, 1) for s_ in ("g", "a")}
        P = D.prepare((s_all < V) & (s_all >= cfg["train_from"]))
        idx = np.where(s_all == V)[0]
        SKB0 = P["SKB"][idx].copy()
        SKB1 = SKB0.copy()
        ids = A["SKID"][idx]
        mask = (A["SKM"][idx] > 0) & np.isin(ids, pr.index.to_numpy())
        raw = A["SK"][idx]
        jj = np.argwhere(mask)
        for (g_, s_, k_) in jj:
            pid = int(ids[g_, s_, k_])
            r = pr.loc[pid]
            p = int(r.pos)
            # the prior's per-60 rates, at the position's baseline ice time
            if MODE == "finishing":     # shots stay at the prior; goals enter through finishing
                scale_g = float(np.clip(r.pred_g / mean_pred[(p, "g")], 0.5, 2.0))
                scale_a = float(np.clip(r.pred_a / mean_pred[(p, "a")], 0.5, 2.0))
                new_p = {"g_per_sog": pri["g_per_sog"][p] * scale_g, "a60": pri["a60"][p] * scale_a}
            elif MODE == "relative":      # keep the trained level: multiply the position prior by the
                # rookie's prior relative to the average earlier rookie of his position
                scale_g = r.pred_g / mean_pred[(p, "g")]
                scale_a = r.pred_a / mean_pred[(p, "a")]
                new_p = {"isog60": pri["isog60"][p] * scale_g, "iatt60": pri["iatt60"][p] * scale_g,
                         "ixg60": pri["ixg60"][p] * scale_g, "a60": pri["a60"][p] * scale_a}
            else:
                toi = pri["toi_ev"][p] + pri["toi_pp"][p] + pri["toi_sh"][p]
                g60 = r.pred_g * 60.0 / toi
                a60 = r.pred_a * 60.0 / toi
                scale_g = g60 / (pri["isog60"][p] * pri["g_per_sog"][p])
                new_p = {"isog60": pri["isog60"][p] * scale_g, "iatt60": pri["iatt60"][p] * scale_g,
                         "ixg60": pri["ixg60"][p] * scale_g, "a60": a60}
            for name, val in new_p.items():
                col, _, kk = SK_BASE[name]
                tag = col.rsplit("_", 1)[1]
                n = raw[g_, s_, k_, sf.index(f"neff_{tag}")]
                x = raw[g_, s_, k_, sf.index(col)]
                n = 0.0 if not np.isfinite(n) else n
                x = val if not np.isfinite(x) else x
                SKB1[g_, s_, k_, sb.index(f"b_{name}")] = (n * x + kk * val) / (n + kk)
        outs = {}
        for tag, skb in (("frozen", SKB0), ("rookie", SKB1)):
            P2 = dict(P)
            P2["SKB"] = P["SKB"].copy()
            P2["SKB"][idx] = skb
            acc = []
            for sd in range(5):
                m = build(D, cfg)
                m.load_state_dict(torch.load(SNAP / f"snap_{V}_{sd}.pt"))
                m.eval()
                acc.append(predict(m, P2, idx))
            outs[tag] = {k: np.mean([a[k] for a in acc], 0) for k in ("g", "a", "isog", "p_home_win")}
        Y = A["SKY"][idx]
        for key, tk in (("g", "g"), ("a", "a"), ("isog", "isog")):
            y = Y[..., st.index(tk)][mask]
            ok = np.isfinite(y)
            rows.append({"season": V, "stat": key, "n": int(ok.sum()),
                         "frozen": float(pois_ll(outs["frozen"][key][mask][ok], y[ok]).mean()),
                         "rookie": float(pois_ll(outs["rookie"][key][mask][ok], y[ok]).mean())})
        has = mask.any(axis=(1, 2))
        yg = y_all[idx][has]
        grows.append({"season": V, "games": int(has.sum()),
                      "frozen": float(nll(outs["frozen"]["p_home_win"][has], yg).mean()),
                      "rookie": float(nll(outs["rookie"]["p_home_win"][has], yg).mean())})
        print(V, "rookies", len(pr), "rookie skater-games", int(mask.sum()), flush=True)
    r = pd.DataFrame(rows)
    g = pd.DataFrame(grows)
    out = {"by_stat": {}, "games": {}}
    for stt, gr in r.groupby("stat"):
        w = gr.n
        fr, rk = np.average(gr.frozen, weights=w), np.average(gr.rookie, weights=w)
        out["by_stat"][stt] = {"n": int(w.sum()), "frozen": float(fr), "rookie": float(rk),
                               "rel_diff": float(rk / fr - 1),
                               "seasons_better": int((gr.rookie < gr.frozen).sum()), "of": int(len(gr))}
    w = g.games
    out["games"] = {"n": int(w.sum()), "frozen": float(np.average(g.frozen, weights=w)),
                    "rookie": float(np.average(g.rookie, weights=w)),
                    "seasons_better": int((g.rookie < g.frozen).sum()), "of": int(len(g))}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
