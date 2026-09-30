"""NeurHL 1.2 R1: does the preseason convention bias season-level box-score levels?
(PLAN_NeurHL_1_2 A1, finding 1.)

For each backtest season V (g1 snapshots trained on seasons < V, cached by
eval/backtest_unified_season.py), every game is assembled from opening-night rows
(eval/backtest_unified_season.preseason_arrays) in two conventions:
  P  the current one: season-progress inputs keep their opening-night values
  C  season-progress inputs advanced per game: days_in and tm_gp_season are the
     game's own; skater gp_season, gp_with_team and gp_career + (team game number - 1);
     skater days_since and goalie gk_days_since = the team's rest days before the game
League-season and team-season means of PP opportunities, PP minutes, shots, xG and
regulation goals are compared with what happened, and win-probability log loss with
the g2027_v1 fallback stack (Elo + NeurHL-G), as in the C2 backtest.
Writes output/neurhl_1_2/season_convention.json.
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
from backtest_unified_season import ALIAS, preseason_arrays, preseason_elo  # noqa: E402
from finetune_g_c4 import SNAP, build  # noqa: E402
from train.train_neurhl_g import DEFAULT, Data, predict  # noqa: E402
from common import TENSORS  # noqa: E402

FIT = [2012, 2014, 2015, 2016, 2017, 2018]
JUDGE = [2019, 2020, 2022, 2023, 2024]
STATS = {"pp_opps": "pp_opps", "pp_m": "pp_m", "sogf": "sogf", "xgf": "xgf_all", "goals": "gf_reg"}
OUT = ROOT / "output" / "neurhl_1_2" / "season_convention.json"


def advance(D, Q, idx, stats):
    """Convention C: overwrite the season-progress inputs of Q (standardised) per game."""
    A, n = D.A, D.names
    meta = D.meta.iloc[idx].reset_index(drop=True)
    C = {k: v.copy() for k, v in Q.items()}
    z = lambda key, col, raw: np.clip((raw - stats[key][0][col]) / stats[key][1][col], -8, 8)  # noqa: E731
    ctx, tf, sf, gf = n["ctx"], n["tm_feat"], n["sk_feat"], n["gk_feat"]
    idn = ctx.index("days_in")
    C["CTX"][:, idn] = z("CTX", idn, np.nan_to_num(A["CTX"][idx, idn]))
    itg, irest = tf.index("tm_gp_season"), tf.index("rest")
    s_cols = [sf.index(c) for c in ("gp_season", "gp_with_team", "gp_career")]
    i_ds, i_gds = sf.index("days_since"), gf.index("gk_days_since")
    first, gno = {}, {}
    for j, r in enumerate(meta.itertuples()):
        for s, t in ((0, r.home_idx), (1, r.away_idx)):
            first.setdefault(int(t), (j, s))
    for j, r in enumerate(meta.itertuples()):
        for s, t in ((0, r.home_idx), (1, r.away_idx)):
            t = int(t)
            k = gno.get(t, 0)
            gno[t] = k + 1
            fj, fs = first[t]
            C["TM"][j, s, itg] = z("TM", itg, float(k))
            rest = A["TM"][idx[j], s, irest]
            rest = float(np.clip(rest, 0, 365)) if np.isfinite(rest) else 365.0
            raw = A["SK"][idx[fj], fs]
            for c in s_cols:
                C["SK"][j, s, :, c] = z("SK", c, np.nan_to_num(raw[:, c]) + k)
            if k > 0:
                C["SK"][j, s, :, i_ds] = z("SK", i_ds, rest)
                C["GK"][j, s, i_gds] = z("GK", i_gds, rest)
            m = Q["SKM"][j, s] == 0
            C["SK"][j, s, m] = Q["SK"][j, s, m]
    return C


def main():
    torch.set_num_threads(4)
    cfg = {**DEFAULT, **json.loads((ROOT / "configs" / "neurhl_g" / "g1.json").read_text())}
    fb = json.loads((ROOT / "checkpoints" / "g" / "g2027_v1" / "bundle.json").read_text())["stack"]["fallback"]
    D = Data("train")
    s_all = D.meta.season_end.to_numpy()
    tidx = json.loads((TENSORS / "maps.json").read_text())["team"]
    abbr = {v: ALIAS.get(k, k) for k, v in tidx.items()}
    tt = D.names["tm_tgt"]
    league, team, ll = [], [], []
    for V in FIT + JUDGE:
        idx = np.where(s_all == V)[0]
        idx = idx[np.argsort(D.meta.date.values[idx], kind="stable")]
        meta = D.meta.iloc[idx].reset_index(drop=True)
        pre, H = preseason_elo(V)
        rh = meta.home_idx.map(lambda t: pre[abbr[int(t)]]).to_numpy()
        ra = meta.away_idx.map(lambda t: pre[abbr[int(t)]]).to_numpy()
        elo = (rh + H - ra) * np.log(10) / 400.0
        P = D.prepare((s_all < V) & (s_all >= cfg["train_from"]))
        stats = dict(D.stats)
        Qp = preseason_arrays(D, P, idx, elo, stats)
        Qc = advance(D, Qp, idx, stats)
        models = []
        for sd in range(5):
            m = build(D, cfg)
            m.load_state_dict(torch.load(SNAP / f"snap_{V}_{sd}.pt"))
            m.eval()
            models.append(m)
        Y = D.A["TMY"][idx]
        y = meta.outcome4.isin([0, 2]).to_numpy().astype(float)
        for conv, Q in (("P", Qp), ("C", Qc)):
            with torch.no_grad():
                outs = [predict(m, Q, np.arange(len(idx))) for m in models]
            o = {k: np.mean([x[k] for x in outs], 0) for k in ("pp_opps", "pp_m", "sogf", "xgf", "goals",
                                                                "p_home_win")}
            pg = np.clip(o["p_home_win"], 1e-6, 1 - 1e-6)
            zz = fb["intercept"] + fb["coef"][0] * elo + fb["coef"][1] * np.log(pg / (1 - pg))
            p = 1 / (1 + np.exp(-zz))
            ll.append({"season": V, "conv": conv, "ll": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))})
            row = {"season": V, "conv": conv}
            for k, tk in STATS.items():
                yk = Y[..., tt.index(tk)]
                ok = np.isfinite(yk)
                row[f"{k}_pred"] = float(o[k][ok].mean())
                row[f"{k}_act"] = float(yk[ok].mean())
                row[f"{k}_bias"] = row[f"{k}_pred"] / row[f"{k}_act"] - 1
                tm = pd.DataFrame({"team": np.r_[meta.home_idx, meta.away_idx],
                                   "pred": np.r_[o[k][:, 0], o[k][:, 1]],
                                   "act": np.r_[yk[:, 0], yk[:, 1]]}).dropna()
                g = tm.groupby("team")[["pred", "act"]].mean()
                team.append({"season": V, "conv": conv, "stat": k,
                             "team_mae": float((g.pred - g.act).abs().mean())})
            league.append(row)
            print(V, conv, {k: round(row[f"{k}_bias"], 4) for k in STATS}, "ll", round(ll[-1]["ll"], 5), flush=True)
    L, T, LL = pd.DataFrame(league), pd.DataFrame(team), pd.DataFrame(ll)
    out = {"rows": league, "team": team, "ll": ll}
    for part, ss in (("fit", FIT), ("judge", JUDGE)):
        sub = L[L.season.isin(ss)]
        out[part] = {c: {k: {"mean_abs_bias": float(sub[sub.conv == c][f"{k}_bias"].abs().mean()),
                             "mean_bias": float(sub[sub.conv == c][f"{k}_bias"].mean()),
                             "ratio_act_pred": float(sub[sub.conv == c][f"{k}_act"].mean()
                                                     / sub[sub.conv == c][f"{k}_pred"].mean()),
                             "team_mae": float(T[T.season.isin(ss) & (T.conv == c) & (T.stat == k)].team_mae.mean())}
                         for k in STATS} | {"ll": float(LL[LL.season.isin(ss) & (LL.conv == c)].ll.mean())}
                     for c in ("P", "C")}
    j = out["judge"]
    adopt_c = (j["C"]["pp_opps"]["mean_abs_bias"] < j["P"]["pp_opps"]["mean_abs_bias"]
               and j["C"]["sogf"]["mean_abs_bias"] < j["P"]["sogf"]["mean_abs_bias"]
               and j["C"]["ll"] <= j["P"]["ll"] + 0.001)
    ch = "C" if adopt_c else "P"
    f = out["fit"][ch]
    out["decision"] = {"convention": ch,
                       "ratios": {k: f[k]["ratio_act_pred"] for k in ("pp_opps", "pp_m", "sogf", "xgf")
                                  if abs(f[k]["mean_bias"]) > 0.03}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("fit", "judge", "decision")}, indent=1))


if __name__ == "__main__":
    main()
