"""Standings backtest: HatTrick's team layer against NeurHL's published numbers.

Two comparisons:

A. NeurHL's own backtest seasons (no market data exists for them):
   - 2012, 2014-2017 (season_matched_comparison.json: v1 Elo 9.17, NeurHL-2 9.99)
   - 2011, 2012, 2014-2017 (season_backtest.json: NeurHL-2 per-season MAE)
   HatTrick uses its top-down (+ bottom-up when available) view, fitted
   walk-forward on seasons before each target.
B. The seasons with preseason market lines (2019, 2020, 2022-2026), which
   include NeurHL's 2019-24 judge window (engine layer 9.42, Elo 9.47):
   HatTrick's leave-one-season-out blend of market, top-down and bottom-up.

Two scales are reported. "rel" MAE is in points per 82 games relative to the
league mean (the blend works on this scale). "raw" MAE is what NeurHL's
season_layer_c2 judge reports: absolute points over the games each team
actually played (2019-20 unscaled), uncentred -- HatTrick's raw prediction adds
the league mean it would have projected from the three previous seasons.
Roster-based views (bottom-up, roster change) are only fitted and scored on
seasons whose roster proxy is the first 10 games (<= 2024): 2025 and 2026
rosters would come from each player's season team, which is in-season
information.

Run: python3 -m hattrick.backtest.teams_bt
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from hattrick import config as C
from hattrick import teams as T

OUT = C.OUT / "backtest" / "teams_bt.json"
BU = C.OUT / "backtest" / "team_components_hist.csv"


def bottom_up_frame() -> pd.DataFrame | None:
    """Bottom-up roster view, mapped to points per 82 walk-forward, if the
    player layer has produced its historical component table."""
    if not BU.exists():
        return None
    from hattrick import team_points_map as TPM
    h = pd.read_csv(BU)
    return TPM.walk_forward_points(h[h.roster_proxy == "first10"])


def section_a(bu) -> dict:
    st = T.standings_all()
    nm = json.loads((C.ROOT / "neurhl/configs/season_backtest.json").read_text())
    neur = {r["season"]: r["mae"] for r in nm["rows"]}
    rows = []
    # 2011 has no earlier season with MoneyPuck process stats to train on
    for V in (2012, 2014, 2015, 2016, 2017):
        p = T.predict_topdown(T.fit_topdown(V, alpha=8.0), V)
        a = st[st.season_end == V].set_index("team")
        p["act"] = p.team.map(a.pts82 - a.pts82.mean())
        row = {"season": V, "n": len(p), "hattrick_topdown_mae": float((p.td_rel82 - p.act).abs().mean()),
               "neurhl2_mae": neur.get(V)}
        if bu is not None:
            q = p.merge(bu[bu.season_end == V], on=["team", "season_end"], how="left")
            if q.bu_rel82.notna().all() and V >= 2014:
                w = _fit_td_bu(bu, V)
                pred = w[0] * q.td_rel82 + w[1] * q.bu_rel82
                row["hattrick_td_bu_mae"] = float((pred - q.act).abs().mean())
        rows.append(row)
    df = pd.DataFrame(rows)
    matched = df[df.season.isin([2012, 2014, 2015, 2016, 2017])]
    sm = json.loads((C.ROOT / "neurhl/configs/season_matched_comparison.json").read_text())
    return {"per_season": rows,
            "matched_5_seasons": {
                "hattrick_topdown": float(matched.hattrick_topdown_mae.mean()),
                **({"hattrick_td_bu": float(matched.hattrick_td_bu_mae.mean())}
                   if "hattrick_td_bu_mae" in matched else {}),
                **{k: v["mae"] for k, v in sm["models"].items()}}}


def _fit_td_bu(bu, V):
    """Weights for top-down and bottom-up from seasons before V (NNLS)."""
    f = T.blend_frame([s for s in range(2012, V) if s not in C.BROKEN_SEASONS],
                      extra=bu[["team", "season_end", "bu_rel82"]])
    return T.fit_blend(f, ["td_rel82", "bu_rel82"])


def roster_frame() -> pd.DataFrame | None:
    """Roster-change correction rd_rel82, walk-forward (hattrick.roster_delta)."""
    from hattrick import roster_delta as RD
    if not RD.OUT.exists():
        return None
    h = pd.read_csv(RD.OUT)
    return RD.walk_forward(h[h.season_end <= 2024])


def crps_normal(mu, sd, x):
    from scipy.stats import norm
    z = (x - mu) / sd
    return sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


CLEAN = [2019, 2020, 2022, 2023, 2024]
ALL_MKT = [2019, 2020, 2022, 2023, 2024, 2025, 2026]


def _raw_scores(f, cols, seasons):
    """LOSO predictions on the raw, uncentred scale NeurHL's judge uses."""
    st = T.standings_all().set_index(["season_end", "team"])
    out = []
    for V in seasons:
        tr = f[(f.season_end != V) & f.season_end.isin(seasons)]
        te = f[f.season_end == V].dropna(subset=cols + ["act_rel82"])
        w = T.fit_blend(tr, cols)
        trd = tr.dropna(subset=cols + ["act_rel82"])
        sd = float(np.sqrt(((trd[cols].to_numpy() @ w - trd.act_rel82) ** 2).mean()))
        gp = np.array([st.loc[(V, t)].gp for t in te.team])
        act = np.array([st.loc[(V, t)].pts for t in te.team])
        mean82 = T.league_points_per_team(V, 82)
        pred = (mean82 + te[cols].to_numpy() @ w) * gp / 82.0
        out.append(pd.DataFrame({"season_end": V, "team": te.team.to_numpy(), "pred": pred,
                                 "act": act, "sd": sd * gp / 82.0}))
    return pd.concat(out)


def section_b(bu) -> dict:
    extra = bu[["team", "season_end", "bu_rel82"]] if bu is not None else None
    f = T.blend_frame(ALL_MKT, extra=extra)
    rd = roster_frame()
    if rd is not None:
        f = f.merge(rd, on=["team", "season_end"], how="left")
        f["rd_rel82"] = f.rd_rel82.fillna(0.0)
        f["tdr_rel82"] = f.td_rel82 + f.rd_rel82
        f.loc[f.season_end > 2024, ["rd_rel82", "tdr_rel82"]] = np.nan
    out = {"note": "LOSO over the listed seasons; roster views only on clean (first-10) seasons"}
    views = [(["mkt_rel82"], ALL_MKT), (["td_rel82"], ALL_MKT), (["mkt_rel82", "td_rel82"], ALL_MKT),
             (["mkt_rel82"], CLEAN), (["td_rel82"], CLEAN), (["mkt_rel82", "td_rel82"], CLEAN)]
    if rd is not None:
        views += [(["tdr_rel82"], CLEAN), (["mkt_rel82", "tdr_rel82"], CLEAN),
                  (["mkt_rel82", "td_rel82", "tdr_rel82"], CLEAN)]
    if bu is not None:
        views += [(["bu_rel82"], CLEAN), (["mkt_rel82", "bu_rel82"], CLEAN),
                  (["mkt_rel82", "td_rel82", "bu_rel82"], CLEAN)]
    for cols, seasons in views:
        g = f[f.season_end.isin(seasons)]
        l = T.loso_blend(g, cols)
        key = "+".join(c.replace("_rel82", "") for c in cols) + ("@clean" if seasons == CLEAN else "@all")
        n = l.n.sum()
        crps = []
        for V in l.season_end:
            tr, te = g[g.season_end != V], g[g.season_end == V].dropna(subset=cols + ["act_rel82"])
            w = T.fit_blend(tr, cols)
            trd = tr.dropna(subset=cols + ["act_rel82"])
            sd = float(np.sqrt(((trd[cols].to_numpy() @ w - trd.act_rel82) ** 2).mean()))
            crps.append((len(te), float(crps_normal(te[cols].to_numpy() @ w, sd, te.act_rel82).mean())))
        raw = _raw_scores(g, cols, [s_ for s_ in seasons if s_ in CLEAN])
        out[key] = {"seasons": seasons, "rel_mae": float((l.blend_mae * l.n).sum() / n),
                    "rel_rmse": float(np.sqrt((l.blend_rmse ** 2 * l.n).sum() / n)),
                    "rel_crps": sum(a * b for a, b in crps) / sum(a for a, _ in crps),
                    "judge_raw_mae": float((raw.pred - raw.act).abs().mean()),
                    "judge_raw_crps": float(crps_normal(raw.pred, raw.sd, raw.act).mean()),
                    "per_season": l.round(4).to_dict("records")}
    c2 = json.loads((C.ROOT / "neurhl/output/neurhl_1_1/season_layer_c2.json").read_text())
    out["neurhl_judge_2019_2024"] = {
        "scale": "raw uncentred points over games played (season_layer_c2.py)",
        "engine_layer_mae": c2["engine"]["judge"]["frozen"]["mae"],
        "engine_layer_crps": c2["engine"]["judge"]["frozen"]["crps"],
        "elo_mae": c2["elo"]["judge"]["frozen"]["mae"],
        "elo_crps": c2["elo"]["judge"]["frozen"]["crps"]}
    clean = {k: v for k, v in out.items() if k.endswith("@clean")}
    best = min(clean, key=lambda k: clean[k]["rel_rmse"])
    out["selected"] = best
    out["talent_sd_82"] = float(np.sqrt(max(clean[best]["rel_rmse"] ** 2 - T.LUCK_SD_82 ** 2, 1.0)))
    return out


def main():
    bu = bottom_up_frame()
    res = {"A_neurhl_backtest_seasons": section_a(bu), "B_market_seasons": section_b(bu)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=float))
    a = res["A_neurhl_backtest_seasons"]
    print(pd.DataFrame(a["per_season"]).round(2).to_string(index=False))
    print("matched 5 seasons:", {k: round(v, 2) for k, v in a["matched_5_seasons"].items()})
    b = res["B_market_seasons"]
    for k, v in b.items():
        if isinstance(v, dict) and "rel_mae" in v:
            print(f"{k:<26} rel MAE {v['rel_mae']:.3f} RMSE {v['rel_rmse']:.3f} CRPS {v['rel_crps']:.3f} | "
                  f"judge raw MAE {v['judge_raw_mae']:.3f} CRPS {v['judge_raw_crps']:.3f}")
    print("NeurHL judge window:", {k: v for k, v in b["neurhl_judge_2019_2024"].items() if k != "scale"},
          "| selected:", b["selected"])


if __name__ == "__main__":
    main()
