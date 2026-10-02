"""Every number in orr/COMPARISON_ORR2_NeurHL13.md that compares the two 2026-27
forecasts directly: ORR's preseason file (orr/output/freeze_2027/, used by ORR 2.0)
against NeurHL 1.3 (neurhl/output/neurhl_1_3/season/), and both against the
preseason points lines in data/market/nhl_totals_ou_2027.csv.

    python3 -m orr.compare_neurhl   ->  orr/output/compare_neurhl_1_3.json
"""
from __future__ import annotations

import json

import pandas as pd

from orr import config as C

O = C.OUT / "freeze_2027"
N = C.ROOT / "neurhl" / "output" / "neurhl_1_3" / "season"
OUT = C.OUT / "compare_neurhl_1_3.json"


def _r(x, n=3):
    return round(float(x), n)


def teams() -> dict:
    to = pd.read_csv(O / "teams_2027.csv").set_index("team")
    tn = pd.read_csv(N / "teams_2027.csv").set_index("team").reindex(to.index)
    mk = pd.read_csv(C.ROOT / "data" / "market" / "nhl_totals_ou_2027.csv").set_index("team").line.reindex(to.index)
    rel = lambda s: s - s.mean()                       # points relative to each source's league mean
    d = (to.points - tn.points).sort_values()
    return {"corr_points": _r(to.points.corr(tn.points)), "rank_corr": _r(to.points.rank().corr(tn.points.rank())),
            "mean_abs_diff": _r((to.points - tn.points).abs().mean(), 2),
            "sd_points": {"orr": _r(to.points.std(), 2), "neurhl": _r(tn.points.std(), 2), "market": _r(mk.std(), 2)},
            "mean_80_width": {"orr": _r((to.points_p90 - to.points_p10).mean(), 1),
                              "neurhl": _r((tn.points_p90 - tn.points_p10).mean(), 1)},
            "market_rel_mae": {"orr": _r((rel(to.points) - rel(mk)).abs().mean(), 2),
                               "neurhl": _r((rel(tn.points) - rel(mk)).abs().mean(), 2)},
            "largest_gaps": [{"team": t, "orr": _r(to.points[t], 1), "neurhl": _r(tn.points[t], 1), "market": _r(mk[t], 1),
                              "orr_playoff": _r(to.playoff_pct[t], 1), "neurhl_playoff": _r(tn.playoff_pct[t], 1)}
                             for t in list(d.index[:3]) + list(d.index[-3:])],
            "cup_top5": {"orr": to.cup_pct.nlargest(5).round(1).to_dict(), "neurhl": tn.cup_pct.nlargest(5).round(1).to_dict()}}


def skaters() -> dict:
    so = pd.read_csv(O / "skaters_2027.csv")
    sn = pd.read_csv(N / "skaters_2027.csv")
    m = so.merge(sn, on="player_id", suffixes=("_o", "_n")).copy()
    m["dp"] = m.p - m.points
    big = m.reindex(m.dp.abs().sort_values(ascending=False).index).head(6)
    top = lambda s, p, g: s.nlargest(5, p)[["name", "team", "gp", g, p]].round(1).to_dict("records")
    return {"n": {"orr": len(so), "neurhl": len(sn), "common": len(m)},
            "corr_points": _r(m.p.corr(m.points)), "mean_abs_diff_points": _r(m.dp.abs().mean(), 2),
            "corr_gp": _r(m.gp_o.corr(m.gp_n)), "mean_abs_diff_gp": _r((m.gp_o - m.gp_n).abs().mean(), 2),
            "max": {"orr": {"goals": _r(so.g.max(), 1), "points": _r(so.p.max(), 1)},
                    "neurhl": {"goals": _r(sn.goals.max(), 1), "points": _r(sn.points.max(), 1)}},
            "n_40_goals": {"orr": int((so.g >= 40).sum()), "neurhl": int((sn.goals >= 40).sum())},
            "top5": {"orr": top(so, "p", "g"), "neurhl": top(sn, "points", "goals")},
            "largest_gaps": [{"name": r.name_o, "team": r.team_o, "orr_gp": _r(r.gp_o, 1), "orr_p": _r(r.p, 1),
                              "neurhl_gp": _r(r.gp_n, 1), "neurhl_p": _r(r.points, 1)} for r in big.itertuples()]}


def games() -> dict:
    go = pd.read_csv(O / "games_2027.csv")
    gn = pd.read_csv(N / "games_2027.csv")
    m = go.merge(gn, on="game_id", suffixes=("_o", "_n"))
    po, pn = m.p_home_win_o, m.p_home_win_n
    return {"n": len(m), "corr": _r(po.corr(pn)), "mean_abs_diff": _r((po - pn).abs().mean(), 4),
            "max_abs_diff": _r((po - pn).abs().max()), "different_favourite": _r(((po > .5) != (pn > .5)).mean()),
            "mean_p_home": {"orr": _r(po.mean()), "neurhl": _r(pn.mean())},
            "sd_p_home": {"orr": _r(po.std()), "neurhl": _r(pn.std())}}


def goalies() -> dict:
    go = pd.read_csv(O / "goalies_2027.csv")
    gn = pd.read_csv(N / "goalies_2027.csv")
    m = go.merge(gn, on="player_id", suffixes=("_o", "_n"))
    return {"common": len(m), "corr_sv": _r(m.sv_pct_o.corr(m.sv_pct_n)),
            "mean_sv": {"orr": _r(go.sv_pct.mean(), 4), "neurhl": _r(gn.sv_pct.mean(), 4)}}


def main():
    out = {"orr": "orr/output/freeze_2027 (ORR 1.0 preseason file, used by ORR 2.0)",
           "neurhl": "neurhl/output/neurhl_1_3/season (NeurHL 1.3)",
           "teams": teams(), "skaters": skaters(), "games": games(), "goalies": goalies()}
    OUT.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
