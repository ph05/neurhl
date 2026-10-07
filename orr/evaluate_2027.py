"""The preregistered 2026-27 evaluation of ORR against NeurHL (orr/EVALUATION_2027.md).

    python3 -m orr.evaluate_2027 [--neurhl-live DIR]   ->  orr/output/evaluation_2027.json

Runs at any time. Comparisons whose data are not complete (team and skater
season totals before the last regular-season game) are reported as
"pending". The SHA-256 of every forecast file is recorded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from orr import config as C
from orr import score as SC

OUT = C.OUT / "evaluation_2027.json"
FILES = {
    "orr_preseason": [f"orr/output/freeze_2027/{f}_2027.csv" for f in ("teams", "games", "skaters", "goalies")],
    "neurhl_1.0": [f"neurhl/output/neurhl_1_0/{f}_2027.csv" for f in ("teams", "games", "skaters", "goalies")],
    "neurhl_1.1": [f"neurhl/output/neurhl_1_1/season/{f}_2027.csv" for f in ("teams", "games", "skaters")],
}


def paired(d: np.ndarray, nb: int = 2000, seed: int = 7) -> dict:
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if not len(d):
        return {"n": 0, "diff": None, "ci95": None, "verdict": "no observations"}
    rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(nb)])
    lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    verdict = "ORR better" if hi < 0 else "NeurHL better" if lo > 0 else "no difference shown"
    return {"n": int(len(d)), "diff": float(d.mean()), "ci95": [lo, hi], "verdict": verdict}


def game_comparison(ep: pd.DataFrame, a: str, b: str, final: bool = False) -> dict:
    if not len(ep) or a not in set(ep.model) or b not in set(ep.model):
        return {"status": "pending", "reason": f"no eligible games for both {a} and {b} yet"}
    ll = ep.assign(ll=-(ep.y * np.log(ep.p.clip(1e-6, 1 - 1e-6)) + (1 - ep.y) * np.log((1 - ep.p).clip(1e-6, 1 - 1e-6))))
    ll["br"] = (ll.p - ll.y) ** 2
    w = ll.pivot_table(index="game_id", columns="model", values="ll")[[a, b]].dropna()
    if not len(w):
        return {"status": "pending", "reason": "no common eligible games yet"}
    br = ll.pivot_table(index="game_id", columns="model", values="br")[[a, b]].dropna()
    r = paired((w[a] - w[b]).to_numpy())
    r["brier_diff"] = float((br[a] - br[b]).mean())            # secondary: reported, not judged
    if not final:                      # a verdict is only read at season end
        r["verdict"] = f"(interim, not a verdict) {r['verdict']}"
    return {"status": "final" if final else "interim", **r}


def season_complete(res: pd.DataFrame) -> bool:
    gp = pd.concat([res.home, res.away]).value_counts()
    return bool(len(gp) == 32 and gp.min() >= C.GAMES_PER_TEAM[C.TARGET_SEASON])   # numpy bool is not JSON


def team_comparison(res: pd.DataFrame, ours: str, theirs: str) -> dict:
    if not season_complete(res):
        return {"status": "pending", "reason": "regular season not complete"}
    act = SC._team_points_so_far(res).pts
    t1 = pd.read_csv(C.ROOT / FILES[ours][0]).set_index("team")
    t2 = pd.read_csv(C.ROOT / FILES[theirs][0]).set_index("team")
    e1, e2 = (t1.points.reindex(act.index) - act).abs(), (t2.points.reindex(act.index) - act).abs()
    c1, c2 = crps_from_quantiles(t1.reindex(act.index), act), crps_from_quantiles(t2.reindex(act.index), act)
    cover = lambda t: float(((t.points_p10.reindex(act.index) <= act) & (act <= t.points_p90.reindex(act.index))).mean())
    return {"status": "final", **paired((e1 - e2).to_numpy()),           # judged: final points MAE
            "crps": paired((c1 - c2).to_numpy()), "cover80": {"orr": cover(t1), "neurhl": cover(t2)}}


def crps_from_quantiles(t: pd.DataFrame, y: pd.Series) -> pd.Series:
    """CRPS of a normal with the file's mean and the sd implied by its 10th-90th
    percentile range. Both files publish those quantiles, so both are scored alike."""
    from scipy import stats
    mu = t.points
    sd = ((t.points_p90 - t.points_p10) / (2 * stats.norm.ppf(0.9))).clip(lower=1e-6)
    z = (y - mu) / sd
    return sd * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))


def skater_comparison(boxes: Path, ours: str, theirs: str, res: pd.DataFrame) -> dict:
    if not season_complete(res) or not boxes.exists():
        return {"status": "pending", "reason": "regular season not complete or no box scores"}
    b = pd.read_csv(boxes)
    b = b[b.pos != "G"]
    act = b.groupby("player_id").agg(gp=("game_id", "nunique"), pts=("g", "sum"))
    act["pts"] += b.groupby("player_id").a.sum()
    s1 = pd.read_csv(C.ROOT / FILES[ours][2]).set_index("player_id")
    s2 = pd.read_csv(C.ROOT / FILES[theirs][2]).set_index("player_id")
    ids = s1.index.intersection(s2.index)
    y = act.pts.reindex(ids).fillna(0)
    d = (s1.p.reindex(ids) - y).abs() - (s2.points.reindex(ids) - y).abs()
    gp40 = act.gp.reindex(ids).fillna(0) >= 40
    return {"status": "final", **paired(d[gp40].to_numpy()), "population": "40+ GP",   # judged
            "all_skaters": paired(d.to_numpy())}


def goalie_comparison(boxes: Path, theirs: str, res: pd.DataFrame, min_sa: int = 1000) -> dict:
    """Secondary: season save % MAE for goalies with at least ``min_sa`` shots against."""
    if not season_complete(res) or not boxes.exists():
        return {"status": "pending", "reason": "regular season not complete or no box scores"}
    b = pd.read_csv(boxes)
    b = b[(b.pos == "G") & b.shots_against.notna()]
    act = b.groupby("player_id").agg(sa=("shots_against", "sum"), ga=("goals_against", "sum"))
    act = act[act.sa >= min_sa]
    sv = 1 - act.ga / act.sa
    g1 = pd.read_csv(C.ROOT / FILES["orr_preseason"][3]).set_index("player_id")
    g2 = pd.read_csv(C.ROOT / FILES[theirs][3]).set_index("player_id")
    ids = sv.index.intersection(g1.index).intersection(g2.index)
    d = (g1.sv_pct.reindex(ids) - sv[ids]).abs() - (g2.sv_pct.reindex(ids) - sv[ids]).abs()
    return {"status": "final", **paired(d.to_numpy())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neurhl-live", default=None)
    a = ap.parse_args()
    if a.neurhl_live:
        SC.NEURHL_LIVE = Path(a.neurhl_live)
    res = SC.load_results()
    ep = SC.eligible_probs(res)
    boxes = C.OUT / "live" / f"boxes_{C.TARGET_SEASON}.csv"
    fin = season_complete(res)
    out = {"protocol": "orr/EVALUATION_2027.md", "evaluated_utc": pd.Timestamp.now("UTC").isoformat(timespec="seconds"),
           "games_played": int(len(res)), "season_complete": fin,
           "files_sha256": {f: hashlib.sha256((C.ROOT / f).read_bytes()).hexdigest()
                            for fs in FILES.values() for f in fs if (C.ROOT / f).exists()},
           "P1_daily_vs_neurhl_G": game_comparison(ep, "orr_inseason", "neurhl_G_pregame", fin),
           "P2_preseason_vs_neurhl_1_1": game_comparison(ep, "orr_preseason", "neurhl_1.1", fin),
           "P3_teams_vs_neurhl_1_1": team_comparison(res, "orr_preseason", "neurhl_1.1"),
           "P4_skaters_vs_neurhl_1_1": skater_comparison(boxes, "orr_preseason", "neurhl_1.1", res),
           "secondary": {"P2_vs_neurhl_1_0": game_comparison(ep, "orr_preseason", "neurhl_1.0", fin),
                         "P3_vs_neurhl_1_0": team_comparison(res, "orr_preseason", "neurhl_1.0"),
                         "P4_vs_neurhl_1_0": skater_comparison(boxes, "orr_preseason", "neurhl_1.0", res),
                         "daily_vs_neurhl_H": game_comparison(ep, "orr_inseason", "neurhl_H_pregame", fin),
                         "daily_vs_elo": game_comparison(ep, "orr_inseason", "elo_pregame", fin),
                         "goalie_sv_vs_neurhl_1_0": goalie_comparison(boxes, "neurhl_1.0", res)}}
    OUT.write_text(json.dumps(out, indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))
    for k, v in out.items():
        if k.startswith("P"):
            print(k, json.dumps(v)[:160])


if __name__ == "__main__":
    main()
