"""Recompute every number the NeurHL review cites from files in this repository.

Standalone: reads only NeurHL's own files, data/, git history, and the
hand-collected preseason points lines in review/data/. It imports nothing
from any competing model.

Run from the repository root:
    python3 review/evidence/recompute.py
Writes review/evidence/recomputed.json and prints a summary.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "recomputed.json"
FIRST_PUCK_DROP = pd.Timestamp("2026-09-29T21:00:00Z")
FRANCHISE = {"ATL": "WPG", "PHX": "UTA", "ARI": "UTA", "L.A": "LAK",
             "N.J": "NJD", "S.J": "SJS", "T.B": "TBL"}


def ll(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def rd(path):
    return pd.read_csv(ROOT / path)


def jl(path):
    return json.loads((ROOT / path).read_text())


def git(*a):
    return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True,
                          text=True, check=True).stdout.strip()


R: dict = {}

# 1. Game-level log losses on NeurHL's own held-out prediction files --------
g = rd("neurhl/output/preds/g_gate_games.csv")
R["g_gate"] = {"n": len(g), "stack": ll(g.p_stack, g.y), "H": ll(g.p_h, g.y),
               "elo": ll(g.p_elo, g.y), "raw_engine": ll(g.p_g, g.y)}
lg = np.log(np.clip(g.p_g, 1e-6, 1 - 1e-6) / (1 - np.clip(g.p_g, 1e-6, 1 - 1e-6)))
ls = np.log(g.p_stack / (1 - g.p_stack))
R["g_gate"]["logit_sd_raw_engine"] = float(lg.std())
R["g_gate"]["logit_sd_stack"] = float(ls.std())
dec = pd.qcut(g.p_g, 10, labels=False)
R["g_gate"]["raw_engine_decile_pred_obs"] = {
    "bottom": [float(g.p_g[dec == 0].mean()), float(g.y[dec == 0].mean())],
    "top": [float(g.p_g[dec == 9].mean()), float(g.y[dec == 9].mean())]}
d_sh = (-(g.y * np.log(g.p_stack) + (1 - g.y) * np.log(1 - g.p_stack))
        + (g.y * np.log(g.p_h) + (1 - g.y) * np.log(1 - g.p_h)))
R["g_gate"]["stack_minus_H"] = [float(d_sh.mean()), float(d_sh.std() / np.sqrt(len(g)))]
h = rd("neurhl/output/preds/hier_restatement_games.csv")
R["hier"] = {"n": len(h), "H": ll(h.p_neurhl_h, h.y), "elo": ll(h.p_elo, h.y),
             "constant": ll(np.full(len(h), h.y.mean()), h.y)}
s = rd("neurhl/output/preds/g_seal_games.csv")
R["seal"] = {"n": len(s), "stack": ll(s.p_g, s.y), "H": ll(s.p_h, s.y),
             "elo": ll(s.p_elo, s.y)}

# 2. NeurHL's own result files ------------------------------------------------
R["eda_lineup_gain"] = {k: v for k, v in jl("neurhl/output/eda_lineup_gain.json")
                        ["by_absence_gap"].items()}
R["s_stop"] = {k: v for k, v in jl("neurhl/output/g_sstop.json").items()
               if k in ("g_minus_elo", "g_minus_h", "seasons_beat_h", "pass")}
cal = jl("neurhl/configs/calibration_1_1.json")
R["goal_slope"] = {"b_hat": cal["goal_slope"]["b_hat"],
                   "n": cal["goal_slope"]["n_team_games"],
                   "deciles": cal["goal_slope"]["deciles"],
                   "sog_coverage_r40": cal["sog_dispersion"]["coverage_r40"],
                   "sog_r_hat": cal["sog_dispersion"]["r_hat"]}
sm = jl("neurhl/configs/season_matched_comparison.json")["models"]
R["season_matched_mae"] = {k: v["mae"] for k, v in sm.items()}
c2 = jl("neurhl/output/neurhl_1_1/season_layer_c2.json")
R["season_layer_c2_judge"] = {"seasons": c2["judge"],
                              "engine_frozen_mae": c2["engine"]["judge"]["frozen"]["mae"],
                              "elo_frozen_mae": c2["elo"]["judge"]["frozen"]["mae"],
                              "n": c2["engine"]["judge"]["frozen"]["n"]}

# 3. Historical shootout share -------------------------------------------------
gm = rd("data/processed/games.csv")
gm = gm[gm.game_type == "R"]
x = gm[(gm.season_end.between(2016, 2026)) & (gm.went_ot | gm.went_so)]
R["so_share_2016_2026"] = float(x.went_so.mean())
R["so_share_by_season"] = x.groupby("season_end").went_so.mean().round(3).to_dict()

# 4. NeurHL 1.3 (current release) sanity numbers -------------------------------
t = rd("neurhl/output/neurhl_1_3/season/teams_2027.csv")
m = rd("data/market/nhl_totals_ou_2027.csv")
tm = t.merge(m, on="team")
sd90 = (tm.points_p90 - tm.points_p10) / 2.563
from scipy.stats import norm  # noqa: E402
p_over = 1 - norm.cdf((tm.line - tm.points) / sd90)
R["release_1_3_teams"] = {
    "points_sd": float(t.points.std()), "market_line_sd": float(m.line.std()),
    "corr_with_market": float(np.corrcoef(tm.points, tm.line)[0, 1]),
    "mean_abs_gap": float((tm.points - tm.line).abs().mean()),
    "gd_sd": float((t.goals_for - t.goals_against).std()),
    "largest_gaps": (tm.assign(gap=tm.points - tm.line).sort_values("gap", key=abs,
                     ascending=False).head(8).set_index("team").gap.round(1).to_dict()),
    "lines_with_p_side_ge_0.65": int(((p_over >= .65) | (p_over <= .35)).sum()),
}
sk = rd("neurhl/output/neurhl_1_3/season/skaters_2027.csv")
mp = pd.read_csv(ROOT / "data/raw/mp_skaters_2025.csv")
mp = mp[mp.situation == "all"].set_index("playerId")
sk["last_ppg"] = sk.player_id.map(mp.I_F_points / mp.games_played)
sk["last_gp"] = sk.player_id.map(mp.games_played)
reg = sk[sk.last_gp >= 60]
R["release_1_3_skaters"] = {
    "n_regulars": int(len(reg)),
    "slope_proj_ppg_on_last_ppg": float(np.polyfit(reg.last_ppg, reg.points_per_gp, 1)[0]),
    "n_100pt": int((sk.points >= 100).sum()), "n_40g": int((sk.goals >= 40).sum()),
    "mcdavid": sk[sk.name == "Connor McDavid"][["gp", "points", "points_p10", "points_p90"]]
    .round(1).iloc[0].to_dict(),
}
go = rd("neurhl/output/neurhl_1_3/season/goalies_2027.csv")
R["release_1_3_goalies_top_starts"] = go.nlargest(4, "starts").set_index("name").starts.round(1).to_dict()

# 5. Preseason sportsbook lines: accuracy on NeurHL's judge seasons -----------
mh = pd.read_csv(ROOT / "review/data/nhl_point_totals_history.csv")
mh["team"] = mh.team.replace(FRANCHISE)
rows = []
for side, other in (("home", "away"), ("away", "home")):
    win = gm[f"{side}_g"] > gm[f"{other}_g"]
    rows.append(pd.DataFrame({"season_end": gm.season_end, "team": gm[side].replace(FRANCHISE),
                              "pts": np.where(win, 2, np.where(gm.went_ot | gm.went_so, 1, 0)),
                              "gp": 1, "gd": gm[f"{side}_g"] - gm[f"{other}_g"]}))
st = pd.concat(rows).groupby(["season_end", "team"], as_index=False).sum()
st["pts82"] = st.pts * 82 / st.gp
st["gd_pg"] = st.gd / st.gp
st["pts_pct"] = st.pts / (2 * st.gp)
mk = mh.merge(st, on=["season_end", "team"])
per = mk.assign(e=(mk.line - mk.pts82).abs()).groupby("season_end").e.mean()
judge = [2019, 2020, 2022, 2023, 2024]
jm = mk[mk.season_end.isin(judge)]
R["market_lines"] = {
    "team_seasons": int(len(mk)), "mae_by_season": per.round(2).to_dict(),
    "mae_on_c2_judge_seasons": float((jm.line - jm.pts82).abs().mean()),
    "n_judge": int(len(jm)),
}


# A simple walk-forward team-history regression, to show what blending buys.
def td_pred(V):
    def feats(s):
        f = st[st.season_end == s][["team"]].copy()
        for lag in (1, 2):
            p = st[st.season_end == s - lag].set_index("team")
            f[f"gd{lag}"] = f.team.map(p.gd_pg).fillna(0)
            f[f"pp{lag}"] = f.team.map(p.pts_pct - p.pts_pct.mean()).fillna(0)
        return f
    tr = []
    for s in range(2011, V):
        if s in (2013, 2020, 2021):
            continue
        f = feats(s)
        y = st[st.season_end == s].set_index("team").pts82
        f["y"] = f.team.map(y - y.mean())
        tr.append(f)
    tr = pd.concat(tr)
    cols = ["gd1", "pp1", "gd2", "pp2"]
    X = np.c_[np.ones(len(tr)), tr[cols].to_numpy()]
    b = np.linalg.lstsq(X, tr.y.to_numpy(), rcond=None)[0]
    f = feats(V)
    f["td"] = np.c_[np.ones(len(f)), f[cols].to_numpy()] @ b
    return f[["team", "td"]]


bl = []
for V in sorted(mk.season_end.unique()):
    p = td_pred(V).merge(mk[mk.season_end == V], on="team")
    p["act"] = p.pts82 - p.pts82.mean()
    p["mkt"] = p.line - p.line.mean()
    bl.append(p)
bl = pd.concat(bl)
R["blend_market_vs_team_history"] = {
    w: float((w * bl.mkt + (1 - w) * bl.td - bl.act).abs().mean()) for w in (0.0, 0.5, 0.75, 1.0)}

# 6. Release timeline -----------------------------------------------------------
tl = {}
for label, c in (("1.1", "9580606"), ("1.2", "a921158"), ("results_09_29", "0b90332"),
                 ("1.3", "420cd71"), ("docs_removed", "d523c85"), ("seal_spent", "43aa58d"),
                 ("seal_in_training", "5816561")):
    when = pd.Timestamp(git("show", "-s", "--format=%aI", c)).tz_convert("UTC")
    tl[label] = {"commit": c, "utc": str(when),
                 "minutes_vs_first_puck_drop": round((when - FIRST_PUCK_DROP).total_seconds() / 60, 1)}
R["timeline"] = tl

# 7. Hidden documents -----------------------------------------------------------
tracked = git("ls-files").split("\n")
refs = 0
for f in tracked:
    if f.endswith((".py", ".json", ".csv", ".sh", ".md", ".html", ".js", ".yml")):
        try:
            txt = (ROOT / f).read_text(errors="ignore")
        except (IsADirectoryError, FileNotFoundError):
            continue
        if "PLAN_" in txt or "NOTES.md" in txt or "EVIDENCE" in txt:
            refs += 1
R["hidden_docs"] = {"gitignore_hides_md": "*.md" in (ROOT / ".gitignore").read_text(),
                    "tracked_md_files": sum(f.endswith(".md") for f in tracked),
                    "tracked_files_referencing_plan_docs": refs}

OUT.write_text(json.dumps(R, indent=1, default=str))
for k, v in R.items():
    print(f"{k}: {json.dumps(v, default=str)[:300]}")
