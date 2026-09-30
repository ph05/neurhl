"""Build docs/data.js for the NeurHL projection site (docs/index.html).

Reads only committed records, so the site always shows exactly what the
repository holds: the frozen 2026-27 predictions, the live results cache and
scorecard, and the evidence files behind each headline number.

Usage: uv run --no-project --python 3.12 --with "pandas<3" \
         python neurhl/site/build_site.py
"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CONFIGS, NOUT, PROJ  # noqa: E402

DOCS = PROJ / "docs"
UNI = NOUT / "neurhl_1_0"          # NeurHL 1.0, the unified model (PLAN_NeurHL_1_0.md)
# PLAN_NeurHL4 A5: the season set of the engine refit on all seasons is the primary forecast
# once it exists and its consistency check has passed; otherwise the frozen 1.0 files
_RELS = [NOUT / f"neurhl_{v}" / "season" for v in ("1_3", "1_2", "1_1")]   # newest release first
_CANDS = [r for r in _RELS if r.exists()] + sorted(UNI.glob("v*_20*"), reverse=True)
for _v in _CANDS:                                          # newest dated set that passed its check
    if (_v / "checks_2027.json").exists() and json.loads((_v / "checks_2027.json").read_text()).get("pass"):
        UNI = _v
        break
FROZEN = ["neurhl/output/games_2027.csv", "neurhl/output/projection_2027.csv",
          "neurhl/output/player_proj_2027.csv"]
FROZEN_1_0 = [str((UNI / f).relative_to(PROJ)) for f in ("games_2027.csv", "teams_2027.csv", "skaters_2027.csv")]


def rd(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def tonight() -> dict:
    """The latest game day's committed live forecasts (pregame where present,
    else morning), with the players most likely to score."""
    base = NOUT / "live" / "2027"
    days = sorted(d for d in base.iterdir() if d.is_dir()) if base.exists() else []
    if not days:
        return {}
    d = days[-1]
    frames, players = [], []
    for stem in ("morning", "preview"):
        if (d / f"{stem}.csv").exists():
            frames.append(pd.read_csv(d / f"{stem}.csv"))
            players.append(pd.read_csv(d / f"{stem}_players.csv").assign(forecast=stem))
    for f in sorted(d.glob("pregame_*.csv")):
        if f.name.endswith("_players.csv"):
            continue
        frames.append(pd.read_csv(f))
        pf = f.with_name(f.stem + "_players.csv")
        if pf.exists():
            players.append(pd.read_csv(pf).assign(forecast="pregame"))
    if not frames:
        return {}
    g = pd.concat(frames, ignore_index=True)
    order = {"pregame": 0, "morning": 1, "preview": 2}
    g = g.sort_values("forecast", key=lambda x: x.map(order)).drop_duplicates("game_id")
    pl = pd.concat(players, ignore_index=True) if players else pd.DataFrame()
    games = []
    for r in g.sort_values("start_utc").itertuples():
        top = []
        if len(pl):
            q = pl[pl.game_id == r.game_id]
            if "forecast" in q:
                q = q[q.forecast == r.forecast] if (q.forecast == r.forecast).any() else q
            q = q.drop_duplicates("player_id").sort_values("p_goal", ascending=False).head(4)
            top = [[x.name, x.team, round(x.sog_mean, 1), round(x.p_goal, 3), round(x.p_point, 3)]
                   for x in q.itertuples()]
        games.append({"id": int(r.game_id), "start": r.start_utc, "home": r.home, "away": r.away,
                      "kind": r.forecast, "g": r.p_home_win_neurhl_g,
                      "h": None if pd.isna(getattr(r, "p_home_win_neurhl_h", None)) else r.p_home_win_neurhl_h,
                      "elo": r.p_home_win_elo, "ot": r.p_ot,
                      "score": [round(r.goals_home, 2), round(r.goals_away, 2)],
                      "xg": [round(r.xgf_home, 2), round(r.xgf_away, 2)],
                      "sog": [round(r.sog_home, 1), round(r.sog_away, 1)],
                      "lineup": [r.lineup_home, r.lineup_away], "top": top})
    return {"date": d.name, "games": games}


def main():
    howe = pd.read_csv(PROJ / "output" / "projections_2026_27_howe.csv")
    meta = howe.set_index("Abbr")[["Team", "Division", "xPts"]]
    unified = (UNI / "teams_2027.csv").exists()
    if unified:     # NeurHL 1.0: one model for players, games and the season
        t = pd.read_csv(UNI / "teams_2027.csv")
        teams = [{"ab": r.team, "name": meta.loc[r.team, "Team"],
                  "div": meta.loc[r.team, "Division"], "conf": r.conf,
                  "pts": round(r.points, 1), "p10": int(round(r.points_p10)), "p90": int(round(r.points_p90)),
                  "wins": round(r.w, 1), "po": round(r.playoff_pct, 1), "cup": round(r.cup_pct, 1),
                  "howe": round(float(meta.loc[r.team, "xPts"]), 1)}
                 for r in t.itertuples()]
    else:
        t = pd.read_csv(NOUT / "projection_2027.csv")
        teams = [{"ab": r.team, "name": meta.loc[r.team, "Team"],
                  "div": meta.loc[r.team, "Division"], "conf": r.conf,
                  "pts": round(r.proj_points, 1), "p10": int(r.p10), "p90": int(r.p90),
                  "wins": round(r.proj_wins, 1), "po": round(r.playoff_pct, 1),
                  "howe": round(float(meta.loc[r.team, "xPts"]), 1)}
                 for r in t.itertuples()]

    g = pd.read_csv((UNI if unified else NOUT) / "games_2027.csv")
    res_p = NOUT / "live" / "results_2027.csv"
    res = pd.read_csv(res_p).set_index("game_id") if res_p.exists() else pd.DataFrame()
    games = []
    for r in g.itertuples():
        row = [int(r.game_id), r.date, r.away, r.home, round(r.p_home_win, 4),
               round(r.p_home_win_elo, 4), round(r.p_ot, 4)]
        if len(res) and r.game_id in res.index:
            x = res.loc[r.game_id]
            row.append([int(x.away_g), int(x.home_g), str(x.last_period)])
        games.append(row)

    if unified:
        p = pd.read_csv(UNI / "skaters_2027.csv")
        p = p[p.gp >= 1].sort_values("points", ascending=False)
        players = [[r.name if isinstance(r.name, str) else str(r.player_id), r.team, r.pos,
                    round(r.gp, 1), round(r.toi_per_gp, 1),
                    round(r.goals, 1), round(r.assists, 1), round(r.points, 1),
                    round(r.points_p10, 1), round(r.points_p90, 1)]
                   for r in p.itertuples()]
    else:
        p = pd.read_csv(NOUT / "player_proj_2027.csv")
        players = [[r.name, r.team, "D" if r.pos_group == 1 else "F",
                    round(r.exp_gp, 1), round(r.toi_per_gp_min, 1),
                    round(r.proj_g, 1), round(r.proj_a, 1), round(r.proj_p, 1),
                    round(r.proj_p_path_a, 1), round(r.proj_p_path_b, 1)]
                   for r in p.itertuples()]

    c1 = rd(NOUT / "hier_restatement.json")
    pg = rd(CONFIGS / "player_game_twoway_2018_2020.json")
    sm = rd(CONFIGS / "season_matched_comparison.json")
    bw = rd(CONFIGS / "hier_both_ways_original.json")
    tune = rd(NOUT / "hier_result.json")
    evidence = {
        "c1": None if c1 is None else {
            "verdict": c1["verdict"],
            **{k: c1["primary"][k] for k in ("n_games", "neurhl_h", "elo", "diff",
                                             "ci95", "p_two_sided", "clustered_p",
                                             "seasons_won", "seasons",
                                             "wild_cluster_p_exact")},
            "per_season": c1["primary"]["per_season"],
            "incl_2021": {k: c1["including_2021"][k] for k in
                          ("n_games", "diff", "p_two_sided")}},
        "tune": {"diff": tune["diff"], "p": tune["p_two_sided"],
                 "n": tune["n_games"],
                 "p_range": [min(s["p_two_sided"] for s in bw["sets"].values()),
                             max(s["p_two_sided"] for s in bw["sets"].values())]},
        "pg": {h: {"rel": v["relative_to_baseline"], "z2": v["z_twoway"]}
               for h, v in pg["heads"].items()} | {"n": pg["n_rows"]},
        "season": {k: {"mae": v["mae"], "crps": v["crps"]}
                   for k, v in sm["models"].items()},
    }

    gg = rd(NOUT / "g_gates.json")
    if gg:
        ss = gg["S_STOP"]
        evidence["g"] = {"n": gg["n_games"], "d_elo": ss["g_minus_elo"], "se_elo": ss["se_g_minus_elo"],
                         "d_h": ss["g_minus_h"], "se_h": ss["se_g_minus_h"],
                         "beat_elo": ss["seasons_beat_elo"], "beat_h": ss["seasons_beat_h"],
                         "pg": {h: v["rel_diff"] for h, v in gg["PG"]["heads"].items()},
                         "sstop": ss["pass"]}
    # full stat lines for the tables (technical page)
    tx, sx, gx = [], [], []
    if unified:
        tt = pd.read_csv(UNI / "teams_2027.csv")
        for r in tt.itertuples():
            tx.append({"ab": r.team, "gp": int(r.gp), "w": round(r.w, 1), "l": round(r.l, 1), "otl": round(r.otl, 1),
                       "rw": round(r.rw, 1), "pts": round(r.points, 1), "p10": int(round(r.points_p10)),
                       "p50": int(round(r.points_p50)), "p90": int(round(r.points_p90)),
                       "gf": round(r.goals_for, 1), "ga": round(r.goals_against, 1),
                       "sf": round(r.sog_for, 1), "sa": round(r.sog_against, 1),
                       "xgf": round(r.xgf_for, 1), "xga": round(r.xgf_against, 1),
                       "ppo": round(r.pp_opps_for, 1), "ppg": round(r.pp_goals_for, 1),
                       "pp": round(r.pp_pct, 1), "pk": round(r.pk_pct, 1),
                       "sh": round(r.shooting_pct, 2), "sv": round(r.save_pct * (100 if r.save_pct < 1 else 1), 2),
                       "po": round(r.playoff_pct, 1), "div_p": round(r.division_pct, 1),
                       "pres": round(r.presidents_pct, 1), "r2": round(r.round2_pct, 1),
                       "cf": round(r.conf_final_pct, 1), "fin": round(r.cup_final_pct, 1), "cup": round(r.cup_pct, 1)})
        ss_ = pd.read_csv(UNI / "skaters_2027.csv")
        ss_ = ss_[ss_.gp >= 1]
        for r in ss_.itertuples():
            sx.append([r.name if isinstance(r.name, str) else str(r.player_id), r.team, r.pos,
                       round(r.gp, 1), round(r.toi_per_gp, 1), round(r.goals, 1), round(r.assists, 1),
                       round(r.points, 1), round(r.points_p10, 1), round(r.points_p90, 1), round(r.sog, 1),
                       round(r.ixg, 1), round(r.shooting_pct, 1), round(r.pim, 1), round(r.hits, 1),
                       round(r.blocks, 1), round(r.fo_won / r.fo_taken * 100, 1) if r.fo_taken > 50 else None,
                       round(r.oi_xgf, 1), round(r.oi_xga, 1)])
        gg_ = pd.read_csv(UNI / "goalies_2027.csv")
        gg_ = gg_[gg_.starts >= 1]
        for r in gg_.itertuples():
            gx.append([r.name if isinstance(r.name, str) else str(r.player_id), r.team, round(r.starts, 1),
                       round(r.wins, 1), round(r.sa, 1), round(r.ga, 1), round(r.sv_pct, 4), round(r.gaa, 2),
                       round(r.shutouts, 1)])
    seal = rd(NOUT / "g_seal_result.json")
    lm = rd(CONFIGS / "live_models.json") or {}
    run = rd(UNI / "run_2027.json") or {}
    meta_x = {"release": lm.get("release", "NeurHL 1.0"), "rosters": run.get("rosters_date"),
              "m0": run.get("goal_mult_m0"), "draws": run.get("draws"), "sims": run.get("sims"),
              "holdout": None if not seal else {"n": seal["n"],
                                            "d_elo": seal["S1"]["diff"], "se_elo": seal["S1"]["se"],
                                            "p_elo": seal["S1"]["p"], "d_h": seal["S2"].get("diff"),
                                            "se_h": seal["S2"].get("se"), "p_h": seal["S2"].get("p")}}
    card = rd(NOUT / "live" / "scorecard_2027.json") or {}
    files = FROZEN_1_0 if unified else FROZEN
    sha = {f: hashlib.sha256((PROJ / f).read_bytes()).hexdigest() for f in files}
    frozen = "2026-09-25"
    if unified:
        frozen = json.loads((UNI / "run_2027.json").read_text())["created_utc"][:10]
    checks = rd(UNI / "checks_2027.json") if unified else None
    data = {"frozen": frozen, "sha256": sha, "game_sha": sha[files[0]], "unified": unified,
            "checks": {k: checks[k] for k in ("pass", "passed", "n")} if checks else None,
            "live": card, "tonight": tonight(),
            "teams": teams, "games": games, "players": players,
            "evidence": evidence, "teams_x": tx, "skaters_x": sx, "goalies": gx, "meta": meta_x}
    DOCS.mkdir(exist_ok=True)
    (DOCS / "data.js").write_text(
        "window.NEURHL = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"docs/data.js: {len(teams)} teams, {len(games)} games, "
          f"{len(players)} skaters; C1 {'recorded' if c1 else 'pending'}; "
          f"{(DOCS / 'data.js').stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
