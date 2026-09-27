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
FROZEN = ["neurhl/output/games_2027.csv", "neurhl/output/projection_2027.csv",
          "neurhl/output/player_proj_2027.csv"]


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
    t = pd.read_csv(NOUT / "projection_2027.csv")
    teams = [{"ab": r.team, "name": meta.loc[r.team, "Team"],
              "div": meta.loc[r.team, "Division"], "conf": r.conf,
              "pts": round(r.proj_points, 1), "p10": int(r.p10), "p90": int(r.p90),
              "wins": round(r.proj_wins, 1), "po": round(r.playoff_pct, 1),
              "howe": round(float(meta.loc[r.team, "xPts"]), 1)}
             for r in t.itertuples()]

    g = pd.read_csv(NOUT / "games_2027.csv")
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

    card = rd(NOUT / "live" / "scorecard_2027.json") or {}
    sha = {f: hashlib.sha256((PROJ / f).read_bytes()).hexdigest() for f in FROZEN}
    data = {"frozen": "2026-09-25", "sha256": sha, "live": card, "tonight": tonight(),
            "teams": teams, "games": games, "players": players,
            "evidence": evidence}
    DOCS.mkdir(exist_ok=True)
    (DOCS / "data.js").write_text(
        "window.NEURHL = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"docs/data.js: {len(teams)} teams, {len(games)} games, "
          f"{len(players)} skaters; C1 {'recorded' if c1 else 'pending'}; "
          f"{(DOCS / 'data.js').stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
