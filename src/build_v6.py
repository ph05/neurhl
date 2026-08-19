"""Assemble v6 derived tables (PLAN_V6): coaches, EDGE players, team_seasons_v6.

Outputs (committed):
  data/processed/coaches.csv        (season_end, team, coaches, n_coaches,
                                     final_coach, coach_new, coach_tenure)
  data/processed/edge_players.csv   per player-season EDGE tracking (2022+)
  data/processed/team_seasons_v6.csv shift metrics + coach features per
                                     (season_end, team)
Cross-checks appended to output/v6_crosschecks.json.
"""
import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJ = Path(__file__).resolve().parents[1]
RAW = PROJ / "data" / "raw"
PROC = PROJ / "data" / "processed"
OUT = PROJ / "output"
HR_FRAN = {"MDA": "ANA", "PHX": "UTA", "ARI": "UTA", "ATL": "WPG", "VEG": "VGK"}


def coaches() -> pd.DataFrame:
    rows = []
    for p in sorted((RAW / "hr_html").glob("team_*_*.html")):
        _, code, yy = p.stem.split("_")
        team = HR_FRAN.get(code, code)
        m = re.search(r"Coach:</strong>(.*?)</p>", p.read_text(), re.DOTALL)
        if not m:
            continue
        names = re.findall(r'/coaches/([a-z0-9]+)c?\.html"?>([^<]+)</a>',
                           m.group(1))
        if not names:
            continue
        rows.append({"season_end": int(yy), "team": team,
                     "coaches": "; ".join(n for _, n in names),
                     "n_coaches": len(names),
                     "final_coach": names[-1][0]})
    df = pd.DataFrame(rows).drop_duplicates(["season_end", "team"]) \
        .sort_values(["team", "season_end"]).reset_index(drop=True)
    prev = df.rename(columns={"final_coach": "prev_final"})
    prev = prev.assign(season_end=prev.season_end + 1)[
        ["season_end", "team", "prev_final"]]
    df = df.merge(prev, on=["season_end", "team"], how="left")
    df["coach_new"] = ((df.n_coaches > 1)
                       | (df.prev_final.notna()
                          & (df.final_coach != df.prev_final))).astype(float)
    tenure = []
    run: dict = {}
    for r in df.itertuples():   # sorted by team, season
        key = r.team
        if run.get(key, (None, 0))[0] == r.final_coach and r.n_coaches == 1:
            run[key] = (r.final_coach, run[key][1] + 1)
        else:
            run[key] = (r.final_coach, 1)
        tenure.append(run[key][1])
    df["coach_tenure"] = tenure
    return df.drop(columns=["prev_final"])


def edge_players() -> pd.DataFrame:
    rows = []
    for p in sorted((RAW / "edge_players").glob("*.json.gz")):
        pid, end = p.stem.replace(".json", "").split("_")
        with gzip.open(p, "rt") as fh:
            d = json.load(fh)
        if not d:
            continue
        ss = d.get("skatingSpeed") or {}
        ds = d.get("totalDistanceSkated") or d.get("distanceSkated") or {}
        row = {"playerId": int(pid), "season_end": int(end)}
        for name, node, key in (("speed_max", ss.get("maxSkatingSpeed"), "imperial"),
                                ("speed_max2", ss.get("speedMax"), "imperial"),
                                ("bursts_20", ss.get("burstsOver20"), "value"),
                                ("bursts_22", ss.get("burstsOver22"), "value"),
                                ("dist_total", ds.get("total"), "imperial")):
            row[name] = (node or {}).get(key, np.nan) \
                if isinstance(node, dict) else np.nan
        row["speed_max"] = row.pop("speed_max") if np.isfinite(
            row.get("speed_max", np.nan)) else row.pop("speed_max2")
        row.pop("speed_max2", None)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    co = coaches()
    co.to_csv(PROC / "coaches.csv", index=False)
    ep = edge_players()
    if len(ep):
        ep.to_csv(PROC / "edge_players.csv", index=False)
    sh = pd.read_csv(PROC / "shift_team_seasons.csv")
    v6 = sh.merge(co[["season_end", "team", "coach_new", "coach_tenure",
                      "n_coaches"]], on=["season_end", "team"], how="outer") \
        .sort_values(["season_end", "team"])
    v6.round(6).to_csv(PROC / "team_seasons_v6.csv", index=False)

    ts = pd.read_csv(PROC / "team_seasons.csv")
    checks = {}
    m = sh.merge(ts[["season_end", "team", "gp"]], on=["season_end", "team"])
    m["hours_pg"] = m.team_toi_hr / m.gp
    checks["shift_toi_identity"] = {
        "n": len(m), "mean_skater_hours_per_game": round(float(m.hours_pg.mean()), 3),
        "note": "5 skaters + 1 goalie x ~60min = ~6h/game expected (penalties "
                "shave slightly; OT adds slightly)",
        "in_band_5p5_6p3": float(((m.hours_pg > 5.5) & (m.hours_pg < 6.3)).mean())}
    cov = co.groupby("season_end").team.nunique()
    exp = ts.groupby("season_end").team.nunique()
    checks["coach_coverage"] = {
        "seasons_full": int((cov.reindex(exp.index) == exp).sum()),
        "seasons_total": len(exp)}
    checks["coach_change_rate"] = round(float(co.coach_new.mean()), 3)
    old = json.loads((OUT / "v6_crosschecks.json").read_text()) \
        if (OUT / "v6_crosschecks.json").exists() else {}
    old.update(checks)
    (OUT / "v6_crosschecks.json").write_text(json.dumps(old, indent=2))
    print(f"coaches: {len(co)} rows; edge_players: {len(ep)}; "
          f"team_seasons_v6: {len(v6)}")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
