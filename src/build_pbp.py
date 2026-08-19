"""Aggregate the raw NHL play-by-play corpus (PLAN_V5 D2) to team-season tables.

Reads data/raw/pbp/<season_end>/<gameId>.json.gz, writes
data/processed/pbp_team_seasons.csv with per (season_end, team):
  fo_w/fo_l + fo_pct_pbp        faceoff wins/losses (owner of faceoff event = winner)
  pen_taken/pen_drawn + net/gp  minors+majors+match (misconducts excluded)
  sat5 f/a + close variants     5v5 shot attempts (situationCode 1551), "close" =
                                score within 1 at event time (running score tracked)
  gp
Blocked-shot team attribution is resolved empirically in build_v5.py against the
official realtime satPct (both attributions are computed here).
"""
import gzip
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from players import MP_FRAN

PROJ = Path(__file__).resolve().parents[1]
RAW = PROJ / "data" / "raw" / "pbp"
PROC = PROJ / "data" / "processed"
PEN_COUNTED = {"MIN", "MAJ", "MAT", "BEN"}   # exclude MIS/GMI (no manpower change)


def parse_season(sdir: Path) -> dict:
    end = int(sdir.name)
    agg: dict = {}

    def A(team_id, id2abbr):
        t = MP_FRAN.get(id2abbr[team_id], id2abbr[team_id])
        key = (end, t)
        if key not in agg:
            agg[key] = {k: 0 for k in
                        ("fo_w", "fo_l", "pen_taken", "pen_drawn", "sat_f", "sat_a",
                         "satc_f", "satc_a", "blk_own", "blk_opp", "gp")}
        return agg[key]

    for f in sorted(sdir.glob("*.json.gz")):
        with gzip.open(f, "rt") as fh:
            g = json.load(fh)
        home, away = g["homeTeam"], g["awayTeam"]
        id2abbr = {home["id"]: home["abbrev"], away["id"]: away["abbrev"]}
        other = {home["id"]: away["id"], away["id"]: home["id"]}
        A(home["id"], id2abbr)["gp"] += 1
        A(away["id"], id2abbr)["gp"] += 1
        hs = as_ = 0
        for p in g["plays"]:
            det = p.get("details") or {}
            own = det.get("eventOwnerTeamId")
            typ = p["typeDescKey"]
            if typ == "goal":
                hs = det.get("homeScore", hs)
                as_ = det.get("awayScore", as_)
            if own not in id2abbr:
                continue
            if typ == "faceoff":
                A(own, id2abbr)["fo_w"] += 1
                A(other[own], id2abbr)["fo_l"] += 1
            elif typ == "penalty":
                if det.get("typeCode") in PEN_COUNTED:
                    A(own, id2abbr)["pen_taken"] += 1
                    A(other[own], id2abbr)["pen_drawn"] += 1
            elif typ in ("shot-on-goal", "missed-shot", "goal", "blocked-shot"):
                sc = p.get("situationCode") or ""
                if sc != "1551":
                    continue
                # close BEFORE this event (goal already applied above for goals —
                # use pre-event margin for goals by backing the goal out)
                margin = abs(hs - as_)
                if typ == "goal":
                    margin = abs((hs - (own == home["id"]))
                                 - (as_ - (own == away["id"])))
                if typ == "blocked-shot":
                    A(own, id2abbr)["blk_own"] += 1
                    A(other[own], id2abbr)["blk_opp"] += 1
                    continue
                A(own, id2abbr)["sat_f"] += 1
                A(other[own], id2abbr)["sat_a"] += 1
                if margin <= 1:
                    A(own, id2abbr)["satc_f"] += 1
                    A(other[own], id2abbr)["satc_a"] += 1
    return agg


def main():
    sdirs = sorted(d for d in RAW.iterdir() if d.is_dir())
    rows = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for agg in ex.map(parse_season, sdirs):
            for (end, t), v in agg.items():
                rows.append({"season_end": end, "team": t, **v})
    df = pd.DataFrame(rows).sort_values(["season_end", "team"])
    df["fo_pct_pbp"] = df.fo_w / (df.fo_w + df.fo_l)
    df["pen_net_g_pbp"] = (df.pen_drawn - df.pen_taken) / df.gp
    # two blocked-shot attributions; build_v5 resolves against official satPct
    df["sat5_pct_own"] = (df.sat_f + df.blk_own) / \
        (df.sat_f + df.blk_own + df.sat_a + df.blk_opp)
    df["sat5_pct_opp"] = (df.sat_f + df.blk_opp) / \
        (df.sat_f + df.blk_opp + df.sat_a + df.blk_own)
    df["sat5_close_pct"] = df.satc_f / (df.satc_f + df.satc_a)
    df.round(6).to_csv(PROC / "pbp_team_seasons.csv", index=False)
    print(f"pbp_team_seasons: {len(df)} rows, seasons "
          f"{df.season_end.min()}-{df.season_end.max()}")
    print(df.groupby("season_end").gp.sum().rename("team-games").to_string())


if __name__ == "__main__":
    main()
