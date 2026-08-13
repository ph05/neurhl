"""Build unified analysis tables from raw Hockey-Reference + MoneyPuck files.

Outputs:
  data/processed/games.csv         one row per game, franchise-coded
  data/processed/team_seasons.csv  per franchise-season: record, points, GF/GA, xG (2007-08+)

Franchise continuity: ATL->WPG (2011 relocation), PHX/ARI->UTA (2024). Codes = current MP codes.
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
RAW = PROJ / "data" / "raw"
OUT = PROJ / "data" / "processed"

NAME2FRAN = {
    "Anaheim Ducks": "ANA", "Mighty Ducks of Anaheim": "ANA",
    "Arizona Coyotes": "UTA", "Phoenix Coyotes": "UTA", "Utah Hockey Club": "UTA",
    "Utah Mammoth": "UTA",
    "Atlanta Thrashers": "WPG", "Winnipeg Jets": "WPG",
    "Boston Bruins": "BOS", "Buffalo Sabres": "BUF", "Calgary Flames": "CGY",
    "Carolina Hurricanes": "CAR", "Chicago Blackhawks": "CHI", "Colorado Avalanche": "COL",
    "Columbus Blue Jackets": "CBJ", "Dallas Stars": "DAL", "Detroit Red Wings": "DET",
    "Edmonton Oilers": "EDM", "Florida Panthers": "FLA", "Los Angeles Kings": "LAK",
    "Minnesota Wild": "MIN", "Montreal Canadiens": "MTL", "Nashville Predators": "NSH",
    "New Jersey Devils": "NJD", "New York Islanders": "NYI", "New York Rangers": "NYR",
    "Ottawa Senators": "OTT", "Philadelphia Flyers": "PHI", "Pittsburgh Penguins": "PIT",
    "San Jose Sharks": "SJS", "Seattle Kraken": "SEA", "St. Louis Blues": "STL",
    "Tampa Bay Lightning": "TBL", "Toronto Maple Leafs": "TOR",
    "Vancouver Canucks": "VAN", "Vegas Golden Knights": "VGK", "Washington Capitals": "WSH",
}
MP_FRAN = {"ATL": "WPG", "ARI": "UTA",  # MoneyPuck code -> franchise code
           "L.A": "LAK", "N.J": "NJD", "S.J": "SJS", "T.B": "TBL"}

EXPECTED_GP = {2013: 48, 2021: 56}  # season_end -> uniform GP; 2020 is ragged (68-71)


def build_games() -> pd.DataFrame:
    frames = []
    for f in sorted(RAW.glob("hr_games_*.csv")):
        frames.append(pd.read_csv(f, keep_default_na=False))
    g = pd.concat(frames, ignore_index=True)
    g["home"] = g["home"].map(NAME2FRAN)
    g["away"] = g["away"].map(NAME2FRAN)
    assert not g["home"].isna().any() and not g["away"].isna().any(), "unmapped team name"
    g["date"] = pd.to_datetime(g["date"])
    g = g.sort_values(["date"]).reset_index(drop=True)
    g["went_ot"] = g["ot"].astype(str).str.contains("OT")
    g["went_so"] = g["ot"].astype(str).eq("SO")
    g["margin"] = (g["home_g"] - g["away_g"]).astype(int)
    assert (g["margin"] != 0).all(), "tie found (impossible post-2005)"
    # duplicate-game check
    dup = g.duplicated(subset=["date", "home", "away"]).sum()
    assert dup == 0, f"{dup} duplicate games"
    return g


def build_team_seasons(g: pd.DataFrame) -> pd.DataFrame:
    r = g[g.game_type == "R"]
    home = r[["season_end", "home", "home_g", "away_g", "went_ot", "went_so"]].rename(
        columns={"home": "team", "home_g": "gf", "away_g": "ga"})
    away = r[["season_end", "away", "away_g", "home_g", "went_ot", "went_so"]].rename(
        columns={"away": "team", "away_g": "gf", "home_g": "ga"})
    t = pd.concat([home, away], ignore_index=True)
    t["win"] = t.gf > t.ga
    t["reg_win"] = t.win & ~t.went_ot & ~t.went_so
    t["otl"] = ~t.win & (t.went_ot | t.went_so)
    t["pts"] = 2 * t.win + 1 * t.otl
    ts = t.groupby(["season_end", "team"]).agg(
        gp=("win", "size"), w=("win", "sum"), reg_w=("reg_win", "sum"),
        otl=("otl", "sum"), pts=("pts", "sum"), gf=("gf", "sum"), ga=("ga", "sum"),
    ).reset_index()
    ts["l"] = ts.gp - ts.w - ts.otl
    ts["pts_pct"] = ts.pts / (2 * ts.gp)
    ts["gf_pct"] = ts.gf / (ts.gf + ts.ga)
    return ts


def add_moneypuck(ts: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for f in sorted(RAW.glob("mp_teams_*.csv")):
        mp = pd.read_csv(f)
        season_end = int(f.stem.split("_")[-1]) + 1
        m5 = mp[mp.situation == "5on5"][
            ["team", "scoreVenueAdjustedxGoalsFor", "scoreVenueAdjustedxGoalsAgainst"]
        ].rename(columns={"scoreVenueAdjustedxGoalsFor": "xgf5_adj",
                          "scoreVenueAdjustedxGoalsAgainst": "xga5_adj"})
        mall = mp[mp.situation == "all"][
            ["team", "xGoalsFor", "xGoalsAgainst", "goalsFor", "goalsAgainst",
             "shotsOnGoalFor", "shotsOnGoalAgainst"]
        ].rename(columns={"xGoalsFor": "xgf_all", "xGoalsAgainst": "xga_all",
                          "goalsFor": "mp_gf", "goalsAgainst": "mp_ga",
                          "shotsOnGoalFor": "sog_f", "shotsOnGoalAgainst": "sog_a"})
        m = m5.merge(mall, on="team")
        m["season_end"] = season_end
        m["team"] = m["team"].replace(MP_FRAN)
        frames.append(m)
    mp = pd.concat(frames, ignore_index=True)
    out = ts.merge(mp, on=["season_end", "team"], how="left")
    out["xg_pct_5v5"] = out.xgf5_adj / (out.xgf5_adj + out.xga5_adj)
    out["xg_pct_all"] = out.xgf_all / (out.xgf_all + out.xga_all)
    # PDO (all situations, from MP shot data; excludes shootout)
    out["sh_pct"] = out.mp_gf / out.sog_f
    out["sv_pct"] = 1 - out.mp_ga / out.sog_a
    out["pdo"] = out.sh_pct + out.sv_pct
    return out


def validate(g: pd.DataFrame, ts: pd.DataFrame):
    for season, grp in ts.groupby("season_end"):
        n = len(grp)
        exp_n = 30 if season <= 2017 else (31 if season <= 2021 else 32)
        assert n == exp_n, f"{season}: {n} teams, expected {exp_n}"
        if season in EXPECTED_GP:
            assert (grp.gp == EXPECTED_GP[season]).all(), f"{season}: bad GP"
        elif season == 2020:
            assert grp.gp.between(68, 71).all(), f"{season}: GP out of range"
        else:
            assert (grp.gp == 82).all(), f"{season}: GP != 82"
    # points identity: league points = 2*games + #OT games (per season)
    r = g[g.game_type == "R"]
    for season, grp in r.groupby("season_end"):
        lp = ts[ts.season_end == season].pts.sum()
        expected = 2 * len(grp) + (grp.went_ot | grp.went_so).sum()
        assert lp == expected, f"{season}: points {lp} != {expected}"
    # xG present for 2008+
    missing = ts[(ts.season_end >= 2008) & ts.xg_pct_5v5.isna()]
    assert missing.empty, f"missing xG:\n{missing[['season_end','team']]}"
    # MP goals vs HR goals cross-check (tolerance: shootout-winner goals + scorekeeping)
    chk = ts.dropna(subset=["mp_gf"])
    diff = (chk.gf - chk.mp_gf).abs()
    frac_big = (diff > 12).mean()
    assert frac_big < 0.02, f"GF cross-check failing for {frac_big:.1%} of team-seasons"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    g = build_games()
    ts = build_team_seasons(g)
    ts = add_moneypuck(ts)
    validate(g, ts)
    g.to_csv(OUT / "games.csv", index=False)
    ts.to_csv(OUT / "team_seasons.csv", index=False)
    print(f"games: {len(g)} rows ({g.season_end.min()}-{g.season_end.max()}), "
          f"team_seasons: {len(ts)} rows")
    print("validation passed")


if __name__ == "__main__":
    main()
