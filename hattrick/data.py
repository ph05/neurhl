"""Data layer: every table HatTrick uses, loaded from the pre-cutoff snapshot.

Conventions
- season_end: the calendar year a season ends (2025-26 -> 2026).
- team codes are normalised to the current franchise (config.FRANCHISE).
- times on ice are in MINUTES.
- MoneyPuck season files are named by the year a season STARTS; they are
  re-keyed to season_end here.
"""
from __future__ import annotations

import functools
import json

import numpy as np
import pandas as pd

from hattrick import config as C
from hattrick.snapshot import path as snap

SITS = {"all": "all", "5on5": "ev", "5on4": "pp", "4on5": "sh", "other": "oth"}


def norm_team(s: pd.Series) -> pd.Series:
    return s.replace(C.FRANCHISE)


# --------------------------------------------------------------------------
# Skaters
# --------------------------------------------------------------------------
_SK_COLS = {
    "games_played": "gp", "icetime": "toi", "I_F_goals": "g",
    "I_F_primaryAssists": "a1", "I_F_secondaryAssists": "a2",
    "I_F_shotsOnGoal": "sog", "I_F_shotAttempts": "att",
    "I_F_unblockedShotAttempts": "fen", "I_F_xGoals": "ixg",
    "OnIce_F_xGoals": "oi_xgf", "OnIce_A_xGoals": "oi_xga",
    "OnIce_F_goals": "oi_gf", "OnIce_A_goals": "oi_ga",
    "OnIce_F_shotAttempts": "oi_cf", "OnIce_A_shotAttempts": "oi_ca",
    "OffIce_F_xGoals": "off_xgf", "OffIce_A_xGoals": "off_xga",
    "timeOnBench": "bench", "I_F_hits": "hits", "shotsBlockedByPlayer": "blk",
    "I_F_takeaways": "tk", "I_F_giveaways": "gv", "penalties": "pen",
    "penaltiesDrawn": "pend", "I_F_penalityMinutes": "pim",
    "faceoffsWon": "fow", "faceoffsLost": "fol",
    "I_F_oZoneShiftStarts": "ozs", "I_F_dZoneShiftStarts": "dzs",
}


@functools.lru_cache(maxsize=None)
def skater_seasons() -> pd.DataFrame:
    """One row per skater-season (all teams combined), wide by situation.

    Columns are <stat>_<sit> for sit in all/ev/pp/sh/oth; toi in minutes.
    """
    frames = []
    for y in range(2008, 2026):
        d = pd.read_csv(snap(f"data/raw/mp_skaters_{y}.csv"),
                        usecols=["playerId", "name", "team", "position",
                                 "situation", *_SK_COLS])
        d = d.rename(columns=_SK_COLS)
        d["toi"] = d.toi / 60.0
        d["bench"] = d.bench / 60.0
        d["season_end"] = y + 1
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["team"] = norm_team(d.team)
    d["sit"] = d.situation.map(SITS)
    stats = list(_SK_COLS.values())
    w = d.pivot_table(index=["playerId", "season_end"], columns="sit",
                      values=stats, aggfunc="sum")
    w.columns = [f"{s}_{t}" for s, t in w.columns]
    w = w.reset_index()
    meta = (d[d.sit == "all"][["playerId", "season_end", "name", "team", "position"]]
            .drop_duplicates(["playerId", "season_end"]))
    w = meta.merge(w, on=["playerId", "season_end"], how="right")
    w["gp"] = w.pop("gp_all")
    for c in [c for c in w.columns if c.startswith("gp_")]:
        del w[c]
    w["pos"] = np.where(w.position == "D", "D", "F")
    w["a_all"] = w.a1_all + w.a2_all
    w["p_all"] = w.g_all + w.a_all
    return w.rename(columns={"playerId": "player_id"})


@functools.lru_cache(maxsize=None)
def skater_team_seasons() -> pd.DataFrame:
    """player_id, season_end, team, toi (min), gp -- split by team (trades)."""
    d = pd.read_csv(snap("data/processed/panel_skater_team.csv"))
    d = d.rename(columns={"playerId": "player_id", "toi_min": "toi",
                          "games_played": "gp", "I_F_points": "pts"})
    d["team"] = norm_team(d.team)
    return d


# --------------------------------------------------------------------------
# Goalies
# --------------------------------------------------------------------------
_G_COLS = {"games_played": "gp", "icetime": "toi", "xGoals": "xga",
           "goals": "ga", "ongoal": "sa", "unblocked_shot_attempts": "fa"}


@functools.lru_cache(maxsize=None)
def goalie_seasons() -> pd.DataFrame:
    frames = []
    for y in range(2008, 2026):
        d = pd.read_csv(snap(f"data/raw/mp_goalies_{y}.csv"),
                        usecols=["playerId", "name", "team", "situation", *_G_COLS])
        d = d.rename(columns=_G_COLS)
        d["toi"] = d.toi / 60.0
        d["season_end"] = y + 1
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["team"] = norm_team(d.team)
    d["sit"] = d.situation.map(SITS)
    w = d.pivot_table(index=["playerId", "season_end"], columns="sit",
                      values=list(_G_COLS.values()), aggfunc="sum")
    w.columns = [f"{s}_{t}" for s, t in w.columns]
    w = w.reset_index()
    meta = (d[d.sit == "all"][["playerId", "season_end", "name", "team"]]
            .drop_duplicates(["playerId", "season_end"]))
    w = meta.merge(w, on=["playerId", "season_end"], how="right")
    w["gp"] = w.pop("gp_all")
    for c in [c for c in w.columns if c.startswith("gp_")]:
        del w[c]
    w["gsax_all"] = w.xga_all - w.ga_all
    return w.rename(columns={"playerId": "player_id"})


@functools.lru_cache(maxsize=None)
def goalie_team_seasons() -> pd.DataFrame:
    d = pd.read_csv(snap("data/processed/panel_goalie_team.csv"))
    d = d.rename(columns={"playerId": "player_id", "toi_min": "toi",
                          "games_played": "gp", "ongoal": "sa", "xGoals": "xga",
                          "goals": "ga"})
    d["team"] = norm_team(d.team)
    return d


# --------------------------------------------------------------------------
# Teams
# --------------------------------------------------------------------------
_T_COLS = {"games_played": "gp", "iceTime": "toi", "goalsFor": "gf",
           "goalsAgainst": "ga", "xGoalsFor": "xgf", "xGoalsAgainst": "xga",
           "shotsOnGoalFor": "sf", "shotsOnGoalAgainst": "sa",
           "shotAttemptsFor": "cf", "shotAttemptsAgainst": "ca",
           "scoreVenueAdjustedxGoalsFor": "axgf",
           "scoreVenueAdjustedxGoalsAgainst": "axga",
           "penaltiesFor": "pen_taken", "penaltiesAgainst": "pen_drawn"}


@functools.lru_cache(maxsize=None)
def team_seasons() -> pd.DataFrame:
    """MoneyPuck team seasons (regular season) by situation, wide."""
    frames = []
    for y in range(2008, 2026):
        d = pd.read_csv(snap(f"data/raw/mp_teams_{y}.csv"))
        d = d[["team", "situation", *[c for c in _T_COLS if c in d.columns]]]
        d = d.rename(columns=_T_COLS)
        d["toi"] = d.toi / 60.0
        d["season_end"] = y + 1
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["team"] = norm_team(d.team)
    d["sit"] = d.situation.map(SITS)
    w = d.pivot_table(index=["team", "season_end"], columns="sit",
                      values=[v for v in _T_COLS.values() if v in d.columns],
                      aggfunc="sum", dropna=False)
    w.columns = [f"{s}_{t}" for s, t in w.columns]
    w = w.reset_index()
    w["gp"] = w.pop("gp_all")
    for c in [c for c in w.columns if c.startswith("gp_")]:
        del w[c]
    return w


@functools.lru_cache(maxsize=None)
def games() -> pd.DataFrame:
    """All games 2005-06 .. 2025-26 with final scores and extra-time flags.

    result columns: home_win (1/0), extra ('REG'|'OT'|'SO').
    """
    g = pd.read_csv(snap("data/processed/games.csv"), parse_dates=["date"])
    g["home"] = norm_team(g.home)
    g["away"] = norm_team(g.away)
    g["home_win"] = (g.home_g > g.away_g).astype(int)
    g["extra"] = np.where(g.went_so, "SO", np.where(g.went_ot, "OT", "REG"))
    g = g.sort_values(["date", "home"]).reset_index(drop=True)
    return g


def regular_season(season_end: int | None = None) -> pd.DataFrame:
    g = games()
    g = g[g.game_type == "R"]
    if season_end is not None:
        g = g[g.season_end == season_end]
    return g.reset_index(drop=True)


@functools.lru_cache(maxsize=None)
def travel() -> pd.DataFrame:
    t = pd.read_csv(snap("data/processed/travel_games.csv"), parse_dates=["date"])
    t["home"] = norm_team(t.home)
    t["away"] = norm_team(t.away)
    return t


# --------------------------------------------------------------------------
# People
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def bios() -> pd.DataFrame:
    """player_id -> birth date, position (panel bios + 2026-27 rosters)."""
    b = pd.read_csv(snap("data/processed/panel_bios.csv"), parse_dates=["birthDate"])
    b = b.rename(columns={"playerId": "player_id", "birthDate": "birth"})
    r = rosters_2027()[["player_id", "birth", "pos_raw"]].rename(
        columns={"pos_raw": "position"})
    b = pd.concat([r, b[["player_id", "birth", "position"]]]).drop_duplicates(
        "player_id")
    return b.reset_index(drop=True)


def age_on(birth: pd.Series, season_end: int) -> pd.Series:
    """Age on Feb 1 of the season (the usual hockey age convention)."""
    ref = pd.Timestamp(f"{season_end}-02-01")
    return (ref - pd.to_datetime(birth)).dt.days / 365.25


@functools.lru_cache(maxsize=None)
def rosters_2027() -> pd.DataFrame:
    r = pd.read_csv(snap("data/raw/rosters/2026-09-29/rosters.csv"),
                    parse_dates=["birthdate"])
    r = r.rename(columns={"birthdate": "birth", "pos": "pos_raw"})
    r["name"] = r["first"] + " " + r["last"]
    r["grp"] = np.where(r.pos_raw == "G", "G", np.where(r.pos_raw == "D", "D", "F"))
    return r


@functools.lru_cache(maxsize=None)
def schedule_2027() -> pd.DataFrame:
    s = pd.read_csv(snap("data/raw/nhl_schedule_20262027.csv"), parse_dates=["date"])
    return s.sort_values(["date", "game_id"]).reset_index(drop=True)


@functools.lru_cache(maxsize=None)
def availability_raw_2027() -> pd.DataFrame:
    """Raw injury fields recorded 2026-09-28 (MoneyPuck list, DailyFaceoff
    status) plus researched overrides and manual statuses from 2026-09-29.

    Only RAW fields are kept (injured flag, return date, DailyFaceoff status);
    NeurHL's modelled columns (p_dress, exp_gp, toi_proj...) are dropped.
    """
    a = pd.read_csv(snap("neurhl/output/neurhl_1_0/availability_2027.csv"))
    a = a[["team", "player_id", "name", "injured", "return_game", "return_date",
           "df_status"]].copy()
    o = pd.read_csv(snap("neurhl/configs/injury_overrides_2027.csv"))
    o = o.rename(columns={"games_out": "override_games_out", "note": "override_note"})
    a = a.merge(o[["player_id", "override_games_out", "override_note"]],
                on="player_id", how="outer")
    s = pd.read_csv(snap("data/manual/player_status_2027.csv"))
    a = a.merge(s[["player_id", "status", "note"]].rename(
        columns={"status": "manual_status", "note": "manual_note"}),
        on="player_id", how="outer")
    return a


@functools.lru_cache(maxsize=None)
def absences() -> pd.DataFrame:
    a = pd.read_csv(snap("data/processed/player_absences.csv"))
    a = a.rename(columns={"playerId": "player_id"})
    a["team"] = norm_team(a.team)
    return a


@functools.lru_cache(maxsize=None)
def prospects() -> pd.DataFrame:
    p = pd.read_csv(snap("data/processed/prospect_production.csv"))
    return p.rename(columns={"playerId": "player_id"})


@functools.lru_cache(maxsize=None)
def draft() -> pd.DataFrame:
    d = pd.read_csv(snap("data/processed/draft_join.csv"))
    return d.rename(columns={"playerId": "player_id"})


# --------------------------------------------------------------------------
# Market
# --------------------------------------------------------------------------
def american_to_prob(a) -> np.ndarray:
    a = np.asarray(a, float)
    return np.where(a < 0, -a / (-a + 100.0), 100.0 / (a + 100.0))


@functools.lru_cache(maxsize=None)
def market_totals_2027() -> pd.DataFrame:
    """Season points over/unders (2026-08-17) with no-vig over probability."""
    m = pd.read_csv(snap("data/market/nhl_totals_ou_2027.csv"))
    po, pu = american_to_prob(m.over_american), american_to_prob(m.under_american)
    m["p_over"] = po / (po + pu)
    return m


@functools.lru_cache(maxsize=None)
def market_cup_2027() -> pd.DataFrame:
    m = pd.read_csv(snap("data/market/nhl_cup_2027.csv"))
    m["p_raw"] = american_to_prob(m.american)
    m["p_cup"] = m.p_raw / m.p_raw.sum()
    return m


# --------------------------------------------------------------------------
# Per-game boxes (fastRhockey, NHL API mirror) 2010-11 .. 2023-24 (partial)
# --------------------------------------------------------------------------
FR = C.ROOT / "data" / "raw" / "fastrhockey"


def _mmss(s: pd.Series) -> pd.Series:
    x = s.fillna("0:00").astype(str).str.split(":", expand=True)
    return x[0].astype(float) + x[1].astype(float) / 60.0


@functools.lru_cache(maxsize=None)
def goalie_games() -> pd.DataFrame:
    """Per regular-season game and team: the starting goalie (most TOI; the
    first listed breaks ties), his TOI, shots and saves. Seasons 2011-2023."""
    frames = []
    for y in range(2011, 2025):
        f = FR / f"player_box_{y}.parquet"
        if not f.exists():
            continue
        p = pd.read_parquet(f, columns=["player_id", "player_full_name",
                                        "position_code", "home_away", "game_id",
                                        "goalie_stats_time_on_ice",
                                        "goalie_stats_shots", "goalie_stats_saves",
                                        "goalie_stats_decision"])
        p = p[p.position_code == "G"].copy()
        p["gtoi"] = _mmss(p.goalie_stats_time_on_ice)
        p["season_end"] = y
        frames.append(p)
    p = pd.concat(frames, ignore_index=True)
    p = p[(p.game_id // 10000) % 100 == 2]          # regular season only
    p = p.sort_values(["game_id", "home_away", "gtoi"], ascending=[True, True, False])
    st = p.drop_duplicates(["game_id", "home_away"]).rename(
        columns={"player_id": "goalie_id", "player_full_name": "goalie"})
    return st[["game_id", "season_end", "home_away", "goalie_id", "goalie",
               "gtoi", "goalie_stats_shots", "goalie_stats_saves"]]


@functools.lru_cache(maxsize=None)
def fr_schedule() -> pd.DataFrame:
    """game_id -> date, home, away (fastRhockey schedules, regular season).

    ``date`` is the local (US Eastern) calendar date, consistent with
    games.csv: fastRhockey's ``game_date`` is the UTC date, which puts every
    North American evening game on the next day, so the date is taken from
    ``game_date_time`` converted to America/New_York. Only game_id type 02
    (regular season) is kept; the 2020 qualifying round-robin is tagged "R"."""
    frames = []
    for y in range(2011, 2025):
        f = FR / f"nhl_schedule_{y}.parquet"
        if f.exists():
            s = pd.read_parquet(f, columns=["game_id", "game_date_time", "home_team_name",
                                            "away_team_name", "home_score",
                                            "away_score", "game_type_abbreviation"])
            s["season_end"] = y
            frames.append(s)
    s = pd.concat(frames, ignore_index=True)
    s = s[(s.game_type_abbreviation == "R") & ((s.game_id // 10000) % 100 == 2)].copy()
    names = team_names()
    s["home"] = s.home_team_name.map(names)
    s["away"] = s.away_team_name.map(names)
    s["date"] = (s.game_date_time.dt.tz_convert("America/New_York")
                 .dt.tz_localize(None).dt.normalize().astype("datetime64[us]"))
    return s[["game_id", "season_end", "date", "home", "away", "home_score",
              "away_score"]]


@functools.lru_cache(maxsize=None)
def opening_rosters(season_end: int, n_games: int = 10) -> pd.DataFrame:
    """Opening-roster proxy: everyone who dressed in any of a team's first
    n_games regular-season games (fastRhockey boxes, 2011-2024; 2024 is
    partial but covers every team's first 10 games).

    Columns: player_id, team, grp ('F'|'D'|'G'), n_dressed. A player who
    dressed for two teams inside the window keeps the one with more games.
    Raises FileNotFoundError when the season's boxes are absent.
    """
    f = FR / f"player_box_{season_end}.parquet"
    if not f.exists():
        raise FileNotFoundError(f"no fastRhockey boxes for {season_end}")
    p = pd.read_parquet(f, columns=["player_id", "position_code", "home_away",
                                    "game_id"])
    p = p[(p.game_id // 10000) % 100 == 2]
    s = fr_schedule()
    s = s[s.season_end == season_end][["game_id", "date", "home", "away"]]
    p = p.merge(s, on="game_id")
    p["team"] = np.where(p.home_away.str.lower() == "home", p.home, p.away)
    tg = p[["team", "game_id", "date"]].drop_duplicates().sort_values(["team", "date", "game_id"])
    tg["k"] = tg.groupby("team").cumcount()
    first = tg[tg.k < n_games][["team", "game_id"]]
    p = p.merge(first, on=["team", "game_id"])
    p["grp"] = np.where(p.position_code == "G", "G",
                        np.where(p.position_code == "D", "D", "F"))
    r = p.groupby(["player_id", "team", "grp"], as_index=False).size().rename(
        columns={"size": "n_dressed"})
    r = r.sort_values("n_dressed", ascending=False).drop_duplicates("player_id")
    return r.reset_index(drop=True)


def team_names() -> dict:
    return {
        "Anaheim Ducks": "ANA", "Mighty Ducks of Anaheim": "ANA", "Arizona Coyotes": "UTA",
        "Phoenix Coyotes": "UTA", "Utah Hockey Club": "UTA", "Utah Mammoth": "UTA",
        "Atlanta Thrashers": "WPG", "Boston Bruins": "BOS", "Buffalo Sabres": "BUF",
        "Calgary Flames": "CGY", "Carolina Hurricanes": "CAR", "Chicago Blackhawks": "CHI",
        "Colorado Avalanche": "COL", "Columbus Blue Jackets": "CBJ", "Dallas Stars": "DAL",
        "Detroit Red Wings": "DET", "Edmonton Oilers": "EDM", "Florida Panthers": "FLA",
        "Los Angeles Kings": "LAK", "Minnesota Wild": "MIN", "Montréal Canadiens": "MTL",
        "Montreal Canadiens": "MTL", "Nashville Predators": "NSH", "New Jersey Devils": "NJD",
        "New York Islanders": "NYI", "New York Rangers": "NYR", "Ottawa Senators": "OTT",
        "Philadelphia Flyers": "PHI", "Pittsburgh Penguins": "PIT", "San Jose Sharks": "SJS",
        "Seattle Kraken": "SEA", "St. Louis Blues": "STL", "Tampa Bay Lightning": "TBL",
        "Toronto Maple Leafs": "TOR", "Vancouver Canucks": "VAN",
        "Vegas Golden Knights": "VGK", "Washington Capitals": "WSH", "Winnipeg Jets": "WPG",
    }


# --------------------------------------------------------------------------
# NeurHL published forecasts (comparison only; never used as model inputs)
# --------------------------------------------------------------------------
def neurhl_release(release: str, table: str) -> pd.DataFrame:
    """Read a NeurHL release table. Pre-cutoff releases (1.0, 1.1) come from the
    snapshot; 1.2 and 1.3 (published after the first puck drop) from HEAD."""
    rel = {"1.0": "neurhl/output/neurhl_1_0/{t}_2027.csv",
           "1.1": "neurhl/output/neurhl_1_1/season/{t}_2027.csv",
           "1.2": "neurhl/output/neurhl_1_2/season/{t}_2027.csv",
           "1.3": "neurhl/output/neurhl_1_3/season/{t}_2027.csv"}[release].format(t=table)
    if release in ("1.0", "1.1"):
        return pd.read_csv(snap(rel))
    return pd.read_csv(C.ROOT / rel)


def write_json(obj, p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, default=float))
