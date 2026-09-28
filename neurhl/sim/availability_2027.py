"""NeurHL 1.0 unified simulator: player availability, depth charts and goalie starts (2026-27).

In every Monte Carlo draw the season simulator needs, for each scheduled team-game,
a dressed lineup of 12 forwards, 6 defencemen and one starting goalie. This module
is that sampling layer. Games played, ice time and every counting stat of a player
then follow from the lineups he dresses in.

  load(rosters_date)                    -> Roster: {team: TeamRoster}
  draw_lineups(games, roster, k, seed)  -> k dicts {game_id: {"home": {"skaters": [18 ids,
                                           12 forward slots then 6 defence slots], "goalie": id},
                                           "away": {...}}}
  expected_lineup(games, roster)        -> {game_id: {...}}, the single most likely lineup
  validate()                            -> 2023-24 backtest, model refit on seasons <= 2023

ROSTER (load). The dated NHL snapshot data/raw/rosters/<date>/rosters.csv, minus
status.unavailable(rosters_date) (data/manual/player_status_2027.csv). Added to it:
  injured   players in the DailyFaceoff "ir" group of the team's latest lines page on
            or before rosters_date who are NOT on the NHL snapshot (the roster endpoint
            omits injured reserve). They miss the team's next IR_RETURN_GAMES games and
            rejoin, at their normal rate, from the following game (return_game).
  reserves  players on the team in the previous snapshot (fetch_rosters.previous: the
            camp roster for the post-deadline snapshot, the August files before that)
            who are on no current roster and are at least RESERVE_MIN_AGE (tier 1), then
            the club's 2025-26 skaters with <= HIST_RES_MAX_GP games, aged <=
            HIST_RES_MAX_AGE, on no current roster (tier 2, its recent call-ups): the
            call-up pool. They dress only to fill a lineup the roster cannot fill.
Depth rank within F / D: projected ice time per game, the 2025-26 and 2024-25 NHL
TOI per game (games-weighted, 2024-25 at TOI_W_PREV) shrunk with TOI_PRIOR_GP games
of weight toward neurhl/output/player_proj_2027.csv toi_per_gp_min, else the
position default. Reserves rank below every roster player.

P_DRESS. The per-game probability that a skater who starts the season on the
roster (not injured) dresses: E[share of the team's games dressed]. A binomial GLM
(logit link) fitted on every opening-night skater of the target seasons
FIT_FROM..2024 (not 2013, 2021) with the realised share of his team's games dressed
(any team, so trades do not count as absences) as the outcome. Opening-night roster
= dressed or listed as a scratch (NHL right-rail) in the team's first game.
Features, all from seasons before the target:
  shrunk share   (sum_l w_l n_l s_l + K m0) / (sum_l w_l n_l + K), l = 1..3 seasons
                 back, w = LAG_W, s_l = games dressed / team games from his first
                 listing that season to its end, n_l those team games, m0 the mean
                 outcome, K chosen on the training fit (deviance);  logit(.) enters
  log(1 + exposure)          sum_l w_l n_l
  age terms       max(age - 29, 0), max(24 - age, 0), age = season_start_year - birth_year
  position        is_D
  usage           z-score of recent TOI per game within position (0 without), and a
                  flag for no NHL game in the last two seasons
2025 and 2026 enter only as features for 2027 (their outcomes are reserved).

SAMPLER (draw_lineups). For each team-game and draw:
  1. each skater is AVAILABLE with probability p_avail (below); injured players are
     not available before their return game;
  2. the top 12 available roster forwards and top 6 defencemen by depth rank dress;
  3. a shortfall is filled from available reserves by depth and the available
     surplus of the other position: an open forward slot goes to the 7th
     defenceman (11F/7D) first in P_SEVEN_D of games, else to a reserve forward
     first; a forward plays defence only after the reserve defencemen. Then, as an
     emergency, unavailable reserves (call-ups), unavailable roster players from the
     BOTTOM of the depth chart, and last injured players (so 18 skaters whenever
     the team has 18);
  4. the starting goalie is sampled from the per-game start probabilities.
p_avail is p_dress adjusted for depth so that the sampler REPRODUCES p_dress: going
down the depth chart, p_avail = min(P_AVAIL_MAX, p_dress / P(a slot is open)),
with P(a slot is open) the Poisson-binomial probability that fewer than 12 (6)
players above him are available. For the top 9 F / top 4 D p_avail equals p_dress to
two decimals; a 13th forward is drawn available more often than he dresses, because
he dresses only when someone above him is out. (Drawing p_dress itself and then
truncating to the depth chart would dress depth players far less than p_dress and
hand the difference to whoever fills.) Reserves have p_avail = RESERVE_P_AVAIL.
Absences persist: a share LONG_FRAC of each player's absences come in multi-game
spells (a two-state Markov chain, mean length LONG_MEAN games, every player healthy
at the first game), the rest are single-game draws. The per-game availability is
unchanged; only the clustering is, which gives the realised spread of games
played. persist=False makes every game independent.

GOALIES. Starter = the available goalie with the most 2025-26 starts (any team),
then career starts; backup = the next. p_start(starter) = his recent start share,
(starts_2026 + 0.5 starts_2025) / (games dressed_2026 + 0.5 games dressed_2025),
clipped to [G_START_MIN, G_START_MAX]; the backup gets the rest; a third goalie gets 0
unless a starter is unavailable (injured goalies are out before their return game,
and the ranking is redone among those available; reserve goalies are used only
when the roster has fewer than two). Second game of a back-to-back (the team played
the previous calendar day, from the full 2026-27 schedule): the starter's probability
times B2B_STARTER_MULT, the difference to the backup.

Constants are declared below with their provenance. Never raises on a missing
DailyFaceoff page (no IR additions, noted); a team with fewer than 18 skaters
dresses what it has.

CLI: python neurhl/sim/availability_2027.py --rosters-date 2026-09-27 --summary
     python neurhl/sim/availability_2027.py --validate
     [--k 64] [--seed 20260929] [--no-persist] [--timing]
Writes neurhl/output/neurhl_1_0/availability_2027.csv (--summary) and
availability_validation_2024.json (--validate).
"""
import argparse
import gzip
import importlib.util
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import NOUT, RAW, TENSORS  # noqa: E402

log = logging.getLogger("neurhl.sim.availability_2027")

LIVE = Path(__file__).resolve().parents[1] / "live"
OUT_DIR = NOUT / "neurhl_1_0"
SEASON = 2027                   # season_end 2027 == 2026-27
N_F, N_D = 12, 6
FWD_POS = {"C", "L", "R"}

# ---- declared constants --------------------------------------------------------
IR_RETURN_GAMES = 15            # DailyFaceoff IR players off the NHL roster miss the team's
                                # next 15 games and rejoin from game 16 (declared; a flat
                                # prior, since DailyFaceoff gives no return dates)
G_START_MIN, G_START_MAX = 0.50, 0.72     # starter's p_start clip (declared)
B2B_STARTER_MULT = 0.54       # tuned on 2022-23: number-one goalies started 54% as often
                              # in the second game of a back-to-back (2023-24: 0.59)
TOI_W_PREV = 0.5                # weight of 2024-25 games against 2025-26 in recent TOI
TOI_PRIOR_GP = 10.0             # games of prior weight on the fallback TOI (small samples)
POS_DEFAULT_TOI = {"F": 10.0, "D": 14.0}  # min/game without NHL minutes or a projection;
                                # below the 2022-24 rookie means (F 12.2, D 16.0 for rookies
                                # with 10+ GP) so established NHLers rank above unknowns
LAG_W = (1.0, 0.5, 0.25)        # evidence weight of the last three seasons in p_dress
K_GRID = (5.0, 10.0, 20.0, 40.0, 80.0)    # prior games in the shrunk share (chosen on fit)
FIT_FROM = 2012                 # first target season (right-rail scratch lists start 2012)
BROKEN = {2013, 2021}           # never targets (48-game lockout; 56-game COVID season)
RIDGE = 1.0                     # L2 on the GLM slopes (not the intercept)
P_DRESS_CLIP = (0.02, 0.99)
P_AVAIL_MAX = 0.99
RESERVE_P_AVAIL = 0.90          # a reserve is available for a call-up 90% of games
P_SEVEN_D = 0.40                # an open forward slot goes to an available 7th defenceman (11F/7D)
                                # before a reserve forward in 40% of games. Tuned on 2022-23: lineups
                                # not 12F/6D 0.076 / 0.086 / 0.102 at 0.25 / 0.35 / 0.5, realised 0.093
RESERVE_MIN_AGE = 20            # younger cut players are assumed returned to junior
HIST_RES_MAX_GP = 40            # tier-2 reserves: the club's 2025-26 skaters with <= 40 GP ...
HIST_RES_MAX_AGE = 30           # ... aged <= 30, on no current roster (its recent call-ups)
# Absence clustering. Measured on single-team regulars (top-9 F / top-4 D by TOI per
# game, 20+ GP), seasons 2014-19 and 2022-23: spells of 4+ games carry 80% of missed
# games and average 10.9 games, with a heavy tail (games-weighted mean spell ~22).
# A geometric spell of mean 11 left the spread of games played too narrow, so the two
# constants were TUNED on 2022-23 (validate(V=2023), p_dress fitted on <= 2022; the
# 2023-24 validation season untouched): with 87% of absences in spells of mean 30
# games, one draw's SD of games played across regulars is 17.42 against a realised
# 17.43 and the GP histogram of opening-night skaters matches (80+ GP 0.295 vs 0.287).
# (grid: mean 11/20/25/30/40/45/60 x share 0.80/0.87/0.90/0.95)
LONG_FRAC = 0.87
LONG_MEAN = 30.0
DF_LOOKBACK_DAYS = 7            # latest DailyFaceoff lines page within a week of the date
SEED = 20260929
TEAM_IDX_KEY = "team"           # maps.json key for the historical team index


# ------------------------------------------------------------------ live helpers (by path)
def _live(name: str):
    """A neurhl/live module loaded by path: src/live.py shadows `live` and neurhl/status.py
    shadows `status`, so neither `import live.x` nor a plain `import status` is safe."""
    key = f"neurhl_live_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, LIVE / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


# ------------------------------------------------------------------ schedule
@lru_cache(maxsize=1)
def load_schedule() -> pd.DataFrame:
    """The 2026-27 regular season (game_id, date, home, away): the same files and dedup as
    sim/project_2027.load_schedule(), not imported because it pulls in the game model."""
    rows = []
    for f in sorted(RAW.glob("nhl_sched_*_20262027.json")):
        d = json.loads(f.read_text())
        for g in d.get("games", d if isinstance(d, list) else []):
            if int(g.get("gameType", 2)) != 2:
                continue
            rows.append({"game_id": int(g["id"]), "date": g.get("gameDate", ""),
                         "home": g["homeTeam"]["abbrev"], "away": g["awayTeam"]["abbrev"]})
    return (pd.DataFrame(rows, columns=["game_id", "date", "home", "away"])
            .drop_duplicates("game_id").sort_values(["date", "game_id"]).reset_index(drop=True))


def _prev_day(d: str) -> str:
    return (date.fromisoformat(str(d)[:10]) - timedelta(days=1)).isoformat()


def team_games(games: pd.DataFrame, context: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """team -> its games in `games` in date order: game_id, date, side, b2b.

    b2b = the team also plays the previous calendar day, looked up in `games` plus
    `context` (default: the full 2026-27 schedule), so a subset of games still sees
    the first leg of a back-to-back."""
    g = games[["game_id", "date", "home", "away"]].copy()
    g["date"] = g["date"].astype(str).str[:10]
    if context is None:
        try:
            context = load_schedule()
        except Exception as e:  # noqa: BLE001
            log.warning("2026-27 schedule unreadable (%s); back-to-backs from `games` only", e)
            context = g
    allg = pd.concat([context[["game_id", "date", "home", "away"]].astype({"date": str}), g])
    allg = allg.drop_duplicates("game_id")
    played = set(zip(allg["home"], allg["date"].str[:10])) | set(zip(allg["away"], allg["date"].str[:10]))
    long = pd.concat([g.assign(team=g["home"], side="home"), g.assign(team=g["away"], side="away")],
                     ignore_index=True)
    long["b2b"] = [(t, _prev_day(d)) in played for t, d in zip(long["team"], long["date"])]
    return {t: d.sort_values(["date", "game_id"]).reset_index(drop=True)[["game_id", "date", "side", "b2b"]]
            for t, d in long.groupby("team")}


# ------------------------------------------------------------------ history tables
@lru_cache(maxsize=None)
def _team_abbrev() -> dict[int, str]:
    m = json.loads((TENSORS / "maps.json").read_text())[TEAM_IDX_KEY]
    return {int(v): k for k, v in m.items()}


@lru_cache(maxsize=None)
def _bios() -> pd.DataFrame:
    b = pd.read_parquet(TENSORS / "career_bios.parquet", columns=["player_id", "pos_group", "birth_year"])
    return b.drop_duplicates("player_id").set_index("player_id")


@lru_cache(maxsize=None)
def _scratch_raw(s: int) -> pd.DataFrame:
    """Players listed as scratches (NHL right-rail gameInfo), one row per team-game."""
    rows = []
    for f in sorted((RAW / "right_rail" / str(s)).glob("*.json.gz")):
        try:
            gi = json.load(gzip.open(f, "rt")).get("gameInfo", {}) or {}
        except (OSError, ValueError):
            continue
        gid = int(f.name.split(".")[0])
        for is_home, key in ((True, "homeTeam"), (False, "awayTeam")):
            for p in (gi.get(key) or {}).get("scratches", []) or []:
                try:
                    rows.append((gid, is_home, int(p["id"])))
                except (KeyError, TypeError, ValueError):
                    continue
    return pd.DataFrame(rows, columns=["game_id", "is_home", "player_id"])


@lru_cache(maxsize=None)
def _season(s: int):
    """(team-games, dressed player-games, scratches) for regular season s; team = history index."""
    gc = pd.read_parquet(TENSORS / f"games_ctx_{s}.parquet",
                         columns=["game_id", "game_type", "date", "home_idx", "away_idx"])
    gc = gc[gc["game_type"] == 2].drop(columns="game_type").copy()
    gc["date"] = gc["date"].astype(str).str[:10]
    tg = pd.concat([gc[["game_id", "date", "home_idx"]].rename(columns={"home_idx": "team"}),
                    gc[["game_id", "date", "away_idx"]].rename(columns={"away_idx": "team"})])
    tg = tg.sort_values(["team", "date", "game_id"]).reset_index(drop=True)
    tg["gno"] = tg.groupby("team").cumcount() + 1
    pg = pd.read_parquet(TENSORS / f"player_games_{s}.parquet",
                         columns=["game_id", "game_type", "player_id", "is_home", "pos_group",
                                  "toi_sec", "goalie_start"])
    pg = pg[pg["game_type"] == 2].drop(columns="game_type").merge(gc, on="game_id")
    pg["team"] = np.where(pg["is_home"], pg["home_idx"], pg["away_idx"]).astype(int)
    sc = _scratch_raw(s).merge(gc, on="game_id")
    sc["team"] = np.where(sc["is_home"], sc["home_idx"], sc["away_idx"]).astype(int)
    return tg, pg, sc


def _pos_of(pids, pg: pd.DataFrame) -> pd.Series:
    """pos_group (0 F, 1 D, 2 G) from the season's dressed games, else career bios."""
    m = pg.groupby("player_id")["pos_group"].agg(lambda x: int(x.mode().iloc[0]))
    out = pd.Series(pids, index=pids).map(m)
    return out.fillna(pd.Series(pids, index=pids).map(_bios()["pos_group"])).fillna(0).astype(int)


@lru_cache(maxsize=None)
def _skater_season(s: int) -> pd.DataFrame:
    """Per skater in season s: D (games dressed, any team), toi_pg (min), n_span (team games
    from his first listing, dressed or scratch, to the season's end), share = D / n_span,
    G (games of his first team), pos_group."""
    tg, pg, sc = _season(s)
    sk = pg[pg["pos_group"] < 2]
    lst = pd.concat([pg[["player_id", "team", "date", "game_id"]], sc[["player_id", "team", "date", "game_id"]]])
    first = lst.sort_values(["date", "game_id"]).groupby("player_id").first()
    pos = _pos_of(first.index.to_numpy(), pg)
    first = first[pos.reindex(first.index).to_numpy() < 2]
    dates = {t: d["date"].to_numpy() for t, d in tg.groupby("team")}
    G = tg.groupby("team").size()
    n_span = np.array([len(dates[t]) - np.searchsorted(dates[t], d, side="left")
                       for t, d in zip(first["team"], first["date"])])
    agg = sk.groupby("player_id").agg(D=("game_id", "size"), toi=("toi_sec", "sum"))
    out = pd.DataFrame({"n_span": np.maximum(n_span, 1), "G": first["team"].map(G).to_numpy(),
                        "team": first["team"].to_numpy()}, index=first.index)
    out = out.join(agg, how="left").fillna({"D": 0, "toi": 0})
    out["D"] = out["D"].astype(int)
    out["share"] = (out["D"] / out["n_span"]).clip(upper=1.0)
    out["toi_pg"] = np.where(out["D"] > 0, out["toi"] / out["D"].clip(lower=1) / 60.0, np.nan)
    out["pos_group"] = pos.reindex(out.index).to_numpy()
    return out


@lru_cache(maxsize=None)
def _goalie_season(s: int) -> pd.DataFrame:
    """Per goalie in season s: starts and games dressed (starter or backup), any team."""
    p = TENSORS / f"player_games_{s}.parquet"
    if not p.exists():
        return pd.DataFrame(columns=["starts", "dressed"])
    g = pd.read_parquet(p, columns=["player_id", "game_type", "pos_group", "goalie_start"])
    g = g[(g["game_type"] == 2) & (g["pos_group"] == 2)]
    return g.groupby("player_id").agg(starts=("goalie_start", "sum"), dressed=("goalie_start", "size"))


@lru_cache(maxsize=None)
def _career_starts(through: int) -> pd.Series:
    parts = [_goalie_season(s)["starts"] for s in range(2008, through + 1)]
    parts = [p for p in parts if len(p)]
    return pd.concat(parts).groupby(level=0).sum() if parts else pd.Series(dtype=float)


def opening_roster(s: int) -> pd.DataFrame:
    """Historical opening-night rosters: dressed or listed as a scratch in the team's first game."""
    tg, pg, sc = _season(s)
    first = tg[tg["gno"] == 1][["team", "game_id"]]
    a = pg.merge(first, on=["team", "game_id"])[["team", "player_id"]]
    b = sc.merge(first, on=["team", "game_id"])[["team", "player_id"]]
    r = pd.concat([a, b]).drop_duplicates("player_id").reset_index(drop=True)
    r["pos_group"] = _pos_of(r["player_id"].to_numpy(), pg).to_numpy()
    return r


# ------------------------------------------------------------------ p_dress model
def _recent_toi(pids, seasons: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """(games-weighted TOI per game in minutes over two seasons, weighted games)."""
    num = np.zeros(len(pids))
    den = np.zeros(len(pids))
    for s, w in zip(seasons, (1.0, TOI_W_PREV)):
        try:
            t = _skater_season(s)
        except FileNotFoundError:
            continue
        D = pd.Series(pids).map(t["D"]).fillna(0).to_numpy(float)
        tp = pd.Series(pids).map(t["toi_pg"]).fillna(0).to_numpy(float)
        num += w * D * tp
        den += w * D
    return np.where(den > 0, num / np.maximum(den, 1e-9), np.nan), den


def features(V: int, pids, pos_group, birth_year) -> pd.DataFrame:
    """p_dress features for season V from seasons V-1..V-3 only."""
    pids = np.asarray(pids, dtype=np.int64)
    f = pd.DataFrame({"player_id": pids, "is_d": (np.asarray(pos_group) == 1).astype(float)})
    by = pd.Series(birth_year, dtype="float64").to_numpy()
    f["age"] = np.where(np.isfinite(by), (V - 1) - by, 26.0)
    num, exp = np.zeros(len(f)), np.zeros(len(f))
    for lag, w in enumerate(LAG_W, start=1):
        try:
            t = _skater_season(V - lag)
        except FileNotFoundError:
            continue
        n = f["player_id"].map(t["n_span"]).fillna(0).to_numpy(float)
        s = f["player_id"].map(t["share"]).fillna(0).to_numpy(float)
        num += w * n * s
        exp += w * n
    f["ev_num"], f["exposure"] = num, exp
    toi, gpw = _recent_toi(pids, (V - 1, V - 2))
    f["toi_rec"], f["toi_gp"] = toi, gpw
    return f


@dataclass
class AvailModel:
    beta: np.ndarray
    k_prior: float
    m0: float
    toi_mu: dict
    toi_sd: dict
    targets: list
    n_obs: int
    names: tuple = ("const", "logit_shrunk", "log1p_exposure", "age_over29", "age_under24",
                    "is_d", "toi_z", "no_recent_nhl")

    def as_dict(self) -> dict:
        return {"beta": dict(zip(self.names, [round(float(b), 5) for b in self.beta])),
                "k_prior": self.k_prior, "m0": round(self.m0, 5), "targets": self.targets,
                "n_obs": self.n_obs, "toi_mu": self.toi_mu, "toi_sd": self.toi_sd}


def _design(f: pd.DataFrame, k: float, m0: float, toi_mu: dict, toi_sd: dict) -> np.ndarray:
    shrunk = (f["ev_num"] + k * m0) / (f["exposure"] + k)
    sh = np.clip(shrunk.to_numpy(float), 0.02, 0.98)
    has = np.isfinite(f["toi_rec"].to_numpy(float))
    mu = np.where(f["is_d"] > 0, toi_mu["D"], toi_mu["F"])
    sd = np.where(f["is_d"] > 0, toi_sd["D"], toi_sd["F"])
    z = np.where(has, (np.nan_to_num(f["toi_rec"].to_numpy(float)) - mu) / sd, 0.0)
    age = f["age"].to_numpy(float)
    return np.column_stack([np.ones(len(f)), np.log(sh / (1 - sh)), np.log1p(f["exposure"].to_numpy(float)),
                            np.maximum(age - 29, 0), np.maximum(24 - age, 0), f["is_d"].to_numpy(float),
                            np.clip(z, -3, 3), (~has).astype(float)])


def _glm(X: np.ndarray, y: np.ndarray, n: np.ndarray, ridge: float = RIDGE) -> tuple[np.ndarray, float]:
    """Binomial logit GLM by IRLS; returns (beta, deviance)."""
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(y.sum() / max(n.sum() - y.sum(), 1e-9))
    P = np.eye(X.shape[1]) * ridge
    P[0, 0] = 0.0
    for _ in range(100):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        W = np.maximum(n * mu * (1 - mu), 1e-9)
        z = eta + (y - n * mu) / W
        b = np.linalg.solve(X.T @ (W[:, None] * X) + P, X.T @ (W * z))
        if np.max(np.abs(b - beta)) < 1e-9:
            beta = b
            break
        beta = b
    mu = np.clip(1.0 / (1.0 + np.exp(-(X @ beta))), 1e-9, 1 - 1e-9)
    with np.errstate(divide="ignore", invalid="ignore"):
        dev = 2 * np.sum(np.where(y > 0, y * np.log(y / (n * mu)), 0.0)
                         + np.where(n - y > 0, (n - y) * np.log((n - y) / (n * (1 - mu))), 0.0))
    return beta, float(dev)


def target_frame(V: int) -> pd.DataFrame:
    """Opening-night skaters of season V with features and the realised dressed share."""
    r = opening_roster(V)
    r = r[r["pos_group"] < 2].reset_index(drop=True)
    by = r["player_id"].map(_bios()["birth_year"])
    f = features(V, r["player_id"], r["pos_group"], by)
    t = _skater_season(V)
    tg, _, _ = _season(V)
    G = tg.groupby("team").size()
    f["team"] = r["team"].to_numpy()
    f["G"] = r["team"].map(G).to_numpy(float)
    f["D"] = np.minimum(f["player_id"].map(t["D"]).fillna(0).to_numpy(float), f["G"])
    f["season"] = V
    return f


@lru_cache(maxsize=4)
def fit_model(last_target: int) -> AvailModel:
    """Fit the p_dress GLM on target seasons FIT_FROM..last_target (not BROKEN)."""
    seasons = [v for v in range(FIT_FROM, last_target + 1) if v not in BROKEN]
    tr = pd.concat([target_frame(v) for v in seasons], ignore_index=True)
    m0 = float(tr["D"].sum() / tr["G"].sum())
    has = np.isfinite(tr["toi_rec"])
    toi_mu = {k: float(tr.loc[has & (tr["is_d"] == v), "toi_rec"].mean()) for k, v in (("F", 0), ("D", 1))}
    toi_sd = {k: float(tr.loc[has & (tr["is_d"] == v), "toi_rec"].std()) for k, v in (("F", 0), ("D", 1))}
    best = None
    for k in K_GRID:
        X = _design(tr, k, m0, toi_mu, toi_sd)
        beta, dev = _glm(X, tr["D"].to_numpy(float), tr["G"].to_numpy(float))
        if best is None or dev < best[2]:
            best = (k, beta, dev)
    k, beta, _ = best
    return AvailModel(beta=beta, k_prior=k, m0=m0, toi_mu=toi_mu, toi_sd=toi_sd,
                      targets=seasons, n_obs=len(tr))


def predict_p_dress(model: AvailModel, f: pd.DataFrame) -> np.ndarray:
    X = _design(f, model.k_prior, model.m0, model.toi_mu, model.toi_sd)
    return np.clip(1.0 / (1.0 + np.exp(-(X @ model.beta))), *P_DRESS_CLIP)


# ------------------------------------------------------------------ depth adjustment / persistence
def depth_adjust(p_dress: np.ndarray, slots: int) -> np.ndarray:
    """Draw probability per player (depth order) such that P(dresses) = p_dress, given that he
    dresses only if fewer than `slots` players above him are available (Poisson-binomial)."""
    dist = np.array([1.0])
    out = np.empty(len(p_dress))
    for i, p in enumerate(p_dress):
        p_open = float(dist[:slots].sum())
        pa = min(P_AVAIL_MAX, max(p, p / max(p_open, 1e-9)))
        out[i] = pa
        dist = np.convolve(dist, [1.0 - pa, pa])
    return out


def persistence(p_avail: np.ndarray, m: int, persist: bool = True) -> tuple[np.ndarray, np.ndarray, float]:
    """(h, s, e): per-game hazard of starting a long spell, single-game absence probability and
    the spell exit probability, so that a player healthy at game 0 averages p_avail over m games."""
    p_avail = np.asarray(p_avail, float)
    q = 1.0 - p_avail
    if not persist or m <= 1:
        return np.zeros_like(q), q, 1.0
    e = 1.0 / LONG_MEAN
    ql = LONG_FRAC * q
    j = np.arange(m)

    def mean_long(h):
        pi = h / (h + e)
        lam = 1.0 - h - e
        return pi * (1.0 - np.mean(lam[:, None] ** j[None, :], axis=1))

    lo, hi = np.zeros_like(q), np.full_like(q, 0.5)
    for _ in range(50):
        mid = (lo + hi) / 2
        big = mean_long(np.maximum(mid, 1e-12)) > ql
        hi = np.where(big, mid, hi)
        lo = np.where(big, lo, mid)
    h = np.where(ql > 0, (lo + hi) / 2, 0.0)
    s = np.clip(1.0 - p_avail / np.maximum(1.0 - ql, 1e-9), 0.0, 1.0)
    return h, s, e


# ------------------------------------------------------------------ roster
@dataclass
class TeamRoster:
    team: str
    skaters: pd.DataFrame      # sampler order: roster F, roster D (depth order), reserve F, reserve D
    goalies: pd.DataFrame      # roster goalies by rank, then reserves
    notes: list = field(default_factory=list)


class Roster(dict):
    """team -> TeamRoster; `meta` holds the snapshot dates and sources."""

    def __init__(self, *a, meta: dict | None = None, **kw):
        super().__init__(*a, **kw)
        self.meta = meta or {}


def _slug(team: str) -> str:
    pdf = _live("parse_dailyfaceoff")
    if team == "UTA":
        return "utah-mammoth"      # snapshot files are always saved under the current Utah slug
    for s, t in pdf.SLUG_TO_NHL.items():
        if t == team:
            return s
    raise KeyError(team)


@lru_cache(maxsize=64)
def _lines(d: str) -> pd.DataFrame:
    return _live("parse_dailyfaceoff").parse_lines(Path(d))


def df_lines_dir(team: str, on_or_before: str) -> Path | None:
    """Directory of the team's latest DailyFaceoff lines page on or before the date:
    <date>/<HHMM>/ (latest intraday) else <date>/, looking back DF_LOOKBACK_DAYS days."""
    snap = _live("parse_dailyfaceoff").SNAP
    if not snap.exists():
        return None
    fname = f"df_lines_{_slug(team)}.html.gz"
    days = sorted(p.name for p in snap.iterdir()
                  if p.is_dir() and re.match(r"^\d{4}-\d\d-\d\d$", p.name) and p.name <= on_or_before)
    lim = (date.fromisoformat(on_or_before[:10]) - timedelta(days=DF_LOOKBACK_DAYS)).isoformat()
    for day in reversed(days):
        if day < lim:
            break
        dd = snap / day
        subs = sorted(s.name for s in dd.iterdir() if s.is_dir() and re.match(r"^\d{4}$", s.name)
                      and (s / fname).exists())
        if subs:
            return dd / subs[-1]
        if (dd / fname).exists():
            return dd
    return None


def _df_ir(teams: list[str], rosters_date: str, rosters: pd.DataFrame, notes: dict) -> pd.DataFrame:
    """DailyFaceoff `ir` group rows mapped to NHL ids, with on_roster."""
    idm = _live("id_map")
    parts = []
    for team in teams:
        try:
            d = df_lines_dir(team, rosters_date)
            if d is None:
                notes.setdefault(team, []).append("no DailyFaceoff lines page within a week; no IR additions")
                continue
            L = _lines(str(d))
            L = L[(L["team_abbrev"] == team) & L["group"].eq("ir")]
            if len(L):
                parts.append(L.assign(df_dir=str(d)))
        except Exception as e:  # noqa: BLE001
            notes.setdefault(team, []).append(f"DailyFaceoff IR unreadable ({type(e).__name__}: {e})")
    if not parts:
        return pd.DataFrame(columns=["team_abbrev", "name", "player_id", "on_roster", "injury_status", "df_dir"])
    ir = pd.concat(parts, ignore_index=True)
    m, un = idm.map_ids(ir, rosters, idm.load_aliases(), idm.load_fallback())
    for r in un.itertuples():
        notes.setdefault(r.team_abbrev, []).append(f"DailyFaceoff IR player unmatched: {r.name}")
    return m[m["player_id"].notna()].drop_duplicates(["team_abbrev", "player_id"])


def _history_reserves(s: int, taken: set) -> list[dict]:
    """Tier-2 reserves: skaters who dressed for a club in season s (their last club) with at most
    HIST_RES_MAX_GP games, aged at most HIST_RES_MAX_AGE at the next season's start, on no current
    roster: the club's recent call-ups, used only after the snapshot reserves."""
    _, pg, _ = _season(s)
    sk = pg[pg["pos_group"] < 2].sort_values(["date", "game_id"])
    last = sk.groupby("player_id").agg(team=("team", "last"), gp=("game_id", "size"),
                                       pos=("pos_group", lambda x: int(x.mode().iloc[0])))
    last = last[(last["gp"] <= HIST_RES_MAX_GP) & ~last.index.isin(taken)]
    by = _bios()["birth_year"]
    ab = _team_abbrev()
    out = []
    for pid, r in last.iterrows():
        b = by.get(pid)
        if b is None or not np.isfinite(b) or s - b > HIST_RES_MAX_AGE or s - b < RESERVE_MIN_AGE:
            continue
        out.append({"team": ab.get(int(r["team"])), "player_id": int(pid), "name": _hist_name(int(pid)),
                    "pos": "D" if r["pos"] == 1 else "C", "birthdate": f"{int(b)}", "status": "reserve", "tier": 2})
    return [o for o in out if o["team"]]


@lru_cache(maxsize=1)
def _names_lookup() -> dict[int, str]:
    out = {}
    try:
        mp = pd.read_csv(RAW / "mp_lookup.csv", usecols=["playerId", "name"])
        out.update({int(a): str(b) for a, b in zip(mp["playerId"], mp["name"])})
    except Exception:  # noqa: BLE001
        pass
    try:
        fr = _live("fetch_rosters")
        for p in sorted(fr.ROSTERS.glob("*/rosters.csv")):
            r = pd.read_csv(p)
            out.update({int(a): f"{b} {c}" for a, b, c in zip(r["player_id"], r["first"], r["last"])})
    except Exception:  # noqa: BLE001
        pass
    return out


def _hist_name(pid: int) -> str:
    return _names_lookup().get(pid, str(pid))


def _fallback_toi() -> dict[int, float]:
    p = NOUT / "player_proj_2027.csv"
    try:
        pp = pd.read_csv(p, usecols=["player_id", "toi_per_gp_min"])
        return {int(a): float(b) for a, b in zip(pp["player_id"], pp["toi_per_gp_min"]) if np.isfinite(b)}
    except Exception as e:  # noqa: BLE001
        log.warning("%s unreadable (%s); position defaults only", p, e)
        return {}


def project_toi(pids, grp, V: int, fallback: dict | None = None) -> tuple[np.ndarray, list[str]]:
    """Projected minutes per game: seasons V-1 (weight 1) and V-2 (TOI_W_PREV) shrunk with
    TOI_PRIOR_GP games toward the fallback (player projection, else position default)."""
    fallback = fallback or {}
    toi, gpw = _recent_toi(np.asarray(pids, dtype=np.int64), (V - 1, V - 2))
    out, src = np.empty(len(toi)), []
    for i, (pid, g) in enumerate(zip(pids, grp)):
        fb = fallback.get(int(pid))
        prior = fb if fb is not None else POS_DEFAULT_TOI[g]
        if np.isfinite(toi[i]) and gpw[i] > 0:
            out[i] = (gpw[i] * toi[i] + TOI_PRIOR_GP * prior) / (gpw[i] + TOI_PRIOR_GP)
            src.append("nhl")
        else:
            out[i] = prior
            src.append("proj" if fb is not None else "default")
    return out, src


def _grp(pos) -> str:
    return "F" if pos in FWD_POS else ("D" if pos == "D" else "G")


def _age(birthdate, pid, season_start: int) -> float:
    try:
        return float(season_start - int(str(birthdate)[:4]))
    except (TypeError, ValueError):
        by = _bios()["birth_year"].get(int(pid))
        return float(season_start - by) if by is not None and np.isfinite(by) else np.nan


def _return_info(team: str, rosters_date: str, sched: pd.DataFrame) -> tuple[int | None, str | None]:
    """(season game number, date) of the team's (IR_RETURN_GAMES + 1)-th game on/after the date."""
    tg = sched[(sched["home"] == team) | (sched["away"] == team)].sort_values(["date", "game_id"])
    tg = tg.reset_index(drop=True)
    after = tg[tg["date"].astype(str) >= rosters_date]
    if len(after) <= IR_RETURN_GAMES:
        return None, None
    row = after.iloc[IR_RETURN_GAMES]
    return int(row.name) + 1, str(row["date"])[:10]


def _goalie_table(team: str, g: pd.DataFrame, prev: int = SEASON - 1) -> pd.DataFrame:
    """Rank and start shares for a team's goalies (roster first, then reserves)."""
    s1, s2 = _goalie_season(prev), _goalie_season(prev - 1)
    career = _career_starts(prev)
    g = g.copy()
    for col, src, c in (("starts_prev", s1, "starts"), ("dressed_prev", s1, "dressed"),
                        ("starts_prev2", s2, "starts"), ("dressed_prev2", s2, "dressed")):
        g[col] = g["player_id"].map(src[c] if len(src) else {}).fillna(0).astype(int)
    g["career_starts"] = g["player_id"].map(career).fillna(0).astype(int)
    den = g["dressed_prev"] + 0.5 * g["dressed_prev2"]
    g["share_recent"] = np.where(den > 0, (g["starts_prev"] + 0.5 * g["starts_prev2"]) / den.clip(lower=1e-9), np.nan)
    g["tier"] = (g["status"] == "reserve").astype(int)
    g = g.sort_values(["tier", "starts_prev", "career_starts", "player_id"],
                      ascending=[True, False, False, True]).reset_index(drop=True)
    return g


def goalie_probs(g: pd.DataFrame, avail: np.ndarray, b2b: bool) -> np.ndarray:
    """Start probabilities over g's rows given which goalies are available (rank order)."""
    p = np.zeros(len(g))
    idx = [i for i in range(len(g)) if avail[i] and g["tier"].iat[i] == 0]
    idx += [i for i in range(len(g)) if avail[i] and g["tier"].iat[i] == 1][:max(0, 2 - len(idx))]
    if not idx:
        return p
    if len(idx) == 1:
        p[idx[0]] = 1.0
        return p
    sh = g["share_recent"].iat[idx[0]]
    ps = float(np.clip(sh if np.isfinite(sh) else G_START_MIN, G_START_MIN, G_START_MAX))
    if b2b:
        ps *= B2B_STARTER_MULT
    p[idx[0]], p[idx[1]] = ps, 1.0 - ps
    return p


def load(rosters_date: str, *, reserves: bool = True, model: AvailModel | None = None) -> Roster:
    """Rosters, depth charts, p_dress and goalie shares per team from the roster snapshot on or
    before rosters_date (see the module docstring)."""
    fr = _live("fetch_rosters")
    st = _live("status")
    rday, R = fr.latest_rosters(rosters_date)
    R = R.copy()
    unavail = set(st.unavailable(rosters_date))
    sched = load_schedule()
    teams = sorted(set(sched["home"]) | set(sched["away"])) or sorted(R["team"].unique())
    notes: dict[str, list] = {}
    excluded = R[R["player_id"].isin(unavail)]
    for r in excluded.itertuples():
        notes.setdefault(r.team, []).append(f"excluded (status file): {r.first} {r.last}")
    R = R[~R["player_id"].isin(unavail)]
    on_any = set(int(x) for x in R["player_id"])

    # DailyFaceoff IR players who are off the NHL roster -> injured with a return game
    ir = _df_ir(teams, rosters_date, R, notes)
    df_status = {(r.team_abbrev, int(r.player_id)): f"ir:{r.injury_status or '-'}" for r in ir.itertuples()}
    ir_off = (ir[~ir["on_roster"].astype(bool) & ~ir["player_id"].astype("int64").isin(unavail)]
              if len(ir) else ir)
    aug = fr.load_august()
    ref = pd.concat([R, aug]).drop_duplicates("player_id").set_index("player_id")
    rows = []
    for r in R.itertuples():
        rows.append({"team": r.team, "player_id": int(r.player_id), "name": f"{r.first} {r.last}",
                     "pos": r.pos, "birthdate": r.birthdate, "status": "roster"})
    for r in ir_off.itertuples():
        pid = int(r.player_id)
        if pid in on_any:
            continue
        pos = ref["pos"].get(pid)
        if pos is None or (isinstance(pos, float) and np.isnan(pos)):
            pg_ = _bios()["pos_group"].get(pid)
            pos = {0: "C", 1: "D", 2: "G"}.get(int(pg_), "C") if pg_ is not None else "C"
        rows.append({"team": r.team_abbrev, "player_id": pid, "name": r.name, "pos": pos,
                     "birthdate": ref["birthdate"].get(pid), "status": "injured"})
    # reserves: previous snapshot, on no current roster, not listed injured, old enough
    pday = None
    if reserves:
        pday, P = fr.previous(rday)
        taken = on_any | set(int(r["player_id"]) for r in rows) | unavail
        P = P[~P["player_id"].isin(taken)].drop_duplicates("player_id")
        for r in P.itertuples():
            age = _age(r.birthdate, r.player_id, SEASON - 1)
            if np.isfinite(age) and age < RESERVE_MIN_AGE:
                continue
            rows.append({"team": r.team, "player_id": int(r.player_id), "name": f"{r.first} {r.last}",
                         "pos": r.pos, "birthdate": r.birthdate, "status": "reserve", "tier": 1})
        # second tier: the club's 2025-26 depth skaters (call-up types) on no current roster
        taken |= set(int(r["player_id"]) for r in rows)
        rows += _history_reserves(SEASON - 1, taken)
    A = pd.DataFrame(rows)
    A["tier"] = A["tier"].fillna(0).astype(int) if "tier" in A else 0
    A["grp"] = A["pos"].map(_grp)
    A["df_status"] = [df_status.get((t, p), "") for t, p in zip(A["team"], A["player_id"])]
    A["age"] = [_age(b, p, SEASON - 1) for b, p in zip(A["birthdate"], A["player_id"])]
    A["return_game"], A["return_date"] = pd.array([pd.NA] * len(A), dtype="Int64"), None
    for i in A.index[A["status"] == "injured"]:
        gno, gdate = _return_info(A.at[i, "team"], rosters_date, sched)
        A.at[i, "return_game"] = gno if gno is not None else pd.NA
        A.at[i, "return_date"] = gdate if gdate is not None else "9999-12-31"
    A["injured"] = A["status"].eq("injured")

    # depth (skaters) and p_dress
    model = model or fit_model(SEASON - 3)
    S = A[A["grp"] != "G"].copy()
    toi, src = project_toi(S["player_id"].to_numpy(), S["grp"].to_numpy(), SEASON, _fallback_toi())
    S["toi_proj"], S["toi_src"] = np.round(toi, 3), src
    by = SEASON - 1 - S["age"]
    f = features(SEASON, S["player_id"], (S["grp"] == "D").astype(int), by)
    S["p_dress"] = predict_p_dress(model, f)
    S["tier"] = S["tier"].astype(int)
    S["gord"] = (S["grp"] == "D").astype(int)
    out = Roster(meta={"rosters_date": rosters_date, "roster_snapshot": rday, "reserve_snapshot": pday,
                       "unavailable": sorted(unavail), "model": model.as_dict(),
                       "df_dirs": sorted(set(ir["df_dir"])) if len(ir) else []})
    for team in teams:
        s = S[S["team"] == team].sort_values(["tier", "gord", "toi_proj", "player_id"],
                                             ascending=[True, True, False, True]).reset_index(drop=True)
        s["depth"] = s.groupby("grp").cumcount() + 1
        s["p_avail"] = s["p_dress"].to_numpy()
        for grp_, slots in (("F", N_F), ("D", N_D)):
            m = (s["grp"] == grp_) & (s["tier"] == 0)
            s.loc[m, "p_avail"] = depth_adjust(s.loc[m, "p_dress"].to_numpy(), slots)
        s.loc[s["tier"] >= 1, "p_avail"] = RESERVE_P_AVAIL
        g = _goalie_table(team, A[(A["team"] == team) & (A["grp"] == "G")])
        base = goalie_probs(g, ~g["injured"].to_numpy(), False) if len(g) else np.zeros(0)
        b2b = goalie_probs(g, ~g["injured"].to_numpy(), True) if len(g) else np.zeros(0)
        g["p_start"], g["p_start_b2b"] = base, b2b
        g["p_start_full"] = goalie_probs(g, np.ones(len(g), bool), False) if len(g) else np.zeros(0)
        live = [i for i in range(len(g)) if base[i] > 0]       # rank order among goalies who start
        g["role"] = ["reserve" if t else "third" for t in g["tier"]]
        for i, r in zip(live, ("starter", "backup")):
            g.loc[i, "role"] = r
        g.loc[g["injured"], "role"] = "injured"
        tn = notes.setdefault(team, [])
        healthy = s[(s["tier"] == 0) & ~s["injured"]]
        nF, nD, ng = int((healthy["grp"] == "F").sum()), int((healthy["grp"] == "D").sum()), int(
            ((g["tier"] == 0) & ~g["injured"]).sum())
        if nF < N_F or nD < N_D:
            tn.append(f"thin: {nF} healthy roster F / {nD} D (fills from reserves)")
        if ng < 2:
            tn.append(f"thin in goal: {ng} healthy roster goalie(s)")
        out[team] = TeamRoster(team=team, skaters=s.drop(columns=["gord"]), goalies=g, notes=tn)
    return out


# ------------------------------------------------------------------ sampler
def _team_arrays(tr: TeamRoster, tg: pd.DataFrame):
    s, g = tr.skaters, tr.goalies
    m = len(tg)
    dates = tg["date"].to_numpy(str)
    ret = s["return_date"].fillna("").to_numpy(str)
    j0 = np.array([np.searchsorted(dates, r, side="left") if r else 0 for r in ret], dtype=int)
    grp = (s["grp"] == "D").to_numpy().astype(int)
    tier = s["tier"].to_numpy().astype(int)
    # goalie probabilities per game (injury-dependent ranking, back-to-backs)
    gret = g["return_date"].fillna("").to_numpy(str) if len(g) else np.zeros(0, str)
    gj0 = np.array([np.searchsorted(dates, r, side="left") if r else 0 for r in gret], dtype=int)
    gp = np.zeros((m, len(g)))
    cache = {}
    for j in range(m):
        av = gj0 <= j
        key = (av.tobytes(), bool(tg["b2b"].iat[j]))
        if key not in cache:
            cache[key] = goalie_probs(g, av, key[1]) if len(g) else np.zeros(0)
        gp[j] = cache[key]
    return s["player_id"].to_numpy(np.int64), grp, tier, s["p_avail"].to_numpy(float), j0, \
        g["player_id"].to_numpy(np.int64) if len(g) else np.zeros(0, np.int64), gp


def _select(avail, inj, grp, tier, seven_d=None):
    """Slot assignment for (k, m, n) availability. Returns (keyF, keyD): slot-order keys, BIG if
    not dressed in that slot group. Player index order is the depth order within a tier/group.
    seven_d (k, m): games where an open forward slot goes to an available 7th defenceman before
    a reserve forward (11F/7D by choice)."""
    k, m, n = avail.shape
    BIG = np.iinfo(np.int32).max
    idx = np.arange(n)
    F0, D0 = (grp == 0) & (tier == 0), (grp == 1) & (tier == 0)
    F1, D1 = (grp == 0) & (tier >= 1), (grp == 1) & (tier >= 1)
    used = np.zeros((k, m, n), bool)
    keyF = np.full((k, m, n), BIG, np.int32)
    keyD = np.full((k, m, n), BIG, np.int32)
    need = {"F": np.full((k, m), N_F), "D": np.full((k, m), N_D)}

    def take(elig, slot, stage, reverse=False):
        e = elig & ~used
        if not e.any():
            return
        if reverse:                  # lowest-ranked first
            sel = e & (np.cumsum(e[..., ::-1], axis=-1)[..., ::-1] <= need[slot][..., None])
        else:
            sel = e & (np.cumsum(e, axis=-1) <= need[slot][..., None])
        if not sel.any():
            return
        used[sel] = True
        key = keyF if slot == "F" else keyD
        key[sel] = (stage * 1000 + np.broadcast_to(idx, sel.shape)[sel]).astype(np.int32)
        need[slot] -= sel.sum(-1)

    ok = avail & ~inj
    nav = ~avail & ~inj              # not available, not injured
    seven = np.zeros((k, m), bool) if seven_d is None else seven_d
    take(ok & D0, "D", 0)
    take(ok & F0, "F", 0)
    take(ok & D1, "D", 1)            # call-ups: available reserves first, then any reserve
    take(nav & D1, "D", 2)           # (the pool stands in for a deep farm team)
    take(ok & D0 & seven[..., None], "F", 1)   # 11F/7D by choice
    take(ok & F1, "F", 2)                      # else a call-up forward
    take(nav & F1, "F", 3)
    take(ok & D0, "F", 4)            # the 7th defenceman when no forward is left
    take(ok & D1, "F", 5)
    take(ok & F0, "D", 3)            # a forward on defence only when no defenceman is left
    take(ok & F1, "D", 4)
    # emergency: unavailable roster players from the BOTTOM of the depth chart, so a short roster
    # never hands phantom games to its stars; then the other position; then injured players
    for slot, same, other in (("F", F0, D0), ("D", D0, F0)):
        take(nav & same, slot, 6, reverse=True)
        take(nav & other, slot, 7, reverse=True)
    for slot in ("F", "D"):
        take(inj, slot, 9)           # last resort: 18 skaters whenever the team has 18
    return keyF, keyD


def _lineup_ids(pids, keyF, keyD):
    BIG = np.iinfo(np.int32).max
    oF = np.argsort(keyF, axis=-1, kind="stable")[..., :N_F]
    oD = np.argsort(keyD, axis=-1, kind="stable")[..., :N_D]
    idsF = np.where(np.take_along_axis(keyF, oF, -1) < BIG, pids[oF], -1)
    idsD = np.where(np.take_along_axis(keyD, oD, -1) < BIG, pids[oD], -1)
    return np.concatenate([idsF, idsD], axis=-1)


def simulate_team(tr: TeamRoster, tg: pd.DataFrame, k: int, rng: np.random.Generator,
                  persist: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """(skater ids (k, m, 18), -1 where short; goalie ids (k, m), -1 if none) for one team."""
    pids, grp, tier, p_av, j0, gids, gp = _team_arrays(tr, tg)
    m, n = len(tg), len(pids)
    h, s, e = persistence(p_av, m, persist)
    u_long = rng.random((k, m, n))
    u_short = rng.random((k, m, n))
    avail = np.empty((k, m, n), bool)
    long_ = np.zeros((k, n), bool)
    for j in range(m):
        if j > 0 and persist:
            long_ = (long_ & (u_long[:, j] >= e)) | (~long_ & (u_long[:, j] < h))
        back = j0 == j                 # injured players return healthy
        if back.any():
            long_[:, back] = False
        avail[:, j] = ~long_ & (u_short[:, j] >= s)
    inj = (np.arange(m)[:, None] < j0[None, :])[None].repeat(k, 0)
    keyF, keyD = _select(avail, inj, grp, tier, rng.random((k, m)) < P_SEVEN_D)
    ids = _lineup_ids(pids, keyF, keyD)
    if len(gids):
        cum = np.cumsum(gp, axis=1)
        u = rng.random((k, m))
        gi = np.minimum((u[..., None] >= cum[None]).sum(-1), len(gids) - 1)
        goal = np.where(cum[None, :, -1] > 0, gids[gi], -1)
    else:
        goal = np.full((k, m), -1, np.int64)
    return ids, goal


def _team_rng(seed: int, team: str) -> np.random.Generator:
    """Per-team stream: a team's draws do not depend on which other teams are simulated."""
    return np.random.default_rng([int(seed), sum((i + 1) * ord(c) for i, c in enumerate(team))])


def draw_arrays(games: pd.DataFrame, roster: Roster, k: int, seed: int = SEED,
                persist: bool = True) -> dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray]]:
    """team -> (its games in order, skater ids (k, m, 18), goalie ids (k, m))."""
    out = {}
    for team, tg in team_games(games).items():
        if team not in roster:
            raise KeyError(f"no roster for {team}")
        ids, goal = simulate_team(roster[team], tg, k, _team_rng(seed, team), persist)
        out[team] = (tg, ids, goal)
    return out


def _assemble(games: pd.DataFrame, arrs: dict, k: int) -> list[dict]:
    pos = {}
    lists = {}
    for team, (tg, ids, goal) in arrs.items():
        for j, gid in enumerate(tg["game_id"].to_numpy()):
            pos[(team, int(gid))] = j
        short = bool((ids < 0).any())
        L = ids.tolist()
        if short:
            L = [[[x for x in row if x >= 0] for row in dr] for dr in L]
        lists[team] = (L, goal.tolist())
    rows = list(zip(games["game_id"].astype(int), games["home"], games["away"]))
    draws = []
    for d in range(k):
        dd = {}
        for gid, h, a in rows:
            jh, ja = pos[(h, gid)], pos[(a, gid)]
            Lh, Gh = lists[h]
            La, Ga = lists[a]
            gh, ga = Gh[d][jh], Ga[d][ja]
            dd[gid] = {"home": {"skaters": Lh[d][jh], "goalie": gh if gh >= 0 else None},
                       "away": {"skaters": La[d][ja], "goalie": ga if ga >= 0 else None}}
        draws.append(dd)
    return draws


def draw_lineups(games: pd.DataFrame, roster: Roster, k: int, seed: int = SEED,
                 persist: bool = True) -> list[dict]:
    """k draws of {game_id: {"home": {"skaters": [...], "goalie": id}, "away": {...}}}.

    skaters: 18 ids, the 12 forward slots (depth order, then fills) then the 6 defence slots.
    Deterministic given (games, roster, k, seed, persist)."""
    return _assemble(games, draw_arrays(games, roster, k, seed, persist), k)


def expected_lineup(games: pd.DataFrame, roster: Roster) -> dict:
    """The single most likely lineup per team-game: healthy (not injured at that game) top 12 F
    and 6 D by depth, filled as in the sampler; the starter in goal except in the 2nd game of a
    back-to-back (the backup, whose probability is then higher)."""
    arrs = {}
    for team, tg in team_games(games).items():
        pids, grp, tier, _, j0, gids, gp = _team_arrays(roster[team], tg)
        m = len(tg)
        inj = (np.arange(m)[:, None] < j0[None, :])[None]
        avail = ~inj & (tier == 0)[None, None, :]
        keyF, keyD = _select(avail, inj, grp, tier)
        ids = _lineup_ids(pids, keyF, keyD)
        goal = (np.where(gp.sum(1) > 0, gids[np.argmax(gp, axis=1)], -1)[None]
                if len(gids) else np.full((1, m), -1))
        arrs[team] = (tg, ids, goal)
    return _assemble(games, arrs, 1)[0]


# ------------------------------------------------------------------ reporting helpers
def season_gp(games: pd.DataFrame, arrs: dict) -> pd.DataFrame:
    """Per (team, player): games dressed per draw -> mean, p10, p90; plus goalie starts."""
    rows = []
    for team, (tg, ids, goal) in arrs.items():
        k = ids.shape[0]
        flat = ids.reshape(k, -1)
        u = np.unique(flat[flat >= 0])
        cnt = np.stack([(flat == p).sum(1) for p in u], axis=1) if len(u) else np.zeros((k, 0))
        for i, p in enumerate(u):
            c = cnt[:, i]
            rows.append((team, int(p), "S", c.mean(), np.percentile(c, 10), np.percentile(c, 90), c.std()))
        gu = np.unique(goal[goal >= 0])
        for p in gu:
            c = (goal == p).sum(1)
            rows.append((team, int(p), "G", c.mean(), np.percentile(c, 10), np.percentile(c, 90), c.std()))
    return pd.DataFrame(rows, columns=["team", "player_id", "kind", "gp_mean", "gp_p10", "gp_p90", "gp_sd"])


def summary_frame(roster: Roster, gp: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for team, tr in roster.items():
        for r in tr.skaters.itertuples():
            rows.append({"team": team, "player_id": r.player_id, "name": r.name, "pos": r.pos, "grp": r.grp,
                         "status": r.status, "tier": r.tier, "depth_rank": r.depth, "toi_proj_min": r.toi_proj,
                         "toi_src": r.toi_src, "p_dress": round(float(r.p_dress), 4),
                         "p_avail": round(float(r.p_avail), 4), "injured": bool(r.injured),
                         "return_game": r.return_game, "return_date": r.return_date or "",
                         "df_status": r.df_status, "p_start": np.nan, "p_start_b2b": np.nan,
                         "p_start_full": np.nan, "role": ""})
        for i, r in enumerate(tr.goalies.itertuples()):
            rows.append({"team": team, "player_id": r.player_id, "name": r.name, "pos": "G", "grp": "G",
                         "status": r.status, "tier": r.tier, "depth_rank": i + 1, "toi_proj_min": np.nan, "toi_src": "",
                         "p_dress": np.nan, "p_avail": np.nan, "injured": bool(r.injured),
                         "return_game": r.return_game, "return_date": r.return_date or "",
                         "df_status": r.df_status, "p_start": round(float(r.p_start), 4),
                         "p_start_b2b": round(float(r.p_start_b2b), 4),
                         "p_start_full": round(float(r.p_start_full), 4), "role": r.role})
    df = pd.DataFrame(rows)
    df.insert(0, "rosters_date", roster.meta.get("rosters_date"))
    if gp is not None and len(gp):
        g = gp.groupby(["team", "player_id"])[["gp_mean", "gp_p10", "gp_p90"]].sum()
        df = df.merge(g.rename(columns={"gp_mean": "exp_gp", "gp_p10": "gp_p10", "gp_p90": "gp_p90"}),
                      left_on=["team", "player_id"], right_index=True, how="left")
        df[["exp_gp", "gp_p10", "gp_p90"]] = df[["exp_gp", "gp_p10", "gp_p90"]].fillna(0).round(2)
    return df


def print_summary(roster: Roster, df: pd.DataFrame) -> None:
    meta = roster.meta
    print(f"rosters {meta['rosters_date']} (snapshot {meta['roster_snapshot']}, reserves from "
          f"{meta['reserve_snapshot']}); DailyFaceoff {', '.join(Path(d).name for d in meta['df_dirs']) or '-'}; "
          f"status-unavailable {meta['unavailable']}")
    b = meta["model"]["beta"]
    print("p_dress GLM (logit): " + ", ".join(f"{k} {v:+.3f}" for k, v in b.items())
          + f"; K {meta['model']['k_prior']:.0f}, m0 {meta['model']['m0']:.3f}, n {meta['model']['n_obs']}")
    has_gp = "exp_gp" in df
    for team in sorted(roster):
        d = df[df["team"] == team]
        sk = d[d["grp"] != "G"]
        n = {s: int((sk["status"] == s).sum()) for s in ("roster", "injured", "reserve")}
        print(f"\n=== {team}: {n['roster']} roster skaters, {n['injured']} injured, {n['reserve']} reserves; "
              f"{int((d['grp'] == 'G').sum())} goalies")
        for grp_ in ("F", "D"):
            x = sk[sk["grp"] == grp_].sort_values("depth_rank")
            line = []
            for r in x.itertuples():
                tag = {"injured": f" INJ->g{r.return_game}", "reserve": " res"}.get(r.status, "")
                gp_ = f" gp {r.exp_gp:.0f}" if has_gp else ""
                line.append(f"{r.depth_rank:>2} {r.name} {r.pos} {r.toi_proj_min:.1f}m p{r.p_dress:.2f}{gp_}{tag}")
            print(f"  {grp_}: " + "\n     ".join(line))
        for r in d[d["grp"] == "G"].itertuples():
            gp_ = f", exp starts {r.exp_gp:.1f}" if has_gp else ""
            inj = f", injured -> game {r.return_game} ({r.return_date})" if r.injured else ""
            full = f", all healthy {r.p_start_full:.3f}" if abs(r.p_start_full - r.p_start) > 1e-9 else ""
            print(f"  G: {r.role:<8} {r.name}: p_start {r.p_start:.3f} (b2b {r.p_start_b2b:.3f}){full}{gp_}{inj}")
        inj = d[d["injured"]]
        if len(inj):
            print("  injured: " + "; ".join(f"{r.name} ({r.pos}, DF {r.df_status}) back game {r.return_game} "
                                            f"({r.return_date})" for r in inj.itertuples()))
        if roster[team].notes:
            print("  notes: " + " | ".join(roster[team].notes))


# ------------------------------------------------------------------ validation (2023-24)
def _hist_roster(V: int, model: AvailModel) -> Roster:
    """A Roster for season V from history: opening-night skaters and goalies (tier 0), reserves =
    the team's skaters of V-1 on no opening-night roster (age < 35), depth from V-1 / V-2 TOI."""
    ab = _team_abbrev()
    op = opening_roster(V)
    op["team_ab"] = op["team"].map(ab)
    prev = _skater_season(V - 1)
    on = set(op["player_id"])
    by_all = _bios()["birth_year"]
    res = prev[(~prev.index.isin(on))].copy()
    res = res[(V - 1 - res.index.map(by_all).fillna(V - 30)) < 35]
    tg_prev, pg_prev, _ = _season(V - 1)
    last_team = pg_prev.sort_values(["date", "game_id"]).groupby("player_id")["team"].last()
    res["team_ab"] = res.index.map(last_team).map(ab)
    rows = [{"team": t, "player_id": int(p), "grp": ("G" if g == 2 else ("D" if g == 1 else "F")), "tier": 0}
            for t, p, g in zip(op["team_ab"], op["player_id"], op["pos_group"])]
    rows += [{"team": t, "player_id": int(p), "grp": ("D" if g == 1 else "F"), "tier": 1}
             for t, p, g in zip(res["team_ab"], res.index, res["pos_group"]) if isinstance(t, str)]
    A = pd.DataFrame(rows)
    teams = sorted(op["team_ab"].unique())
    A = A[A["team"].isin(teams)]
    S = A[A["grp"] != "G"].copy()
    toi, src = project_toi(S["player_id"].to_numpy(), S["grp"].to_numpy(), V, {})
    S["toi_proj"], S["toi_src"] = toi, src
    f = features(V, S["player_id"], (S["grp"] == "D").astype(int), S["player_id"].map(by_all))
    S["p_dress"] = predict_p_dress(model, f)
    S["status"] = np.where(S["tier"] == 1, "reserve", "roster")
    S["injured"], S["return_date"], S["return_game"] = False, None, pd.NA
    S["gord"] = (S["grp"] == "D").astype(int)
    out = Roster(meta={"season": V})
    for team in teams:
        s = S[S["team"] == team].sort_values(["tier", "gord", "toi_proj", "player_id"],
                                             ascending=[True, True, False, True]).reset_index(drop=True)
        s["depth"] = s.groupby("grp").cumcount() + 1
        s["p_avail"] = s["p_dress"].to_numpy()
        for grp_, slots in (("F", N_F), ("D", N_D)):
            m = (s["grp"] == grp_) & (s["tier"] == 0)
            s.loc[m, "p_avail"] = depth_adjust(s.loc[m, "p_dress"].to_numpy(), slots)
        s.loc[s["tier"] >= 1, "p_avail"] = RESERVE_P_AVAIL
        g = A[(A["team"] == team) & (A["grp"] == "G")].copy()
        g["status"], g["injured"], g["return_date"], g["return_game"] = "roster", False, None, pd.NA
        g = _goalie_table(team, g, prev=V - 1)
        out[team] = TeamRoster(team=team, skaters=s.drop(columns=["gord"]), goalies=g)
    return out


def _metrics(pred: np.ndarray, act: np.ndarray) -> dict:
    ok = np.isfinite(pred) & np.isfinite(act)
    p, a = pred[ok], act[ok]
    return {"n": int(ok.sum()), "mae": round(float(np.mean(np.abs(p - a))), 4),
            "rmse": round(float(np.sqrt(np.mean((p - a) ** 2))), 4),
            "corr": round(float(np.corrcoef(p, a)[0, 1]), 4), "bias": round(float(np.mean(p - a)), 4)}


def _deciles(pred: np.ndarray, act: np.ndarray) -> list[dict]:
    q = pd.qcut(pd.Series(pred).rank(method="first"), 10, labels=False)
    d = pd.DataFrame({"q": q, "p": pred, "a": act}).groupby("q").agg(n=("p", "size"), pred=("p", "mean"),
                                                                     act=("a", "mean"))
    return [{"decile": int(i) + 1, "n": int(r.n), "pred": round(float(r.pred), 3), "act": round(float(r.act), 3)}
            for i, r in d.iterrows()]


def validate(V: int = 2024, k: int = 64, seed: int = SEED, persist: bool = True) -> dict:
    """Refit p_dress on target seasons <= V-1, predict V; run the sampler over V's schedule with
    V's opening-night rosters; compare with the realised season. V must not be 2025 or 2026."""
    assert V not in (2025, 2026), "2025 and 2026 are reserved"
    t0 = time.time()
    model = fit_model(V - 1)
    tf = target_frame(V)
    tf["p_dress"] = predict_p_dress(model, tf)
    tf["act"] = tf["D"] / tf["G"]
    prev = _skater_season(V - 1)
    tgp, _, _ = _season(V - 1)
    Gp = tgp.groupby("team").size().max()
    Dp = tf["player_id"].map(prev["D"]).fillna(0).to_numpy(float)
    tf["base"] = np.minimum(Dp / Gp, 1.0)
    has_prev = Dp > 0
    res = {"season": V, "model": model.as_dict(), "k": k, "seed": seed, "persist": persist,
           "constants": {"LONG_FRAC": LONG_FRAC, "LONG_MEAN": LONG_MEAN, "RESERVE_P_AVAIL": RESERVE_P_AVAIL,
                         "TOI_PRIOR_GP": TOI_PRIOR_GP, "POS_DEFAULT_TOI": POS_DEFAULT_TOI, "LAG_W": LAG_W}}
    p, a, b = tf["p_dress"].to_numpy(), tf["act"].to_numpy(), tf["base"].to_numpy()
    res["head"] = {"all": _metrics(p, a), "with_prev_season": _metrics(p[has_prev], a[has_prev]),
                   "deciles": _deciles(p, a)}
    res["baseline_last_season_share"] = {"all": _metrics(b, a), "with_prev_season": _metrics(b[has_prev], a[has_prev]),
                                         "deciles": _deciles(b, a)}

    # the full layer: sample V's season with V's opening-night rosters
    roster = _hist_roster(V, model)
    ab = _team_abbrev()
    tg, pg, _ = _season(V)
    gc = pd.read_parquet(TENSORS / f"games_ctx_{V}.parquet", columns=["game_id", "game_type", "date",
                                                                       "home_idx", "away_idx"])
    gc = gc[gc["game_type"] == 2]
    games = pd.DataFrame({"game_id": gc["game_id"].astype(int), "date": gc["date"].astype(str).str[:10],
                          "home": gc["home_idx"].map(ab), "away": gc["away_idx"].map(ab)})
    games = games.sort_values(["date", "game_id"]).reset_index(drop=True)
    t1 = time.time()
    arrs = {}
    for team, tgt in team_games(games, context=games).items():
        ids, goal = simulate_team(roster[team], tgt, k, _team_rng(seed, team), persist)
        arrs[team] = (tgt, ids, goal)
    res["sampler_seconds"] = round(time.time() - t1, 2)
    gp = season_gp(games, arrs)
    sk = gp[gp["kind"] == "S"].groupby("player_id")[["gp_mean", "gp_p10", "gp_p90", "gp_sd"]].sum()
    G = float(len(next(iter(arrs.values()))[0]))
    tf["sim"] = tf["player_id"].map(sk["gp_mean"]).fillna(0).to_numpy() / G
    s_ = tf["sim"].to_numpy()
    res["sampler"] = {"all": _metrics(s_, a), "with_prev_season": _metrics(s_[has_prev], a[has_prev]),
                      "deciles": _deciles(s_, a),
                      "sim_minus_p_dress_mean_abs": round(float(np.mean(np.abs(s_ - p))), 4)}
    # per-player GP distribution: realised GP inside the sampler's 10-90% band
    lo = tf["player_id"].map(sk["gp_p10"]).fillna(0).to_numpy()
    hi = tf["player_id"].map(sk["gp_p90"]).fillna(0).to_numpy()
    act_gp = tf["D"].to_numpy()
    res["sampler"]["coverage_p10_p90"] = round(float(np.mean((act_gp >= lo) & (act_gp <= hi))), 4)
    rank = roster_depth = pd.concat([tr.skaters.assign(team=t) for t, tr in roster.items()])
    rank = roster_depth.set_index("player_id")
    tf["depth"] = tf["player_id"].map(rank["depth"])
    tf["grp"] = tf["player_id"].map(rank["grp"])
    reg = ((tf["grp"] == "F") & (tf["depth"] <= 9)) | ((tf["grp"] == "D") & (tf["depth"] <= 4))
    res["regulars_gp_sd"] = {"sim_within_player_mean": round(float(tf.loc[reg, "player_id"].map(sk["gp_sd"]).mean()), 2),
                             "sim_across_players_one_draw": None,
                             "real_across_players": round(float(tf.loc[reg, "D"].std()), 2),
                             "sim_mean_gp": round(float(tf.loc[reg, "sim"].mean() * G), 2),
                             "real_mean_gp": round(float(tf.loc[reg, "D"].mean()), 2)}
    # one draw's spread across regulars (comparable to the realised spread)
    d0 = {}
    for team, (tgt, ids, goal) in arrs.items():
        u, c = np.unique(ids[0][ids[0] >= 0], return_counts=True)
        d0.update(dict(zip(u.tolist(), c.tolist())))
    res["regulars_gp_sd"]["sim_across_players_one_draw"] = round(
        float(pd.Series(tf.loc[reg, "player_id"].map(d0)).fillna(0).std()), 2)

    # team-level: distinct skaters, top-6 F, GP histogram, opening-roster share, 11F/7D
    ab_team = {v: k_ for k_, v in ab.items()}
    real_team = pg[pg["pos_group"] < 2].copy()
    real_team["team_ab"] = real_team["team"].map(ab)
    distinct_real = real_team.groupby("team_ab")["player_id"].nunique()
    op = opening_roster(V)
    op["team_ab"] = op["team"].map(ab)
    op_sk = op[op["pos_group"] < 2]
    real_op = real_team.merge(op_sk[["team_ab", "player_id"]], on=["team_ab", "player_id"])
    distinct_real_op = real_op.groupby("team_ab")["player_id"].nunique()
    distinct_sim = {t: float(np.mean([len(np.unique(ids[d][ids[d] >= 0])) for d in range(ids.shape[0])]))
                    for t, (tgt, ids, goal) in arrs.items()}
    share_op_real = len(real_op) / len(real_team)
    opset = set(zip(op_sk["team_ab"], op_sk["player_id"]))
    share_op_sim = float(np.mean([np.isin(ids, [p for (tt, p) in opset if tt == t]).sum() / (ids >= 0).sum()
                                  for t, (tgt, ids, goal) in arrs.items()]))
    top6 = rank[(rank["grp"] == "F") & (rank["depth"] <= 6) & (rank["tier"] == 0)]
    t6 = tf[tf["player_id"].isin(top6.index)]
    # realised share with the same team, and games with all six dressed
    real_by_team = real_team.groupby(["team_ab", "player_id"]).size()
    t6_same = [real_by_team.get((t, p), 0) / G for t, p in zip(top6["team"], top6.index)]
    all6_real, all6_sim = [], []
    for team, (tgt, ids, goal) in arrs.items():
        six = top6[top6["team"] == team].index.to_numpy()
        gids = tgt["game_id"].to_numpy()
        dressed = real_team[(real_team["team_ab"] == team) & real_team["player_id"].isin(six)]
        cnt = dressed.groupby("game_id").size().reindex(gids).fillna(0)
        all6_real.append(float((cnt == len(six)).mean()))
        all6_sim.append(float(np.isin(ids[..., :N_F], six).sum(-1).__eq__(len(six)).mean()))
    n_f_real = real_team[real_team["pos_group"] == 0].groupby(["game_id", "team_ab"]).size()
    cross = []
    for team, (tgt, ids, goal) in arrs.items():
        dset = roster[team].skaters.loc[roster[team].skaters["grp"] == "D", "player_id"].to_numpy()
        isd = np.isin(ids, dset)
        cross.append(float((isd[..., :N_F].any(-1) | (~isd[..., N_F:] & (ids[..., N_F:] >= 0)).any(-1)).mean()))
    res["team"] = {
        "distinct_skaters_per_team": {"real_all": round(float(distinct_real.mean()), 2),
                                      "real_opening_roster_players": round(float(distinct_real_op.mean()), 2),
                                      "sim": round(float(np.mean(list(distinct_sim.values()))), 2)},
        "share_of_skater_games_by_opening_roster": {"real_same_team": round(share_op_real, 4),
                                                    "sim": round(share_op_sim, 4)},
        "top6_F_dressed_share": {"real_any_team": round(float(t6["act"].mean()), 4),
                                 "real_same_team": round(float(np.mean(t6_same)), 4),
                                 "sim": round(float(t6["sim"].mean()), 4)},
        "games_with_all_top6_F": {"real": round(float(np.mean(all6_real)), 4),
                                  "sim": round(float(np.mean(all6_sim)), 4)},
        "games_not_12F_6D": {"real": round(float((n_f_real != N_F).mean()), 4),
                             "sim": round(float(np.mean(cross)), 4)},
    }
    bins = [-1, 9, 39, 59, 69, 79, 100]
    lab = ["0-9", "10-39", "40-59", "60-69", "70-79", "80+"]
    real_hist = pd.cut(tf["D"], bins, labels=lab).value_counts(normalize=True).reindex(lab)
    sim_counts = []
    for d in range(k):
        dd = {}
        for team, (tgt, ids, goal) in arrs.items():
            u, c = np.unique(ids[d][ids[d] >= 0], return_counts=True)
            dd.update(dict(zip(u.tolist(), c.tolist())))
        sim_counts.append(pd.cut(tf["player_id"].map(dd).fillna(0), bins, labels=lab).value_counts(normalize=True))
    sim_hist = pd.concat(sim_counts, axis=1).mean(1).reindex(lab)
    res["gp_histogram_opening_roster"] = {b_: {"real": round(float(real_hist[b_]), 4), "sim": round(float(sim_hist[b_]), 4)}
                                          for b_ in lab}
    # goalies: realised share of team starts by the preseason starter; back-to-back 2nd games
    gs = pg[pg["pos_group"] == 2].copy()
    gs["team_ab"] = gs["team"].map(ab)
    real_st = gs.groupby(["team_ab", "player_id"])["goalie_start"].sum()
    gstat = gp[gp["kind"] == "G"].set_index(["team", "player_id"])["gp_mean"]
    st_rows = []
    for team, tr in roster.items():
        g = tr.goalies
        if not len(g):
            continue
        sid = int(g["player_id"].iat[0])
        st_rows.append((team, sid, float(g["p_start"].iat[0]) if "p_start" in g else np.nan,
                        real_st.get((team, sid), 0) / G, gstat.get((team, sid), 0) / G))
    st = pd.DataFrame(st_rows, columns=["team", "pid", "p", "real", "sim"])
    tgs = team_games(games, context=games)
    b2b_real = []
    for team, tgt in tgs.items():
        sid = st.loc[st["team"] == team, "pid"]
        if not len(sid):
            continue
        x = gs[(gs["team_ab"] == team) & (gs["player_id"] == int(sid.iat[0]))].set_index("game_id")["goalie_start"]
        for gid_, b2 in zip(tgt["game_id"], tgt["b2b"]):
            b2b_real.append((b2, float(x.get(gid_, 0)), gid_ in x.index))
    b2 = pd.DataFrame(b2b_real, columns=["b2b", "start", "dressed"])
    dr = b2[b2["dressed"]]
    # the same for each team's realised number-one goalie (most starts in V): the back-to-back
    # effect itself, free of preseason misclassification
    top = real_st.groupby(level=0).idxmax()
    b2t = []
    for team, tgt in tgs.items():
        if team not in top.index:
            continue
        x = gs[(gs["team_ab"] == team) & (gs["player_id"] == top[team][1])].set_index("game_id")["goalie_start"]
        b2t += [(b2_, float(x[gid_])) for gid_, b2_ in zip(tgt["game_id"], tgt["b2b"]) if gid_ in x.index]
    b2t = pd.DataFrame(b2t, columns=["b2b", "start"])
    res["goalies"] = {"preseason_starter_start_share": {"real": round(float(st["real"].mean()), 4),
                                                        "sim": round(float(st["sim"].mean()), 4),
                                                        "mae_team": round(float(np.mean(np.abs(st["sim"] - st["real"]))), 4)},
                      "starter_start_rate_b2b_second_game": round(float(b2.loc[b2["b2b"], "start"].mean()), 4),
                      "starter_start_rate_other_games": round(float(b2.loc[~b2["b2b"], "start"].mean()), 4),
                      "when_dressed_b2b_second_game": round(float(dr.loc[dr["b2b"], "start"].mean()), 4),
                      "when_dressed_other_games": round(float(dr.loc[~dr["b2b"], "start"].mean()), 4),
                      "realised_no1_when_dressed_b2b_second_game": round(float(b2t.loc[b2t["b2b"], "start"].mean()), 4),
                      "realised_no1_when_dressed_other_games": round(float(b2t.loc[~b2t["b2b"], "start"].mean()), 4)}
    res["seconds"] = round(time.time() - t0, 1)
    return res


def print_validation(res: dict) -> None:
    V = res["season"]
    print(f"VALIDATION {V - 1}-{str(V)[2:]}: p_dress refit on target seasons {res['model']['targets']} "
          f"(n {res['model']['n_obs']}, K {res['model']['k_prior']:.0f})")
    print("coefficients: " + ", ".join(f"{k} {v:+.3f}" for k, v in res["model"]["beta"].items()))
    print(f"\nshare of team games dressed, {V} opening-night skaters:")
    print(f"  {'':<34}{'n':>5}{'MAE':>8}{'RMSE':>8}{'corr':>8}{'bias':>8}")
    for name, key in (("p_dress (model head)", "head"), ("sampler (p_dress + depth chart)", "sampler"),
                      ("baseline: last season's share", "baseline_last_season_share")):
        for sub in ("all", "with_prev_season"):
            m = res[key][sub]
            print(f"  {name + (' [prev GP>0]' if sub != 'all' else ''):<34}{m['n']:>5}{m['mae']:>8.4f}"
                  f"{m['rmse']:>8.4f}{m['corr']:>8.4f}{m['bias']:>+8.4f}")
    print("\n  calibration by decile of the prediction (mean predicted / mean realised):")
    print(f"  {'dec':>4} {'p_dress':>15} {'sampler':>15} {'baseline':>15}")
    for a, b, c in zip(res["head"]["deciles"], res["sampler"]["deciles"], res["baseline_last_season_share"]["deciles"]):
        print(f"  {a['decile']:>4} {a['pred']:>7.3f}/{a['act']:<7.3f} {b['pred']:>7.3f}/{b['act']:<7.3f} "
              f"{c['pred']:>7.3f}/{c['act']:<7.3f}")
    print(f"\n  sampler reproduces p_dress: mean |sim share - p_dress| = {res['sampler']['sim_minus_p_dress_mean_abs']:.4f}")
    print(f"  realised GP inside the sampler's per-player 10-90% band: {res['sampler']['coverage_p10_p90']:.3f} "
          f"(nominal 0.80, discrete GP)")
    r = res["regulars_gp_sd"]
    print(f"  regulars (top-9 F / top-4 D): mean GP sim {r['sim_mean_gp']} vs real {r['real_mean_gp']}; "
          f"GP sd across players: one sim draw {r['sim_across_players_one_draw']} vs real {r['real_across_players']}; "
          f"sim within-player sd {r['sim_within_player_mean']}")
    t = res["team"]
    print("\nteam level:")
    for k_, v in t.items():
        print(f"  {k_}: " + ", ".join(f"{a} {b}" for a, b in v.items()))
    print("  GP histogram, opening-night skaters (share real / sim): " + ", ".join(
        f"{b_} {v['real']:.3f}/{v['sim']:.3f}" for b_, v in res["gp_histogram_opening_roster"].items()))
    g = res["goalies"]
    print(f"\ngoalies: preseason starter's share of team starts real {g['preseason_starter_start_share']['real']:.3f} "
          f"/ sim {g['preseason_starter_start_share']['sim']:.3f} (team MAE {g['preseason_starter_start_share']['mae_team']:.3f}); "
          f"realised starter start rate, 2nd game of a back-to-back {g['starter_start_rate_b2b_second_game']:.3f} "
          f"vs other games {g['starter_start_rate_other_games']:.3f}; when he dressed "
          f"{g['when_dressed_b2b_second_game']:.3f} vs {g['when_dressed_other_games']:.3f}; each team's realised "
          f"number one when dressed {g['realised_no1_when_dressed_b2b_second_game']:.3f} vs "
          f"{g['realised_no1_when_dressed_other_games']:.3f}")
    print(f"\nsampler {res['sampler_seconds']} s for k={res['k']} over the season; total {res['seconds']} s")


# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser(description="NeurHL 1.0 availability, depth charts and goalie starts.")
    ap.add_argument("--rosters-date", default=pd.Timestamp.now().strftime("%Y-%m-%d"),
                    help="roster snapshot date (latest on/before is used)")
    ap.add_argument("--summary", action="store_true", help="print depth charts and write the CSV")
    ap.add_argument("--validate", action="store_true", help="2023-24 backtest")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-persist", action="store_true", help="independent games (no absence spells)")
    ap.add_argument("--timing", action="store_true", help="time draw_lineups on the full schedule")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    persist = not a.no_persist
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if a.validate:
        res = validate(k=a.k, seed=a.seed, persist=persist)
        print_validation(res)
        p = OUT_DIR / "availability_validation_2024.json"
        p.write_text(json.dumps(res, indent=1, default=str))
        print(f"-> {p}")
    if a.summary or a.timing:
        t0 = time.time()
        roster = load(a.rosters_date)
        t1 = time.time()
        games = load_schedule()
        arrs = draw_arrays(games, roster, a.k, a.seed, persist)
        gp = season_gp(games, arrs)
        t2 = time.time()
        if a.timing:
            draws = _assemble(games, arrs, a.k)
            t3 = time.time()
            n_sk = {len(s["skaters"]) for d in draws for g in d.values() for s in (g["home"], g["away"])}
            print(f"load {t1 - t0:.1f} s; sample k={a.k} x {len(games)} games {t2 - t1:.1f} s; "
                  f"assemble dicts {t3 - t2:.1f} s; skaters per lineup {sorted(n_sk)}")
        if a.summary:
            df = summary_frame(roster, gp)
            print_summary(roster, df)
            p = OUT_DIR / "availability_2027.csv"
            df.to_csv(p, index=False)
            (OUT_DIR / "availability_model_2027.json").write_text(json.dumps(
                {"meta": {k: v for k, v in roster.meta.items() if k != "model"}, "model": roster.meta["model"],
                 "k": a.k, "seed": a.seed, "persist": persist}, indent=1, default=str))
            print(f"\n-> {p}")
    if not (a.validate or a.summary or a.timing):
        ap.print_help()


if __name__ == "__main__":
    main()
