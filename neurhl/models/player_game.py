"""NeurHL-3 — M1: per player-game projection frame and models.

Targets per skater-game (regular season, skaters only), the recorded H4
definitions so the recorded baseline is comparable:

    toi_share   player toi_sec / team dressed-skater toi_sec
    shots       SOG count (Poisson)
    goal1       1{goals >= 1}
    assist1     1{assists >= 1}

Every feature is STRICTLY PRE-GAME: derived state is shifted within its group
ordering (player / team / goalie), so row i of a group sees only games before
it. The dressed roster and announced starter of TODAY are the conditioning
variables (INGAME-at-puck-drop, exactly the recorded H4 / Tier-0 protocol);
their in-game outcomes are not.

Feature blocks (PLAN_NeurHL3 PG):
  B0 self-form: EWMAs of the four target quantities at alpha {0.02, 0.1, 0.3}
     (0.1 = the baseline's own constant -> the model NESTS the baseline),
     last-1/3/5-game means, gp_todate, days since last game, age, pos, home.
  B1 usage structure (usage_*): EV/PP/SH-share EWMAs, line rank + short-vs-
     long delta, PP-unit-1 recency, linemate churn, linemate quality.
  B2 opportunity (absences_*): vacated same-position EWMA-TOI, absent-regular
     count, an above-me-absent flag, own return-from-absence ramp.
  B3 context: opponent Elo, opponent goals-against form, opposing starter
     quality (gq / GSAx, NaN until goalie_games exists), rest, b2b, schedule
     density, travel, timezone, era columns.

Chained GBMs (opportunity -> volume -> conversion): toi_share first; its
prediction feeds shots; both feed the binaries. HistGradientBoosting is
NaN-native, so sparse blocks enter under availability masks without imputation.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import TENSORS  # noqa: E402

ALPHAS = (0.02, 0.1, 0.3)
BASE_ALPHA = 0.1             # the recorded baseline's constant
SEASONS_ALL = list(range(2008, 2027))
TARGETS = ["toi_share", "shots", "goal1", "assist1"]


def _ewm_shifted(g: pd.core.groupby.SeriesGroupBy, alpha: float) -> pd.Series:
    return g.transform(lambda s: s.ewm(alpha=alpha, adjust=False)
                       .mean().shift(1))


def _roll_shifted(g, n: int) -> pd.Series:
    return g.transform(lambda s: s.rolling(n, min_periods=1).mean().shift(1))


def load_bios() -> pd.Series:
    """player_id -> birth year, best-effort from on-disk sources."""
    for cand in ("career_bios.parquet", "careers.parquet"):
        p = TENSORS / cand
        if p.exists():
            d = pd.read_parquet(p)
            for c in ("birth_year", "birthYear"):
                if c in d.columns and "player_id" in d.columns:
                    return d.set_index("player_id")[c]
            if "birthDate" in d.columns and "player_id" in d.columns:
                return d.set_index("player_id").birthDate.str[:4].astype(float)
    lk = Path(TENSORS).parents[1] / "data" / "raw" / "mp_lookup.csv"
    if lk.exists():
        d = pd.read_csv(lk)
        if "birthDate" in d.columns:
            return d.set_index("playerId").birthDate.str[:4].astype(float)
    return pd.Series(dtype=float)


def build_frames(seasons=SEASONS_ALL) -> pd.DataFrame:
    """One modeling frame over `seasons`, all features strictly pre-game."""
    from sim.game_model import run_elo

    pgs, gcs, usages, absencess, goalies = [], [], [], [], []
    for s in seasons:
        pg = pd.read_parquet(TENSORS / f"player_games_{s}.parquet")
        pg = pg[(pg.game_type == 2) & (pg.pos_group != 2)]
        pgs.append(pg.assign(season_end=s))
        gc = pd.read_parquet(TENSORS / f"games_ctx_{s}.parquet")
        gcs.append(gc[gc.game_type == 2])
        up = TENSORS / f"usage_{s}.parquet"
        if up.exists():
            usages.append(pd.read_parquet(up))
        ap = TENSORS / f"absences_{s}.parquet"
        if ap.exists():
            absencess.append(pd.read_parquet(ap))
        gp = TENSORS / f"goalie_games_{s}.parquet"
        if gp.exists():
            goalies.append(pd.read_parquet(gp))
    pg = pd.concat(pgs, ignore_index=True)
    gc = pd.concat(gcs, ignore_index=True)
    gc["date"] = pd.to_datetime(gc.date)

    # ---- targets and identity
    pg = pg.merge(gc[["game_id", "date", "home_idx", "away_idx", "home_rest",
                      "away_rest", "home_km3d", "away_km3d", "home_dtz",
                      "away_dtz", "days_in", "prior_gpg", "prior_ot_share",
                      "prior_margin_abs", "prior_parity", "season_scaled"]],
                  on="game_id", how="inner")
    pg["team"] = np.where(pg.is_home, pg.home_idx, pg.away_idx).astype("int64")
    pg["opp"] = np.where(pg.is_home, pg.away_idx, pg.home_idx).astype("int64")
    team_toi = pg.groupby(["game_id", "team"]).toi_sec.transform("sum")
    pg["toi_share"] = pg.toi_sec / team_toi.clip(lower=1)
    pg["shots"] = pg.sog.astype("float64")
    pg["goal1"] = (pg.goals >= 1).astype("float64")
    pg["assist1"] = (pg.assists >= 1).astype("float64")
    pg["goal_rate"] = pg.goal1
    pg["assist_rate"] = pg.assist1

    pg = pg.sort_values(["player_id", "date", "game_id"],
                        kind="stable").reset_index(drop=True)
    gpl = pg.groupby("player_id", sort=False)

    # ---- B0 self-form (shifted)
    for a in ALPHAS:
        tag = str(a).replace("0.", "a")
        for col in ("toi_share", "shots", "goal_rate", "assist_rate"):
            pg[f"{col}_ew{tag}"] = _ewm_shifted(gpl[col], a)
    for n in (1, 3, 5):
        pg[f"toi_share_l{n}"] = _roll_shifted(gpl["toi_share"], n)
        pg[f"shots_l{n}"] = _roll_shifted(gpl["shots"], n)
    pg["gp_todate"] = gpl.cumcount()
    pg["days_since"] = (pg.date - gpl.date.shift(1)).dt.days
    bios = load_bios()
    pg["age"] = pg.season_end - 1 - pg.player_id.map(bios)

    # ---- B1 usage structure (shifted)
    if usages:
        us = pd.concat(usages, ignore_index=True)
        us = us.merge(gc[["game_id", "date"]], on="game_id", how="left")
        us["pp_share"] = us.pp_toi / us.toi_all.clip(lower=1)
        us["ev_share"] = us.ev_toi / us.toi_all.clip(lower=1)
        us["pp1"] = ((us.pp_rank >= 1) & (us.pp_rank <= 5)).astype(float)
        us = us.sort_values(["player_id", "date", "game_id"], kind="stable")
        ug = us.groupby("player_id", sort=False)
        us["pp_share_ew"] = _ewm_shifted(ug["pp_share"], BASE_ALPHA)
        us["ev_share_ew"] = _ewm_shifted(ug["ev_share"], BASE_ALPHA)
        us["rank5_last"] = ug.rank5.shift(1)
        us["rank5_r3"] = _roll_shifted(ug["rank5"], 3)
        us["rank5_r25"] = _roll_shifted(ug["rank5"], 25)
        us["pp1_l5"] = _roll_shifted(ug["pp1"], 5)
        # linemate churn: Jaccard of {l1,l2} vs previous game's, shifted
        l1p, l2p = ug.l1.shift(1), ug.l2.shift(1)
        cur = np.stack([us.l1.to_numpy(), us.l2.to_numpy()], axis=1)
        prv = np.stack([l1p.to_numpy(), l2p.to_numpy()], axis=1)
        inter = ((cur[:, :, None] == prv[:, None, :]) & (cur[:, :, None] > 0)
                 ).any(-1).sum(1)
        union = ((cur > 0).sum(1) + (prv > 0).sum(1) - inter).clip(min=1)
        churn = 1.0 - inter / union
        # churn as computed compares TODAY's realised lines with yesterday's —
        # in-game information historically. Shift by one so the feature is
        # "how churny were my lines COMING IN": (t-1 vs t-2).
        us["lm_churn"] = pd.Series(churn, index=us.index).where(
            l1p.notna(), np.nan)
        us["lm_churn"] = us.groupby("player_id", sort=False)["lm_churn"].shift(1)
        # linemate quality: l1/l2 mapped to their own shifted pts/60 proxy
        us["pts_ind"] = np.nan  # filled from pg below
        pq = pg[["game_id", "player_id", "goal_rate", "assist_rate",
                 "toi_share_ewa1"]].copy()
        pq["prod_ew"] = pg["goal_rate_ewa1"].fillna(0) \
            + pg["assist_rate_ewa1"].fillna(0)
        us = us.merge(pq[["game_id", "player_id", "prod_ew"]]
                      .rename(columns={"player_id": "l1", "prod_ew": "l1_q"}),
                      on=["game_id", "l1"], how="left")
        us = us.merge(pq[["game_id", "player_id", "prod_ew"]]
                      .rename(columns={"player_id": "l2", "prod_ew": "l2_q"}),
                      on=["game_id", "l2"], how="left")
        us["lm_quality"] = us[["l1_q", "l2_q"]].mean(axis=1)
        # lm_quality uses the LINEMATES' shifted (pre-game) form of the SAME
        # game -- pre-game info about today's most-recent line, but the line
        # membership itself is in-game; shift it:
        us = us.sort_values(["player_id", "date", "game_id"], kind="stable")
        us["lm_quality"] = us.groupby("player_id", sort=False)["lm_quality"] \
            .shift(1)
        ucols = ["game_id", "player_id", "pp_share_ew", "ev_share_ew",
                 "rank5_last", "rank5_r3", "rank5_r25", "pp1_l5", "lm_churn",
                 "lm_quality"]
        pg = pg.merge(us[ucols], on=["game_id", "player_id"], how="left")
    else:
        for c in ("pp_share_ew", "ev_share_ew", "rank5_last", "rank5_r3",
                  "rank5_r25", "pp1_l5", "lm_churn", "lm_quality"):
            pg[c] = np.nan

    # ---- B2 opportunity (absences are pre-game facts about TODAY)
    if absencess:
        ab = pd.concat(absencess, ignore_index=True)
        agg = ab.groupby(["game_id", "team", "pos_group"], as_index=False).agg(
            vacated=("ewma_toi_sec", "sum"), n_absent=("player_id", "size"),
            max_absent_ew=("ewma_toi_sec", "max"))
        pg = pg.merge(agg, on=["game_id", "team", "pos_group"], how="left")
        pg["vacated"] = pg.vacated.fillna(0.0)
        pg["n_absent"] = pg.n_absent.fillna(0).astype(float)
        # same units on both sides: the absent player's EWMA toi_sec against
        # MY shifted EWMA toi_sec
        my_toi_ew = pg.groupby("player_id", sort=False)["toi_sec"] \
            .transform(lambda s: s.ewm(alpha=BASE_ALPHA, adjust=False)
                       .mean().shift(1))
        pg["above_me_out"] = (pg.max_absent_ew.fillna(0.0)
                              > my_toi_ew.fillna(0.0)).astype(float)
        # own return ramp: was I absent >=3 of my team's last games?
        pg["ret_gap"] = (pg.days_since > 8).astype(float)
    else:
        for c in ("vacated", "n_absent", "above_me_out", "ret_gap"):
            pg[c] = np.nan

    # ---- B3 context
    elo = run_elo(seasons)
    pg = pg.merge(elo[["game_id", "elo_h", "elo_a"]], on="game_id", how="left")
    pg["opp_elo"] = np.where(pg.is_home, pg.elo_a, pg.elo_h)
    pg["own_elo"] = np.where(pg.is_home, pg.elo_h, pg.elo_a)
    pg["rest"] = np.where(pg.is_home, pg.home_rest, pg.away_rest)
    pg["b2b"] = (pg.rest <= 1).astype(float)
    pg["km3d"] = np.where(pg.is_home, pg.home_km3d, pg.away_km3d)
    pg["dtz"] = np.where(pg.is_home, pg.home_dtz, pg.away_dtz)
    # opponent defensive form: shifted EWMA of goals against per team-game
    tg = gc.melt(id_vars=["game_id", "date"],
                 value_vars=["home_idx", "away_idx"], value_name="t_team")
    ga = np.where(tg.variable == "home_idx",
                  gc.set_index("game_id").away_g.reindex(tg.game_id).to_numpy(),
                  gc.set_index("game_id").home_g.reindex(tg.game_id).to_numpy())
    tg["ga"] = ga
    tg = tg.sort_values(["t_team", "date", "game_id"], kind="stable")
    tg["ga_ew"] = tg.groupby("t_team", sort=False)["ga"].transform(
        lambda s: s.ewm(alpha=BASE_ALPHA, adjust=False).mean().shift(1))
    pg = pg.merge(tg[["game_id", "t_team", "ga_ew"]]
                  .rename(columns={"t_team": "opp", "ga_ew": "opp_ga_ew"}),
                  on=["game_id", "opp"], how="left")
    if goalies:
        gl = pd.concat(goalies, ignore_index=True)
        st = gl[gl.goalie_start == 1][["game_id", "team", "gq", "gsax60",
                                       "starts_7d"]]
        pg = pg.merge(st.rename(columns={"team": "opp", "gq": "opp_gq",
                                         "gsax60": "opp_gsax60",
                                         "starts_7d": "opp_g_starts7"}),
                      on=["game_id", "opp"], how="left")
    else:
        for c in ("opp_gq", "opp_gsax60", "opp_g_starts7"):
            pg[c] = np.nan
    return pg


FEATS_B0 = (["toi_share_ewa02", "toi_share_ewa1", "toi_share_ewa3",
             "shots_ewa02", "shots_ewa1", "shots_ewa3",
             "goal_rate_ewa02", "goal_rate_ewa1", "goal_rate_ewa3",
             "assist_rate_ewa02", "assist_rate_ewa1", "assist_rate_ewa3",
             "toi_share_l1", "toi_share_l3", "toi_share_l5",
             "shots_l1", "shots_l3", "shots_l5",
             "gp_todate", "days_since", "age", "pos_group", "is_home"])
FEATS_B1 = ["pp_share_ew", "ev_share_ew", "rank5_last", "rank5_r3",
            "rank5_r25", "pp1_l5", "lm_churn", "lm_quality"]
FEATS_B2 = ["vacated", "n_absent", "above_me_out", "ret_gap"]
FEATS_B3 = ["opp_elo", "own_elo", "rest", "b2b", "km3d", "dtz", "days_in",
            "opp_ga_ew", "opp_gq", "opp_gsax60", "opp_g_starts7",
            "prior_gpg", "prior_ot_share", "prior_margin_abs", "prior_parity",
            "season_scaled"]
FEATURES = FEATS_B0 + FEATS_B1 + FEATS_B2 + FEATS_B3
