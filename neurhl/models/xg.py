"""NeurHL-2 — L0 expected goals: shot quality, walk-forward, no player identity.

xG here is deliberately a model of the SHOT, not of the shooter. Finishing skill
is a player property and belongs in the player layer (RAPM / hazard modulation),
where it can be shrunk and pooled properly; folding it into xG would make the
same quantity do two jobs and would leak player identity into a feature that the
player model then consumes. So no shooter id, no team id — geometry, shot type,
and the state of play only.

Coverage. Every feature comes from the MoneyPuck shot corpus, which is complete
back to 2008 with ~0% missingness. That matters because the NHL JSON feed carries
no coordinates before 2012, so an xG model built on the native feed would exist
only for 2012+ and the whole pre-2012 half of the corpus would be unusable for
anything geometric.

Two deliberate constraints:

  * **Unblocked attempts only** (SOG + missed + goals, i.e. Fenwick). MoneyPuck
    excludes blocked shots, and blocked shots have no recorded location anyway —
    the coordinate is where the BLOCK happened, not the shot. Corsi therefore
    still has to come from the event shards; xG cannot replace it silently.
  * **Everything is oriented to the SHOOTER.** Raw fields are home/away
    (homeSkatersOnIce, homeTeamGoals); a model fed those learns "the home team
    scores more" instead of "a 2-man advantage scores more". Skater counts, score
    state and empty nets are all flipped to for/against before use.

The baseline that gate X1 measures against is a logistic on distance and |angle|
alone — the classic two-variable xG. Beating it is the whole claim.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import TENSORS  # noqa: E402

TARGET = "goal"

# geometry available in every season, 2008-2026
GEOM = ["shotDistance", "shotAngleAbs", "shotAngleAdjusted",
        "shotAnglePlusRebound", "shotAngleReboundRoyalRoad"]
# state of play, oriented to the shooter
STATE = ["skaters_for", "skaters_against", "skater_diff", "score_diff",
         "empty_net_target", "own_net_empty", "period", "is_home",
         "shooter_toi"]
# how the shot was created
CONTEXT = ["shotRush", "shotRebound", "timeSinceLastEvent", "timeSinceFaceoff",
           "speedFromLastEvent", "distanceFromLastEvent",
           "lastEventShotAngle", "lastEventShotDistance", "offWing"]
CATS = ["shotType", "shooterLeftRight", "lastEventCategory", "shooterPos"]
# A9 pre-committed fallback: recording-regime covariates in P5's sanctioned
# form (walk-forward computable, no season one-hots). For each game, the share
# of league shots recorded <10 ft and the rebound-flag share over the trailing
# ERA_TRAIL_GAMES league games STRICTLY BEFORE it -- input-side quantities
# only, so the GBM can condition on the recording regime it is scoring under.
ERA = ["era_close_share", "era_rebound_share"]
# A11 rung 1: per-rink recording bias. Recorded location is produced by
# rink-specific scorers with persistent biases (eda_02: venue means -2.31 to
# +2.10 ft, worst venue-season -6.67), invisible to any league-pooled
# calibrator. House-built walk-forward expanding offset -- MoneyPuck's fitted
# arenaAdjusted* columns remain P3-excluded.
RINK = ["rink_dist_offset"]
# A11 rung 2: how NOISY is this rink's recorded location right now? Trailing
# per-venue mean |dist_NHL − dist_MP| over the same shots (dual sources exist
# 2012+; the corpus-level join reconciles at corr 0.99972). High disagreement
# = uncertain recorded geometry; the GBM can discount distance accordingly.
# NaN before 2012 and below the min-shot floor — unmeasured is not zero.
DISAGREE = ["rink_disagree"]
# A11 rung 3: the two sides of the regime drift A9's pair left unconditioned —
# the rush-flag collapse (0.19% -> 0.06%, the sharpest marker of the
# definition change) and the REALIZED conversion of close shots (what an
# "8 ft" shot is currently worth), both trailing-300-league-games,
# strictly pre-game, no season one-hots (P5 form).
RUNG3 = ["era_rush_share", "era_close_conv"]
NUM = GEOM + STATE + CONTEXT + ERA + RINK + DISAGREE + RUNG3
FEATURES = NUM + CATS
BASELINE = ["shotDistance", "shotAngleAbs"]
ERA_TRAIL_GAMES = 300        # ~quarter season; fixed a priori, not tuned
ERA_CLOSE_FT = 10.0
RINK_MIN_SHOTS = 300         # A11: venue offset is 0 until this many prior shots


def rink_offsets(seasons) -> pd.DataFrame:
    """game_id -> rink_dist_offset (A11 rung 1).

    Expanding per-home-venue mean of (recorded shotDistance − trailing league
    mean), strictly pre-game; 0.0 until the venue has RINK_MIN_SHOTS prior
    shots. Walk-forward safe by construction: each game's value derives only
    from games before it."""
    rows = []
    for s in seasons:
        p = TENSORS / f"mp_shots_{s}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p, columns=["game_id", "isPlayoffGame", "period",
                                        "shotDistance"])
        d = d[(d.isPlayoffGame == 0) & (d.period <= 4)]
        g = d.groupby("game_id").agg(n=("shotDistance", "size"),
                                     sd_sum=("shotDistance", "sum")).reset_index()
        g["season_end"] = s
        rows.append(g)
    t = pd.concat(rows, ignore_index=True)
    metas = []
    for s in sorted(set(t.season_end)):
        p = TENSORS / f"games_ctx_{s}.parquet"
        if p.exists():
            metas.append(pd.read_parquet(p, columns=["game_id", "date",
                                                     "home_idx"]))
    gm = pd.concat(metas, ignore_index=True).drop_duplicates("game_id")
    t = t.merge(gm, on="game_id", how="left")
    t["date"] = pd.to_datetime(t.date)
    t["home_idx"] = t.home_idx.fillna(-1).astype("int64")
    t = t.sort_values(["date", "game_id"], kind="stable").reset_index(drop=True)
    ln = t.n.rolling(ERA_TRAIL_GAMES, min_periods=30).sum().shift(1)
    ls = t.sd_sum.rolling(ERA_TRAIL_GAMES, min_periods=30).sum().shift(1)
    lg_mean = ls / ln                              # NaN for the earliest games
    dev_sum = t.sd_sum - t.n * lg_mean
    valid = lg_mean.notna()
    t["_dev"] = np.where(valid, dev_sum, 0.0)
    t["_nc"] = np.where(valid, t.n, 0)
    cum_dev = t.groupby("home_idx")["_dev"].cumsum() - t._dev   # strictly prior
    cum_n = t.groupby("home_idx")["_nc"].cumsum() - t._nc
    off = np.divide(cum_dev.to_numpy(float), cum_n.to_numpy(float),
                    out=np.zeros(len(t)),
                    where=cum_n.to_numpy(float) >= RINK_MIN_SHOTS)
    t["rink_dist_offset"] = np.nan_to_num(off).astype("float32")
    return t.set_index("game_id")[RINK]


def rink_disagreement(seasons) -> pd.DataFrame:
    """game_id -> rink_disagree (A11 rung 2), strictly pre-game.

    Per game: NHL-feed shots (SOG/miss/goal with coordinates, 2012+) matched to
    MoneyPuck shots on (game_id, period, shooter, second-in-period);
    |hypot(89−|x|, y) − shotDistance| summed. Per home venue: expanding mean
    over PRIOR games (shifted), NaN until RINK_MIN_SHOTS matched shots."""
    per_game = []
    for s in seasons:
        if s < 2012:
            continue
        ep = TENSORS / f"events_{s}.parquet"
        mp = TENSORS / f"mp_shots_{s}.parquet"
        if not (ep.exists() and mp.exists()):
            continue
        ev = pd.read_parquet(ep, columns=["game_id", "game_type", "event_type",
                                          "period", "t", "p1", "x", "y",
                                          "has_coord"])
        ev = ev[(ev.game_type == 2) & ev.event_type.isin([7, 9, 14])
                & (ev.has_coord > 0) & (ev.period <= 4)].copy()
        ev["sec_in_p"] = (ev.t.astype("int64")
                          - (ev.period.astype("int64") - 1) * 1200)
        ev["dist_nhl"] = np.hypot(89.0 - ev.x.abs(), ev.y)
        m = pd.read_parquet(mp, columns=["game_id", "isPlayoffGame", "period",
                                         "time", "shooterPlayerId",
                                         "shotDistance"])
        m = m[(m.isPlayoffGame == 0) & (m.period <= 4)].copy()
        m["sec_in_p"] = (m.time.astype("int64")
                         - (m.period.astype("int64") - 1) * 1200)
        m = m.rename(columns={"shooterPlayerId": "p1"})
        j = ev.merge(m[["game_id", "period", "p1", "sec_in_p", "shotDistance"]],
                     on=["game_id", "period", "p1", "sec_in_p"], how="inner")
        j["adiff"] = (j.dist_nhl - j.shotDistance).abs()
        g = j.groupby("game_id", as_index=False).agg(nd=("adiff", "size"),
                                                     sd=("adiff", "sum"))
        g["season_end"] = s
        per_game.append(g)
    if not per_game:
        return pd.DataFrame(columns=DISAGREE)
    t = pd.concat(per_game, ignore_index=True)
    metas = []
    for s in sorted(set(t.season_end)):
        p = TENSORS / f"games_ctx_{s}.parquet"
        if p.exists():
            metas.append(pd.read_parquet(p, columns=["game_id", "date",
                                                     "home_idx"]))
    gm = pd.concat(metas, ignore_index=True).drop_duplicates("game_id")
    t = t.merge(gm, on="game_id", how="left")
    t["date"] = pd.to_datetime(t.date)
    t["home_idx"] = t.home_idx.fillna(-1).astype("int64")
    t = t.sort_values(["date", "game_id"], kind="stable").reset_index(drop=True)
    cum_sd = t.groupby("home_idx")["sd"].cumsum() - t.sd
    cum_n = t.groupby("home_idx")["nd"].cumsum() - t.nd
    vals = np.where(cum_n.to_numpy(float) >= RINK_MIN_SHOTS,
                    cum_sd.to_numpy(float)
                    / np.maximum(cum_n.to_numpy(float), 1.0), np.nan)
    t["rink_disagree"] = vals.astype("float32")
    return t.set_index("game_id")[DISAGREE]


def era_covariates(seasons) -> pd.DataFrame:
    """game_id -> (era_close_share, era_rebound_share), strictly pre-game.

    One table over all seasons is walk-forward safe by construction: each
    game's value derives only from games before it, so a training row never
    sees anything its own vantage could not have seen. Earliest games get NaN
    (min_periods), which HistGradientBoosting handles natively."""
    rows = []
    for s in seasons:
        p = TENSORS / f"mp_shots_{s}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p, columns=["game_id", "isPlayoffGame", "period",
                                        "shotDistance", "shotRebound"])
        d = d[(d.isPlayoffGame == 0) & (d.period <= 4)].copy()
        d["close"] = (d.shotDistance < ERA_CLOSE_FT).astype("int64")
        d["reb"] = d.shotRebound.astype("float64")
        g = d.groupby("game_id").agg(n=("close", "size"), close=("close", "sum"),
                                     reb=("reb", "sum")).reset_index()
        g["season_end"] = s
        rows.append(g)
    t = pd.concat(rows, ignore_index=True)
    dates = []
    for s in sorted(set(t.season_end)):
        p = TENSORS / f"games_ctx_{s}.parquet"
        if p.exists():
            dates.append(pd.read_parquet(p, columns=["game_id", "date"]))
    dm = pd.concat(dates, ignore_index=True).drop_duplicates("game_id")
    t = t.merge(dm, on="game_id", how="left")
    t["date"] = pd.to_datetime(t.date)
    t = t.sort_values(["date", "game_id"], kind="stable").reset_index(drop=True)
    n = t.n.rolling(ERA_TRAIL_GAMES, min_periods=30).sum().shift(1)
    t["era_close_share"] = (t.close.rolling(ERA_TRAIL_GAMES, min_periods=30)
                            .sum().shift(1) / n).astype("float32")
    t["era_rebound_share"] = (t.reb.rolling(ERA_TRAIL_GAMES, min_periods=30)
                              .sum().shift(1) / n).astype("float32")
    return t.set_index("game_id")[ERA]


def regime_covariates(seasons) -> pd.DataFrame:
    """game_id -> (era_rush_share, era_close_conv), strictly pre-game (A11
    rung 3). Same trailing-window machinery as era_covariates."""
    rows = []
    for s in seasons:
        p = TENSORS / f"mp_shots_{s}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p, columns=["game_id", "isPlayoffGame", "period",
                                        "shotDistance", "shotRush", "goal"])
        d = d[(d.isPlayoffGame == 0) & (d.period <= 4)].copy()
        d["rush"] = d.shotRush.astype("float64")
        d["close"] = (d.shotDistance < ERA_CLOSE_FT).astype("int64")
        d["close_goal"] = d.close * d.goal.astype("int64")
        g = d.groupby("game_id").agg(n=("rush", "size"), rush=("rush", "sum"),
                                     close=("close", "sum"),
                                     cg=("close_goal", "sum")).reset_index()
        g["season_end"] = s
        rows.append(g)
    t = pd.concat(rows, ignore_index=True)
    metas = []
    for s in sorted(set(t.season_end)):
        p = TENSORS / f"games_ctx_{s}.parquet"
        if p.exists():
            metas.append(pd.read_parquet(p, columns=["game_id", "date"]))
    dm = pd.concat(metas, ignore_index=True).drop_duplicates("game_id")
    t = t.merge(dm, on="game_id", how="left")
    t["date"] = pd.to_datetime(t.date)
    t = t.sort_values(["date", "game_id"], kind="stable").reset_index(drop=True)
    n = t.n.rolling(ERA_TRAIL_GAMES, min_periods=30).sum().shift(1)
    cl = t.close.rolling(ERA_TRAIL_GAMES, min_periods=30).sum().shift(1)
    t["era_rush_share"] = (t.rush.rolling(ERA_TRAIL_GAMES, min_periods=30)
                           .sum().shift(1) / n).astype("float32")
    t["era_close_conv"] = (t.cg.rolling(ERA_TRAIL_GAMES, min_periods=30)
                           .sum().shift(1) / cl.clip(lower=1)).astype("float32")
    return t.set_index("game_id")[RUNG3]


def build_features(d: pd.DataFrame, era: pd.DataFrame | None = None
                   ) -> pd.DataFrame:
    """Shooter-oriented feature frame. Input is a parsed mp_shots shard.
    `era` is the era_covariates() table; None leaves the two columns NaN."""
    x = pd.DataFrame(index=d.index)
    home = d.isHomeTeam.astype(bool)

    x["shotDistance"] = d.shotDistance.astype("float32")
    x["shotAngleAbs"] = d.shotAngle.abs().astype("float32")
    x["shotAngleAdjusted"] = d.shotAngleAdjusted.abs().astype("float32")
    x["shotAnglePlusRebound"] = d.shotAnglePlusRebound.astype("float32")
    x["shotAngleReboundRoyalRoad"] = d.shotAngleReboundRoyalRoad.astype("float32")

    # orient to the shooter: "for" is always the shooting side
    sk_h = d.homeSkatersOnIce.astype("float32")
    sk_a = d.awaySkatersOnIce.astype("float32")
    x["skaters_for"] = np.where(home, sk_h, sk_a)
    x["skaters_against"] = np.where(home, sk_a, sk_h)
    x["skater_diff"] = x.skaters_for - x.skaters_against
    g_h = d.homeTeamGoals.astype("float32")
    g_a = d.awayTeamGoals.astype("float32")
    x["score_diff"] = np.where(home, g_h - g_a, g_a - g_h)
    # the net being SHOT AT, versus the shooter's own net
    x["empty_net_target"] = np.where(home, d.awayEmptyNet, d.homeEmptyNet)
    x["own_net_empty"] = np.where(home, d.homeEmptyNet, d.awayEmptyNet)
    x["period"] = d.period.astype("float32")
    x["is_home"] = home.astype("float32")
    x["shooter_toi"] = d.shooterTimeOnIce.astype("float32")

    for c in CONTEXT:
        src = "offWing" if c == "offWing" else c
        x[c] = d[src].astype("float32")

    if era is not None:
        x["era_close_share"] = d.game_id.map(era.era_close_share).astype("float32")
        x["era_rebound_share"] = d.game_id.map(era.era_rebound_share).astype("float32")
    else:
        x["era_close_share"] = np.float32("nan")
        x["era_rebound_share"] = np.float32("nan")
    if era is not None and "rink_dist_offset" in era.columns:
        x["rink_dist_offset"] = (d.game_id.map(era.rink_dist_offset)
                                 .fillna(0.0).astype("float32"))
    else:
        x["rink_dist_offset"] = np.float32(0.0)
    if era is not None and "rink_disagree" in era.columns:
        x["rink_disagree"] = d.game_id.map(era.rink_disagree).astype("float32")
    else:
        x["rink_disagree"] = np.float32("nan")
    for c in RUNG3:
        if era is not None and c in era.columns:
            x[c] = d.game_id.map(era[c]).astype("float32")
        else:
            x[c] = np.float32("nan")

    x["shotType"] = d.shotType.fillna("UNK").astype("category")
    x["shooterLeftRight"] = d.shooterLeftRight.fillna("UNK").astype("category")
    x["lastEventCategory"] = d.lastEventCategory.fillna("UNK").astype("category")
    x["shooterPos"] = (d.playerPositionThatDidEvent.fillna("UNK")
                       .astype("category"))
    return x[FEATURES]


def load_shots(seasons, regular_only=True) -> pd.DataFrame:
    parts = []
    for s in seasons:
        p = TENSORS / f"mp_shots_{s}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        if regular_only:
            d = d[d.isPlayoffGame == 0]
        d = d[d.period <= 4]                  # exclude the shootout
        parts.append(d)
    if not parts:
        raise FileNotFoundError(f"no mp_shots shards for {list(seasons)}")
    return pd.concat(parts, ignore_index=True)


def align_categories(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Force test categoricals onto the training vocabulary.

    A category seen only in the test season would otherwise shift the encoding
    and silently corrupt every downstream column.
    """
    out = test.copy()
    for c in CATS:
        out[c] = pd.Categorical(out[c].astype(str),
                                categories=list(train[c].cat.categories))
    return out


def logloss(y, p, eps=1e-15) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
