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
NUM = GEOM + STATE + CONTEXT
FEATURES = NUM + CATS
BASELINE = ["shotDistance", "shotAngleAbs"]


def build_features(d: pd.DataFrame) -> pd.DataFrame:
    """Shooter-oriented feature frame. Input is a parsed mp_shots shard."""
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
