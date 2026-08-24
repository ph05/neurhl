"""NeurHL-2 — learned player projection: rates, ice time and availability.

Replaces a hand-tuned shrinkage scheme that compressed the whole distribution.
The failure was worth recording, because every part of it was a constant chosen
by hand rather than estimated:

  * a single shrink constant (90,000 s) pulled a 3,300-minute veteran 31% toward
    the positional mean, turning a 4.5 P/60 player into 3.6. The empirically
    justified weight for a regular is 0.83, not 0.69;
  * total ice time was regressed toward a positional mean, which conflates
    "plays fewer minutes" with "played fewer games" and cost a first-line
    forward ~15% of his minutes;
  * raw per-60 rates were averaged across an era boundary. League points per
    game ran 14.5-14.9 in 2012-2017 and 16.3-17.1 in 2018-2026 — a 13% shift —
    so the history dragged projections toward the lower-scoring era.

Net effect, measured against 2025-26: the top projected scorer came out at 97
points against an actual league-leading 139, and the whole top ten was
compressed.

So nothing is hand-tuned here. Four quantities are LEARNED, each walk-forward:

    toi_per_gp   ice time per game played
    gp           availability, which is a different process from usage
    rel_g60      goal rate as a RATIO to the season's league rate
    rel_a60      assist rate, likewise

Separating usage from availability matters: a player who misses half a season
through injury has not become a worse player, and a model that predicts total
ice time cannot tell those apart. Modelling rates as ratios to the league level
is the era control — the model learns talent, and the league level for the
projected season is supplied separately.

Gradient boosting, because this is exactly the tabular small-n regime where
Grinsztajn et al. (NeurIPS 2022) put trees ahead of deep nets, and the same
argument that kept S1 at 5M parameters applies with more force to ~10k
player-seasons.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import TENSORS  # noqa: E402

POS_G = 2
# A10: assists recovered at source for 2008-2011 (build_htm assist fix;
# configs/assist_recovery.json — 2012 ground-truth p2/p3 agreement
# 0.9994/0.9989, zero invented). The full history is assist-complete now.
ASSISTS_FROM = 2008
# Train on EVERY player who appeared, not just regulars. Filtering to >=10 games
# meant the model never saw how little a depth player actually gets, so it
# over-predicted their usage by 46% (share 0.0257 against an actual 0.0176 for
# roster ranks 16+). With ~346 such players per season their inflation pushed the
# sum of predicted team shares to 1.046, and normalising then took those minutes
# straight off the top line -- the opposite of the intended effect.
MIN_TRAIN_GP = 1
TARGETS = ["toi_share", "gp_share", "rel_g60", "rel_a60"]

# A team's skater minutes are a FIXED BUDGET, not a free parameter: five skaters
# on the ice for sixty minutes is 300 minutes per team-game, and the measured
# value is 296.8-298.2 across 2016-2026 (slightly under 300 because a team kills
# penalties four-strong). Projecting each player's minutes independently ignores
# that constraint and the errors do not cancel -- they compounded into a 28%
# under-projection of ice time for first-line forwards, which was the single
# largest source of compression in the point totals.
#
# So usage is modelled as a SHARE of that budget and normalised within each
# roster. Availability is then absorbed too: a player who misses twenty games
# does not vanish from the ice, his minutes go to somebody else on the same
# roster, and a share model conserves them automatically.
# ...and the pools are SEPARATE by position. Measured 2023-2026, remarkably
# stable: forwards draw 180.5-181.3 minutes per team-game and defencemen
# 116.7-117.0, a 60.8 / 39.2 split. Normalising across the whole roster instead
# let a star on a thin roster absorb minutes that structurally belong to
# defencemen -- it put a first-line forward at 28.4 min/game, when no forward in
# the league exceeds 23.0 and only a number-one defenceman reaches 27.
POS_BUDGET_MIN_PER_GAME = {0: 181.0, 1: 116.9}          # 0 = forward, 1 = defence
SKATER_MIN_PER_TEAM_GAME = sum(POS_BUDGET_MIN_PER_GAME.values())
# observed ceilings, used only as a sanity bound on an individual projection
MAX_TOI_PER_GP_MIN = {0: 23.2, 1: 28.0}

# GAMES ARE A BUDGET TOO. Exactly 18 skaters dress per team per game, so the sum
# of every skater's EXPECTED games is a fixed total: 18 * 84 = 1,512 per team.
# Measured 2024-2026, the players on a club's announced roster take 0.966-0.972
# of them and the rest go to call-ups, so a projection covering only the
# announced roster should sum to ~0.968 of the budget. Left unnormalised the
# projection came to 0.979 of that target -- a real shortfall, and the reason
# projected games looked systematically light.
#
# What normalising does NOT do is push anyone to a full season, and it should
# not: a player who just appeared in 80+ games averages 71.9 the next year and
# repeats 80+ only 44% of the time, so an EXPECTATION above 80 would be wrong
# however unfamiliar it looks next to a realised season.
DRESSED_SKATERS_PER_GAME = 18
ROSTER_SHARE_OF_GAMES = 0.968
# No player's EXPECTED games should approach a full season. The most durable
# cohort -- players who just appeared in 80+ games -- averages 71.9 the next
# year and repeats 80+ only 44% of the time, so 78 (on an 82-game basis) is a
# generous ceiling on an expectation, not a conservative one.
MAX_EXPECTED_GP = 78.0


def load_player_seasons(seasons) -> pd.DataFrame:
    """One row per player-season, skaters only, regular season only."""
    rows = []
    for s in seasons:
        p = TENSORS / f"player_games_{s}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p, columns=["player_id", "game_type", "toi_sec",
                                        "goals", "assists", "sog", "pos_group"])
        d = d[(d.game_type == 2) & (d.pos_group != POS_G)]
        d = d.astype({"goals": "float64", "assists": "float64",
                      "sog": "float64"})
        g = d.groupby("player_id", as_index=False).agg(
            toi=("toi_sec", "sum"), g=("goals", "sum"), a=("assists", "sum"),
            sh=("sog", "sum"), gp=("goals", "size"),
            pos_group=("pos_group", "first"))
        g["season_end"] = s
        # SCHEDULE LENGTH normalisation. 2013 was 48 games and 2021 was 56, so a
        # raw games-played feature tells the model "this player appears in ~56
        # games" and it duly predicts a short season for the year AFTER a
        # shortened one. Measured: vantage 2022, the season following COVID,
        # carried a -7.37 point bias against -0.57 to -3.84 elsewhere. Availability
        # is therefore modelled as a SHARE of the schedule, exactly as scoring is
        # modelled as a share of the era's rate.
        g["sched_games"] = float(np.ceil(g.gp.max() / 2.0) * 2) if len(g) else 82.0
        # share of one team's seasonal budget FOR THAT POSITION
        n_teams = max(round(g.toi.sum() / (SKATER_MIN_PER_TEAM_GAME * 60 * 82)), 1)
        pos_tot = g.groupby("pos_group").toi.transform("sum")
        g["team_budget"] = pos_tot / n_teams
        g["toi_share"] = g.toi / g.team_budget
        rows.append(g)
    h = pd.concat(rows, ignore_index=True)
    lg = h.groupby("season_end").apply(
        lambda d: pd.Series({"lg_g60": 3600 * d.g.sum() / d.toi.sum(),
                             "lg_a60": 3600 * d.a.sum() / d.toi.sum()}),
        include_groups=False)
    h = h.merge(lg, left_on="season_end", right_index=True)
    sched = h.groupby("season_end").gp.max().clip(lower=40)
    h["sched_games"] = h.season_end.map(sched).astype(float)
    h["gp_share"] = h.gp / h.sched_games
    h["toi_per_gp"] = h.toi / h.gp
    h["g60"] = 3600 * h.g / h.toi
    h["a60"] = 3600 * h.a / h.toi
    h["rel_g60"] = h.g60 / h.lg_g60
    h["rel_a60"] = h.a60 / h.lg_a60
    return h


# Structurally broken seasons must not define the scoring environment. A4 says
# they train but never score; the same logic applies to the era estimate. 2021
# (56 games, division-only, no crowds) ran 15.77 points per game against 16.92
# the following year, and averaging it in made the model under-predict the whole
# 2021-22 season -- a -7.3 point bias where every other vantage sat between -0.5
# and -3.4.
BROKEN_SEASONS = {2013, 2021}


def league_level(hist: pd.DataFrame, target: int, window=3) -> tuple:
    """League g60 / a60 expected in `target`, from recent INTACT seasons."""
    h = hist[(hist.season_end < target)
             & (~hist.season_end.isin(BROKEN_SEASONS))]
    r = h.groupby("season_end")[["lg_g60", "lg_a60"]].first().tail(window)
    return float(r.lg_g60.mean()), float(r.lg_a60.mean())


def _ewma(vals, w):
    w = np.asarray(w, float)
    return float(np.sum(np.asarray(vals, float) * w) / w.sum()) if w.sum() else np.nan


def build_features(hist: pd.DataFrame, target: int, bios=None,
                   ids=None) -> pd.DataFrame:
    """Features for predicting season `target`, from seasons < target ONLY."""
    prior = hist[hist.season_end < target]
    if not len(prior):
        return pd.DataFrame()
    rapm = {}
    rp = TENSORS / f"rapm_prior_{target}.parquet"
    if rp.exists():
        r = pd.read_parquet(rp)
        r = r[~r.is_replacement.astype(bool)]
        rapm = {"off": dict(zip(r.player_id, r.cf_off)),
                "def": dict(zip(r.player_id, r.cf_def))}

    rows = []
    for pid, d in prior.groupby("player_id"):
        if ids is not None and pid not in ids:
            continue
        d = d.sort_values("season_end")
        age = np.nan
        if bios and pid in bios:
            age = target - 1 - bios[pid]["birth_year"]
        rec = {"player_id": pid, "pos_group": int(d.pos_group.iloc[-1]),
               "age": age, "n_seasons": len(d),
               "career_toi": float(d.toi.sum()),
               "gap": float(target - 1 - d.season_end.max())}
        for hl, tag in ((1.0, "h1"), (2.0, "h2"), (4.0, "h4")):
            w = 0.5 ** ((target - 1 - d.season_end) / hl) * d.toi
            rec[f"rel_g60_{tag}"] = _ewma(d.rel_g60, w)
            da = d[d.season_end >= ASSISTS_FROM]
            wa = 0.5 ** ((target - 1 - da.season_end) / hl) * da.toi
            rec[f"rel_a60_{tag}"] = _ewma(da.rel_a60, wa) if len(da) else np.nan
            rec[f"tpg_{tag}"] = _ewma(d.toi_per_gp,
                                      0.5 ** ((target - 1 - d.season_end) / hl))
        for lag in (1, 2, 3):
            s = d[d.season_end == target - lag]
            rec[f"gp_{lag}"] = float(s.gp_share.iloc[0]) if len(s) else np.nan
            rec[f"tpg_{lag}"] = float(s.toi_per_gp.iloc[0]) if len(s) else np.nan
            rec[f"relg_{lag}"] = float(s.rel_g60.iloc[0]) if len(s) else np.nan
            rec[f"rela_{lag}"] = (float(s.rel_a60.iloc[0])
                                  if len(s) and s.season_end.iloc[0] >= ASSISTS_FROM
                                  else np.nan)
        rec["rapm_off"] = rapm.get("off", {}).get(pid, np.nan)
        rec["rapm_def"] = rapm.get("def", {}).get(pid, np.nan)
        rows.append(rec)
    return pd.DataFrame(rows)


FEATS = None


def feature_cols(df) -> list:
    return [c for c in df.columns if c not in ("player_id",)]


def fit_predict(hist: pd.DataFrame, target: int, train_from: int,
                bios=None, ids=None, seed=0):
    """Train on every player-season strictly before `target`, predict `target`."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    X_tr, Y_tr = [], {t: [] for t in TARGETS}
    for V in range(train_from, target):
        f = build_features(hist, V, bios=bios)
        if not len(f):
            continue
        act = hist[hist.season_end == V].set_index("player_id")
        f = f[f.player_id.isin(act.index)].copy()
        f = f[f.player_id.map(act.gp) >= MIN_TRAIN_GP]
        if not len(f):
            continue
        X_tr.append(f)
        for t in TARGETS:
            Y_tr[t].append(f.player_id.map(act[t]).to_numpy(float))
    if not X_tr:
        raise RuntimeError("no training rows")
    X = pd.concat(X_tr, ignore_index=True)
    cols = feature_cols(X)
    models = {}
    use_cols = {}
    for t in TARGETS:
        y = np.concatenate(Y_tr[t])
        ok = np.isfinite(y)
        # a feature with <2 distinct observed values on THIS fit subset
        # cannot be binned (early vantages: lag-3 columns are all-NaN when
        # only one training vantage exists) -- drop per target, consistently
        # applied at predict time below
        tc = [c for c in cols if X.loc[ok, c].dropna().nunique() >= 2]
        use_cols[t] = tc
        m = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=40, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.12, random_state=seed)
        m.fit(X.loc[ok, tc], y[ok])
        models[t] = m

    f = build_features(hist, target, bios=bios, ids=ids)
    if not len(f):
        return pd.DataFrame(), models
    out = f[["player_id", "pos_group"]].copy()
    for t in TARGETS:
        out[t] = models[t].predict(f[use_cols[t]])
    return out, models


def fit_top_calibration(hist, target, train_from, bios, team_of_fn, games=82):
    """One monotone correction on projected points, fitted WALK-FORWARD.

    Gradient boosting cannot extrapolate past its training range and shrinks
    extremes, which leaves the very best players systematically under-projected:
    measured over four held-out seasons, the top decile came in 4.1 points low
    (69.0 against 73.1) and the top 2% 3.9 low, while deciles 0-8 were
    calibrated. A single slope-and-intercept fitted on the SEASON BEFORE the
    target corrects it without touching the target's own outcomes.
    """
    V = target - 1
    try:
        pred, _ = fit_predict(hist, V, train_from=train_from, bios=bios)
    except RuntimeError:
        return (0.0, 1.0)
    lg_g, lg_a = league_level(hist, V)
    tot = to_totals(pred, lg_g, lg_a, games=games, team_of=team_of_fn(V),
                    calib=None)
    act = hist[hist.season_end == V].set_index("player_id")
    m = tot[tot.player_id.isin(act.index)].copy()
    y = m.player_id.map(act.g + act.a).to_numpy(float)
    x = m.proj_p.to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 100:
        return (0.0, 1.0)
    b, a = np.polyfit(x[ok], y[ok], 1)
    return (float(a), float(np.clip(b, 0.9, 1.25)))


def to_totals(pred: pd.DataFrame, lg_g60: float, lg_a60: float,
              games: int, team_of=None, calib=None) -> pd.DataFrame:
    """Convert predicted shares and rates into season totals.

    If `team_of` is supplied, predicted usage shares are NORMALISED WITHIN EACH
    ROSTER so they sum to one, then multiplied by the team's seasonal
    skater-minute budget. That enforces the conservation law exactly: every
    minute allotted to one player is taken from a team-mate. Without it the
    per-player errors all leaned the same way and first-line ice time came out
    28% short.
    """
    d = pred.copy()
    d["toi_share"] = d.toi_share.clip(lower=0.0)
    d["pos_budget"] = (d.pos_group.map(POS_BUDGET_MIN_PER_GAME)
                       .fillna(POS_BUDGET_MIN_PER_GAME[0]) * 60.0 * games)
    if team_of is not None:
        d["team"] = d.player_id.map(team_of)
        tot = d.groupby(["team", "pos_group"]).toi_share.transform("sum")
        d["toi_share_norm"] = np.where(tot > 0, d.toi_share / tot, 0.0)
        d["toi_proj"] = d.toi_share_norm * d.pos_budget
    else:
        d["toi_proj"] = d.toi_share * d.pos_budget
    d["exp_gp"] = d.gp_share.clip(0.02, 1.0) * games
    if team_of is not None:
        # Normalise LEAGUE-WIDE, not per team. Announced rosters vary in size,
        # and the games a short roster does not cover go to call-ups rather than
        # to its own players -- forcing every club to the same total pushed a
        # 19-man roster to 77 games a man and pinned four Detroit players at a
        # full 84. A single league factor fixes the identity without inventing
        # durability for whoever happens to sit on a thin roster.
        # Only players WITH a roster place share the budget. The projection path
        # is already filtered to the announced rosters, but the backtest path is
        # not -- it carries every player in history -- and normalising that whole
        # set to one team's-worth of games crushed everyone's ice time (MAE 9.55
        # -> 15.66, top projections from ~110 to ~60). The scope of a
        # conservation law has to be exactly the population it conserves over.
        on_roster = d["team"].notna() if "team" in d else pd.Series(True, index=d.index)
        n_teams = max(len(set(team_of.values())), 1)
        budget_gp = (DRESSED_SKATERS_PER_GAME * games * n_teams
                     * ROSTER_SHARE_OF_GAMES)
        cap_gp = MAX_EXPECTED_GP * games / 82.0
        for _ in range(4):
            tot = float(d.loc[on_roster, "exp_gp"].sum())
            if tot <= 0:
                break
            d.loc[on_roster, "exp_gp"] = (
                d.loc[on_roster, "exp_gp"] * budget_gp / tot).clip(upper=cap_gp)
            if abs(float(d.loc[on_roster, "exp_gp"].sum()) - budget_gp) < 5:
                break
    # No skater exceeds the observed positional ceiling, and minutes taken off a
    # capped player are RETURNED to his position-mates rather than deleted --
    # the budget is conserved, so capping one player must feed the rest.
    cap = (d.pos_group.map(MAX_TOI_PER_GP_MIN).fillna(MAX_TOI_PER_GP_MIN[0])
           * 60.0 * d.exp_gp)
    if team_of is not None:
        for _ in range(3):
            over = (d.toi_proj - cap).clip(lower=0)
            if float(over.sum()) < 1.0:
                break
            d["toi_proj"] = np.minimum(d.toi_proj, cap)
            head = (cap - d.toi_proj).clip(lower=0)
            grp = [d.team, d.pos_group]
            spare = over.groupby(grp).transform("sum")
            room = head.groupby(grp).transform("sum")
            d["toi_proj"] = d.toi_proj + np.where(
                room > 0, head * (spare / room.replace(0, np.nan)).fillna(0), 0)
    d["toi_proj"] = np.minimum(d.toi_proj, cap)
    d["g60"] = (d.rel_g60.clip(0, 6) * lg_g60)
    d["a60"] = (d.rel_a60.clip(0, 6) * lg_a60)
    d["proj_g"] = d.g60 * d.toi_proj / 3600
    d["proj_a"] = d.a60 * d.toi_proj / 3600
    d["proj_p"] = d.proj_g + d.proj_a
    if calib is not None:
        a, b = calib
        scale = np.where(d.proj_p > 0, (a + b * d.proj_p) / d.proj_p.clip(lower=1e-6), 1.0)
        scale = np.clip(scale, 0.8, 1.4)
        for c in ("proj_g", "proj_a", "proj_p"):
            d[c] = d[c] * scale
    d["proj_toi_min"] = d.toi_proj / 60
    d["toi_per_gp_min"] = d.proj_toi_min / d.exp_gp.clip(lower=1)
    return d
