"""NeurHL-G live inputs: tonight's games in the master-tensor format.

Given tonight's games and resolved lineups (neurhl/live/lineup_resolver.py),
builds the same raw arrays build_g_tensors.py builds for history, through the
same audited state code: each lineup player gets a row for tonight appended to
his history, so his state is read after every game already played this season
and in earlier seasons (data/build_g_state.py, extra=...).

Context: home Elo logit from the house Elo carried through every played game
(season carry-over phi_s, then in-season updates from 2026-27 results), the
2027 era vector (prior-season league rates, as tensorize_games.era_table
defines it), days into the season, and each side's rest, 3-day travel and
timezone change from the published schedule.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import PROC, PROJ, TENSORS  # noqa: E402
import data.build_g_state as BS  # noqa: E402
from data.build_g_tensors import NS, shrink  # noqa: E402

SEASON = 2027


def team_index() -> dict:
    return json.loads((TENSORS / "maps.json").read_text())["team"]


def elo_logits(games: pd.DataFrame, results: pd.DataFrame = None) -> np.ndarray:
    """Pre-game home Elo logit for each of `games` (home, away abbreviations),
    from the house Elo over all history plus any 2026-27 results."""
    sys.path.insert(0, str(PROJ / "src"))
    import engine as E
    v1 = json.loads((PROJ / "output" / "params.json").read_text())
    g = pd.read_csv(PROC / "games.csv", keep_default_na=False, parse_dates=["date"])
    if results is not None and len(results):
        r = results.rename(columns={"home_g": "home_g", "away_g": "away_g"})
        extra = pd.DataFrame({
            "date": pd.to_datetime(r.date), "away": r.away, "away_g": r.away_g,
            "home": r.home, "home_g": r.home_g, "ot": "", "game_type": "R",
            "season_end": SEASON, "went_ot": r.last_period.eq("OT"),
            "went_so": r.last_period.eq("SO"), "margin": r.home_g - r.away_g})
        g = pd.concat([g, extra], ignore_index=True)
    g = g.sort_values("date").reset_index(drop=True)
    _, end, _ = E.run_elo(g, K=v1["K"], H=v1["H"], phi_s=v1["phi_s"])
    if SEASON in end:
        rt = end[SEASON]
    else:
        last = max(end)
        rt = {t: E.MEAN + v1["phi_s"] * (x - E.MEAN) for t, x in end[last].items()}
    d = np.array([rt.get(h, E.MEAN) + v1["H"] - rt.get(a, E.MEAN)
                  for h, a in zip(games.home, games.away)])
    e = 1.0 / (1.0 + 10 ** (-d / 400.0))
    return np.log(e / (1 - e))


def era_2027() -> dict:
    from data.tensorize_games import era_table
    games = pd.read_csv(PROC / "games.csv")
    ts = pd.read_csv(PROC / "team_seasons.csv")
    r = games[(games.game_type == "R") & (games.season_end == SEASON - 1)]
    return {"prior_gpg": float((r.home_g + r.away_g).mean()),
            "prior_ot_share": float(r.went_ot.astype(bool).mean()),
            "prior_so_share": float(r.went_so.astype(bool).mean()),
            "prior_margin_abs": float(r.margin.abs().mean()),
            "prior_parity": float(ts[ts.season_end == SEASON - 1].pts_pct.std()),
            "flag_3v3": 1.0, "flag_covid": 0.0,
            "season_scaled": (SEASON - 2006) / 20.0}


def build(games: pd.DataFrame, lineups: dict, results: pd.DataFrame = None,
          hproj: pd.DataFrame = None):
    """games: game_id, date (YYYY-MM-DD), home, away (NHL abbreviations).
    lineups: {game_id: {"home": {"skaters": [...], "goalie": id},
                        "away": {...}}}.  Returns (arrays, meta, names)."""
    names = json.loads((TENSORS / "g_names.json").read_text())
    tidx = team_index()
    bios = pd.read_parquet(TENSORS / "career_bios.parquet").set_index("player_id")
    date = pd.Timestamp(games.date.iloc[0])
    sk_x, gk_x, tm_x = [], [], []
    for r in games.itertuples():
        for side, is_home, team in (("home", 1, r.home), ("away", 0, r.away)):
            lu = lineups[r.game_id][side]
            for pid in lu["skaters"]:
                sk_x.append({"game_id": r.game_id, "player_id": int(pid),
                             "is_home": is_home, "team": tidx[team],
                             "pos_group": bios.pos_group.get(int(pid), 0),
                             "date": date, "season_end": SEASON, "game_type": 2})
            if lu.get("goalie"):
                gk_x.append({"game_id": r.game_id, "player_id": int(lu["goalie"]),
                             "is_home": is_home, "team": tidx[team], "date": date,
                             "season_end": SEASON})
            tm_x.append({"game_id": r.game_id, "is_home": is_home, "date": date,
                         "home_idx": tidx[r.home], "away_idx": tidx[r.away],
                         "season_end": SEASON})
    dates = BS.game_dates()
    ids = set(games.game_id)
    sk = BS.skater_state(dates, pd.DataFrame(sk_x))
    sk = sk[sk.game_id.isin(ids)].copy()
    sk = pd.concat([sk, shrink(sk, names["priors"])], axis=1)
    from sim.rookie_hook import apply as rookie_priors      # no-op unless enabled (PLAN_NeurHL_1_1 A15)
    sk = rookie_priors(sk, names["priors"])
    gk = BS.goalie_state(dates, pd.DataFrame(gk_x)) if gk_x else pd.DataFrame()
    gk = gk[gk.game_id.isin(ids)] if len(gk) else gk
    tm = BS.team_state(dates, pd.DataFrame(tm_x))
    tm = tm[tm.game_id.isin(ids)].copy()

    from sim.project_2027 import load_schedule
    from sim.schedule_context import build as sched_ctx
    sc = sched_ctx(load_schedule(), SEASON).set_index("game_id")
    N = len(games)
    gi = {g: i for i, g in enumerate(games.game_id)}
    Fs, Fb = names["sk_feat"], names["sk_base"]
    # a live bundle trained with rookie inputs (candidate g1rk, PLAN_NeurHL_1_1 A16) lists them in
    # its own feature names; fill them from the 2026-27 rookie table. Other bundles: unchanged.
    try:
        _b = json.loads((ROOT / "configs" / "live_models.json").read_text())["neurhl_g"]
        _bn = json.loads((ROOT / "checkpoints" / "g" / _b / "bundle.json").read_text())["names"]["sk_feat"]
    except (OSError, KeyError, ValueError):
        _bn = Fs
    if "rk_flag" in _bn:
        _rk = pd.read_csv(ROOT / "configs" / "rookie_priors_2027.csv").set_index("player_id")
        sk["rk_pred_g"] = sk.player_id.map(_rk.pred_g)
        sk["rk_pred_a"] = sk.player_id.map(_rk.pred_a)
        sk["rk_flag"] = sk.player_id.isin(_rk.index).astype(float)
        Fs = _bn
    if "px_g" in _bn:                                  # candidate g1x (A19): non-NHL records, all players
        _px = pd.read_parquet(ROOT / "data" / "tensors" / "prior_features.parquet")
        _px = _px[_px.season_end == SEASON].set_index("player_id")
        sk["px_g"] = sk.player_id.map(_px.px_g)
        sk["px_a"] = sk.player_id.map(_px.px_a)
        sk["px_gp"] = sk.player_id.map(_px.px_gp).fillna(0.0)
        Fs = _bn
    SK = np.full((N, 2, NS, len(Fs)), np.nan, np.float32)
    SKB = np.full((N, 2, NS, len(Fb)), np.nan, np.float32)
    SKM = np.zeros((N, 2, NS), np.float32)
    SKP = np.zeros((N, 2, NS), np.int8)
    SKID = np.zeros((N, 2, NS), np.int64)
    sk["gi"] = sk.game_id.map(gi)
    sk["side"] = 1 - sk.is_home.astype(int)
    sk = sk.sort_values(["gi", "side", "b_toi_ev"], ascending=[True, True, False])
    sk["slot"] = sk.groupby(["gi", "side"]).cumcount()
    sk = sk[sk.slot < NS]
    for c in Fs:
        if c not in sk:
            sk[c] = np.nan
    ii, ss, kk = sk.gi.to_numpy(), sk.side.to_numpy(), sk.slot.to_numpy()
    SK[ii, ss, kk] = sk[Fs].to_numpy(np.float32)
    SKB[ii, ss, kk] = sk[Fb].to_numpy(np.float32)
    SKM[ii, ss, kk] = 1.0
    SKP[ii, ss, kk] = sk.pos_group.clip(0, 1).to_numpy()
    SKID[ii, ss, kk] = sk.player_id.to_numpy()

    GK = np.full((N, 2, len(names["gk_feat"])), np.nan, np.float32)
    GKID = np.zeros((N, 2), np.int64)
    if len(gk):
        gk = gk.assign(gi=gk.game_id.map(gi), side=1 - gk.is_home.astype(int))
        GK[gk.gi, gk.side] = gk[names["gk_feat"]].to_numpy(np.float32)
        GKID[gk.gi, gk.side] = gk.player_id.to_numpy()

    tm = tm.assign(gi=tm.game_id.map(gi), side=1 - tm.is_home.astype(int))
    h = tm.side == 0
    for col, hc, ac in (("rest", "home_rest", "away_rest"), ("km3d", "home_km3d", "away_km3d"),
                        ("dtz", "home_dtz", "away_dtz")):
        tm[col] = np.where(h, tm.game_id.map(sc[hc]), tm.game_id.map(sc[ac]))
    tm["b2b"] = (tm.rest <= 1).astype(float)
    # NeurHL-H lineup projections for tonight (sim/h_live.lineup_projection)
    if hproj is not None:
        tm["h_cfpct"] = np.where(h, tm.game_id.map(hproj.cfpct_h), tm.game_id.map(hproj.cfpct_a))
        tm["h_clshare"] = np.where(h, tm.game_id.map(hproj.clsh_h), tm.game_id.map(hproj.clsh_a))
    for c in names["tm_feat"]:
        if c not in tm:
            tm[c] = np.nan
    TM = np.full((N, 2, len(names["tm_feat"])), np.nan, np.float32)
    TM[tm.gi, tm.side] = tm[names["tm_feat"]].to_numpy(np.float32)

    era = era_2027()
    elo = elo_logits(games, results)
    ctx = []
    for k, r in enumerate(games.itertuples()):
        row = {"elo_logit": elo[k], **era,
               "days_in": float(sc.days_in.get(r.game_id, 0))}
        ctx.append([row[c] for c in names["ctx"]])
    CTX = np.array(ctx, np.float32)
    arrays = {"SK": SK, "SKB": SKB, "SKM": SKM, "SKP": SKP, "SKID": SKID,
              "GK": GK, "GKID": GKID, "TM": TM, "CTX": CTX}
    meta = games.copy()
    meta["season_end"] = SEASON
    return arrays, meta, names
