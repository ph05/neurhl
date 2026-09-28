"""Parity of neurhl/live/nhl_api_shots.py with MoneyPuck on 2025-26 (season 2026).

Both sources exist for every 2025-26 regular-season game: the NHL
play-by-play and shift charts under data/raw/{pbp,shifts}/2026/, and the
MoneyPuck-derived neurhl/data/tensors/mp_shots_2026.parquet. The API rows are
built with handedness from sources older than 2025-26 (MoneyPuck 2008-2025,
player landing files, roster snapshots), as they would be for a live season.

  counts    unblocked attempts per game in each source; rows matched on
            (game, period, time, shooter, event); what the unmatched rows are
  fields    every raw column the xG model or a live builder reads, and the 33
            model features after models/xg.build_features (matched rows):
            exact-match rate, correlation, mean absolute difference
  xG        the frozen production GBM (neurhl/checkpoints/xg_live_v2027.pkl,
            sha256 checked) on both sources with the same covariate table and
            its opening calibrator (the isotonic seed: what the first 25 league
            games of 2026-27 get): per shot (matched rows), per team-game and
            per player-game (each source's own rows). Its seed is 2025-26
            itself, so the MoneyPuck side is calibrated in sample; both sides
            go through the same map, which is what the comparison needs
  live path each source through the live code in its own sandbox: covariates
            recomputed from that source, train_xg.seq_calibrate over the
            season, build_stream, build_xg_games, build_goalie_games.season_shots;
            compared on tgx, pgx and goalie xG faced
  arbiter   where the empty-net flags or skater counts disagree, which source
            the shift charts (events_2026 on-ice slots) side with
  gate      the fallback in ingest_2027.py is justified only if team-game xG
            correlation >= 0.98 and |bias| <= 2% (both xG views above)

Writes neurhl/configs/nhl_api_shots_parity.json. Sandboxes
neurhl/data/tensors/_sandbox_ingest/apishots_{api,mp}/ hold symlinks to the
real tables and real files only for mp_shots_2026 (API side) and
xg_shots_2026; nothing else is written. Exit 1 if the gate fails.

Usage: uv run --no-project --python 3.12 --with numpy --with "pandas<3" \
         --with pyarrow --with requests --with scikit-learn==1.9.1 --with scipy \
         python neurhl/live/test_nhl_api_shots.py
"""
import hashlib
import json
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
NRL = HERE.parent
for p in (str(HERE), str(NRL)):
    if p not in sys.path:
        sys.path.insert(0, p)
import common  # noqa: E402
import models.xg as XG  # noqa: E402
import train.train_xg as TX  # noqa: E402
import data.build_stream as BSTR  # noqa: E402
import data.build_xg_games as XGG  # noqa: E402
import data.build_goalie_games as GG  # noqa: E402
import ingest_2027 as ING  # noqa: E402
import nhl_api_shots as NAS  # noqa: E402

S = 2026
TENSORS = common.TENSORS
SANDBOX = TENSORS / "_sandbox_ingest"
CKPT = common.CKPT / "xg_live_v2027.pkl"
OUT = common.CONFIGS / "nhl_api_shots_parity.json"
KEY = ["game_id", "period", "time", "shooterPlayerId", "event"]
GATE_CORR, GATE_BIAS = 0.98, 0.02
# raw columns read by models/xg.build_features, the covariates or a live builder
NUM_FIELDS = ["xCord", "yCord", "shotDistance", "shotAngle", "shotAngleAdjusted",
              "shotAnglePlusRebound", "shotAngleReboundRoyalRoad", "homeSkatersOnIce",
              "awaySkatersOnIce", "homeTeamGoals", "awayTeamGoals", "homeEmptyNet",
              "awayEmptyNet", "isHomeTeam", "goal", "shooterTimeOnIce", "shotRush",
              "shotRebound", "timeSinceLastEvent", "timeSinceFaceoff", "speedFromLastEvent",
              "distanceFromLastEvent", "lastEventShotAngle", "lastEventShotDistance",
              "offWing", "goalieIdForShot", "shotOnEmptyNet", "shotWasOnGoal"]
CAT_FIELDS = ["shotType", "shooterLeftRight", "lastEventCategory",
              "playerPositionThatDidEvent", "teamCode"]
# not reproduced by nhl_api_shots (left NaN); none is read by the xG model
NOT_REPRODUCED = [c for c in NAS.COLUMNS if c.startswith(("shooting", "defending"))
                  or c in ("averageRestDifference", "timeDifferenceSinceChange",
                           "homePenalty1Length", "homePenalty1TimeLeft",
                           "awayPenalty1Length", "awayPenalty1TimeLeft",
                           "shotGeneratedRebound", "shotGoalieFroze", "shotPlayStopped",
                           "shotPlayContinuedInZone", "shotPlayContinuedOutsideZone")]


def log(msg: str) -> None:
    print(msg, flush=True)


def r6(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 6)


def corr(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or a[ok].std() == 0 or b[ok].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def agree(a: pd.Series, m: pd.Series) -> dict:
    """API vs MoneyPuck on matched rows: bias is mean(api - mp), rel_bias
    sum(api) / sum(mp) - 1."""
    a, m = a.astype(float).to_numpy(), m.astype(float).to_numpy()
    d = a - m
    return {"n": int(len(a)), "corr": r6(corr(a, m)), "mae": r6(np.nanmean(np.abs(d))),
            "bias": r6(np.nanmean(d)), "rel_bias": r6(np.nansum(a) / np.nansum(m) - 1),
            "mean_mp": r6(np.nanmean(m)), "sd_mp": r6(np.nanstd(m))}


def paired(api: pd.DataFrame, mp: pd.DataFrame, keys: list, col: str) -> dict:
    """Sum `col` by `keys` in each source (its own rows), outer-join, compare."""
    a = api.groupby(keys)[col].sum().rename("a")
    m = mp.groupby(keys)[col].sum().rename("m")
    j = pd.concat([a, m], axis=1).fillna(0.0)
    return agree(j.a, j.m)


def match(api: pd.DataFrame, mp: pd.DataFrame) -> pd.DataFrame:
    a, m = api.copy(), mp.copy()
    a["_k"] = a.groupby(KEY).cumcount()
    m["_k"] = m.groupby(KEY).cumcount()
    a["_ia"], m["_im"] = np.arange(len(a)), np.arange(len(m))
    return a[KEY + ["_k", "_ia"]].merge(m[KEY + ["_k", "_im"]], on=KEY + ["_k"],
                                        how="outer", indicator=True)


# ------------------------------------------------------------ xG machinery
def load_ckpt() -> dict:
    want = Path(str(CKPT) + ".sha256").read_text().split()[0]
    got = hashlib.sha256(CKPT.read_bytes()).hexdigest()
    if want != got:
        raise RuntimeError(f"{CKPT} does not match its sha256 sidecar")
    with open(CKPT, "rb") as f:
        ck = pickle.load(f)
    import sklearn
    if sklearn.__version__ != ck["sklearn"]:
        raise RuntimeError(f"need scikit-learn=={ck['sklearn']}, have {sklearn.__version__}")
    return ck


def p_raw(ck: dict, d: pd.DataFrame, era: pd.DataFrame) -> np.ndarray:
    """ingest_2027.build_xg up to the calibrator."""
    X = XG.build_features(d, era)
    for c in XG.CATS:
        X[c] = pd.Categorical(X[c].astype(str), categories=ck["cats"][c])
    X = X.drop(columns=ck["dead"])[ck["columns"]]
    return ck["model"].predict_proba(X)[:, 1]


def opening_map(ck: dict):
    """seq_calibrate's first block: isotonic on the trailing seq_w seed shots."""
    from sklearn.isotonic import IsotonicRegression
    return IsotonicRegression(out_of_bounds="clip").fit(
        np.asarray(ck["seed_p"][-ck["seq_w"]:], float),
        np.asarray(ck["seed_y"][-ck["seq_w"]:], float))


def point_modules(d: Path) -> None:
    for m in (XG, TX, BSTR, XGG, GG):
        m.TENSORS = d


def sandbox(name: str, api_rows: pd.DataFrame = None) -> Path:
    """Symlinks to the real history the live path reads; mp_shots_2026 is the
    API table (a real file) on the API side."""
    d = SANDBOX / name
    assert SANDBOX in d.parents
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for p in TENSORS.glob("*.parquet"):
        fam, _, yr = p.stem.rpartition("_")
        if not yr.isdigit():
            continue
        yr = int(yr)
        keep = ((fam == "mp_shots" and yr < S) or (fam == "games_ctx" and yr <= S)
                or (fam == "events" and 2012 <= yr <= S)
                or (fam in ("player_games", "stints") and yr == S)
                or (fam == "mp_shots" and yr == S and api_rows is None))
        if keep:
            (d / p.name).symlink_to(p)
    if api_rows is not None:
        api_rows.to_parquet(d / f"mp_shots_{S}.parquet", index=False)
    return d


def live_path(ck: dict, d: Path, maps: dict) -> dict:
    """The live code over season S in sandbox `d`: covariates, GBM,
    seq_calibrate, xg_shots, stream, pgx, tgx, goalie xG faced."""
    point_modules(d)
    era = ING.era_features(S)
    te = XG.load_shots([S])
    pr = p_raw(ck, te, era)
    y = te[XG.TARGET].to_numpy()
    xg = TX.seq_calibrate(ck["seed_p"], ck["seed_y"], ck["seed_gid"], pr, y,
                          te.game_id.to_numpy(), S + 1, w=ck["seq_w"], k=ck["seq_k"])
    xs = pd.DataFrame({
        "game_id": te.game_id.to_numpy(), "is_home": te.isHomeTeam.to_numpy(),
        "goal": y, "xg": xg, "sk_h": te.homeSkatersOnIce.to_numpy(),
        "sk_a": te.awaySkatersOnIce.to_numpy(),
        "en": (te.homeEmptyNet | te.awayEmptyNet).to_numpy(),
        "time": te.time.to_numpy(), "period": te.period.to_numpy(),
        "shooter": te.shooterPlayerId.to_numpy(), "team_code": te.teamCode.to_numpy()})
    xs.to_parquet(d / f"xg_shots_{S}.parquet", index=False)
    stream = BSTR.build_season(S, maps)
    pgx = XGG.build_player(stream, XGG.goalie_ids(S), S)
    tgx = XGG.build_team(stream, S)
    gk = GG.season_shots(S)
    shots = stream[stream.event_type.isin([7, 9, 14]) & (stream.game_type == 2)]
    return {"xs": xs, "pgx": pgx, "tgx": tgx, "gk": gk,
            "stream_xg_cov": float(shots.has_xg.mean())}


# -------------------------------------------------------------------- main
def main() -> int:
    t0 = time.time()
    res = {"season": S, "checkpoint": CKPT.name, "gate": {}}
    mp = pd.read_parquet(TENSORS / f"mp_shots_{S}.parquet")
    mp = mp[(mp.isPlayoffGame == 0) & (mp.period <= 4)].reset_index(drop=True)
    gids = sorted(set(mp.game_id.astype(int)))
    hand = NAS.handedness(S)
    api = NAS.build(gids, S, common.RAW, hand)
    api = api[(api.isPlayoffGame == 0) & (api.period <= 4)].reset_index(drop=True)
    log(f"season {S}: {len(gids)} regular-season games with MoneyPuck rows; "
        f"MoneyPuck {len(mp):,} rows, NHL API {len(api):,} rows "
        f"({time.time() - t0:.0f}s)")

    # ---- counts and matching
    na, nm = api.groupby("game_id").size(), mp.groupby("game_id").size()
    dn = na.reindex(gids, fill_value=0) - nm.reindex(gids, fill_value=0)
    j = match(api, mp)
    both = j[j._merge == "both"]
    ao = api.iloc[j.loc[j._merge == "left_only", "_ia"].astype(int)]
    mo = mp.iloc[j.loc[j._merge == "right_only", "_im"].astype(int)]
    rekey = ao.merge(mo, on=["game_id", "period", "time", "shooterPlayerId"])
    near = ao.merge(mo, on=["game_id", "period", "shooterPlayerId", "event"])
    near = near[(near.time_x - near.time_y).abs().between(1, 5)]
    res["counts"] = {
        "games": len(gids), "rows_api": len(api), "rows_mp": len(mp),
        "rows_matched": len(both),
        "matched_share_of_mp": r6(len(both) / len(mp)),
        "matched_share_of_api": r6(len(both) / len(api)),
        "games_equal_count": r6((dn == 0).mean()),
        "per_game_diff_api_minus_mp": {"mean": r6(dn.mean()), "mean_abs": r6(dn.abs().mean()),
                                       "min": int(dn.min()), "max": int(dn.max())},
        "min_game_ratio_api_over_mp": r6((na.reindex(gids) / nm.reindex(gids)).min()),
        "unmatched": {"api_only": len(ao), "mp_only": len(mo),
                      "same_shot_other_event_type": len(rekey),
                      "same_shooter_event_within_5s": len(near)}}
    c = res["counts"]
    log(f"\nrows matched {c['rows_matched']:,} = {c['matched_share_of_mp']:.2%} of MoneyPuck, "
        f"{c['matched_share_of_api']:.2%} of API; games with equal counts "
        f"{c['games_equal_count']:.1%}, per-game API-MP mean {dn.mean():+.2f} "
        f"(min {dn.min()}, max {dn.max()})")
    log(f"unmatched: {len(ao)} API-only, {len(mo)} MoneyPuck-only; of these "
        f"{len(rekey)} are the same shot with SHOT/MISS swapped and {len(near)} the "
        f"same shot 1-5 s apart (feed revisions between the two downloads)")

    # ---- fields on matched rows
    A = api.iloc[both._ia.astype(int)].reset_index(drop=True)
    M = mp.iloc[both._im.astype(int)].reset_index(drop=True)
    fields = {}
    log(f"\n{'field':<30}{'exact':>9}{'corr':>9}{'mae':>10}")
    for f in NUM_FIELDS:
        a, m = A[f].astype(float), M[f].astype(float)
        ex = (np.isclose(a, m, atol=1e-6) | (a.isna() & m.isna())).mean()
        fields[f] = {"exact": r6(ex), "corr": r6(corr(a, m)), "mae": r6((a - m).abs().mean())}
        log(f"{f:<30}{ex:>9.4f}{fields[f]['corr'] or float('nan'):>9.4f}"
            f"{fields[f]['mae']:>10.4f}")
    for f in CAT_FIELDS:
        a, m = A[f].fillna("UNK").astype(str), M[f].fillna("UNK").astype(str)
        fields[f] = {"exact": r6((a == m).mean()),
                     "mp_missing": int(M[f].isna().sum()), "api_missing": int(A[f].isna().sum())}
        log(f"{f:<30}{fields[f]['exact']:>9.4f}   (missing: MoneyPuck "
            f"{fields[f]['mp_missing']}, API {fields[f]['api_missing']})")
    res["fields"] = fields
    res["not_reproduced"] = NOT_REPRODUCED

    # ---- model features and xG, same covariate table (real, MoneyPuck-based)
    ck = load_ckpt()
    point_modules(TENSORS)
    era = ING.era_features(S)
    XA, XM = XG.build_features(A, era), XG.build_features(M, era)
    feats = {}
    for f in XG.FEATURES:
        if f in XG.CATS:
            feats[f] = r6((XA[f].astype(str) == XM[f].astype(str)).mean())
        else:
            a, m = XA[f].astype(float), XM[f].astype(float)
            feats[f] = r6((np.isclose(a, m, atol=1e-5) | (a.isna() & m.isna())).mean())
    res["features_exact"] = feats
    log(f"\nmodel features: exact-match rate min {min(feats.values()):.4f} "
        f"({min(feats, key=feats.get)}), median {np.median(list(feats.values())):.4f}")
    iso = opening_map(ck)
    for d in (api, mp):
        d["p_raw"] = p_raw(ck, d, era)
        d["xg"] = iso.predict(d.p_raw.to_numpy())
    A = api.iloc[both._ia.astype(int)].reset_index(drop=True)
    M = mp.iloc[both._im.astype(int)].reset_index(drop=True)
    dx = A.xg.to_numpy() - M.xg.to_numpy()
    shot = agree(A.xg, M.xg)
    shot.update({"raw_gbm_corr": r6(corr(A.p_raw, M.p_raw)),
                 "share_absdiff_gt_0.01": r6((np.abs(dx) > 0.01).mean()),
                 "share_absdiff_gt_0.05": r6((np.abs(dx) > 0.05).mean()),
                 "max_absdiff": r6(np.abs(dx).max())})
    team = paired(api, mp, ["game_id", "isHomeTeam"], "xg")
    player = paired(api, mp, ["game_id", "shooterPlayerId"], "xg")
    res["xg_opening_calibrator"] = {
        "per_shot_matched": shot, "team_game": team, "player_game_ixg": player,
        "season_total": {"api": r6(api.xg.sum()), "mp": r6(mp.xg.sum()),
                         "goals_api": int(api.goal.sum()), "goals_mp": int(mp.goal.sum())}}
    log(f"\nxG, frozen GBM + opening calibrator (same covariates):")
    log(f"  per shot (matched)    corr {shot['corr']:.4f}  MAE {shot['mae']:.5f}  "
        f"bias {shot['bias']:+.5f}  |diff|>0.01 {shot['share_absdiff_gt_0.01']:.2%}")
    log(f"  team-game             corr {team['corr']:.4f}  MAE {team['mae']:.4f}  "
        f"bias {team['bias']:+.4f} ({team['rel_bias']:+.2%})  [mean {team['mean_mp']:.3f}]")
    log(f"  player-game ixG       corr {player['corr']:.4f}  MAE {player['mae']:.4f}  "
        f"bias {player['bias']:+.5f} ({player['rel_bias']:+.2%})")

    # ---- which differing inputs carry the per-shot xG differences
    attr = {}
    for f in NUM_FIELDS + CAT_FIELDS:
        if f in ("goal", "goalieIdForShot", "shotOnEmptyNet", "shotWasOnGoal", "teamCode",
                 "xCord", "yCord"):
            continue
        if f in CAT_FIELDS:
            neq = A[f].fillna("UNK").astype(str) != M[f].fillna("UNK").astype(str)
        else:
            a, m = A[f].astype(float), M[f].astype(float)
            neq = ~(np.isclose(a, m, atol=1e-6) | (a.isna() & m.isna()))
        attr[f] = {"rows": int(neq.sum()), "sum_abs_dxg": r6(np.abs(dx[neq]).sum()),
                   "net_dxg": r6(dx[neq].sum())}
    res["xg_diff_attribution"] = dict(sorted(attr.items(), key=lambda kv: -kv[1]["sum_abs_dxg"]))
    res["xg_diff_totals"] = {"matched_sum_abs_dxg": r6(np.abs(dx).sum()),
                             "matched_net_dxg": r6(dx.sum()),
                             "api_only_xg": r6(api.xg.iloc[ao.index].sum()) if len(ao) else 0.0,
                             "mp_only_xg": r6(mp.xg.iloc[mo.index].sum()) if len(mo) else 0.0}
    log("  largest contributors to per-shot differences (rows, sum |dxG|, net):")
    for f, v in list(res["xg_diff_attribution"].items())[:6]:
        log(f"    {f:<28}{v['rows']:>6}{v['sum_abs_dxg']:>9.2f}{v['net_dxg']:>+9.2f}")

    # ---- shift-chart arbiter for empty nets and skater counts
    ev = pd.read_parquet(TENSORS / f"events_{S}.parquet",
                         columns=["game_id", "event_type", "t", "p1"]
                         + [f"h_on{i}" for i in range(7)] + [f"a_on{i}" for i in range(7)])
    ev = ev[ev.event_type.isin([7, 9, 14])].drop_duplicates(["game_id", "t", "p1"])
    sh = pd.DataFrame({"game_id": ev.game_id.astype("int64"), "time": ev.t.astype("int64"),
                       "shooterPlayerId": ev.p1.astype("int64"),
                       "h_en": (ev.h_on0 == 0).astype(int), "a_en": (ev.a_on0 == 0).astype(int),
                       "h_sk": (ev[[f"h_on{i}" for i in range(1, 7)]] > 0).sum(1),
                       "a_sk": (ev[[f"a_on{i}" for i in range(1, 7)]] > 0).sum(1)})
    keyc = ["game_id", "time", "shooterPlayerId"]
    AA = A[keyc].astype("int64").merge(sh, on=keyc, how="left")
    arb = {}
    for col, sc in (("homeEmptyNet", "h_en"), ("awayEmptyNet", "a_en"),
                    ("homeSkatersOnIce", "h_sk"), ("awaySkatersOnIce", "a_sk")):
        dis = (A[col].astype(float) != M[col].astype(float)).to_numpy()
        s_ = AA[sc].to_numpy(float)[dis]
        arb[col] = {"disagreements": int(dis.sum()),
                    "shifts_side_with_api": int((s_ == A[col].to_numpy(float)[dis]).sum()),
                    "shifts_side_with_mp": int((s_ == M[col].to_numpy(float)[dis]).sum())}
    res["arbiter_shift_charts"] = arb
    log("\nshift charts on the disagreements (with API / with MoneyPuck / total):")
    for col, v in arb.items():
        log(f"  {col:<18}{v['shifts_side_with_api']:>5} /{v['shifts_side_with_mp']:>4} /"
            f"{v['disagreements']:>5}")

    # ---- the live path for each source
    maps = json.loads((TENSORS / "maps.json").read_text())
    log("\nlive path per source (covariates from the source, seq_calibrate, stream, "
        "pgx, tgx, goalie xG faced) ...")
    t1 = time.time()
    out = {}
    for src, rows in (("api", api[NAS.COLUMNS]), ("mp", None)):
        out[src] = live_path(ck, sandbox(f"apishots_{src}", rows), maps)
    point_modules(TENSORS)
    lp = {}
    ta, tm = out["api"]["tgx"].set_index(["game_id", "is_home"]), \
        out["mp"]["tgx"].set_index(["game_id", "is_home"])
    for c in ("xgf_all", "xgf_ev", "xgf_pp", "xgf_sh"):
        lp[f"tgx_{c}"] = agree(ta[c], tm[c].reindex(ta.index))
    pa, pm = out["api"]["pgx"].set_index(["game_id", "player_id"]), \
        out["mp"]["pgx"].set_index(["game_id", "player_id"])
    for c in ("ixg_all", "ixg_ev", "ixg_pp", "oi_xgf_ev", "oi_xga_ev"):
        lp[f"pgx_{c}"] = agree(pa[c], pm[c].reindex(pa.index))
    ga = out["api"]["gk"].set_index(["game_id", "goalie"]).xgf
    gm = out["mp"]["gk"].set_index(["game_id", "goalie"]).xgf
    j2 = pd.concat([ga.rename("a"), gm.rename("m")], axis=1).fillna(0.0)
    lp["goalie_xg_faced"] = agree(j2.a, j2.m)
    lp["stream_shots_with_xg"] = {"api": r6(out["api"]["stream_xg_cov"]),
                                  "mp": r6(out["mp"]["stream_xg_cov"])}
    res["xg_live_path"] = lp
    log(f"  ({time.time() - t1:.0f}s) shots in the stream carrying xG: API "
        f"{lp['stream_shots_with_xg']['api']:.2%}, MoneyPuck {lp['stream_shots_with_xg']['mp']:.2%}")
    for k in ("tgx_xgf_all", "tgx_xgf_ev", "tgx_xgf_pp", "pgx_ixg_all", "pgx_ixg_ev",
              "pgx_oi_xgf_ev", "goalie_xg_faced"):
        v = lp[k]
        log(f"  {k:<18} corr {v['corr']:.4f}  MAE {v['mae']:.4f}  bias {v['bias']:+.5f} "
            f"({v['rel_bias']:+.2%})  [mean {v['mean_mp']:.3f}]")

    # ---- gate
    checks = {"opening_team_game": team, "live_tgx_xgf_all": lp["tgx_xgf_all"]}
    ok = all(v["corr"] >= GATE_CORR and abs(v["rel_bias"]) <= GATE_BIAS for v in checks.values())
    res["gate"] = {"corr_min": GATE_CORR, "abs_rel_bias_max": GATE_BIAS,
                   "checked": {k: {"corr": v["corr"], "rel_bias": v["rel_bias"]}
                               for k, v in checks.items()}, "pass": bool(ok)}
    res["runtime_s"] = round(time.time() - t0)
    OUT.write_text(json.dumps(res, indent=1, default=str))
    log(f"\ngate (team-game xG corr >= {GATE_CORR}, |bias| <= {GATE_BIAS:.0%}): "
        f"{'PASS' if ok else 'FAIL'}  -> {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
