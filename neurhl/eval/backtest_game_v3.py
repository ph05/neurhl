"""NeurHL-3 — N5: the game-outcome reattempt, an exact extension of the
incumbent v1-H thin head (eval/backtest_hier.py) with information no prior
attempt used.

Config ladder (ledger v2, cumulative): g-01 +replacement-strength delta
(absences x rapm_prior); g-02 +goalie gq (realized starters — the Tier-0
precedent; GSAx excluded per GQ1); g-03 +schedule-density block (games_7d,
team toi_7d, km3d, SIGNED tz shift from arena longitudes); g-04
+sched_strength_todate (mean opponent pre-game Elo faced). The incumbent
COLS_DEFAULT head is refit IN THE SAME RUN on the same games — exact nesting,
floor 0.67314 by construction.

Pre-gate window: every scoreable season with proj_team coverage and >=900
prior training rows ({2011, 2012, 2014-2017}). G-STOP (candidate − v1-H <=
-0.0015, season-clustered agreement) decides whether C1 is ever spent.
Writes configs/game_v3_pregate.json.

Run: uv run --no-project --python 3.12 --with numpy --with "pandas<3" \
     --with pyarrow --with scikit-learn --with scipy \
     python neurhl/eval/backtest_game_v3.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import TENSORS  # noqa: E402
from train.train_game import elo_features, load_frames  # noqa: E402
from eval.backtest_hier import COLS_DEFAULT, NO_SCORE, CS, nll  # noqa: E402
from sim.game_model import run_elo  # noqa: E402

CFG = ROOT / "configs"
BLOCKS = {
    "g-01": ["repl_delta_diff"],
    "g-02": ["gq_diff"],
    "g-03": ["games7d_diff", "toi7d_diff", "km3d_diff", "dtz_signed_diff"],
    "g-04": ["sched_str_diff"],
}


def arenas_utc() -> pd.DataFrame:
    a = pd.read_csv(ROOT.parent / "data" / "raw" / "arenas.csv")
    return a


def team_utc_map(a: pd.DataFrame, maps: dict) -> dict:
    """(team_idx, season_end) -> utc offset of the team's arena."""
    t2i = maps["team"]
    out = {}
    for r in a.itertuples():
        idx = t2i.get(r.team)
        if idx is None:
            continue
        for se in range(int(r.start_end), int(r.end_end) + 1):
            out[(int(idx), se)] = float(r.utc_std)
    return out


def build_frame(seasons) -> pd.DataFrame:
    gc, _ = load_frames(seasons)
    elo = elo_features(gc)
    have = sorted(int(p.stem.split("_")[-1])
                  for p in TENSORS.glob("proj_team_*.parquet"))
    proj = pd.concat([pd.read_parquet(TENSORS / f"proj_team_{s}.parquet")
                      for s in have])
    d = gc.set_index("game_id").join(elo).join(proj)
    d = d.dropna(subset=["elo_logit", "cfpct_h", "cfpct_a"]).reset_index()
    d["y"] = ((d.outcome4 == 0) | (d.outcome4 == 2)).astype(int)
    d["proj_diff"] = d.cfpct_h - d.cfpct_a
    cl_h = d.p_clf_h / (d.p_clf_h + d.p_cla_h).clip(lower=1e-6)
    cl_a = d.p_clf_a / (d.p_clf_a + d.p_cla_a).clip(lower=1e-6)
    d["proj_cl_diff"] = cl_h - cl_a
    d["rest_diff"] = (d.home_rest.clip(upper=7)
                      - d.away_rest.clip(upper=7)) / 7
    d["date"] = pd.to_datetime(d.date)

    # ---- block 1: replacement-strength delta (absent regulars' RAPM value)
    vals = []
    for s in sorted(d.season_end.unique()):
        ap = TENSORS / f"absences_{s}.parquet"
        rp = TENSORS / f"rapm_prior_{s}.parquet"
        if not ap.exists():
            continue
        ab = pd.read_parquet(ap)
        if rp.exists():
            r = pd.read_parquet(rp)
            r = r[~r.is_replacement.astype(bool)]
            r["net"] = r.cf_off - r.cf_def
            ab = ab.merge(r[["player_id", "net"]], on="player_id", how="left")
        else:
            ab["net"] = np.nan
        # value of an absent regular = his net rating weighted by his EWMA TOI
        # share of a game; unknown net -> the (negative-ish) replacement level
        # proxy 0 keeps him neutral rather than invented
        ab["val"] = ab.net.fillna(0.0) * (ab.ewma_toi_sec / 3600.0)
        g = ab.groupby(["game_id", "team"], as_index=False).val.sum()
        vals.append(g)
    if vals:
        av = pd.concat(vals, ignore_index=True)
        hm = av.rename(columns={"team": "home_idx", "val": "repl_h"})
        am = av.rename(columns={"team": "away_idx", "val": "repl_a"})
        d = d.merge(hm, on=["game_id", "home_idx"], how="left")
        d = d.merge(am, on=["game_id", "away_idx"], how="left")
        d["repl_delta_diff"] = (-d.repl_h.fillna(0.0)) - (-d.repl_a.fillna(0.0))
    else:
        d["repl_delta_diff"] = 0.0

    # ---- block 2: realized-starter goalie quality
    gg = []
    for s in sorted(d.season_end.unique()):
        p = TENSORS / f"goalie_games_{s}.parquet"
        if p.exists():
            gg.append(pd.read_parquet(p))
    if gg:
        g = pd.concat(gg, ignore_index=True)
        st = g[g.goalie_start == 1][["game_id", "team", "gq"]]
        d = d.merge(st.rename(columns={"team": "home_idx", "gq": "gq_h"}),
                    on=["game_id", "home_idx"], how="left")
        d = d.merge(st.rename(columns={"team": "away_idx", "gq": "gq_a"}),
                    on=["game_id", "away_idx"], how="left")
        d["gq_diff"] = d.gq_h.fillna(0.0) - d.gq_a.fillna(0.0)
    else:
        d["gq_diff"] = 0.0

    # ---- block 3: schedule density + signed tz
    maps = json.loads((TENSORS / "maps.json").read_text())
    utc = team_utc_map(arenas_utc(), maps)
    long_rows = pd.concat([
        d[["game_id", "date", "season_end", "home_idx"]]
        .rename(columns={"home_idx": "team"}).assign(is_home=1),
        d[["game_id", "date", "season_end", "away_idx"]]
        .rename(columns={"away_idx": "team"}).assign(is_home=0)],
        ignore_index=True).sort_values(["team", "date", "game_id"],
                                       kind="stable")
    counts = []
    for team, sd in long_rows.groupby("team", sort=False):
        dt = sd.date.to_numpy()
        n7 = np.array([(np.abs((dt[i] - dt[max(0, i - 8):i])
                               / np.timedelta64(1, "D")) <= 7).sum()
                       for i in range(len(dt))])
        counts.append(pd.DataFrame({"game_id": sd.game_id, "team": sd.team,
                                    "g7": n7}))
    c7 = pd.concat(counts, ignore_index=True)
    d = d.merge(c7.rename(columns={"team": "home_idx", "g7": "g7_h"}),
                on=["game_id", "home_idx"], how="left")
    d = d.merge(c7.rename(columns={"team": "away_idx", "g7": "g7_a"}),
                on=["game_id", "away_idx"], how="left")
    d["games7d_diff"] = d.g7_h.fillna(0) - d.g7_a.fillna(0)
    d["toi7d_diff"] = d.games7d_diff * 1.0   # team-level proxy: games are the
    # load unit at team level; player-level toi_7d lives in the PG layer
    d["km3d_diff"] = ((d.home_km3d.fillna(0) - d.away_km3d.fillna(0)) / 1000.0)
    # signed tz: current venue utc minus each side's PREVIOUS venue utc
    venue_utc = d.apply(lambda r: utc.get((int(r.home_idx),
                                           int(r.season_end)), np.nan), axis=1)
    d["venue_utc"] = venue_utc
    vmap = d.set_index("game_id").venue_utc
    lr = long_rows.merge(d[["game_id", "venue_utc"]], on="game_id", how="left")
    lr["prev_utc"] = lr.groupby("team", sort=False).venue_utc.shift(1)
    lr["tz_shift"] = lr.venue_utc - lr.prev_utc
    d = d.merge(lr[lr.is_home == 1][["game_id", "tz_shift"]]
                .rename(columns={"tz_shift": "tzs_h"}), on="game_id",
                how="left")
    d = d.merge(lr[lr.is_home == 0][["game_id", "tz_shift"]]
                .rename(columns={"tz_shift": "tzs_a"}), on="game_id",
                how="left")
    d["dtz_signed_diff"] = d.tzs_h.fillna(0.0) - d.tzs_a.fillna(0.0)

    # ---- block 4: schedule strength to date (mean opponent pre-game Elo)
    e = run_elo(sorted(d.season_end.unique()))
    d = d.merge(e[["game_id", "elo_h", "elo_a"]], on="game_id", how="left")
    lr2 = pd.concat([
        d[["game_id", "date", "season_end", "home_idx", "elo_a"]]
        .rename(columns={"home_idx": "team", "elo_a": "opp_elo"}),
        d[["game_id", "date", "season_end", "away_idx", "elo_h"]]
        .rename(columns={"away_idx": "team", "elo_h": "opp_elo"})],
        ignore_index=True).sort_values(["team", "date", "game_id"],
                                       kind="stable")
    grp = lr2.groupby(["season_end", "team"], sort=False)
    lr2["sched"] = (grp.opp_elo.cumsum() - lr2.opp_elo) \
        / grp.cumcount().clip(lower=1)
    d = d.merge(lr2[["game_id", "team", "sched"]]
                .rename(columns={"team": "home_idx", "sched": "sched_h"}),
                on=["game_id", "home_idx"], how="left")
    d = d.merge(lr2[["game_id", "team", "sched"]]
                .rename(columns={"team": "away_idx", "sched": "sched_a"}),
                on=["game_id", "away_idx"], how="left")
    d["sched_str_diff"] = (d.sched_h.fillna(1500) - d.sched_a.fillna(1500)) \
        / 100.0
    return d


def fit_eval(d: pd.DataFrame, cols, seasons_score) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    rows = []
    for T in seasons_score:
        if T in NO_SCORE:
            continue
        tr = d[d.season_end < T]
        te = d[d.season_end == T]
        if len(tr) < 900 or not len(te):
            continue
        sc = StandardScaler().fit(tr[cols])
        best = (9.0, None)
        for C in CS:
            m = LogisticRegression(C=C, max_iter=4000).fit(
                sc.transform(tr[cols]), tr.y)
            l_tr = nll(m.predict_proba(sc.transform(tr[cols]))[:, 1],
                       tr.y.to_numpy()).mean()
            if l_tr < best[0]:
                best = (l_tr, m)
        p = best[1].predict_proba(sc.transform(te[cols]))[:, 1]
        rows.append(pd.DataFrame({"season": T, "ll": nll(p, te.y.to_numpy())}))
    R = pd.concat(rows, ignore_index=True)
    per = R.groupby("season").ll.mean()
    return {"pooled": float(R.ll.mean()), "n": int(len(R)),
            "per_season": {int(k): round(float(v), 5)
                           for k, v in per.items()},
            "_rows": R}


def main():
    from scipy import stats
    seasons = list(range(2008, 2018))
    d = build_frame(seasons)
    score = [2010, 2011, 2012, 2014, 2015, 2016, 2017]
    base = fit_eval(d, COLS_DEFAULT, score)
    print(f"v1-H (nested incumbent, same games): {base['pooled']:.5f} "
          f"n={base['n']:,}")
    results = {"v1H": {k: v for k, v in base.items() if k != "_rows"}}
    cols = list(COLS_DEFAULT)
    best_name, best = "v1H", base
    for name, block in BLOCKS.items():
        cols = cols + block
        r = fit_eval(d, cols, score)
        dd = (r["_rows"].ll - base["_rows"].ll).to_numpy()
        se = dd.std(ddof=1) / np.sqrt(len(dd))
        z = dd.mean() / se
        cl = (r["_rows"].assign(d=dd).groupby("season").d.mean())
        cl_t = cl.mean() / (cl.std(ddof=1) / np.sqrt(len(cl)))
        cl_p = float(2 * stats.t.sf(abs(cl_t), df=len(cl) - 1))
        print(f"{name} (+{','.join(block)}): {r['pooled']:.5f}  "
              f"diff {dd.mean():+.5f}  z {z:+.2f}  cl_p {cl_p:.3f}")
        results[name] = {"pooled": r["pooled"], "n": r["n"],
                         "diff_vs_v1H": round(float(dd.mean()), 6),
                         "z": round(float(z), 3),
                         "cl_p": round(cl_p, 4),
                         "cl_direction_neg": bool(cl.mean() < 0),
                         "cols": cols.copy(),
                         "per_season": r["per_season"]}
        if r["pooled"] < best["pooled"]:
            best_name, best = name, r
    gstop = None
    if best_name != "v1H":
        dd = (best["_rows"].ll - base["_rows"].ll).to_numpy()
        cl = best["_rows"].assign(d=dd).groupby("season").d.mean()
        gstop = {"candidate": best_name,
                 "mean_diff": round(float(dd.mean()), 6),
                 "clustered_direction_neg": bool(cl.mean() < 0),
                 "clears": bool(dd.mean() <= -0.0015 and cl.mean() < 0)}
    results["G_STOP"] = gstop or {"candidate": None, "clears": False,
                                  "note": "no candidate beat v1-H pooled"}
    (CFG / "game_v3_pregate.json").write_text(
        json.dumps(results, indent=1, default=float))
    print(f"\nbest: {best_name}  G-STOP "
          f"{'CLEARS' if results['G_STOP']['clears'] else 'FIRES (C1 stays unspent)'}")
    print("-> configs/game_v3_pregate.json")


if __name__ == "__main__":
    main()
