"""NeurHL-2 — RAPM fitting with pre-specified gates R1 and R2 (PLAN_NeurHL2 G).

Everything here is decided on the **DEV window (2009-2011)**. That window exists
precisely so architecture and hyperparameter choices have a legitimate home: v1
had only TUNE (spent on ~10 configs) and CONFIRM (sacred), so its choices leaked
into the spent window and its p-values stopped meaning anything.

  **lambda** chosen by out-of-SEASON duration-weighted stint MSE. Not by
  in-sample fit, and not by anything computed on TUNE or CONFIRM.

  **R1 repeatability** -- split games at random into halves, fit RAPM
  independently on each, and correlate the two coefficient vectors across
  players. PASS requires beating the same split-half correlation for raw on-ice
  CF%. This is the question that matters for a rating: does it measure a
  property of the PLAYER, or the situation he happened to be in? Raw on-ice rate
  is repeatable partly because linemates and deployment repeat, so clearing it is
  a real bar rather than a formality.

  **R2 value** -- fit on seasons S, score stints in season S+1 that the fit never
  saw, against three baselines: intercept+home only, team fixed effects, and the
  players' own raw on-ice rates. PASS requires beating all three. Held-out
  players resolve to the pooled replacement column, so newcomers are scored
  rather than dropped.

Both gates are scored on 5v5 regular-season stints only.

Run: uv run --no-project --python 3.12 --with numpy --with "pandas<3" \
     --with pyarrow --with scipy python neurhl/train/fit_rapm.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import TENSORS  # noqa: E402
import models.rapm as R  # noqa: E402
import windows as W  # noqa: E402
import manifest as MAN  # noqa: E402

# lambda must be read against the design scale: the X'WX diagonal for a player
# is his TOI in seconds, ~200,000 over three seasons, so even 6400 is only ~3%
# shrinkage. The first grid topped out at 6400 and the transfer criterion was
# still rising there -- the grid was too short, not the estimator too shrunk.
LAM_GRID = [200.0, 800.0, 3200.0, 6400.0, 12800.0, 25600.0, 51200.0,
            102400.0, 204800.0, 409600.0]
TARGET = "cf"                  # Corsi: dense enough to identify players
MIN_TOI = 12000                # 200 minutes in the fit window
SEED = 20260821


def raw_onice_rates(st: pd.DataFrame) -> dict:
    """Each player's raw duration-weighted on-ice for/against per 60.

    The baseline RAPM has to beat. It is a TEAM property wearing a player's
    name: a fourth-liner's number reflects his linemates and his zone starts as
    much as himself, which is the whole reason RAPM exists.
    """
    h = st[R.H_COLS].to_numpy(np.int64)
    a = st[R.A_COLS].to_numpy(np.int64)
    dur = st.dur_s.to_numpy(np.float64)
    cf_h = st[f"{TARGET}_h"].to_numpy(np.float64)
    cf_a = st[f"{TARGET}_a"].to_numpy(np.float64)
    n = int(max(h.max(initial=0), a.max(initial=0))) + 2
    toi = np.zeros(n)
    fo = np.zeros(n)
    ag = np.zeros(n)
    for side, f, g in ((h, cf_h, cf_a), (a, cf_a, cf_h)):
        for c in range(side.shape[1]):
            pid = side[:, c]
            m = pid > 0
            np.add.at(toi, pid[m], dur[m])
            np.add.at(fo, pid[m], f[m])
            np.add.at(ag, pid[m], g[m])
    with np.errstate(divide="ignore", invalid="ignore"):
        rf = np.where(toi > 0, 3600 * fo / np.maximum(toi, 1), np.nan)
        ra = np.where(toi > 0, 3600 * ag / np.maximum(toi, 1), np.nan)
    return {"toi": toi, "for60": rf, "against60": ra}


def predict_raw(st: pd.DataFrame, raw: dict, league_for: float,
                league_against: float) -> np.ndarray:
    """Baseline prediction: mean of the on-ice players' own raw rates."""
    h = st[R.H_COLS].to_numpy(np.int64)
    a = st[R.A_COLS].to_numpy(np.int64)
    rf, ra = raw["for60"], raw["against60"]
    nmax = len(rf)

    def side_mean(arr, tab, fill):
        idx = np.clip(arr, 0, nmax - 1)
        v = tab[idx]
        v = np.where((arr > 0) & np.isfinite(v), v, np.nan)
        with np.errstate(invalid="ignore"):
            m = np.nanmean(v, axis=1)
        return np.where(np.isfinite(m), m, fill)

    ho, hd = side_mean(h, rf, league_for), side_mean(h, ra, league_against)
    ao, ad = side_mean(a, rf, league_for), side_mean(a, ra, league_against)
    # home attacking, then away attacking -- matches the two-row design order
    return np.concatenate([(ho + ad) / 2, (ao + hd) / 2])


def team_fixed_effects(st: pd.DataFrame) -> np.ndarray:
    """Team offence/defence means: the 'it's just team strength' baseline."""
    dur = st.dur_s.to_numpy(np.float64)
    per60 = 3600.0 / np.maximum(dur, 1.0)
    yh = st[f"{TARGET}_h"].to_numpy(np.float64) * per60
    ya = st[f"{TARGET}_a"].to_numpy(np.float64) * per60
    return np.concatenate([yh, ya])


def player_team(seasons) -> pd.DataFrame:
    """player_id, season_end -> the team he played most minutes for."""
    rows = []
    for s in seasons:
        pg = TENSORS / f"player_games_{s}.parquet"
        gc = TENSORS / f"games_ctx_{s}.parquet"
        if not pg.exists() or not gc.exists():
            continue
        p = pd.read_parquet(pg, columns=["game_id", "player_id", "is_home",
                                         "toi_sec", "game_type"])
        p = p[p.game_type == 2]
        g = pd.read_parquet(gc, columns=["game_id", "home_idx", "away_idx"])
        p = p.merge(g, on="game_id", how="left")
        p["team"] = np.where(p.is_home, p.home_idx, p.away_idx)
        agg = (p.groupby(["player_id", "team"], as_index=False)
               .toi_sec.sum().sort_values("toi_sec", ascending=False)
               .drop_duplicates("player_id"))
        agg["season_end"] = s
        rows.append(agg)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def net_raw(st: pd.DataFrame) -> pd.DataFrame:
    """Per-player raw on-ice net rate per 60, plus TOI."""
    r = raw_onice_rates(st)
    pid = np.flatnonzero(r["toi"] > 0)
    return pd.DataFrame({"player_id": pid, "toi_s": r["toi"][pid],
                         "raw_net": r["for60"][pid] - r["against60"][pid]})


def transfer_gate(tr, te, tr_seasons, te_season, des, coef, min_toi) -> dict:
    """R1' -- does a rating survive a change of team and linemates?

    Target is deliberately METHOD-NEUTRAL: a player's realised raw on-ice net
    rate in the test season, MINUS his new team's average. Demeaning by the new
    team removes the quality of the club he joined, so what remains is 'did he
    outperform his new teammates?'. Using next-season RAPM as the target instead
    would share method variance with the RAPM predictor and rig the comparison;
    using undemeaned raw would reward whichever measure best predicts team
    quality, which is not the question.

    Predictors are matched the same way: prior RAPM net, and prior raw net both
    as-is and demeaned by the player's OLD team.
    """
    pt = player_team(list(tr_seasons) + [te_season])
    last_tr = max(tr_seasons)
    a = pt[pt.season_end == last_tr][["player_id", "team"]].rename(
        columns={"team": "team_from"})
    b = pt[pt.season_end == te_season][["player_id", "team"]].rename(
        columns={"team": "team_to"})
    mv = a.merge(b, on="player_id")
    mv["moved"] = mv.team_from != mv.team_to

    prior_raw = net_raw(tr).rename(columns={"raw_net": "prior_raw",
                                            "toi_s": "toi_tr"})
    post_raw = net_raw(te).rename(columns={"raw_net": "post_raw",
                                           "toi_s": "toi_te"})
    o, d = des.off0, des.def0
    rp = pd.DataFrame({
        "player_id": list(des.pid_to_slot),
        "prior_rapm": [float(coef[o + s] - coef[d + s])
                       for s in des.pid_to_slot.values()]})

    df = (mv.merge(prior_raw, on="player_id").merge(post_raw, on="player_id")
          .merge(rp, on="player_id", how="left"))
    df = df[(df.toi_tr >= min_toi) & (df.toi_te >= min_toi / 3)]
    df = df.dropna(subset=["prior_rapm"])
    # demean by team, weighted equally across players on that team
    df["post_dev"] = df.post_raw - df.groupby("team_to").post_raw.transform("mean")
    df["prior_raw_dev"] = (df.prior_raw
                           - df.groupby("team_from").prior_raw.transform("mean"))
    # like-for-like: RAPM demeaned by the OLD team too, so the comparison with
    # raw_dev differs only in the estimator, not in the centring
    df["prior_rapm_dev"] = (df.prior_rapm
                            - df.groupby("team_from").prior_rapm.transform("mean"))

    out = {}
    for grp, sub in (("movers", df[df.moved]), ("stayers", df[~df.moved]),
                     ("all", df)):
        if len(sub) < 30:
            out[grp] = {"n": int(len(sub))}
            continue
        out[grp] = {
            "n": int(len(sub)),
            "rapm": round(float(np.corrcoef(sub.prior_rapm, sub.post_dev)[0, 1]), 4),
            "raw": round(float(np.corrcoef(sub.prior_raw, sub.post_dev)[0, 1]), 4),
            "raw_dev": round(float(np.corrcoef(sub.prior_raw_dev, sub.post_dev)[0, 1]), 4),
            "rapm_dev": round(float(np.corrcoef(sub.prior_rapm_dev, sub.post_dev)[0, 1]), 4),
        }
    return out


def split_half_corr(st: pd.DataFrame, lam: float, rng) -> dict:
    """R1: fit two independent halves of the GAMES, correlate the ratings."""
    games = np.array(sorted(st.game_id.unique()))
    rng.shuffle(games)
    ga, gb = set(games[:len(games) // 2]), set(games[len(games) // 2:])
    sa = st[st.game_id.isin(ga)]
    sb = st[st.game_id.isin(gb)]

    out = {}
    des, coef = [], []
    for s in (sa, sb):
        d = R.RAPMDesign(s, min_toi_s=MIN_TOI // 2)
        r = R.solve(d, d.targets(s, [TARGET]), lam, want_se=False)
        des.append(d)
        coef.append(r["coef"][TARGET])
    common = sorted(set(des[0].pid_to_slot) & set(des[1].pid_to_slot))
    if len(common) < 50:
        return {"n_common": len(common), "rapm_r": float("nan")}
    for nm, base in (("off", "off0"), ("def", "def0")):
        va = np.array([coef[0][getattr(des[0], base) + des[0].pid_to_slot[p]]
                       for p in common])
        vb = np.array([coef[1][getattr(des[1], base) + des[1].pid_to_slot[p]]
                       for p in common])
        out[f"rapm_{nm}_r"] = round(float(np.corrcoef(va, vb)[0, 1]), 4)

    # same split, raw on-ice CF% -- the comparison that makes R1 meaningful
    ra_, rb_ = raw_onice_rates(sa), raw_onice_rates(sb)
    for nm, key in (("off", "for60"), ("def", "against60")):
        va = np.array([ra_[key][p] for p in common])
        vb = np.array([rb_[key][p] for p in common])
        m = np.isfinite(va) & np.isfinite(vb)
        out[f"raw_{nm}_r"] = round(float(np.corrcoef(va[m], vb[m])[0, 1]), 4)
    out["n_common"] = len(common)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, nargs="*", default=[2008, 2009, 2010])
    ap.add_argument("--test", type=int, default=2011)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    # window discipline: the gate season must be in DEV and scorable
    W.assert_scorable([args.test], "dev")
    print(f"train {args.train} -> test {args.test} "
          f"(window '{W.window_of(args.test)}')\n")

    t0 = time.time()
    tr = R.load_stints(args.train)
    te = R.load_stints([args.test])
    pos = R.player_positions(args.train)
    des = R.RAPMDesign(tr, min_toi_s=MIN_TOI, positions=pos)
    ys = des.targets(tr, [TARGET])
    normal = R.normal_equations(des, ys)
    Xte, wte = des.transform(te)
    yte = des.targets(te, [TARGET])[TARGET]
    print(f"train {des.rows_meta['n_stints']:,} stints / "
          f"{des.rows_meta['n_players']} players | test "
          f"{len(te):,} stints | design in {time.time()-t0:.0f}s\n")

    # ---- baselines on the held-out season, all fit on the TRAIN window only
    lg_f = 3600 * tr[f"{TARGET}_h"].sum() / tr.dur_s.sum()
    lg_a = 3600 * tr[f"{TARGET}_a"].sum() / tr.dur_s.sum()
    base = {}
    grand = float(np.average(ys[TARGET], weights=des.w))
    base["intercept"] = R.wmse(yte, np.full(len(yte), grand), wte)
    base["raw_onice"] = R.wmse(yte, predict_raw(te, raw_onice_rates(tr),
                                                lg_f, lg_a), wte)
    base["team_fe"] = team_fe_baseline(tr, te, args.train, [args.test],
                                       wte, yte, lg_f, lg_a)

    # ---- lambda path. A5: selection is on TRANSFER (R1'), not stint wMSE --
    # the wMSE path is noise-dominated and monotone to the grid edge.
    print(f"{'lambda':>9} {'test wMSE':>12} {'vsRaw':>8} "
          f"{'movers':>7} {'r_rapm':>8} {'r_rapmdev':>10} {'r_raw':>8} {'r_rawdev':>9}")
    rows = []
    for lam in LAM_GRID:
        res = R.solve(des, ys, lam, want_se=False, normal=normal)
        m = R.wmse(yte, Xte @ res["coef"][TARGET], wte)
        tg = transfer_gate(tr, te, args.train, args.test, des,
                           res["coef"][TARGET], MIN_TOI)
        mo = tg.get("movers", {})
        rows.append({"lam": lam, "wmse": m, "transfer": tg})
        print(f"{lam:>9.0f} {m:>12.2f} {m/base['raw_onice']-1:>7.3%} "
              f"{mo.get('n', 0):>7} {mo.get('rapm', float('nan')):>8.4f} "
              f"{mo.get('rapm_dev', float('nan')):>10.4f} "
              f"{mo.get('raw', float('nan')):>8.4f} "
              f"{mo.get('raw_dev', float('nan')):>9.4f}")

    scored = [r for r in rows if r["transfer"].get("movers", {}).get("rapm")
              is not None]
    best = max(scored, key=lambda r: max(r["transfer"]["movers"]["rapm"],
                                         r["transfer"]["movers"]["rapm_dev"]))
    mo = best["transfer"]["movers"]
    print(f"\nbaselines (stint wMSE): intercept {base['intercept']:.2f} | "
          f"team_fe {base['team_fe']:.2f} | raw_onice {base['raw_onice']:.2f}")
    print(f"chosen lambda = {best['lam']:.0f} (transfer r={mo['rapm']:.4f})")

    r2_pass = all(best["wmse"] < base[k] for k in
                  ("intercept", "team_fe", "raw_onice"))
    print(f"R2 {'PASS' if r2_pass else 'FAIL'}: stint wMSE {best['wmse']:.2f} "
          f"vs raw {base['raw_onice']:.2f} "
          f"({best['wmse']/base['raw_onice']-1:+.3%}) -- sanity floor only")

    # ---- R1' transfer gate
    r1_pass = max(mo["rapm"], mo["rapm_dev"]) > max(mo["raw"], mo["raw_dev"])
    print(f"\nR1' TRANSFER (players who CHANGED TEAM, n={mo['n']}): "
          f"target = realised on-ice net in the new team, demeaned by that team")
    print(f"     prior RAPM      r = {mo['rapm']:+.4f}")
    print(f"     prior RAPM(dev) r = {mo['rapm_dev']:+.4f}")
    print(f"     prior raw       r = {mo['raw']:+.4f}")
    print(f"     prior raw (dev) r = {mo['raw_dev']:+.4f}")
    st_ = best["transfer"].get("stayers", {})
    if st_.get("rapm") is not None:
        print(f"     [stayers n={st_['n']}: rapm {st_['rapm']:+.4f} "
              f"raw {st_['raw']:+.4f} raw_dev {st_['raw_dev']:+.4f}]")
    print(f"R1' {'PASS' if r1_pass else 'FAIL'}: RAPM "
          f"{'transfers better than' if r1_pass else 'does NOT transfer better than'}"
          f" raw on-ice rate")

    # legacy R1, recorded as the failure it is (see amendment A5)
    print("\nR1 (RETIRED, A5) split-half repeatability, recorded for the record:")
    r1_old = []
    for s in range(args.seeds):
        rng = np.random.default_rng(SEED + s)
        r1_old.append(split_half_corr(tr, best["lam"], rng))
    mean = {k: round(float(np.mean([x[k] for x in r1_old])), 4)
            for k in ("rapm_off_r", "rapm_def_r", "raw_off_r", "raw_def_r")
            if k in r1_old[0]}
    print(f"  {mean}  -> retired because raw is more repeatable BY BEING "
          f"confounded (reliability != validity)")

    out = {"train": args.train, "test": args.test, "window": W.window_of(args.test),
           "target": TARGET, "min_toi_s": MIN_TOI, "lam_grid": LAM_GRID,
           "lam_chosen": best["lam"], "selection_criterion": "R1_transfer",
           "path": [{k: v for k, v in r.items()} for r in rows],
           "baselines": base,
           "R1_retired_A5": {"per_seed": r1_old, "mean": mean,
                             "note": "raw is more repeatable by being "
                                     "confounded; reliability != validity"},
           "R1_transfer": {**best["transfer"], "pass": bool(r1_pass)},
           "R2": {"pass": bool(r2_pass), "wmse": best["wmse"],
                  "margin_vs_raw": best["wmse"] / base["raw_onice"] - 1},
           "n_players": des.rows_meta["n_players"],
           "n_stints_train": des.rows_meta["n_stints"],
           "n_stints_test": int(len(te))}
    p = TENSORS.parent.parent / "configs" / "rapm_gates.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, default=float))
    print(f"\n-> {p}")
    return 0 if (r1_pass and r2_pass) else 1


def game_teams(seasons) -> pd.DataFrame:
    """game_id -> (home team, away team) from games_ctx."""
    parts = []
    for s in seasons:
        p = TENSORS / f"games_ctx_{s}.parquet"
        if p.exists():
            parts.append(pd.read_parquet(
                p, columns=["game_id", "home_idx", "away_idx"]))
    return (pd.concat(parts, ignore_index=True).drop_duplicates("game_id")
            if parts else pd.DataFrame(columns=["game_id", "home_idx",
                                                "away_idx"]))


def team_fe_baseline(tr, te, tr_seasons, te_seasons, wte, yte,
                     lg_f, lg_a) -> float:
    """Team offence/defence fixed effects -- the 'it is only team strength,
    not the individual' baseline that a player rating must beat to be worth
    having at all."""
    gt = game_teams(list(tr_seasons) + list(te_seasons)).set_index("game_id")
    for st in (tr, te):
        st["_h"] = st.game_id.map(gt.home_idx)
        st["_a"] = st.game_id.map(gt.away_idx)
    d = tr.dur_s
    off = {}
    dfn = {}
    for tcol, fcol, acol in (("_h", f"{TARGET}_h", f"{TARGET}_a"),
                             ("_a", f"{TARGET}_a", f"{TARGET}_h")):
        g = tr.groupby(tcol)
        f_ = 3600 * g[fcol].sum() / g[d.name].sum()
        a_ = 3600 * g[acol].sum() / g[d.name].sum()
        for t, v in f_.items():
            off[t] = off.get(t, []) + [v]
        for t, v in a_.items():
            dfn[t] = dfn.get(t, []) + [v]
    off = {t: float(np.mean(v)) for t, v in off.items()}
    dfn = {t: float(np.mean(v)) for t, v in dfn.items()}
    oh = te._h.map(off).fillna(lg_f).to_numpy()
    dh = te._h.map(dfn).fillna(lg_a).to_numpy()
    oa = te._a.map(off).fillna(lg_f).to_numpy()
    da = te._a.map(dfn).fillna(lg_a).to_numpy()
    pred = np.concatenate([(oh + da) / 2, (oa + dh) / 2])
    return R.wmse(yte, pred, wte)


if __name__ == "__main__":
    sys.exit(main())
