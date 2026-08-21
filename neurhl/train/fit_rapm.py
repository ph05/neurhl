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

LAM_GRID = [25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0, 6400.0]
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

    # ---- lambda path
    print(f"{'lambda':>8} {'test wMSE':>12} {'vs raw':>9} {'vs team':>9}")
    rows = []
    for lam in LAM_GRID:
        res = R.solve(des, ys, lam, want_se=False, normal=normal)
        pred = Xte @ res["coef"][TARGET]
        m = R.wmse(yte, pred, wte)
        rows.append({"lam": lam, "wmse": m})
        print(f"{lam:>8.0f} {m:>12.2f} {m/base['raw_onice']-1:>8.3%} "
              f"{m/base['team_fe']-1:>8.3%}")
    best = min(rows, key=lambda r: r["wmse"])
    print(f"\nbaselines: intercept {base['intercept']:.2f} | "
          f"team_fe {base['team_fe']:.2f} | raw_onice {base['raw_onice']:.2f}")
    print(f"chosen lambda = {best['lam']:.0f} (test wMSE {best['wmse']:.2f})")

    r2_pass = all(best["wmse"] < base[k] for k in
                  ("intercept", "team_fe", "raw_onice"))
    print(f"\nR2 {'PASS' if r2_pass else 'FAIL'}: RAPM beats "
          f"{'all three' if r2_pass else 'NOT all'} baselines out of season")

    # ---- R1 repeatability
    print("\nR1 split-half repeatability (games split at random):")
    r1 = []
    for s in range(args.seeds):
        rng = np.random.default_rng(SEED + s)
        r1.append(split_half_corr(tr, best["lam"], rng))
        print(f"  seed {s}: rapm off {r1[-1].get('rapm_off_r')} "
              f"def {r1[-1].get('rapm_def_r')} | raw off "
              f"{r1[-1].get('raw_off_r')} def {r1[-1].get('raw_def_r')} "
              f"(n={r1[-1].get('n_common')})")
    mean = {k: round(float(np.mean([x[k] for x in r1])), 4)
            for k in ("rapm_off_r", "rapm_def_r", "raw_off_r", "raw_def_r")
            if k in r1[0]}
    r1_pass = (mean["rapm_off_r"] > mean["raw_off_r"] and
               mean["rapm_def_r"] > mean["raw_def_r"])
    print(f"  mean: {mean}")
    print(f"R1 {'PASS' if r1_pass else 'FAIL'}: RAPM "
          f"{'more' if r1_pass else 'NOT more'} repeatable than raw on-ice rate")

    out = {"train": args.train, "test": args.test, "window": W.window_of(args.test),
           "target": TARGET, "min_toi_s": MIN_TOI, "lam_grid": LAM_GRID,
           "lam_chosen": best["lam"], "path": rows, "baselines": base,
           "R1": {"per_seed": r1, "mean": mean, "pass": bool(r1_pass)},
           "R2": {"pass": bool(r2_pass), "wmse": best["wmse"]},
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
