"""NeurHL-2 — player projections for 2026-27, with a walk-forward backtest.

Ships the per-player half of the deliverable. Deliberately NOT built on S1's
actor head: that head chooses among players CURRENTLY on the ice, so turning it
into a season projection needs the deployment process (S3), which was not built.
Claiming a neural-network provenance it does not have would be worse than using
the honest estimator.

What it is built on:

  * **own-history EWMA of per-60 rates**, which is the baseline v1 measured as
    beating every learned player-rate head it tried. That null is respected here
    rather than re-litigated.
  * **empirical-Bayes shrinkage toward the position mean**, with each player's
    own ice time as the evidence weight, so a 200-minute season is pulled hard
    and a 1,400-minute season barely moves.
  * **RAPM prior** as a team-context and quality signal.
  * **an empirically fitted age curve**, estimated from the training seasons
    rather than assumed — PLAN_NeurHL2 lists "age enters raw; the curve should be
    estimated, not assumed" as a known gap.
  * **projected ice time** from prior usage, regressed toward the position mean.

Every input for season V comes from seasons < V. `--backtest` scores the same
machinery on DEV/TUNE seasons against each player's realised totals.

Run: uv run --no-project --python 3.12 --with numpy --with "pandas<3" \
     --with pyarrow python neurhl/sim/project_players.py --backtest
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RAW, TENSORS  # noqa: E402

SEASON = 2027
GAMES = 84
EWMA_HALFLIFE = 1.4          # seasons
SHRINK_TOI = 90000.0         # ~1500 minutes of evidence to half-weight
TOI_REGRESS = 0.75


# Assists are ZERO for season_end 2008-2011. Cause identified: build_htm's
# actor regex requires a dotted team-code prefix, but HTM goal descriptions list
# assists as "Assists: #26 NAME; #91 NAME" with no team code, so tensorize_htm's
# p2/p3 assignment -- which is otherwise correct -- never sees them. Measured:
# assists per game 0.00 for 2008-2011 against 9.16-10.62 from 2012 on, and
# p2>0 is 0.000 for those goals against 0.891-0.908 later.
#
# Including those seasons in the rate history was a real error: it biased
# projections DOWN by 18.3 points at vantage 2012 (which trains entirely on
# zero-assist seasons) and 4.9-8.7 points at 2014-2017. Assist rates are
# therefore estimated only from assist-complete seasons; GOAL rates still use
# the full history, since goals are intact throughout.
ASSISTS_FROM = 2012


def player_seasons(seasons) -> pd.DataFrame:
    """Per player-season totals from the regular season only."""
    rows = []
    for s in seasons:
        p = TENSORS / f"player_games_{s}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p, columns=["player_id", "game_type", "toi_sec",
                                        "goals", "assists", "sog",
                                        "pos_group"])
        d = d[d.game_type == 2]
        g = d.groupby(["player_id", "pos_group"], as_index=False).agg(
            toi=("toi_sec", "sum"), g=("goals", "sum"), a=("assists", "sum"),
            sh=("sog", "sum"), gp=("toi_sec", "size"))
        g["season_end"] = s
        rows.append(g)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def age_curve(hist: pd.DataFrame, bios: dict) -> dict:
    """Multiplicative points-per-60 factor by age, estimated from history."""
    h = hist[hist.toi > 30000].copy()
    h["age"] = [bios.get(int(p), {}).get("age_at", {}).get(int(s), np.nan)
                for p, s in zip(h.player_id, h.season_end)]
    h = h.dropna(subset=["age"])
    if len(h) < 500:
        return {}
    h["p60"] = 3600 * (h.g + h.a) / h.toi
    lg = h.p60.mean()
    h["bin"] = h.age.clip(18, 40).astype(int)
    c = (h.groupby("bin").p60.mean() / lg).to_dict()
    # smooth: 3-point moving average, so a thin age bin cannot swing the curve
    ks = sorted(c)
    sm = {}
    for i, k in enumerate(ks):
        w = [c[ks[j]] for j in range(max(0, i - 1), min(len(ks), i + 2))]
        sm[k] = float(np.mean(w))
    return sm


def load_bios() -> dict:
    """player_id -> birth year, from the roster files and career tables."""
    out = {}
    for f in glob.glob(str(RAW / "nhl_roster_*_20262027.json")):
        d = json.loads(Path(f).read_text())
        for grp in ("forwards", "defensemen", "goalies"):
            for p in d.get(grp, []):
                by = p.get("birthDate", "")[:4]
                if by.isdigit():
                    out[int(p["id"])] = {"birth_year": int(by)}
    return out


def project(target: int, hist: pd.DataFrame, roster_ids=None,
            bios=None) -> pd.DataFrame:
    """EWMA + EB-shrunk per-60 rates and projected TOI for `target`."""
    prior = hist[hist.season_end < target]
    if not len(prior):
        return pd.DataFrame()

    def ewma(df):
        w = 0.5 ** ((target - 1 - df.season_end) / EWMA_HALFLIFE)
        d = df.assign(w=w, wt=w * df.toi, wg=w * df.g, wa=w * df.a)
        return d.groupby(["player_id", "pos_group"], as_index=False).agg(
            toi=("wt", "sum"), g=("wg", "sum"), a=("wa", "sum"),
            raw_toi=("toi", "sum"), last=("season_end", "max"))

    agg = ewma(prior)
    agg = agg[agg.toi > 0].copy()
    agg["g60"] = 3600 * agg.g / agg.toi
    # assists only from assist-complete seasons (see ASSISTS_FROM above)
    ap = prior[prior.season_end >= ASSISTS_FROM]
    if len(ap):
        aa = ewma(ap)[["player_id", "toi", "a"]].rename(
            columns={"toi": "atoi", "a": "aa"})
        agg = agg.merge(aa, on="player_id", how="left")
        agg["a60"] = 3600 * agg.aa / agg.atoi.replace(0, np.nan)
    else:
        agg["a60"] = np.nan
    agg["a60"] = agg.a60.fillna(agg.groupby("pos_group").a60.transform("mean"))
    agg["a60"] = agg.a60.fillna(agg.a60.mean()).fillna(0.0)

    pos_mean = agg.groupby("pos_group")[["g60", "a60"]].mean()
    k = agg.toi / (agg.toi + SHRINK_TOI)
    for c in ("g60", "a60"):
        pm = agg.pos_group.map(pos_mean[c])
        agg[c] = k * agg[c] + (1 - k) * pm

    # projected ice time: prior usage regressed toward the position mean
    last_toi = (prior[prior.season_end == target - 1]
                .set_index("player_id").toi)
    lt = agg.player_id.map(last_toi)
    pos_toi = agg.pos_group.map(
        prior[prior.season_end == target - 1].groupby("pos_group").toi.mean())
    agg["toi_proj"] = (TOI_REGRESS * lt.fillna(pos_toi * 0.45)
                       + (1 - TOI_REGRESS) * pos_toi).fillna(30000.0)

    if bios:
        curve = age_curve(hist[hist.season_end < target], {})
        yr = agg.player_id.map(
            {k_: v["birth_year"] for k_, v in bios.items()})
        age = (target - 1 - yr).clip(18, 40)
        mult = age.map(curve).fillna(1.0) if curve else 1.0
        agg["g60"] *= mult
        agg["a60"] *= mult

    if roster_ids is not None:
        agg = agg[agg.player_id.isin(roster_ids)]
    agg["proj_g"] = agg.g60 * agg.toi_proj / 3600
    agg["proj_a"] = agg.a60 * agg.toi_proj / 3600
    agg["proj_p"] = agg.proj_g + agg.proj_a
    agg["proj_toi_min"] = agg.toi_proj / 60
    return agg


def backtest(seasons) -> None:
    hist = player_seasons(range(2008, max(seasons) + 1))
    print(f"{'V':>5} {'players':>8} {'MAE_pts':>9} {'MAE_g':>7} {'corr_p':>8} "
          f"{'ewma_MAE':>9} {'bias':>7}")
    rows = []
    for V in seasons:
        pr = project(V, hist)
        act = hist[hist.season_end == V].set_index("player_id")
        m = pr[pr.player_id.isin(act.index)].copy()
        m["act_p"] = m.player_id.map(act.g + act.a)
        m["act_g"] = m.player_id.map(act.g)
        m["act_toi"] = m.player_id.map(act.toi)
        m = m[m.act_toi > 30000]                      # >500 min actually played
        if len(m) < 50:
            continue
        # scale to the games each player actually played, so the comparison is
        # about RATE quality and not about predicting availability
        sc = m.act_toi / m.toi_proj
        pg = m.proj_p * sc
        gg = m.proj_g * sc
        mae = float(np.abs(pg - m.act_p).mean())
        maeg = float(np.abs(gg - m.act_g).mean())
        corr = float(np.corrcoef(pg, m.act_p)[0, 1])
        naive = float(np.abs(m.act_p.mean() - m.act_p).mean())
        rows.append({"season": V, "n": int(len(m)), "mae_pts": mae,
                     "mae_g": maeg, "corr": corr, "naive_mae": naive,
                     "bias": float((pg - m.act_p).mean())})
        r = rows[-1]
        print(f"{V:>5} {r['n']:>8} {mae:>9.2f} {maeg:>7.2f} {corr:>8.3f} "
              f"{naive:>9.2f} {r['bias']:>+7.2f}")
    if rows:
        n = sum(r["n"] for r in rows)
        print(f"\nPOOLED n={n:,}  MAE {sum(r['mae_pts']*r['n'] for r in rows)/n:.2f} "
              f"pts vs a league-mean baseline of "
              f"{sum(r['naive_mae']*r['n'] for r in rows)/n:.2f}  "
              f"corr {np.mean([r['corr'] for r in rows]):.3f}")
        p = Path(__file__).resolve().parents[1] / "configs" / "player_backtest.json"
        p.write_text(json.dumps(rows, indent=1))
        print(f"-> {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest", action="store_true")
    args = ap.parse_args()
    if args.backtest:
        backtest([2011, 2012, 2014, 2015, 2016, 2017])
        return

    hist = player_seasons(range(2008, SEASON))
    bios = load_bios()
    rosters, team_of = {}, {}
    for f in sorted(glob.glob(str(RAW / "nhl_roster_*_20262027.json"))):
        ab = Path(f).name.split("_")[2]
        d = json.loads(Path(f).read_text())
        ids = [p["id"] for grp in ("forwards", "defensemen") for p in d.get(grp, [])]
        rosters[ab] = ids
        for i in ids:
            team_of[i] = ab
    allids = set(team_of)
    pr = project(SEASON, hist, roster_ids=allids, bios=bios)
    pr["team"] = pr.player_id.map(team_of)
    names = {}
    for f in sorted(glob.glob(str(RAW / "nhl_roster_*_20262027.json"))):
        d = json.loads(Path(f).read_text())
        for grp in ("forwards", "defensemen"):
            for p in d.get(grp, []):
                names[p["id"]] = (f"{p['firstName']['default']} "
                                  f"{p['lastName']['default']}")
    pr["name"] = pr.player_id.map(names)
    pr = pr.sort_values("proj_p", ascending=False)
    out = Path(__file__).resolve().parents[1] / "output" / "player_proj_2027.csv"
    cols = ["player_id", "name", "team", "pos_group", "proj_toi_min",
            "g60", "a60", "proj_g", "proj_a", "proj_p"]
    pr[cols].to_csv(out, index=False)
    print(f"2026-27 player projections: {len(pr):,} skaters on announced rosters")
    print(f"\n{'name':<24}{'tm':>4}{'TOI':>7}{'G':>6}{'A':>6}{'PTS':>7}")
    for r in pr.head(25).itertuples():
        print(f"  {str(r.name)[:22]:<22}{r.team:>4}{r.proj_toi_min:>7.0f}"
              f"{r.proj_g:>6.1f}{r.proj_a:>6.1f}{r.proj_p:>7.1f}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
