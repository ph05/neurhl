"""Walk-forward check of goalie save-talent projections and start shares.

For every season V (2011-2026 minus the shortened 2013/2020/2021), goalies
who faced >= 1000 shots on goal in V (and, separately, backups and call-ups
with 100-999 shots) are scored on:
  * GSAx/60 against league-calibrated xG (so an average goalie is 0),
  * save percentage,
comparing ORR's projection (orr.goalies.project_goalies, history
< V only) with two baselines: NAIVE (league average: 0 GSAx, last season's
league sv%) and LAST SEASON (the goalie's own previous season if he faced
>= 300 unblocked shots, else league average).

The season-weight decay is chosen on V <= 2017 only (3 values, ledger in the
output); 2022-2026 are held out.

Start shares (seasons with FULL box scores and first-10-game rosters,
2012-2023 minus the shortened seasons; 2024's boxes stop after ~10 games):
projected starts from orr.goalies.deploy_goalies for each team's
opening goalies vs realised starts; also how many goalies are projected /
realised at 55+ starts per 82 games (NeurHL capped everyone at 52.7).

Run: python3 -m orr.backtest.goalies_bt
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from orr import config as C
from orr import data as D
from orr import goalies as GL

SEASONS = [V for V in range(2011, 2027) if V not in C.BROKEN_SEASONS]
TUNE = [V for V in SEASONS if V <= 2017]
TEST = [V for V in SEASONS if V >= 2022]
MIN_SA = 1000
LOW_SA = (100, 1000)      # backups and call-ups, reported separately


def season_rows(V: int, decay: float, sa_range=(MIN_SA, np.inf)) -> pd.DataFrame:
    g = GL.goalie_panel()
    a = g[(g.season_end == V) & (g.sa_all >= sa_range[0]) & (g.sa_all < sa_range[1])].copy()
    pj = GL.project_goalies(V, ids=a.player_id, decay=decay)
    a = a.merge(pj[["player_id", "gsax_per_fa", "gsax60", "sv_pct", "gsax_per_fa_sd"]],
                on="player_id")
    lg = GL.league_goalie(V)
    a["act_gsax60"] = a.gsax_adj / a.toi_all * 60
    a["act_sv"] = 1 - a.ga_all / a.sa_all
    last = g[(g.season_end == V - 1) & (g.fa_all >= 300)].set_index("player_id")
    lr = (last.gsax_adj / last.fa_all).reindex(a.player_id).to_numpy()
    a["last_gsax60"] = np.where(np.isfinite(lr), lr * lg["fa_per60"], 0.0)
    a["last_sv"] = np.where(np.isfinite(lr), lg["sv"] + lr * lg["fa_per_sa"], lg["sv"])
    a["naive_gsax60"] = 0.0
    a["naive_sv"] = lg["sv"]
    # 80% interval for the season GSAx/60 (talent + binomial noise)
    fa = a.fa_all.to_numpy()
    p = lg["xga_per_fa"]
    noise = np.sqrt(p * (1 - p) / fa)
    sd = np.sqrt(a.gsax_per_fa_sd ** 2 + noise ** 2) * lg["fa_per60"]
    a["cover80"] = (a.act_gsax60 - a.gsax60).abs() <= 1.2816 * sd
    a["V"] = V
    return a


def metrics(d: pd.DataFrame) -> dict:
    out = {"n": int(len(d)), "cov80_gsax60": float(d.cover80.mean())}
    for m in ("", "last_", "naive_"):
        g = d[f"{m}gsax60"] if m else d.gsax60
        s = d[f"{m}sv"] if m else d.sv_pct
        k = m.rstrip("_") or "hattrick"
        out[k] = {"gsax60_mae": float((g - d.act_gsax60).abs().mean()),
                  "gsax60_rmse": float(np.sqrt(((g - d.act_gsax60) ** 2).mean())),
                  "gsax60_corr": float(np.corrcoef(g, d.act_gsax60)[0, 1]) if g.std() > 0 else None,
                  "sv_mae": float((s - d.act_sv).abs().mean()),
                  "sv_corr": float(np.corrcoef(s, d.act_sv)[0, 1]) if s.std() > 0 else None}
    return out


def full_box_seasons() -> list[int]:
    """Seasons whose box scores cover (nearly) the whole schedule; 2024's
    fastRhockey file holds only the first ~10 games per team."""
    gg = D.goalie_games()
    n = gg.groupby("season_end").size()
    full = []
    for V, k in n.items():
        if V in C.BROKEN_SEASONS:
            continue
        sched = D.team_seasons().query("season_end == @V").gp.sum()
        if k >= 0.95 * sched:
            full.append(int(V))
    return full


def starts_check(seasons=None) -> dict:
    seasons = [V for V in full_box_seasons() if V >= 2012] if seasons is None else seasons
    gg = D.goalie_games()
    sch = D.fr_schedule()[["game_id", "home", "away"]]
    gg = gg.merge(sch, on="game_id")
    gg["team"] = np.where(gg.home_away.str.lower() == "home", gg.home, gg.away)
    rows = []
    for V in seasons:
        if V in C.BROKEN_SEASONS:
            continue
        r = D.opening_rosters(V)
        r = r[r.grp == "G"][["player_id", "team"]]
        dep = GL.deploy_goalies(V, r, 82)
        act = gg[gg.season_end == V].groupby("goalie_id").size()
        dep["act_starts"] = dep.player_id.map(act).fillna(0)
        dep["V"] = V
        rows.append(dep[["V", "player_id", "team", "starts", "act_starts"]])
    d = pd.concat(rows)
    per = d.groupby("V").agg(proj55=("starts", lambda x: int((x >= 55).sum())),
                             act55=("act_starts", lambda x: int((x >= 55).sum())),
                             proj_max=("starts", "max"), act_max=("act_starts", "max"))
    top = d.sort_values("starts", ascending=False).groupby(["V", "team"]).head(1)
    return {"seasons": seasons,
            "starts_mae": float((d.starts - d.act_starts).abs().mean()),
            "starts_bias": float((d.starts - d.act_starts).mean()),
            "no1_proj_mean": float(top.starts.mean()), "no1_act_mean": float(top.act_starts.mean()),
            "per_season_55plus": per.reset_index().to_dict(orient="records"),
            "n": int(len(d))}


def main():
    ledger = []
    for decay in (0.5, 0.7, 0.85):
        d = pd.concat([season_rows(V, decay) for V in TUNE])
        m = metrics(d)
        ledger.append({"decay": decay, "tune_gsax60_mae": m["hattrick"]["gsax60_mae"]})
        print(f"decay {decay}: tune GSAx/60 MAE {m['hattrick']['gsax60_mae']:.4f}")
    best = min(ledger, key=lambda r: r["tune_gsax60_mae"])["decay"]
    res = {"decay_ledger": ledger, "decay_chosen": best,
           "decay_in_goalies_module": GL.GOALIE_DECAY, "min_sa": MIN_SA, "seasons": {}}
    allrows, lowrows = [], []
    for V in SEASONS:
        d = season_rows(V, best)
        allrows.append(d)
        lowrows.append(season_rows(V, best, LOW_SA))
        res["seasons"][V] = metrics(d)
    A = pd.concat(allrows)
    Lw = pd.concat(lowrows)
    for name, vs in (("tune", TUNE), ("confirm", [2018, 2019]), ("test", TEST)):
        res[f"pooled_{name}"] = metrics(A[A.V.isin(vs)])
        res[f"pooled_{name}_lowSA"] = metrics(Lw[Lw.V.isin(vs)])
    res["low_sa_range"] = list(LOW_SA)
    lw = Lw[Lw.V.isin(TEST)]
    res["lowSA_test_sv_distribution"] = {
        "proj_sv_min": float(lw.sv_pct.min()), "proj_sv_p10": float(lw.sv_pct.quantile(0.1)),
        "proj_sv_median": float(lw.sv_pct.median()), "act_sv_median": float(lw.act_sv.median()),
        "mean_bias_sv": float((lw.sv_pct - lw.act_sv).mean())}
    res["starts"] = starts_check()
    res["talent_prior_2027"] = GL.talent_prior(C.TARGET_SEASON)
    D.write_json(res, C.OUT / "backtest" / "goalies_bt.json")
    print(f"\n>= {MIN_SA} shots, and '_lowSA' = {LOW_SA[0]}-{LOW_SA[1]} shots")
    print(f"{'window':<8}{'n':>5} | {'GSAx/60 MAE: HT':>16}{'last':>7}{'naive':>7} | "
          f"{'corr HT':>8}{'last':>7} | {'sv% MAE HT':>11}{'last':>8}{'naive':>8} | cov80")
    for name in ("tune", "confirm", "test", "tune_lowSA", "confirm_lowSA", "test_lowSA"):
        m = res[f"pooled_{name}"]
        print(f"{name:<8}{m['n']:>5} | {m['hattrick']['gsax60_mae']:>16.4f}{m['last']['gsax60_mae']:>7.4f}"
              f"{m['naive']['gsax60_mae']:>7.4f} | {m['hattrick']['gsax60_corr']:>8.3f}"
              f"{m['last']['gsax60_corr']:>7.3f} | {m['hattrick']['sv_mae']:>11.4f}"
              f"{m['last']['sv_mae']:>8.4f}{m['naive']['sv_mae']:>8.4f} | {m['cov80_gsax60']:.2f}")
    s = res["starts"]
    print(f"\nstarts (opening goalies, full box seasons {s['seasons']}, n={s['n']}): MAE {s['starts_mae']:.1f}, bias "
          f"{s['starts_bias']:+.1f}; team No.1 projected {s['no1_proj_mean']:.1f} vs realised "
          f"{s['no1_act_mean']:.1f}")
    print("55+ starts per season (projected expectation vs realised):",
          [(r["V"], r["proj55"], r["act55"]) for r in s["per_season_55plus"]])
    print("->", C.OUT / "backtest" / "goalies_bt.json")


if __name__ == "__main__":
    main()
