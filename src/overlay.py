"""Roster-delta overlay: value the KNOWN July-2026 roster changes and convert to team Elo
deltas, gated by a retrospective mechanism test on train. Also writes the pre-registration
block into params_v2.json (must run BEFORE confirm).

Mechanism test (train, h=1, T=2013..2017): movers = players whose majority team changed
T-1 -> T. Arrivals weighted by TOI share actually played with the new team in T (mitigates
midseason-trade contamination); departures full weight. Team net value x (goals) ->
expected points = chain * x. Regress v1 residual on it (through origin) -> rho_hat.
Placebo: prior-season residual on the same x -> must be small.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as E
import players as P
from features import FeatureBuilder

PROJ = Path(__file__).resolve().parents[1]
RAW = PROJ / "data" / "raw"
OUT = PROJ / "output"
V_PROD = 2026  # production vantage (2025-26 season)
SHOTS_SEASON = 2500.0
TANDEM = {0: 0.58, 1: 0.30}


def points_per_goal(sk: pd.DataFrame, ts: pd.DataFrame, max_season: int) -> float:
    p = sk[sk.season_end <= max_season].points.sum()
    gf = ts[ts.season_end <= max_season].gf.sum()
    return float(p / gf)


def skater_value_goals(marcel: pd.DataFrame, repl: dict, ppg: float) -> pd.Series:
    """Projected goals/season above positional replacement, per player."""
    v = (marcel.proj_pts60 - marcel.pos_group.map(repl)).clip(lower=-1.0) \
        * marcel.toi82_proj / 60.0 / ppg
    return pd.Series(v.to_numpy(), index=marcel.playerId.to_numpy())


def mover_table_2026(sk_val: pd.Series, goalie_proj: pd.DataFrame):
    """Diff August-2026 API rosters vs MoneyPuck 2025-26 majority teams."""
    ros = pd.read_csv(RAW / "nhl_rosters_20262027.csv")
    skaters, goalies, skt, got, bios = P.load_panels()
    maj_sk = P.majority_team(skt[skt.season_end == V_PROD]).set_index("playerId").team
    maj_go = P.majority_team(got[got.season_end == V_PROD]).set_index("playerId").team
    gproj = goalie_proj.set_index("playerId")

    rows = []
    on_roster = set(ros.playerId)
    # goalie tandem rank within each API roster (by career window shots)
    ros_g = ros[ros.position == "G"].copy()
    ros_g["shots_win"] = ros_g.playerId.map(gproj.shots_win).fillna(0.0)
    ros_g["rank"] = ros_g.groupby("team").shots_win.rank(ascending=False, method="first") - 1

    for r in ros.itertuples():
        if r.position == "G":
            prev = maj_go.get(r.playerId)
            theta = gproj.theta.get(r.playerId, 0.0)
            rk = int(ros_g.loc[ros_g.playerId == r.playerId, "rank"].iloc[0])
            share = TANDEM.get(rk, 0.12)
            val = theta * share * SHOTS_SEASON
            kind = "G"
        else:
            prev = maj_sk.get(r.playerId)
            val = float(sk_val.get(r.playerId, 0.0))
            kind = "S"
        if prev is None:
            continue  # rookie / no 2025-26 NHL sample: value carried by draft_cap
        if prev != r.team:
            rows.append({"playerId": r.playerId, "kind": kind, "from": prev, "to": r.team,
                         "val_goals": val, "limbo": False})
    # limbo: played 2025-26, on no August roster (UFA/retired/Europe) -> 0.5-weight departure
    played = set(maj_sk.index) | set(maj_go.index)
    for pid in played - on_roster:
        if pid in maj_go.index:
            theta = gproj.theta.get(pid, 0.0)
            val, kind, prev = theta * 0.44 * SHOTS_SEASON, "G", maj_go[pid]
        else:
            val, kind, prev = float(sk_val.get(pid, 0.0)), "S", maj_sk[pid]
        rows.append({"playerId": pid, "kind": kind, "from": prev, "to": None,
                     "val_goals": val, "limbo": True})
    return pd.DataFrame(rows)


def team_net_goals(movers: pd.DataFrame) -> pd.Series:
    net = {}
    for _, r in movers.iterrows():
        w = 0.5 if r["limbo"] else 1.0
        if pd.notna(r["to"]) and r["to"] is not None:
            net[r["to"]] = net.get(r["to"], 0.0) + w * r["val_goals"]
        if pd.notna(r["from"]) and r["from"] is not None:
            net[r["from"]] = net.get(r["from"], 0.0) - w * r["val_goals"]
    return pd.Series(net).sort_values()


def mechanism_test(fb: FeatureBuilder, p2: dict, g, ts) -> dict:
    import backtest2 as B
    sk, goalies, skt, got, bios = P.load_panels()
    ppg = points_per_goal(sk, ts, 2017)
    chain = p2["k"] * p2["c"] / 82.0
    preds, end_r, _ = E.run_elo(g, K=B.V1["K"], H=B.V1["H"], phi_s=B.V1["phi_s"])
    v1p = B.v1_predictions(end_r, preds, 1, list(range(2012, 2018)))

    xs, ys, ys_placebo = [], [], []
    for T in range(2013, 2018):
        V = T - 1
        marcel = fb.marcel(V, 1)
        repl = P.replacement_rates(sk, V)
        sk_val = skater_value_goals(marcel, repl, ppg)
        maj_prev = P.majority_team(skt[skt.season_end == V]).set_index("playerId").team
        cur = skt[skt.season_end == T]
        team_toi = cur.groupby("team").toi_min.sum()
        x = {}
        for r in cur.itertuples():
            prev = maj_prev.get(r.playerId)
            if prev is None or prev == r.team:
                continue
            share = r.toi_min / team_toi[r.team]
            val = float(sk_val.get(r.playerId, 0.0))
            w_arr = min(share * 18.0, 1.0)  # ~full-season player has share ~1/18 of team TOI
            x[r.team] = x.get(r.team, 0.0) + w_arr * val
            x[prev] = x.get(prev, 0.0) - val
        act = ts[ts.season_end == T].set_index("team")
        ydev = (act.pts_pct - act.pts_pct.mean()) * 164
        act_prev = ts[ts.season_end == V].set_index("team")
        ydev_prev = (act_prev.pts_pct - act_prev.pts_pct.mean()) * 164
        for team, xv in x.items():
            if team in v1p[T].index and team in ydev.index:
                xs.append(chain * xv)
                ys.append(float(ydev[team] - v1p[T][team]))
                if team in ydev_prev.index and team in v1p.get(V, pd.Series(dtype=float)).index:
                    ys_placebo.append((chain * xv,
                                       float(ydev_prev[team] - v1p[V][team])))
    xs, ys = np.array(xs), np.array(ys)
    rho = float((xs @ ys) / (xs @ xs))
    se = float(np.sqrt(((ys - rho * xs) ** 2).sum() / (len(xs) - 1) / (xs @ xs)))
    if ys_placebo:
        px = np.array([a for a, _ in ys_placebo])
        py = np.array([b for _, b in ys_placebo])
        placebo = float((px @ py) / (px @ px))
    else:
        placebo = np.nan
    return {"rho_hat": round(rho, 3), "se": round(se, 3), "placebo": round(placebo, 3),
            "n": len(xs), "chain": round(chain, 3), "ppg": round(ppg, 3)}


def mechanism_partial(fb: FeatureBuilder, p2: dict, g, ts, arrival_weight: str) -> dict:
    """v3.1 amendment as committed code (was ad hoc; review finding B3), plus the
    production-consistent re-estimate (B4).

    Partial regression on train (T=2013..2017, h=1): v1 residual ~ rho*x + gamma*z,
    z = prior-season v1 residual (controls buyer-selection: teams that add value are
    selected underperformers, which biased the raw through-origin estimate down and
    the placebo negative). arrival_weight: 'toi_share' = original v3.1 spec
    (min(share*18,1) realized-TOI weighting); 'full' = production-consistent
    (August movers are valued at full weight when the overlay is applied).
    """
    import backtest2 as B
    sk, goalies, skt, got, bios = P.load_panels()
    ppg = points_per_goal(sk, ts, 2017)
    chain = p2["k"] * p2["c"] / 82.0
    preds, end_r, _ = E.run_elo(g, K=B.V1["K"], H=B.V1["H"], phi_s=B.V1["phi_s"])
    v1p = B.v1_predictions(end_r, preds, 1, list(range(2012, 2018)))

    xs, ys, zs = [], [], []
    for T in range(2013, 2018):
        V = T - 1
        marcel = fb.marcel(V, 1)
        repl = P.replacement_rates(sk, V)
        sk_val = skater_value_goals(marcel, repl, ppg)
        maj_prev = P.majority_team(skt[skt.season_end == V]).set_index("playerId").team
        cur = skt[skt.season_end == T]
        team_toi = cur.groupby("team").toi_min.sum()
        x = {}
        for r in cur.itertuples():
            prev = maj_prev.get(r.playerId)
            if prev is None or prev == r.team:
                continue
            val = float(sk_val.get(r.playerId, 0.0))
            if arrival_weight == "toi_share":
                w_arr = min(r.toi_min / team_toi[r.team] * 18.0, 1.0)
            else:
                w_arr = 1.0
            x[r.team] = x.get(r.team, 0.0) + w_arr * val
            x[prev] = x.get(prev, 0.0) - val
        act = ts[ts.season_end == T].set_index("team")
        ydev = (act.pts_pct - act.pts_pct.mean()) * 164
        act_prev = ts[ts.season_end == V].set_index("team")
        ydev_prev = (act_prev.pts_pct - act_prev.pts_pct.mean()) * 164
        for team, xv in x.items():
            if (team in v1p[T].index and team in ydev.index
                    and team in ydev_prev.index and team in v1p[V].index):
                xs.append(chain * xv)
                ys.append(float(ydev[team] - v1p[T][team]))
                zs.append(float(ydev_prev[team] - v1p[V][team]))
    # Intercept matters for toi_share x (share-weighted arrivals minus full-weight
    # departures make x negative-sum, mean ~ -2.3 pts/season; through-origin attenuates).
    # For full-weight x the league sum is exactly zero and the intercept is moot.
    M = np.column_stack([xs, zs, np.ones(len(xs))])
    yv = np.array(ys)
    beta = np.linalg.solve(M.T @ M, M.T @ yv)
    resid = yv - M @ beta
    cov = np.linalg.inv(M.T @ M) * (resid @ resid) / (len(yv) - M.shape[1])
    se = np.sqrt(np.diag(cov))
    return {"rho_partial": round(float(beta[0]), 3), "se": round(float(se[0]), 3),
            "gamma_reversion": round(float(beta[1]), 3),
            "intercept": round(float(beta[2]), 3), "n": len(yv),
            "arrival_weight": arrival_weight}


def amend_v4():
    """B3/B4: reproduce the v3.1 amendment, re-estimate with production-consistent
    weights, and regenerate the overlay deltas as overlay_team_deltas_v4.csv
    (v3 artifacts untouched)."""
    p2 = json.loads((OUT / "params_v2.json").read_text())
    g, ts = E.load()
    v1 = json.loads((OUT / "params.json").read_text())
    preds, end_r, _ = E.run_elo(g, K=v1["K"], H=v1["H"], phi_s=v1["phi_s"])
    fb = FeatureBuilder(end_r, ts, goalie_hp=p2["goalie_hp"], skater_delta=p2["skater_delta"])

    orig = mechanism_partial(fb, p2, g, ts, "toi_share")
    full = mechanism_partial(fb, p2, g, ts, "full")
    print(f"repro (toi_share): rho_partial={orig['rho_partial']} (se {orig['se']}), "
          f"gamma={orig['gamma_reversion']}, n={orig['n']}  [documented: 1.011, se 0.260, n 150]")
    print(f"production-consistent (full): rho_partial={full['rho_partial']} "
          f"(se {full['se']}), gamma={full['gamma_reversion']}, n={full['n']}")
    reproduced = abs(orig["rho_partial"] - 1.011) <= 0.15
    print(f"ACCEPTANCE (PLAN_V4 B3): {'REPRODUCED' if reproduced else 'NOT REPRODUCED'}")

    rho_star = float(np.clip(0.5 * full["rho_partial"], 0.0, 0.75))
    print(f"rho* (v4, from production-consistent estimate) = {rho_star:.3f} "
          f"(v3.1 applied 0.506)")

    # regenerate production deltas with the v4 rho* (mover table inputs unchanged)
    sk, goalies, skt, got, bios = P.load_panels()
    marcel = fb.marcel(V_PROD, 1)
    repl = P.replacement_rates(sk, V_PROD)
    ppg = points_per_goal(sk, ts, V_PROD)
    sk_val = skater_value_goals(marcel, repl, ppg)
    gproj = fb.goalie_proj(V_PROD)
    movers = mover_table_2026(sk_val, gproj)
    net = team_net_goals(movers)
    delo = (p2["k"] * net / 82.0 * rho_star).clip(-25, 25)
    delo = delo - delo.mean()
    pd.DataFrame({"net_goals": net, "dElo": delo.reindex(net.index).fillna(0)}).to_csv(
        OUT / "overlay_team_deltas_v4.csv")
    block = {"repro_toi_share": orig, "production_consistent_full": full,
             "reproduced_within_band": bool(reproduced),
             "rho_star_v4": round(rho_star, 3), "rho_star_v31": 0.506,
             "history": "raw rho 0.743 placebo-blocked (v2, prereg working); v3.1 "
                        "'partial-regression identification' reproduced at 1.011 WITH "
                        "INTERCEPT — decomposition shows the gain was the intercept "
                        "absorbing the negative-sum artifact of TOI-share weighting "
                        "(league mean x ~ -2.3), NOT the selection control (gamma ~ "
                        "0.006). v4: production-consistent full weights make x exactly "
                        "zero-sum; rho is spec-robust at ~0.84 (origin/intercept/"
                        "z-control all agree). PLAN_V4 B3+B4."}
    (OUT / "overlay_v4.json").write_text(json.dumps(block, indent=2))
    print(f"wrote overlay_team_deltas_v4.csv + overlay_v4.json "
          f"(biggest: {delo.abs().idxmax()} {delo[delo.abs().idxmax()]:+.1f} Elo)")
    return block


def main():
    p2 = json.loads((OUT / "params_v2.json").read_text())
    g, ts = E.load()
    v1 = json.loads((OUT / "params.json").read_text())
    preds, end_r, _ = E.run_elo(g, K=v1["K"], H=v1["H"], phi_s=v1["phi_s"])
    fb = FeatureBuilder(end_r, ts, goalie_hp=p2["goalie_hp"], skater_delta=p2["skater_delta"])

    mech = mechanism_test(fb, p2, g, ts)
    print(f"mechanism test: rho_hat={mech['rho_hat']} (se {mech['se']}), "
          f"placebo={mech['placebo']}, n={mech['n']}")
    ok = (0.3 <= mech["rho_hat"] <= 1.2) and (abs(mech["placebo"]) < mech["rho_hat"] / 2)
    rho_star = float(np.clip(0.5 * mech["rho_hat"], 0.0, 0.75)) if ok else 0.0
    print(f"overlay {'APPLIED' if ok else 'NOT applied'} (rho*={rho_star})")

    # production overlay from August 2026 rosters
    sk, goalies, skt, got, bios = P.load_panels()
    marcel = fb.marcel(V_PROD, 1)
    repl = P.replacement_rates(sk, V_PROD)
    ppg = points_per_goal(sk, ts, V_PROD)
    sk_val = skater_value_goals(marcel, repl, ppg)
    gproj = fb.goalie_proj(V_PROD)
    movers = mover_table_2026(sk_val, gproj)
    net = team_net_goals(movers)
    delo_raw = p2["k"] * net / 82.0 * rho_star
    delo = delo_raw.clip(-25, 25)
    delo = delo - delo.mean()
    movers["names"] = movers.playerId.map(
        pd.concat([sk.drop_duplicates("playerId").set_index("playerId")["name"],
                   goalies.drop_duplicates("playerId").set_index("playerId")["name"]]
                  ).groupby(level=0).first())
    movers.to_csv(OUT / "roster_delta_2026.csv", index=False)
    pd.DataFrame({"net_goals": net, "dElo": delo.reindex(net.index).fillna(0)}).to_csv(
        OUT / "overlay_team_deltas.csv")
    print("\nbiggest movers (net goals):")
    print(net.head(4).round(1).to_string())
    print(net.tail(4).round(1).to_string())

    p2["prereg"] = {
        "written_before_confirm": True,
        "confirm_gate": "h1 deviation-MAE excl 2021: rung MAE <= v1_MAE + 0.10 AND "
                        "spearman >= v1 - 0.02; adopt highest qualifying rung (R3>R2>R1)",
        "final_window": "2022-2026, single scripted run, no decisions taken on it",
        "overlay_rule": "apply iff rho_hat in [0.3,1.2] and |placebo| < rho_hat/2; "
                        "rho* = clip(0.5*rho_hat, 0, 0.75); dElo = clip(rho**k*x/82, +-25), "
                        "recentered; never inside backtests",
        "hetero_rule": "keep iff train 80% coverage tertile spread improves vs uniform "
                       f"(result: keep={p2['hetero']['keep']})",
        "mechanism": mech, "overlay_applied": ok, "rho_star": rho_star,
    }
    (OUT / "params_v2.json").write_text(json.dumps(p2, indent=2))
    print("pre-registration written to params_v2.json")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "amend_v4":
        amend_v4()
    else:
        main()
