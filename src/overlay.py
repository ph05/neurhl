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
    main()
