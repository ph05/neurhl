"""Checks for the player projection layer (skaters, goalies, deployment).

Run: python3 -m hattrick.tests.test_players
"""
from __future__ import annotations

import functools
import time

import numpy as np
import pandas as pd

from hattrick import config as C
from hattrick import data as D
from hattrick import deploy as DP
from hattrick import goalies as GL
from hattrick import players as PL

V_TEST = 2016


def _clear_player_caches():
    """Drop every memoised intermediate that depends on the panel."""
    PL._RAW_CACHE.clear()
    for name in dir(PL):
        f = getattr(PL, name)
        if isinstance(f, functools._lru_cache_wrapper) and name != "panel":
            f.cache_clear()


def test_walk_forward_no_future_data():
    """project(V) must not change when every season >= V is scrambled."""
    prm = PL.load_params()
    P = PL.panel()
    ids = P.ids[PL._hist_gp(P, V_TEST) > 0][:400]
    _clear_player_caches()
    before = PL.project(V_TEST, prm, ids=ids).set_index("player_id").sort_index()
    saved = {k: v.copy() for k, v in P.x.items()}
    try:
        j = P.j(V_TEST)
        rng = np.random.default_rng(0)
        for k, v in P.x.items():
            v[:, j:] = v[:, j:] * rng.uniform(0.2, 5.0, size=v[:, j:].shape)
        _clear_player_caches()
        after = PL.project(V_TEST, prm, ids=ids).set_index("player_id").sort_index()
    finally:
        for k in saved:
            P.x[k][:] = saved[k]
        _clear_player_caches()
    num = [c for c in before.columns if before[c].dtype.kind in "fi"]
    diff = (before[num] - after[num]).abs().max().max()
    assert diff < 1e-9, f"projection for {V_TEST} moved by {diff} when seasons >= {V_TEST} changed"


def _deploy_2016():
    prm = PL.load_params()
    ros, flag = DP.historical_roster(V_TEST, "opening")
    proj = PL.project(V_TEST, prm, ids=ros.player_id)
    ros = ros[ros.player_id.isin(proj.player_id)]
    dep = DP.deploy(proj, ros, V_TEST, 82, prm.dress_noise, prm.q_scale,
                    cover=DP.coverage(V_TEST, "opening"))
    return prm, proj, dep


def test_toi_conservation():
    """Each team's skater minutes by position and situation (roster plus
    call-ups) equal games x the league budget per team-game."""
    prm, proj, dep = _deploy_2016()
    bud, rep = DP.budgets_for(V_TEST)
    for team, t in dep.groupby("team"):
        for pos in ("F", "D"):
            s = t[t.pos == pos]
            rg = float(t[f"rep_gp_{pos}"].iloc[0])
            for k in DP.SITS:
                got = float(s[f"toi_{k}"].sum()) + rg * rep[(pos, k)]
                need = 82 * bud[(pos, k)]
                assert abs(got - need) < 1e-6 * need + 1e-6, (team, pos, k, got, need)
    # ice-time ceilings respected
    tpg = sum(dep[f"tpg_{k}_n"] for k in DP.SITS)
    assert (tpg[dep.pos == "F"] <= DP.TOI_CAP["F"] + 1e-6).all()
    assert (tpg[dep.pos == "D"] <= DP.TOI_CAP["D"] + 1e-6).all()


def test_gp_caps_and_dressed_games():
    """No player exceeds the schedule (minus games ruled out); a team never
    dresses more than 12 F / 6 D per game in expectation."""
    prm, proj, dep = _deploy_2016()
    assert (dep.gp >= 0).all() and (dep.gp <= 82 + 1e-9).all()
    for (team, pos), t in dep.groupby(["team", "pos"]):
        assert t.gp.sum() <= DP.N_DRESS[pos] * 82 + 1e-6, (team, pos, t.gp.sum())
    # with injuries: games out are never played
    ros = D.rosters_2027()
    sk = ros[ros.grp != "G"][["player_id", "team"]]
    ex = ros[ros.grp != "G"].assign(pos=lambda d: np.where(d.grp == "D", "D", "F"))
    p27 = PL.project(2027, prm, ids=sk.player_id, extra=ex[["player_id", "pos", "birth", "name"]])
    go = DP.games_out_2027(sk, 84)
    d27 = DP.deploy(p27, sk, 2027, 84, prm.dress_noise, prm.q_scale, go,
                    cover=DP.coverage(2027, "opening"))
    lim = 84 - d27.games_out
    assert (d27.gp <= lim + 1e-6).all()
    assert (d27.gp <= 84 + 1e-9).all()


def test_nonnegative_rates_and_totals():
    prm, proj, dep = _deploy_2016()
    r = [c for c in proj.columns if c.startswith(("r60_", "sd60_", "tpg_"))]
    assert (proj[r].to_numpy() >= 0).all()
    assert np.isfinite(proj[r].to_numpy()).all()
    tot = DP.add_totals(dep, proj)
    for c in ("g", "a", "p", "sog", "toi", "hits", "blk", "pim", "fow", "fol", "tk", "gv",
              "ppg", "ppa"):
        assert (tot[c] >= 0).all(), c
    assert np.allclose(tot.p, tot.g + tot.a1 + tot.a2)


def test_simulation_bands():
    prm, proj, dep = _deploy_2016()
    tot = DP.add_totals(dep, proj)
    t = tot[tot.team.isin(["BOS", "PIT"])]
    q = DP.simulate(t, proj, V_TEST, 82, prm.mc(), S=200)
    assert (q.gp_p90 <= 82 + 1e-9).all() and (q.gp_p10 >= 0).all()
    assert (q.p_p10 <= q.p_p90 + 1e-9).all()
    m = t.set_index("player_id").p
    qq = q.set_index("player_id")
    inside = ((m >= qq.p_p10 - 1) & (m <= qq.p_p90 + 1)).mean()
    assert inside > 0.95, inside


def test_goalie_starts_identity():
    """Rostered goalies' starts plus call-up starts fill every game."""
    r = D.opening_rosters(V_TEST)
    g = r[r.grp == "G"][["player_id", "team"]]
    dep = GL.deploy_goalies(V_TEST, g, 82)
    for team, t in dep.groupby("team"):
        total = t.starts.sum() + t.callup_starts.iloc[0]
        assert abs(total - 82) < 1e-6, (team, total)
    assert (dep.starts >= 0).all()


def test_availability_rules_2027():
    """LTIR / suspended players get no games; researched estimates win."""
    ros = PL.roster_2027()
    go, rng_, src = DP.games_out_2027(ros, 84, with_range=True)
    av = D.availability_raw_2027()
    for pid in av[av.manual_status.isin(["LTIR", "SUSPENDED_NOT_REPORTING"])].player_id:
        if pid in go.index:
            assert go[pid] == 84, pid
    for pid, lo_hi in rng_.items():
        assert lo_hi[0] <= go[pid] <= lo_hi[1] or lo_hi[0] == lo_hi[1], pid
    # injured players carried off the 23-man roster are included (Bedard)
    assert 8484144 in set(ros.player_id)
    # Hellebuyck: WPG mixture, 10% chance of playing there, from game ~13
    h = GL.HOLDOUTS[8476945]
    assert h["team"] == "WPG" and 0 < h["p_present"] < 1


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t0 = time.time()
        t()
        print(f"ok  {t.__name__}  ({time.time() - t0:.1f}s)")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
