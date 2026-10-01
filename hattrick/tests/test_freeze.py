"""Acceptance checks on the 2026-27 freeze outputs.

NeurHL's acceptance batteries test bookkeeping identities and wide bands. These
also test the properties its forecasts lacked: no input after the cutoff,
regression toward the mean, a team spread no wider than the market's, goalie
usage without a hard cap, and agreement between player and team totals.

Run (after python3 -m hattrick.freeze): python3 -m hattrick.tests.test_freeze
"""
import json
from datetime import datetime

import numpy as np
import pandas as pd

from hattrick import config as C
from hattrick import data as D

F = C.OUT / "freeze_2027"


def _load(name):
    return pd.read_csv(F / name)


def test_no_input_after_cutoff():
    m = json.loads((F / "manifest_2027.json").read_text())
    latest = datetime.fromisoformat(m["latest_input_change"])
    assert latest < C.CUTOFF_UTC, latest
    for rel, meta in m["files"].items():
        assert datetime.fromisoformat(meta["last_changed"]) < C.CUTOFF_UTC, rel
    inj = pd.read_csv(C.PKG / "data_injuries_2027.csv")
    assert (pd.to_datetime(inj.report_date) <= pd.Timestamp("2026-09-29")).all()


def test_games_file():
    g = _load("games_2027.csv")
    assert len(g) == 1344 and g.game_id.is_unique
    tot = g.p_home_reg + g.p_away_reg + g.p_ot
    assert np.allclose(tot, 1.0, atol=1e-6)
    assert g.p_home_win.between(0.2, 0.85).all()
    assert 0.52 < g.p_home_win.mean() < 0.56          # history: 0.53-0.55
    assert 0.20 < g.p_ot.mean() < 0.26                # history: 0.22-0.25
    for t in C.TEAMS_2027:
        assert ((g.home == t) | (g.away == t)).sum() == 84


def test_standings_sums_and_spread():
    t = _load("teams_2027.csv")
    assert abs(t.playoff_pct.sum() - 1600) < 0.5
    assert abs(t.division_pct.sum() - 400) < 0.5
    assert abs(t.cup_pct.sum() - 100) < 0.05
    assert abs(t.presidents_pct.sum() - 100) < 0.05
    # every game hands out 2 points + 1 if it goes past regulation
    g = _load("games_2027.csv")
    assert abs(t.points.sum() - (2 * len(g) + g.p_ot.sum())) < 5
    # spread of expectations must not exceed the market's (NeurHL 1.3: 10.4 vs 9.4)
    m = D.market_totals_2027()
    assert t.points.std() <= m.line.std() + 0.25, (t.points.std(), m.line.std())
    # within-team uncertainty from backtest error, not an assumed constant
    assert 11.0 < t.points_sd.mean() < 14.0


def test_player_team_consistency():
    t = _load("teams_2027.csv").set_index("team")
    sk = _load("skaters_2027.csv")
    comp = _load("team_components_2027.csv").set_index("team")
    share = comp.gf_pg_skaters / comp.gf_pg_total
    gf_real = t.gf - (t.w - t.row)                     # no shootout "goals"
    got = sk.groupby("team").g.sum()
    want = gf_real * share.reindex(t.index)
    assert np.allclose(got.reindex(t.index), want, rtol=0.01)


def test_regression_toward_mean():
    sk = _load("skaters_2027.csv").copy()
    last = D.skater_seasons()
    last = last[last.season_end == 2026].set_index("player_id")
    sk["last_ppg"] = sk.player_id.map(last.p_all / last.gp)
    sk["last_gp"] = sk.player_id.map(last.gp)
    reg = sk[(sk.last_gp >= 60) & (sk.gp > 40)]
    slope = np.polyfit(reg.last_ppg, reg.p / reg.gp, 1)[0]
    assert slope < 0.92, slope                         # NeurHL 1.3: 0.96


def test_intervals_not_poisson_narrow():
    sk = _load("skaters_2027.csv")
    top = sk[sk.p > 60]
    width = (top.p_p90 - top.p_p10) / top.p
    poisson = 2 * 1.2816 * np.sqrt(top.p) / top.p
    assert (width / poisson).median() > 1.3            # NeurHL 1.3: ~1.17


def test_goalies():
    gl = _load("goalies_2027.csv")
    per_team = gl.groupby("team").starts.sum() + gl.groupby("team").callup_starts.first()
    assert np.allclose(per_team, 84.0, atol=0.05), per_team.describe()
    assert gl.starts.max() > 55                        # no 52.7-start cap
    # low-history backups regress toward a low prior (e.g. DiPietro .858)
    assert gl.sv_pct[gl.starts > 20].between(0.85, 0.93).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
