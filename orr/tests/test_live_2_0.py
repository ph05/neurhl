"""Checks for ORR 2.0: the preregistered evaluation, the reproducibility check
and the model card.

Run: python3 -m orr.tests.test_live_2_0
"""
import hashlib
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from orr import evaluate_2027 as EV
from orr import inseason as IS
from orr import reproduce as RP
from orr.site import build_model as BM


def test_paired_verdicts():
    rng = np.random.default_rng(1)
    assert EV.paired(-0.05 + 0.01 * rng.standard_normal(300))["verdict"] == "ORR better"
    assert EV.paired(0.05 + 0.01 * rng.standard_normal(300))["verdict"] == "NeurHL better"
    r = EV.paired(np.r_[np.full(150, 0.1), np.full(150, -0.1)])
    assert r["verdict"] == "no difference shown" and abs(r["diff"]) < 1e-12


def test_game_comparison_interim_then_final():
    rng = np.random.default_rng(0)
    n = 120
    y = rng.integers(0, 2, n)
    ep = pd.concat([pd.DataFrame({"model": "orr_inseason", "game_id": range(n), "p": np.where(y == 1, 0.7, 0.3), "y": y}),
                    pd.DataFrame({"model": "neurhl_G_pregame", "game_id": range(n), "p": 0.5, "y": y})])
    a = EV.game_comparison(ep, "orr_inseason", "neurhl_G_pregame", final=False)
    b = EV.game_comparison(ep, "orr_inseason", "neurhl_G_pregame", final=True)
    assert a["status"] == "interim" and a["verdict"].startswith("(interim, not a verdict)")
    assert b["status"] == "final" and b["verdict"] == "ORR better" and b["n"] == n
    assert EV.game_comparison(ep, "orr_inseason", "elo_pregame")["status"] == "pending"


def test_team_crps_from_quantiles():
    from scipy import stats
    q = stats.norm.ppf(0.9)
    t = pd.DataFrame({"points": [90.0, 90.0], "points_p10": [90 - 10 * q, 90 - 10 * q], "points_p90": [90 + 10 * q, 90 + 10 * q]})
    c = EV.crps_from_quantiles(t, pd.Series([90.0, 100.0]))
    assert abs(c.iloc[0] - 10 * (2 * stats.norm.pdf(0) - 1 / np.sqrt(np.pi))) < 1e-9
    assert c.iloc[1] > c.iloc[0]


def test_season_end_comparisons_run_on_a_complete_season():
    """P3, P4 and the goalie comparison on a synthetic full season, so the
    season-end branches are exercised now, not first in April."""
    from orr import config as C
    sch = pd.read_csv(C.OUT / "freeze_2027" / "schedule_2027.csv")
    rng = np.random.default_rng(3)
    res = pd.DataFrame({"game_id": sch.game_id, "home": sch.home, "away": sch.away,
                        "home_g": rng.integers(0, 6, len(sch)), "away_g": rng.integers(0, 6, len(sch))})
    tie = res.home_g == res.away_g
    res.loc[tie, "home_g"] += 1
    res["extra"] = np.where(tie, "OT", "REG")
    assert EV.season_complete(res)
    p3 = EV.team_comparison(res, "orr_preseason", "neurhl_1.1")
    assert p3["status"] == "final" and p3["n"] == 32 and p3["crps"]["n"] == 32 and "verdict" in p3
    sk = pd.read_csv(C.ROOT / EV.FILES["orr_preseason"][2]).head(300)
    gl = pd.read_csv(C.ROOT / EV.FILES["orr_preseason"][3])
    rows = [pd.DataFrame({"game_id": 1, "player_id": sk.player_id, "pos": "F", "g": 10, "a": 20,
                          "shots_against": np.nan, "goals_against": np.nan}),
            pd.DataFrame({"game_id": 1, "player_id": gl.player_id, "pos": "G", "g": 0, "a": 0,
                          "shots_against": 1500, "goals_against": 140})]
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "boxes.csv"
        pd.concat(rows).to_csv(f, index=False)
        p4 = EV.skater_comparison(f, "orr_preseason", "neurhl_1.1", res)
        gs = EV.goalie_comparison(f, "neurhl_1.0", res)
    assert p4["status"] == "final" and p4["all_skaters"]["n"] > 0 and p4["n"] == 0   # one game each: nobody has 40 GP
    assert gs["status"] == "final" and gs["n"] > 0


def test_season_complete_needs_every_game():
    teams = [f"T{i}" for i in range(32)]
    res = pd.DataFrame({"home": teams[:16], "away": teams[16:]})
    # every team has played: the check must return a Python bool, which json can write
    # (a numpy bool broke the daily Action from 2026-10-04)
    assert EV.season_complete(res) is False
    json.dumps({"season_complete": EV.season_complete(res)})


def test_reproduce_compare_ignores_only_the_time_stamp():
    a = b"game_id,p,created_utc\n1,0.5,2026-10-02T00:00\n"
    b = b"game_id,p,created_utc\n1,0.5,2026-10-02T09:00\n"
    c = b"game_id,p,created_utc\n1,0.51,2026-10-02T00:00\n"
    assert RP.compare(a, a)["identical"]
    assert RP.compare(a, b)["identical"]
    r = RP.compare(a, c)
    assert not r["identical"] and abs(r["max_abs_diff"] - 0.01) < 1e-12


def test_reproduce_gathers_exactly_the_recorded_lineups():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        src = d / "src" / "2027" / "2026-10-01"
        src.mkdir(parents=True)
        f = src / "pregame_1_lineups.json"
        f.write_text('{"x": 1}')
        (src / "pregame_2_lineups.json").write_text('{"later": 1}')     # not recorded: must not be copied
        rec = [{"path": str(f), "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}]
        dest = d / "dest"
        rows = RP.gather_lineups(rec, None, dest, d)
        assert rows[0]["matched"] and (dest / "2026-10-01" / "pregame_1_lineups.json").exists()
        assert not (dest / "2026-10-01" / "pregame_2_lineups.json").exists()


def test_2_0_forecasts_as_1_9():
    strip = lambda c: {k: v for k, v in c.items() if k not in ("version", "accepted_items")}
    assert IS.DEFAULT_MODEL == "2.0" and strip(IS.MODELS["2.0"]) == strip(IS.MODELS["1.9"])


def test_model_card_states_follow_the_params():
    html = BM.switch_rows()
    for sw in ("absence", "player_calibration", "standings_sharp"):
        row = html.split(f"<code>{sw}</code>")[1].split("</tr>")[0]
        assert "off (by its test)" in row, sw
    row = html.split("<code>lineups</code>")[1].split("</tr>")[0]
    assert "<b>live</b>" in row and ">1.1<" in row
    assert set(BM.SWITCHES) >= {k for k, v in IS.MODELS["2.0"].items() if v is True}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
