"""Checks for the daily pipeline: results ingest, the evaluation file and the
reproducibility check.

Run: python3 -m orr.tests.test_pipeline
"""
import datetime as dt
import json
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from orr import config as C
from orr import evaluate_2027 as EV
from orr import ingest as IG
from orr import reproduce as RP


class _Resp:
    def __init__(self, js):
        self.js = js

    def json(self):
        return self.js


def _fake_api(day: pd.DataFrame, calls: list):
    """The NHL score and boxscore endpoints for one date's finished games."""
    def get(url, timeout=30):
        calls.append(url)
        if "/score/" in url:
            return _Resp({"games": [{"id": int(r.game_id), "gameType": 2, "gameState": "OFF",
                                     "homeTeam": {"abbrev": r.home, "score": 3, "sog": 31},
                                     "awayTeam": {"abbrev": r.away, "score": 2, "sog": 27},
                                     "gameOutcome": {"lastPeriodType": "REG"}} for r in day.itertuples()]})
        gid = int(url.split("/gamecenter/")[1].split("/")[0])
        r = day.set_index("game_id").loc[gid]
        side = lambda pid: {"forwards": [{"playerId": pid, "goals": 1, "assists": 0, "sog": 3, "toi": "15:00"}],
                            "defense": [], "goalies": [{"playerId": pid + 1, "toi": "60:00", "starter": True,
                                                        "saveShotsAgainst": "25/27", "goalsAgainst": 2}]}
        return _Resp({"homeTeam": {"abbrev": r.home}, "awayTeam": {"abbrev": r.away},
                      "playerByGameStats": {"homeTeam": side(gid * 10), "awayTeam": side(gid * 10 + 5)}})
    return get


def test_ingest_backfills_games_without_box_scores():
    """Games stored from another file (no shots, no box score) are fetched again,
    and a complete game is never fetched twice."""
    import requests
    sch = pd.read_csv(C.OUT / "freeze_2027" / "schedule_2027.csv")
    day = sch[sch.date.astype(str) == "2026-09-29"][["game_id", "date", "home", "away"]]
    saved = IG.PATH, IG.BOXES, requests.get, IG.time.sleep
    with tempfile.TemporaryDirectory() as d:
        IG.PATH, IG.BOXES = Path(d) / "results.csv", Path(d) / "boxes.csv"
        day.assign(home_g=3, away_g=2, last_period="REG", source="neurhl file").to_csv(IG.PATH, index=False)
        calls = []
        requests.get, IG.time.sleep = _fake_api(day, calls), (lambda s: None)
        try:
            assert IG.complete_ids() == set()
            IG.save(IG.fetch_api(dt.date(2026, 9, 29)))
            r, b = pd.read_csv(IG.PATH), pd.read_csv(IG.BOXES)
            assert len(r) == len(day) and r.shots_home.notna().all() and (r.source == "api-web.nhle.com").all()
            assert set(b.game_id) == set(day.game_id)
            assert IG.complete_ids() == set(day.game_id)
            calls.clear()
            assert len(IG.fetch_api(dt.date(2026, 9, 29))) == 0 and not calls
        finally:
            IG.PATH, IG.BOXES, requests.get, IG.time.sleep = saved


def test_evaluation_file_is_json_when_every_team_has_played():
    teams = [f"T{i}" for i in range(32)]
    res = pd.DataFrame({"home": teams[:16], "away": teams[16:]})
    json.dumps({"season_complete": EV.season_complete(res)})


def _git(*a):
    return subprocess.run(["git", *a], cwd=C.ROOT, capture_output=True, text=True, check=True).stdout.strip()


def test_reproduce_finds_code_after_a_history_rewrite():
    head, tree = _git("rev-parse", "HEAD"), _git("rev-parse", "HEAD^{tree}")
    assert RP.resolve_code({"code": head[:7]}) == ("commit", head[:7])
    missing = "0" * 40
    assert RP.resolve_code({"code": missing[:7], "code_tree": tree}) == ("tree", tree)
    saved = RP.REWRITES
    with tempfile.TemporaryDirectory() as d:
        RP.REWRITES = Path(d) / "rewrites.json"
        RP.REWRITES.write_text(json.dumps({missing: head}))
        try:
            assert RP.resolve_code({"code": missing[:7]}) == ("commit", head)
        finally:
            RP.REWRITES = saved


def test_every_committed_run_can_be_reproduced_from_this_history():
    """The code named by each published daily run is still in the repository, directly,
    by its tree, or through orr/output/reproduce/rewritten_commits.json."""
    runs = sorted((C.OUT / "live").glob("*/run_*.json"))
    assert runs
    for f in runs:
        kind, ref = RP.resolve_code(json.loads(f.read_text()))
        assert _git("cat-file", "-t", ref) in ("commit", "tree"), f


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
