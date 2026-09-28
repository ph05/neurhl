"""Tests for neurhl/eval/score_live_g_2027.py (PLAN_NeurHL4 LIVE scoring).

Each scenario builds a throwaway git repository in a temporary directory, with
synthetic forecast files committed at controlled times (GIT_COMMITTER_DATE,
GIT_AUTHOR_DATE) and a synthetic results file, runs the scorer on it with
--root and --offline, and reads the scorecard it writes there. The user's git
configuration is not read (GIT_CONFIG_GLOBAL=/dev/null), and nothing is
written under the real neurhl/output (checked before and after).

  validity   a late pregame forecast is invalid and its game is MISSED for the
             primary analysis while its valid morning forecast still counts;
             a morning row is invalid only for the game that had started
             before its commit; an uncommitted file never counts; a file is
             scored as first committed (a later edit or deletion changes
             nothing)
  fetched    a forecast the publishing clone pushed counts once origin/main
             is fetched, though the untracked local copy never does
  pairing    games without NeurHL-H leave G - H and H - Elo only
  sheet      stat-sheet MAE against synthetic tgx_2027 / player_games_2027,
             degraded games left out; goal_mult by date
  starts     a cached NHL API start overrides the file's start_utc, earlier
             or later
  stamps     a cached Actions stamp after the start is reported and changes no
             validity; a sha without a stamp is asked again only within two
             days of its commit (gh itself is never called)
  bootstrap  the week-block bootstrap is deterministic: the same seed gives the
             same interval, and two scorer runs give identical scorecards
  empty      zero completed games (and no results file): a valid scorecard
             with games_completed 0, exit 0

Run: uv run --no-project --python 3.12 --with numpy --with "pandas<3" \
     --with pyarrow --with requests python neurhl/eval/test_score_live_g.py
"""
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

NRL = Path(__file__).resolve().parents[1]
SCORER = NRL / "eval" / "score_live_g_2027.py"
FC = "neurhl/output/live/2027"
CARD = "neurhl/output/live/scorecard_g_2027.json"
RES = []


def check(name, ok, detail=""):
    RES.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


def close(a, b, tol=1e-9):
    return a is not None and b is not None and abs(a - b) < tol


class Repo:
    """A scratch git repository whose commits carry the dates we give them."""

    def __init__(self, base: Path):
        self.root = base
        base.mkdir(parents=True)
        self.env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
                    "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "test@example.invalid",
                    "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "test@example.invalid"}
        self.git("init", "-q", "-b", "main")

    def git(self, *a, env=None):
        p = subprocess.run(["git", "-C", str(self.root), *a], capture_output=True, text=True,
                           env=env or self.env)
        if p.returncode != 0:
            raise RuntimeError(f"git {' '.join(a)}: {p.stderr}")
        return p.stdout

    def write(self, rel: str, frame: pd.DataFrame) -> str:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(p, index=False)
        return rel

    def commit(self, when: str, *rels):
        """Stage rels (additions, edits or deletions) and commit them dated `when`."""
        self.git("add", "-A", "--", *rels)
        self.git("commit", "-q", "--no-verify", "-m", f"live: {when}",
                 env={**self.env, "GIT_AUTHOR_DATE": when, "GIT_COMMITTER_DATE": when})

    def score(self, *extra):
        p = subprocess.run([sys.executable, str(SCORER), "--root", str(self.root), "--offline",
                            *extra], capture_output=True, text=True, env=self.env)
        path = self.root / CARD
        return p, (json.loads(path.read_text()) if path.exists() else None)


def fc(gid, date, start, g, elo, h=np.nan, p1=np.nan, kind="pregame", src=("NHL_API", "NHL_API"),
       goals=(3.0, 2.8), sog=(30.0, 28.0), xgf=(3.1, 2.9), mult=0.9):
    """One forecast row with the columns neurhl/live/forecast.py writes."""
    return {"game_id": gid, "date": date, "start_utc": start, "home": "AAA", "away": "BBB",
            "forecast": kind, "p_home_win_neurhl_g": g, "p_home_win_elo": elo,
            "p_home_win_neurhl_h": h, "p_home_win_1p0": p1, "p_ot": 0.22,
            "stack_used": "elo+g" if math.isnan(h) else "elo+g+h",
            "goals_home": goals[0], "goals_away": goals[1],
            "goals_home_raw": goals[0] / mult, "goals_away_raw": goals[1] / mult,
            "xgf_home": xgf[0], "xgf_away": xgf[1], "sog_home": sog[0], "sog_away": sog[1],
            "goal_mult": mult, "lineup_home": src[0], "lineup_away": src[1],
            "goalie_home": 8400001, "goalie_away": 8400002, "goalie_src_home": "NHL_API",
            "goalie_src_away": "NHL_API", "bundle": "g_test", "bundle_sha": "0" * 16,
            "code": "0000000", "created_utc": start}


def results(rows) -> pd.DataFrame:
    r = pd.DataFrame(rows, columns=["game_id", "date", "home_g", "away_g", "last_period"])
    return r.assign(home="AAA", away="BBB")[["game_id", "date", "home", "away", "home_g",
                                             "away_g", "last_period"]]


def ll(p, y):
    return -math.log(p if y else 1 - p)


# ------------------------------------------------------------------ scenarios
G1, G2, G3, G4, G5, G6 = (2026020000 + i for i in range(1, 7))
D = "2026-10-01"
T = {G1: f"{D}T16:00:00Z", G2: f"{D}T23:00:00Z", G3: f"{D}T23:30:00Z", G4: f"{D}T23:00:00Z",
     G5: f"{D}T23:00:00Z"}


def build_main(repo: Repo):
    """Five completed games on 2026-10-01 (G1 16:00Z, the rest 23:00-23:30Z):

      G1  pregame 15:05 (valid; edited after the game), morning row late (16:30)
      G2  pregame 23:05 (late), morning valid
      G3  pregame 22:40 without NeurHL-H (valid; deleted after the game), morning valid
      G4  no forecast at all
      G5  pregame only in the working tree (never committed), morning valid
    plus G6 (2026-10-02, after --as-of) and a preseason game in the results."""
    d = f"{FC}/{D}"
    p1 = repo.write(f"{d}/pregame_{G1}.csv", pd.DataFrame([fc(
        G1, D, T[G1], 0.62, 0.55, 0.60, 0.58, src=("NHL_API", "DF_CONFIRMED"),
        goals=(3.1, 2.6), sog=(31, 27), xgf=(3.2, 2.5))]))
    pl = repo.write(f"{d}/pregame_{G1}_players.csv", pd.DataFrame(
        [{"game_id": G1, "team": t, "player_id": pid, "name": "", "toi_ev": ev, "toi_pp": pp,
          "toi_sh": sh, "sog_mean": s, "goals_mean": g, "assists_mean": a, "points_mean": g + a,
          "p_goal": 0.3, "p_point": 0.5}
         for t, pid, ev, pp, sh, s, g, a in (("AAA", 101, 15, 2, 1, 2.5, 0.4, 0.5),
                                              ("AAA", 102, 12, 0, 2, 1.5, 0.2, 0.3),
                                              ("BBB", 201, 16, 3, 0, 3.0, 0.5, 0.4),
                                              ("BBB", 999, 10, 0, 0, 1.0, 0.1, 0.1))]))
    repo.commit("2026-10-01T15:05:00+00:00", p1, pl)
    src = ("DF_PROJECTED", "DF_PROJECTED")
    mo = repo.write(f"{d}/morning.csv", pd.DataFrame([
        fc(G1, D, T[G1], 0.61, 0.55, 0.60, 0.58, "morning", src),
        fc(G2, D, T[G2], 0.45, 0.50, 0.47, 0.52, "morning", src),
        fc(G3, D, T[G3], 0.53, 0.51, 0.52, 0.50, "morning", src),
        fc(G5, D, T[G5], 0.40, 0.45, 0.42, 0.44, "morning", src)]))
    repo.commit("2026-10-01T16:30:00+00:00", mo)
    p3 = repo.write(f"{d}/pregame_{G3}.csv", pd.DataFrame([fc(
        G3, D, T[G3], 0.52, 0.50, np.nan, 0.51, goals=(2.9, 2.8), sog=(29, 30), xgf=(2.8, 3.0))]))
    repo.commit("2026-10-01T22:40:00+00:00", p3)
    p2 = repo.write(f"{d}/pregame_{G2}.csv", pd.DataFrame([fc(G2, D, T[G2], 0.44, 0.50, 0.46,
                                                              0.52)]))
    repo.commit("2026-10-01T23:05:00+00:00", p2)
    repo.write(p1, pd.DataFrame([fc(G1, D, T[G1], 0.99, 0.55, 0.60, 0.58)]))    # edited later
    repo.commit("2026-10-02T12:00:00+00:00", p1)
    (repo.root / p3).unlink()                                                    # deleted later
    repo.commit("2026-10-02T12:05:00+00:00", p3)
    repo.write(f"{d}/pregame_{G5}.csv", pd.DataFrame([fc(G5, D, T[G5], 0.41, 0.45, 0.43, 0.44)]))
    repo.write("neurhl/output/live/results_2027.csv", results([
        (G1, D, 3, 2, "REG"), (G2, D, 1, 4, "REG"), (G3, D, 3, 2, "OT"), (G4, D, 2, 1, "SO"),
        (G5, D, 2, 5, "REG"), (G6, "2026-10-02", 4, 1, "REG"), (2026010050, D, 5, 0, "REG")]))
    tens = repo.root / "neurhl/data/tensors"
    tens.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"game_id": [G1, G1, G3, G3], "is_home": [1.0, 0.0, 1.0, 0.0],
                  "sogf": [33.0, 25.0, 28.0, 31.0], "xgf_all": [3.0, 2.0, 0.0, 0.0]}
                 ).to_parquet(tens / "tgx_2027.parquet", index=False)
    pd.DataFrame({"game_id": [G1] * 5, "game_type": 2, "player_id": [101, 102, 201, 103, 8400001],
                  "is_home": [True, True, False, True, True], "pos_group": [0, 1, 0, 0, 2],
                  "toi_sec": [1140, 780, 1200, 600, 3600], "goalie_start": [0, 0, 0, 0, 1],
                  "goals": [1, 0, 0, 0, 0], "sog": [3, 0, 4, 1, 0], "assists": [0, 1, 2, 0, 0]}
                 ).to_parquet(tens / "player_games_2027.parquet", index=False)
    (tens / "_ingest_2027").mkdir(exist_ok=True)
    (tens / "_ingest_2027" / "state.json").write_text(json.dumps({"degraded": {str(G3): "mp"}}))


def test_validity_pairing_sheet(base: Path):
    print("\nVALIDITY, PAIRING, STAT SHEET")
    repo = Repo(base / "main")
    build_main(repo)
    p, card = repo.score("--as-of", "2026-10-02")
    check("scorer exits 0, nothing on stderr", p.returncode == 0 and not p.stderr.strip(),
          p.stderr[-400:])
    if card is None:
        check("scorecard written", False)
        return repo
    c, pre, mor = card["counts"], card["pregame"], card["morning"]
    check("games completed: 5 regular-season games before --as-of",
          c["games_completed"] == 5, str(c["games_completed"]))
    check("late pregame is invalid: G2 MISSED for the primary (every model)",
          card["missed_game_ids"]["pregame"] == [G2, G4, G5]
          and c["missed_by_reason"]["pregame"] == {"late": 1, "none": 1, "uncommitted": 1},
          f"{card['missed_game_ids']['pregame']} {c['missed_by_reason']['pregame']}")
    check("primary scores G1 and G3 only, the same games for G, Elo and 1.0",
          c["valid"]["pregame"] == 2 and all(pre["models"][m]["n"] == 2
                                             for m in ("neurhl_g", "elo", "neurhl_1p0")))
    check("valid morning still counts: G2 in the morning analysis",
          c["valid"]["morning"] == 3 and G2 not in card["missed_game_ids"]["morning"])
    check("morning row late for G1 only (committed after G1's start)",
          card["missed_game_ids"]["morning"] == [G1, G4]
          and c["missed_by_reason"]["morning"] == {"late": 1, "none": 1})
    check("late forecasts listed: G1 morning row and G2 pregame",
          sorted((r["game_id"], r["forecast"]) for r in card["late_forecasts"])
          == [(G1, "morning"), (G2, "pregame")])
    check("uncommitted file never counts and is reported",
          card["files"]["uncommitted"] == [f"{FC}/{D}/pregame_{G5}.csv"])
    exp_g = (ll(0.62, 1) + ll(0.52, 1)) / 2
    check("file scored as first committed (a later edit is ignored)",
          close(pre["models"]["neurhl_g"]["log_loss"], exp_g)
          and card["files"]["edited_after_add"] == [f"{FC}/{D}/pregame_{G1}.csv"],
          f"{pre['models']['neurhl_g']['log_loss']} vs {exp_g}")
    check("a file deleted after the game still counts (G3 in the primary)",
          G3 not in card["missed_game_ids"]["pregame"]
          and card["files"]["deleted_after_add"] == [f"{FC}/{D}/pregame_{G3}.csv"])
    exp = {"g_minus_elo": (2, (ll(.62, 1) - ll(.55, 1) + ll(.52, 1) - ll(.50, 1)) / 2),
           "g_minus_h": (1, ll(.62, 1) - ll(.60, 1)),
           "h_minus_elo": (1, ll(.60, 1) - ll(.55, 1))}
    got = {k: (pre["paired"][k]["n"], pre["paired"][k]["log_loss"]) for k in exp}
    check("pairing: G3 (no NeurHL-H) leaves G - H and H - Elo only",
          all(got[k][0] == exp[k][0] and close(got[k][1], exp[k][1]) for k in exp), str(got))
    check("NeurHL-H missing reported (G3)",
          pre["models"]["neurhl_h"]["n"] == 1 and pre["models"]["neurhl_h"]["missing_game_ids"]
          == [G3] and c["neurhl_h_missing"] == {"pregame": 1, "morning": 0})
    check("Brier: G - Elo on the same games",
          close(pre["paired"]["g_minus_elo"]["brier"],
                ((0.38 ** 2 - 0.45 ** 2) + (0.48 ** 2 - 0.50 ** 2)) / 2))
    exp_m = (ll(0.45, 0) + ll(0.53, 1) + ll(0.40, 0)) / 3
    check("morning: G log loss on G2, G3, G5; G - H paired on all three",
          close(mor["models"]["neurhl_g"]["log_loss"], exp_m)
          and mor["paired"]["g_minus_h"]["n"] == 3)
    check("under 31 games: no interval yet (interim)",
          pre["paired"]["g_minus_elo"]["ci95"] is None and card["status"] == "interim")
    check("lineup source per game is the less certain side",
          c["lineup_source"]["pregame"] == {"games": {"NHL_API": 1, "DF_CONFIRMED": 1},
                                            "sides": {"NHL_API": 3, "DF_CONFIRMED": 1}}
          and set(pre["by_lineup_source"]) == {"NHL_API", "DF_CONFIRMED"},
          str(c["lineup_source"]["pregame"]))
    ss = pre["stat_sheet"]
    raw = [3.1 / .9, 2.6 / .9, 2.9 / .9, 2.8 / .9]
    check("regulation goals MAE (OT winner's goal removed), scaled and raw",
          close(ss["goals_reg"]["mae"], (0.1 + 0.6 + 0.9 + 0.8) / 4)
          and close(ss["goals_reg"]["mae_raw"], np.mean(np.abs(np.array(raw) - [3, 2, 2, 2])))
          and close(ss["goals_reg"]["mean_actual"], 2.25), str(ss["goals_reg"]))
    check("SOG and xGF MAE from tgx_2027, degraded G3 left out",
          ss["sog"]["n_team_games"] == 2 and close(ss["sog"]["mae"], 2.0)
          and close(ss["xgf"]["mae"], (0.2 + 0.5) / 2), f"{ss['sog']} {ss['xgf']}")
    pl = ss["players"]
    check("skater lines against player_games_2027 (goalie and undressed skater apart)",
          pl["n_skater_games"] == 3 and pl["not_dressed"] == 1
          and close(pl["mae"]["toi"], 1.0) and close(pl["mae"]["sog"], 1.0)
          and close(pl["mae"]["goals"], 1.3 / 3) and close(pl["mae"]["assists"], 2.8 / 3)
          and close(pl["mae"]["points"], 1.7 / 3), str(pl))
    check("A1 multiplier in force recorded by date",
          ss["goal_mult"]["by_date"] == {D: 0.9} and ss["goal_mult"]["n"] == 2)
    return repo


def test_api_starts(repo: Repo):
    print("\nNHL API START TIMES (cached)")
    cache = repo.root / "neurhl/data/tensors/_live_g_2027/api_starts_2027.csv"
    cache.parent.mkdir(parents=True, exist_ok=True)
    # G3 really started 22:00 (pregame committed 22:40: late); G2 at 23:30 (23:05: on time)
    pd.DataFrame({"date": [D, D], "game_id": [G3, G2],
                  "start_utc": [f"{D}T22:00:00Z", f"{D}T23:30:00Z"]}).to_csv(cache, index=False)
    p, card = repo.score("--as-of", "2026-10-02")
    cache.unlink()
    check("scorer exits 0, nothing on stderr", p.returncode == 0 and not p.stderr.strip(),
          p.stderr[-400:])
    miss = card["missed_game_ids"] if card else {}
    check("API start earlier than the file's: G3 pregame late; later: G2 pregame on time",
          miss.get("pregame") == [G3, G4, G5], str(miss.get("pregame")))
    check("morning G3 (16:30) still valid against the API start",
          miss.get("morning") == [G1, G4], str(miss.get("morning")))
    check("start mismatches recorded",
          card and {m["game_id"] for m in card["start_check"]["mismatches"]} == {G2, G3})


def scorer():
    spec = importlib.util.spec_from_file_location("score_live_g_2027", SCORER)
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    return S


def test_stamps(repo: Repo, base: Path):
    print("\nACTIONS STAMPS")
    sha = lambda rel: repo.git("log", "--diff-filter=A", "--format=%H", "--", rel).split()[-1]
    s1, s3 = sha(f"{FC}/{D}/pregame_{G1}.csv"), sha(f"{FC}/{D}/pregame_{G3}.csv")
    cache = repo.root / "neurhl/data/tensors/_live_g_2027/stamps_2027.csv"
    cache.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sha": [s1, s3], "stamp_utc": [f"{D}T16:10:00+00:00", f"{D}T22:41:00+00:00"],
                  "checked_utc": [f"{D}T23:59:00+00:00"] * 2}).to_csv(cache, index=False)
    p, card = repo.score("--as-of", "2026-10-02")
    cache.unlink()
    st = card["pregame"]["stamps"] if card else {}
    check("stamp after the start reported (G1), validity unchanged",
          st == {"with_stamp": 2, "stamp_before_start": 1, "stamp_late_game_ids": [G1],
                 "without_stamp": 0} and card["counts"]["valid"]["pregame"] == 2, str(st))
    S, calls = scorer(), []

    def fake(s):
        calls.append(s)
        return pd.Timestamp(f"{D}T22:00:00Z") if s == "a" else None
    now = pd.Timestamp.now(tz="UTC")
    commits = {"a": now - dt.timedelta(hours=1), "b": now - dt.timedelta(hours=1),
               "c": now - dt.timedelta(days=3)}
    (base / "stamps").mkdir()
    got, status = S.gh_stamps(base / "stamps", commits, False, fake)
    again, _ = S.gh_stamps(base / "stamps", commits, False, fake)
    off, _ = S.gh_stamps(base / "stamps", commits, True, fake)
    check("stamp cache: found kept, recent miss asked again, old miss not, offline asks nothing",
          got == again == off == {"a": f"{D}T22:00:00+00:00"} and status == "ok"
          and calls == ["a", "b", "c", "b"], f"{calls} {got}")


def test_fetched(base: Path):
    print("\nPUBLISHED, NOT PULLED (origin/main)")
    gid, rel = 2026020007, f"{FC}/{D}/pregame_2026020007.csv"
    row = pd.DataFrame([fc(gid, D, f"{D}T23:00:00Z", 0.57, 0.52, 0.55, 0.50)])
    bot = Repo(base / "bot")                       # the publishing clone
    bot.write(rel, row)
    bot.commit("2026-10-01T22:05:00+00:00", rel)
    here = Repo(base / "here")
    (here.root / "README").write_text("x\n")
    here.commit("2026-09-01T00:00:00+00:00", "README")
    here.write(rel, row)                           # untracked, as forecast.py leaves it
    here.write("neurhl/output/live/results_2027.csv", results([(gid, D, 3, 1, "REG")]))
    p, card = here.score("--as-of", "2026-10-02")
    check("before the fetch: the untracked copy does not count",
          p.returncode == 0 and card["missed_game_ids"]["pregame"] == [gid]
          and card["counts"]["missed_by_reason"]["pregame"] == {"uncommitted": 1})
    here.git("remote", "add", "origin", str(bot.root))
    here.git("fetch", "-q", "origin")
    p, card = here.score("--as-of", "2026-10-02")
    check("after git fetch: the published commit on origin/main counts",
          p.returncode == 0 and card["counts"]["valid"]["pregame"] == 1
          and not card["files"]["uncommitted"], str(card["counts"]["valid"]))


def test_bootstrap(base: Path):
    print("\nBOOTSTRAP")
    repo = Repo(base / "boot")
    rng = np.random.default_rng(5)
    days = pd.date_range("2026-10-05", "2026-11-01")               # four ISO weeks
    days = [d for d in days if d.dayofweek < 5]
    rows, res = [], []
    for i in range(40):
        day = days[i // 2].strftime("%Y-%m-%d")
        gid = 2026020101 + i
        g, e = rng.uniform(0.3, 0.7, 2)
        rows.append((day, fc(gid, day, f"{day}T23:00:00Z", round(g, 4), round(e, 4),
                             np.nan if i % 7 == 0 else round((g + e) / 2, 4), 0.5)))
        hg = int(rng.integers(0, 6))                               # 5 goals: never a tie
        res.append((gid, day, hg, 5 - hg, "REG"))
    rels = [repo.write(f"{FC}/{d}/pregame_{r['game_id']}.csv", pd.DataFrame([r]))
            for d, r in rows]
    repo.commit("2026-10-05T00:00:00+00:00", *rels)
    repo.write("neurhl/output/live/results_2027.csv", results(res))
    p1, c1 = repo.score("--as-of", "2026-11-02")
    t1 = (repo.root / CARD).read_text()
    p2, c2 = repo.score("--as-of", "2026-11-02")
    t2 = (repo.root / CARD).read_text()
    check("scorer exits 0 twice, nothing on stderr", p1.returncode == 0 and p2.returncode == 0
          and not (p1.stderr + p2.stderr).strip(), p1.stderr[-400:])
    ci = c1["pregame"]["paired"]["g_minus_elo"] if c1 else {}
    check("40 games in 4 ISO weeks: interval computed", ci.get("weeks") == 4
          and ci.get("ci95") is not None and ci["ci95"][0] <= ci["ci95"][1], str(ci))
    check("two scorer runs give identical scorecards", t1 == t2)
    y = np.array([1.0 if r[2] > r[3] else 0.0 for r in res])
    pg = np.array([r["p_home_win_neurhl_g"] for _, r in rows])
    pe = np.array([r["p_home_win_elo"] for _, r in rows])
    d = -(y * np.log(pg) + (1 - y) * np.log(1 - pg)) + (y * np.log(pe) + (1 - y) * np.log(1 - pe))
    check("paired mean matches the per-game differences", close(ci.get("log_loss"), d.mean()))
    h_n = c1["pregame"]["paired"]["g_minus_h"]["n"] if c1 else None
    check("G - H pairs only the games with NeurHL-H (34 of 40)", h_n == 34, str(h_n))
    S = scorer()
    # 20 weeks, so the resampling distribution is fine enough for seeds to differ
    x, wk = np.random.default_rng(9).normal(size=200), np.repeat(np.arange(20), 10)
    a, b = S.week_bootstrap([x], wk), S.week_bootstrap([x], wk)
    check("week_bootstrap: same seed, same interval; another seed differs",
          a == b and a != S.week_bootstrap([x], wk, seed=712), f"{a}")


def test_empty(base: Path):
    print("\nZERO GAMES")
    repo = Repo(base / "empty")                    # no commit at all
    repo.write("neurhl/output/live/results_2027.csv", results([]))
    p, card = repo.score("--as-of", "2026-09-29")
    ok = p.returncode == 0 and card is not None and card["games_completed"] == 0 \
        and card["counts"]["valid"] == {"pregame": 0, "morning": 0} \
        and all(k in card for k in ("as_of", "counts", "pregame", "morning", "missed_game_ids"))
    check("header-only results: exit 0, valid scorecard, games_completed 0", ok,
          "" if ok else (p.stdout + p.stderr)[-400:])
    (repo.root / "neurhl/output/live/results_2027.csv").unlink()
    p, card = repo.score()
    ok = p.returncode == 0 and card is not None and card["games_completed"] == 0
    check("no results file: exit 0, games_completed 0", ok,
          "" if ok else (p.stdout + p.stderr)[-400:])


def fingerprint() -> dict:
    """The scorer's outputs in the real repository, to prove the tests leave them alone."""
    out = {}
    for rel in (CARD, "neurhl/data/tensors/_live_g_2027"):
        p = NRL.parent / rel
        if p.is_file():
            out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
        elif p.is_dir():
            out[rel] = sorted(x.name for x in p.iterdir())
        else:
            out[rel] = None
    return out


def main():
    print("score_live_g_2027 tests")
    before = fingerprint()
    base = Path(tempfile.mkdtemp(prefix="score_live_g_"))
    try:
        repo = test_validity_pairing_sheet(base)
        test_api_starts(repo)
        test_stamps(repo, base)
        test_fetched(base)
        test_bootstrap(base)
        test_empty(base)
    finally:
        if os.environ.get("KEEP_TMP"):
            print(f"\nkept {base}")
        else:
            shutil.rmtree(base, ignore_errors=True)
    print("\nISOLATION")
    check("real neurhl/output and caches untouched", fingerprint() == before)
    k = sum(ok for _, ok in RES)
    print(f"\n{k}/{len(RES)} checks pass")
    sys.exit(0 if k == len(RES) else 1)


if __name__ == "__main__":
    main()
