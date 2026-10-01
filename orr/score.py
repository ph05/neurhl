"""Scorecard: ORR against every NeurHL release on the 2026-27 season.

Scores whatever has been played so far (and the full season at the end):

- games: log loss, Brier and accuracy of P(home win), on the preseason files
  of every model (ORR; NeurHL 09-25 freeze, 1.0, 1.1, 1.2, 1.3) and on the
  game-day forecasts (ORR in-season; NeurHL-G/H pregame).
  Timing is reported, not assumed: NeurHL 1.0 and 1.1 were committed before
  the first puck drop; ORR's preseason file uses only pre-cutoff
  information but was BUILT on 2026-09-30/10-01, and NeurHL 1.2/1.3 were
  published after games had started. Each file is therefore also scored on the
  games played after its own publication time ("after_publication").
- teams (interim): actual points vs each model's expected points FOR THE
  GAMES PLAYED (sum of the per-game expected points), which is a fair interim
  measure; at season end, points MAE/RMSE and CRPS.
- skaters (season end): points, goals and assists MAE over every projected
  player, and over the >=40 GP subset NeurHL reports.

Results come from a CSV (game_id, date, home, away, home_g, away_g,
last_period in REG/OT/SO); by default NeurHL's own results file, which is
only ever read for scoring.

Run: python3 -m orr.score [--results path] [--out path]
"""
from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from orr import config as C

RESULTS_DEFAULT = C.OUT / "live" / "results_2027.csv"
FREEZE = C.OUT / "freeze_2027"

# (path, probability column, publication time UTC: the commit that first added
# the file; ORR's from its own state file)
GAME_FILES = {
    "orr_preseason": (FREEZE / "games_2027.csv", "p_home_win", None),
    "neurhl_0925_freeze": (C.ROOT / "neurhl/output/games_2027.csv", "p_home_win", "2026-09-25T00:00:00Z"),
    "neurhl_1.0": (C.ROOT / "neurhl/output/neurhl_1_0/games_2027.csv", "p_home_win", "2026-09-29T02:12:11Z"),
    "neurhl_1.1": (C.ROOT / "neurhl/output/neurhl_1_1/season/games_2027.csv", "p_home_win", "2026-09-29T20:53:49Z"),
    "neurhl_1.2": (C.ROOT / "neurhl/output/neurhl_1_2/season/games_2027.csv", "p_home_win", "2026-09-30T01:41:45Z"),
    "neurhl_1.3": (C.ROOT / "neurhl/output/neurhl_1_3/season/games_2027.csv", "p_home_win", "2026-09-30T21:06:08Z"),
}
FIRST_PUCK_DROP = pd.Timestamp("2026-09-29T21:00:00Z")
LIVE_DEADLINE = pd.Timedelta(hours=15)
# NeurHL's committed live forecasts. ORR 1.6: --neurhl-live points this at an
# extract of upstream's files (git archive origin/main neurhl/output/live).
NEURHL_LIVE = C.ROOT / "neurhl/output/live/2027"


@functools.lru_cache(maxsize=1)
def _first_start_by_date() -> dict:
    """Earliest scheduled regular-season start (UTC, naive) per date, from
    the NHL club schedules in data/raw (e.g. a Global Series game in Europe
    starts at 13:00 UTC)."""
    starts = {}
    for f in sorted((C.ROOT / "data" / "raw").glob("nhl_sched_*_20262027.json")):
        for g in json.loads(f.read_text()).get("games", []):
            if g.get("gameType") == 2 and g.get("startTimeUTC"):
                t = pd.Timestamp(g["startTimeUTC"]).tz_convert(None)
                d = pd.Timestamp(g["gameDate"])
                starts[d] = min(starts.get(d, t), t)
    return starts


@functools.lru_cache(maxsize=1)
def _start_by_game() -> dict:
    """Scheduled start (UTC, naive) of every 2026-27 regular-season game."""
    out = {}
    for f in sorted((C.ROOT / "data" / "raw").glob("nhl_sched_*_20262027.json")):
        for g in json.loads(f.read_text()).get("games", []):
            if g.get("gameType") == 2 and g.get("startTimeUTC"):
                out[int(g["id"])] = pd.Timestamp(g["startTimeUTC"]).tz_convert(None)
    return out


def _game_deadline(gid, date) -> pd.Timestamp:
    """A forecast counts for a game when published before its scheduled
    puck drop (the rule NeurHL's pregame forecasts also follow); without a
    known start time, the date-level rule below."""
    return _start_by_game().get(int(gid), _deadline(date))


def _deadline(date) -> pd.Timestamp:
    """Latest publication (UTC, naive) that still precedes every game on
    `date`: the earlier of 15:00 UTC and that date's first scheduled start."""
    d = pd.Timestamp(date)
    first = FIRST_PUCK_DROP.tz_convert(None)
    if d == first.normalize():
        return first
    return min(d + LIVE_DEADLINE, _first_start_by_date().get(d, d + LIVE_DEADLINE))
# A game counts as "after publication" when the file was published before
# the game's scheduled puck drop (_game_deadline).
TEAM_FILES = {
    "orr_preseason": FREEZE / "teams_2027.csv",
    "neurhl_1.0": C.ROOT / "neurhl/output/neurhl_1_0/teams_2027.csv",
    "neurhl_1.1": C.ROOT / "neurhl/output/neurhl_1_1/season/teams_2027.csv",
    "neurhl_1.2": C.ROOT / "neurhl/output/neurhl_1_2/season/teams_2027.csv",
    "neurhl_1.3": C.ROOT / "neurhl/output/neurhl_1_3/season/teams_2027.csv",
}


def load_results(path=RESULTS_DEFAULT) -> pd.DataFrame:
    r = pd.read_csv(path)
    r["home_win"] = (r.home_g > r.away_g).astype(int)
    r["extra"] = r.last_period.map({"REG": "REG", "OT": "OT", "SO": "SO"}).fillna("REG")
    return r


def _ll(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _published(name, pub):
    if pub is not None:
        return pd.Timestamp(pub)
    st = FREEZE / "state_2027.json"
    return pd.Timestamp(json.loads(st.read_text())["created_utc"]) if st.exists() else None


def _metrics(p, y):
    return {"n": int(len(y)), "log_loss": _ll(p, y), "brier": float(((p - y) ** 2).mean()),
            "accuracy": float(((p > 0.5) == y).mean())}


def score_games(res: pd.DataFrame) -> dict:
    out = {}
    y = res.set_index("game_id").home_win
    dates = pd.to_datetime(res.set_index("game_id").date)
    for name, (path, col, pub) in GAME_FILES.items():
        if not Path(path).exists():
            continue
        g = pd.read_csv(path).set_index("game_id")
        common = y.index.intersection(g.index)
        if not len(common):
            continue
        published = _published(name, pub)
        row = {**_metrics(g.loc[common, col].to_numpy(float), y.loc[common].to_numpy()),
               "published_utc": str(published),
               "published_before_first_game": bool(published is not None and published < FIRST_PUCK_DROP)}
        if published is not None:
            later = [gid for gid in common
                     if published.tz_convert(None) < _game_deadline(gid, dates[gid])]
            row["after_publication"] = (_metrics(g.loc[later, col].to_numpy(float), y.loc[later].to_numpy())
                                        if later else {"n": 0})
        out[name] = row
    # game-day forecasts
    live = NEURHL_LIVE
    rows = []
    for f in sorted(live.glob("*/pregame_*.csv")):
        if f.name.endswith(("_players.csv", "_lineups.csv")):
            continue
        d = pd.read_csv(f)
        rows.append(d[["game_id", "p_home_win_neurhl_g", "p_home_win_elo",
                       "p_home_win_neurhl_h"]])
    if rows:
        d = pd.concat(rows).drop_duplicates("game_id").set_index("game_id")
        common = y.index.intersection(d.index)
        for col, name in (("p_home_win_neurhl_g", "neurhl_G_pregame"),
                          ("p_home_win_neurhl_h", "neurhl_H_pregame"),
                          ("p_home_win_elo", "elo_pregame")):
            p = d.loc[common, col].to_numpy(float)
            yy = y.loc[common].to_numpy()
            out[name] = {"n": int(len(common)), "log_loss": _ll(p, yy),
                         "brier": float(((p - yy) ** 2).mean()),
                         "accuracy": float(((p > 0.5) == yy).mean())}
    hl = C.OUT / "live"
    if hl.exists():
        rows = [pd.read_csv(f) for f in sorted(hl.glob("*/games_*.csv"))]
        if rows:
            d = pd.concat(rows)
            # a forecast counts only if created before its game date's deadline
            # (_deadline); created_utc is the file's own stamp, and the pushed
            # commit time (git log) is the external evidence that it held
            d = d[pd.to_datetime(d.created_utc).dt.tz_convert(None)
                  < pd.Series([_game_deadline(i, x) for i, x in zip(d.game_id, d.date)], index=d.index)]
            # the LATEST forecast made before the deadline counts (all candidates
            # precede the game, so this is the best-informed pre-game forecast)
            d = d.sort_values("created_utc").drop_duplicates("game_id", keep="last").set_index("game_id")
            common = y.index.intersection(d.index)
            if len(common):
                p = d.loc[common, "p_home_win"].to_numpy(float)
                yy = y.loc[common].to_numpy()
                out["orr_inseason"] = {"n": int(len(common)), "log_loss": _ll(p, yy),
                                            "brier": float(((p - yy) ** 2).mean()),
                                            "accuracy": float(((p > 0.5) == yy).mean())}
    return out


def eligible_probs(res: pd.DataFrame) -> pd.DataFrame:
    """ORR 1.5: one row per (model, game) with the probability each forecast
    gave a game it is eligible for: preseason files published before the
    game's puck drop, ORR's latest daily forecast created before it, and
    NeurHL's pregame forecasts. Columns: model, game_id, date, p, y."""
    y = res.set_index("game_id").home_win
    dates = res.set_index("game_id").date
    rows = []
    for name, (path, col, pub) in GAME_FILES.items():
        if not Path(path).exists():
            continue
        g = pd.read_csv(path).set_index("game_id")
        published = _published(name, pub)
        for gid in y.index.intersection(g.index):
            if published is None or published.tz_convert(None) < _game_deadline(gid, dates[gid]):
                rows.append((name, gid, dates[gid], float(g.at[gid, col]), int(y[gid])))
    pre = []
    for f in sorted(NEURHL_LIVE.glob("*/pregame_*.csv")):
        if not f.name.endswith(("_players.csv", "_lineups.csv")):
            pre.append(pd.read_csv(f))
    if pre:
        d = pd.concat(pre).drop_duplicates("game_id", keep="last").set_index("game_id")
        for col, name in (("p_home_win_neurhl_g", "neurhl_G_pregame"), ("p_home_win_neurhl_h", "neurhl_H_pregame"),
                          ("p_home_win_elo", "elo_pregame")):
            for gid in y.index.intersection(d.index):
                if pd.notna(d.at[gid, col]):
                    rows.append((name, gid, dates[gid], float(d.at[gid, col]), int(y[gid])))
    hl = C.OUT / "live"
    lf = sorted(hl.glob("*/games_*.csv")) if hl.exists() else []
    if lf:
        d = pd.concat([pd.read_csv(f) for f in lf])
        ok = pd.to_datetime(d.created_utc).dt.tz_convert(None) < pd.Series(
            [_game_deadline(i, x) for i, x in zip(d.game_id, d.date)], index=d.index)
        d = d[ok].sort_values("created_utc").drop_duplicates("game_id", keep="last").set_index("game_id")
        for gid in y.index.intersection(d.index):
            rows.append(("orr_inseason", gid, dates[gid], float(d.at[gid, "p_home_win"]), int(y[gid])))
    return pd.DataFrame(rows, columns=["model", "game_id", "date", "p", "y"])


def running_by_date(ep: pd.DataFrame) -> dict:
    """Cumulative log loss of each model by date, on the games it is eligible for."""
    if not len(ep):
        return {}
    ep = ep.assign(ll=-(ep.y * np.log(ep.p.clip(1e-6, 1 - 1e-6)) + (1 - ep.y) * np.log((1 - ep.p).clip(1e-6, 1 - 1e-6))))
    out = {}
    for m, x in ep.groupby("model"):
        x = x.sort_values("date")
        by = x.groupby("date").agg(n=("ll", "size"), s=("ll", "sum"))
        out[m] = [{"date": str(dt), "n": int(n), "log_loss": float(s / n)}
                  for dt, n, s in zip(by.index, by.n.cumsum(), by.s.cumsum())]
    return out


def reliability(ep: pd.DataFrame, model: str = "orr_inseason") -> list:
    x = ep[ep.model == model]
    if not len(x):
        return []
    bins = [0, 0.35, 0.45, 0.55, 0.65, 1.0]
    x = x.assign(bin=pd.cut(x.p, bins, include_lowest=True))
    return [{"bin": f"{iv.left:.2f}-{iv.right:.2f}", "n": int(len(g)), "mean_p": float(g.p.mean()),
             "home_win_rate": float(g.y.mean())} for iv, g in x.groupby("bin", observed=True)]


def _team_points_so_far(res):
    rows = []
    for side, other, sign in (("home", "away", 1), ("away", "home", -1)):
        win = (res.home_g > res.away_g) if sign == 1 else (res.away_g > res.home_g)
        pts = np.where(win, 2, np.where(res.extra != "REG", 1, 0))
        rows.append(pd.DataFrame({"team": res[side], "pts": pts, "gp": 1}))
    return pd.concat(rows).groupby("team").sum()


def score_teams_interim(res: pd.DataFrame) -> dict:
    """Actual points so far vs each model's expected points in those games."""
    out = {}
    for name, (path, col, pub_s) in GAME_FILES.items():
        if not Path(path).exists():
            continue
        g = pd.read_csv(path)
        g = g[g.game_id.isin(res.game_id)]
        published = _published(name, pub_s)
        if published is not None:      # only games that started after publication
            pub = published.tz_convert(None)
            g = g[[pub < _game_deadline(i, d) for i, d in zip(g.game_id, g.date)]]
        if not len(g):
            out[name] = {"teams": 0, "note": "no games after publication"}
            continue
        if {"p_home_reg", "p_away_reg"}.issubset(g.columns):
            p_tie = 1 - g.p_home_reg - g.p_away_reg
            e_h = 2 * g.p_home_win + (p_tie - (g.p_home_win - g.p_home_reg))
            e_a = 2 * (1 - g.p_home_win) + (g.p_home_win - g.p_home_reg)
        else:
            e_h = 2 * g.p_home_win + 0.11
            e_a = 2 * (1 - g.p_home_win) + 0.11
        e = pd.concat([pd.Series(e_h.to_numpy(), index=g.home),
                       pd.Series(e_a.to_numpy(), index=g.away)]).groupby(level=0).sum()
        act = _team_points_so_far(res[res.game_id.isin(g.game_id)])
        d = act.pts - e.reindex(act.index)
        out[name] = {"teams": int(d.notna().sum()), "mae_points_so_far": float(d.abs().mean()),
                     "rmse_points_so_far": float(np.sqrt((d ** 2).mean()))}
    return out


def crps_normal(mu, sd, x):
    from scipy.stats import norm
    z = (x - mu) / sd
    return sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def score_teams_final(res: pd.DataFrame) -> dict:
    act = _team_points_so_far(res)
    if act.gp.min() < C.GAMES_PER_TEAM[C.TARGET_SEASON]:
        return {"status": "season not complete"}
    out = {}
    for name, path in TEAM_FILES.items():
        if not Path(path).exists():
            continue
        t = pd.read_csv(path).set_index("team")
        mu = t.points.reindex(act.index)
        sd = t["points_sd"] if "points_sd" in t else (t.points_p90 - t.points_p10) / 2.563
        e = act.pts - mu
        out[name] = {"mae": float(e.abs().mean()), "rmse": float(np.sqrt((e ** 2).mean())),
                     "crps": float(crps_normal(mu, sd.reindex(act.index), act.pts).mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS_DEFAULT))
    ap.add_argument("--out", default=str(C.OUT / "scorecard_2027.json"))
    ap.add_argument("--neurhl-live", default=None,
                    help="NeurHL's live/2027 folder (default: this checkout's neurhl/output/live/2027)")
    a = ap.parse_args()
    global NEURHL_LIVE
    if a.neurhl_live:
        NEURHL_LIVE = Path(a.neurhl_live)
    res = load_results(a.results)
    ep = eligible_probs(res)
    card = {"through": str(res.date.max()), "games_played": int(len(res)),
            "games": score_games(res), "teams_interim": score_teams_interim(res),
            "running": running_by_date(ep), "reliability": reliability(ep),
            "teams_final": score_teams_final(res)}
    Path(a.out).write_text(json.dumps(card, indent=1))
    g = pd.DataFrame(card["games"]).T
    print(f"through {card['through']}: {card['games_played']} games")
    print(g.drop(columns=[c for c in ("after_publication",) if c in g]).to_string())
    print(pd.DataFrame(card["teams_interim"]).T.to_string())


if __name__ == "__main__":
    main()
