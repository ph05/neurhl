"""Roster change: how much better or worse is this year's roster than last year's?

Team history (hattrick.teams top-down) says what a team was. The market line
says what the consensus thinks it will be. What neither sees fully is how the
roster itself changed. This module evaluates TWO rosters with the SAME
player projections for season V:

  previous roster  every player's most-played team in V-1 (the team that
                   produced last season's results)
  current roster   the opening roster of V (first-10-games proxy in
                   backtests; the 2026-09-29 roster and injury list for 2027)

and reports the difference in the bottom-up components
(hattrick.team_components). Aging, departures, arrivals, injuries and
call-ups all show up in the delta; the player model's level errors largely
cancel because both rosters are projected by the same model.

The same machinery prices the news the 2026-27 market line (recorded
2026-08-17) had not seen: the roster and injury state at the market date vs
at the cutoff.
"""
from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from hattrick import config as C
from hattrick import data as D
from hattrick import team_components as TC
from hattrick.team_points_map import COMPONENTS

OUT = C.OUT / "backtest" / "roster_delta_hist.csv"


def previous_roster(V: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Skaters and goalies on the team they played most for in V-1."""
    sk = D.skater_team_seasons()
    sk = (sk[sk.season_end == V - 1].sort_values("toi", ascending=False)
          .drop_duplicates("player_id")[["player_id", "team"]])
    gl = D.goalie_team_seasons()
    gl = (gl[gl.season_end == V - 1].sort_values("toi", ascending=False)
          .drop_duplicates("player_id")[["player_id", "team"]])
    return sk, gl


@functools.lru_cache(maxsize=None)
def previous_components(V: int, games: int = 82) -> pd.DataFrame:
    sk, gl = previous_roster(V)
    comp, _ = TC.components(V, sk, gl, games, roster_kind="opening")
    return comp


def history(seasons=range(2012, 2027)) -> pd.DataFrame:
    """Delta = opening roster minus previous roster, both projected for V."""
    cur = pd.read_csv(C.OUT / "backtest" / "team_components_hist.csv")
    rows = []
    for V in seasons:
        prev = previous_components(V).set_index("team")
        now = cur[cur.season_end == V].set_index("team")
        d = pd.DataFrame(index=now.index)
        for c in COMPONENTS:
            if c in now and c in prev:
                d[f"d_{c}"] = now[c] - prev[c].reindex(now.index)
        d["season_end"] = V
        rows.append(d.reset_index())
        print(f"{V}: roster delta for {len(d)} teams", flush=True)
    h = pd.concat(rows, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    h.round(5).to_csv(OUT, index=False)
    return h


def to_points(hist_delta: pd.DataFrame, target: int, alpha: float = 8.0) -> dict:
    """Walk-forward ridge: residual of (actual - top-down) on the deltas.

    The delta is meant to explain what team history misses, so it is fitted to
    the top-down view's residual, then used as an additive correction."""
    from hattrick import teams as T
    cols = [c for c in hist_delta.columns if c.startswith("d_")]
    frames = []
    for s in range(2012, target):
        if s in C.BROKEN_SEASONS or s not in set(hist_delta.season_end):
            continue
        f = T.blend_frame([s])
        f = f.merge(hist_delta[hist_delta.season_end == s], on=["team", "season_end"])
        f["resid"] = f.act_rel82 - f.td_rel82
        frames.append(f)
    tr = pd.concat(frames).dropna(subset=cols + ["resid"])
    X = tr[cols].to_numpy(float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    beta = np.linalg.solve(Z.T @ Z + alpha * len(Z) / 100 * np.eye(len(cols)),
                           Z.T @ (tr.resid.to_numpy() - tr.resid.mean()))
    return {"cols": cols, "mu": mu, "sd": sd, "beta": beta, "n": len(tr)}


def apply(fit: dict, delta: pd.DataFrame) -> pd.Series:
    Z = (delta[fit["cols"]].to_numpy(float) - fit["mu"]) / fit["sd"]
    return pd.Series(Z @ fit["beta"], index=delta.team.to_numpy())


def walk_forward(hist_delta: pd.DataFrame, first: int = 2014) -> pd.DataFrame:
    """rd_rel82: the roster-change correction for each season, fitted on
    earlier seasons only."""
    out = []
    for V in sorted(hist_delta.season_end.unique()):
        if V < first:
            continue
        f = to_points(hist_delta, V)
        if f["n"] < 60:
            continue
        d = hist_delta[hist_delta.season_end == V]
        out.append(pd.DataFrame({"team": d.team.to_numpy(), "season_end": V,
                                 "rd_rel82": apply(f, d).to_numpy()}))
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    history()
