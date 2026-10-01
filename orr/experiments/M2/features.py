"""M2 step 1: per-game stacking features, 2012-2024.

For each filter variant (gs = goals and shots, go = goals only) and each
starting prior (ht = team-history table, every season; ship = the shipped
market-anchored preseason ratings, the clean market seasons 2019, 2020,
2022-24, built exactly as orr/backtest/inseason_bt.py builds them):

    p_in   in-season pregame P(home win)      (run_filter + predict_probs)
    p_pre  preseason-frozen P(home win)       (same run, frozen=True)
plus, shared by all rows, p_elo from ratings.elo_run with the frozen
elo_hp.json.

No outcome is scored here. The only check is that the gs/ht probabilities
reproduce the committed games_bt_preds.csv.gz (identity of predictions, not a
score).

Run: python3 -m orr.experiments.M2.features
Writes orr/experiments/M2/features.csv.gz.
"""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from orr import config as C
from orr import ratings as R
from orr import structural as S
from orr.backtest import gamefile_bt as GB

HERE = Path(__file__).resolve().parent
FEAT = HERE / "features.csv.gz"
SEASONS = list(range(2012, 2025))
VIEWS = ["mkt_rel82", "td_rel82"]          # shipped views (inseason_bt.VIEWS)


def elo_hp() -> R.EloHP:
    f = C.PARAMS / "elo_hp.json"
    return R.EloHP(**json.loads(f.read_text())["hp"])


def market_override(hp: R.HP) -> dict:
    """{V: (ratings, mu)} for the clean market seasons, as inseason_bt.main."""
    hist = GB.hist_frame()
    over = {}
    for V in GB.SEASONS:
        P_V = R.fit_gamemodel_params(V, hp, write=False)
        _, lg, r = GB.game_file_season(V, hist, VIEWS, P_V, with_ratings=True)
        over[V] = (r, lg["mu"])
    return over


def build() -> pd.DataFrame:
    t0 = time.time()
    hp0 = R.load_hp()
    over = market_override(hp0)
    print(f"market-anchored priors built ({time.time() - t0:.0f}s)", flush=True)
    g = S.game_frame()[["gid", "game_id", "date", "season_end", "home", "away", "home_win"]]
    elo = R.elo_run(elo_hp())[["gid", "p_elo"]]
    frames = []
    for var, hp in (("gs", hp0), ("go", dataclasses.replace(hp0, use_shots=False))):
        base = R.run_filter(hp, SEASONS, use_goalie=False)
        ot = {}
        pin = R.predict_probs(base, hp, ot_params=ot)
        pfr = R.predict_probs(base, hp, frozen=True, ot_params=ot)
        ship = R.run_filter(hp, GB.SEASONS, use_goalie=False, pre_override=over)
        sin = R.predict_probs(ship, hp, ot_params=dict(ot))
        sfr = R.predict_probs(ship, hp, frozen=True, ot_params=dict(ot))
        for src, a, b, keep in (("ht", pin, pfr, SEASONS), ("ship", sin, sfr, GB.SEASONS)):
            f = (a[["gid", "p_home_win"]].rename(columns={"p_home_win": "p_in"})
                 .merge(b[["gid", "p_home_win"]].rename(columns={"p_home_win": "p_pre"}), on="gid"))
            f = g.merge(f, on="gid").merge(elo, on="gid", how="left")
            f = f[f.season_end.isin(keep)].assign(variant=var, src=src)
            frames.append(f)
        print(f"variant {var} done ({time.time() - t0:.0f}s)", flush=True)
    out = pd.concat(frames, ignore_index=True)
    assert out[["p_in", "p_pre", "p_elo"]].notna().all().all()
    return out


def check_against_games_bt(f: pd.DataFrame) -> dict:
    """The gs/ht in-season, frozen and Elo probabilities must equal the
    committed games_bt predictions (written at 5 decimals)."""
    c = pd.read_csv(C.OUT / "backtest" / "games_bt_preds.csv.gz")
    m = f[(f.variant == "gs") & (f.src == "ht")].merge(c, on="gid", suffixes=("", "_c"))
    return {"n": int(len(m)),
            "max_abs_in": float((m.p_in - m.ht_p_home_win).abs().max()),
            "max_abs_pre": float((m.p_pre - m.ht_frozen_p_home_win).abs().max()),
            "max_abs_elo": float((m.p_elo - m.p_elo_c).abs().max())}


def load() -> pd.DataFrame:
    return pd.read_csv(FEAT, parse_dates=["date"])


def main():
    f = build()
    chk = check_against_games_bt(f)
    print("identity check vs games_bt_preds:", chk)
    f.to_csv(FEAT, index=False, float_format="%.7f")
    (HERE / "features_check.json").write_text(json.dumps(chk, indent=1))
    print(f.groupby(["variant", "src"]).season_end.agg(["min", "max", "size"]))


if __name__ == "__main__":
    main()
