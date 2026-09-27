"""Causality audit for the NeurHL-G state builders (PLAN_NeurHL4, section Q).

The A8 standard: corrupt every outcome after a cut date, rebuild, and require
every feature read on or before the cut to be bit-identical. A feature that
moves has read the future.

Checks, on the real tables:
  1. skater state: cut 2016-01-15 (mid-season)
  2. skater state: cut at the 2024/2025 boundary (sealed seasons corrupted)
  3. goalie state and team state: cut 2016-01-15
Writes configs/leakage_audit_g.json; exits nonzero on any failure.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import data.build_g_state as B  # noqa: E402


def corrupt(df, cols, date_col, cut, seed=7):
    rng = np.random.default_rng(seed)
    out = df.copy()
    m = out[date_col] > cut
    for c in cols:
        out.loc[m, c] = rng.random(int(m.sum())) * 50
    return out


def compare(name, a, b, keys, date_col, cut, results):
    feat = [c for c in a.columns if c not in keys and a[c].dtype.kind == "f"]
    aa = a[a[date_col] <= cut].set_index(keys)[feat].sort_index()
    bb = b[b[date_col] <= cut].set_index(keys)[feat].sort_index()
    same = aa.shape == bb.shape and np.allclose(aa.to_numpy(), bb.to_numpy(),
                                                 equal_nan=True, rtol=0, atol=0)
    results[name] = {"rows_checked": int(len(aa)), "features": len(feat),
                     "identical": bool(same)}
    print(f"  [{'PASS' if same else 'FAIL'}] {name}: {len(aa):,} rows x {len(feat)} features")


def main():
    dates = B.game_dates()
    res = {}
    base = B.skater_rows(dates)
    full = B.add_state(base, B.SK_X, B.SK_RATES, B.SK_TOI)
    for cut in (pd.Timestamp("2016-01-15"), pd.Timestamp("2024-06-30")):
        bad = corrupt(base, B.SK_X[:-1], "date", cut)
        st = B.add_state(bad, B.SK_X, B.SK_RATES, B.SK_TOI)
        state_cols = [c for c in st.columns if c not in base.columns]
        compare(f"skater state, cut {cut.date()}",
                full[["player_id", "game_id", "date"] + state_cols],
                st[["player_id", "game_id", "date"] + state_cols],
                ["player_id", "game_id"], "date", cut, res)

    # goalie / team: rebuild from corrupted source columns via the same code
    cut = pd.Timestamp("2016-01-15")
    gk_full = B.goalie_state(dates)
    orig = B.goalie_state.__globals__["pd"].read_parquet

    def rp_corrupt(path, *a, **k):
        df = orig(path, *a, **k)
        name = Path(path).name
        if name.startswith(("goalie_games_", "tgx_")) or name.startswith("games_ctx_"):
            if "game_id" in df:
                d = df[["game_id"]].merge(dates[["game_id", "date"]].rename(columns={"date": "_cut_date"}), on="game_id", how="left")
                late = (d["_cut_date"] > cut).to_numpy()
                for c in ("sf", "ga", "xgf", "toi_sec", "xgf_all", "xga_all",
                          "sogf", "soga", "home_g", "away_g", "pp_opps", "xgf_ev"):
                    if c in df:
                        df.loc[late, c] = 99
        return df

    B.pd.read_parquet = rp_corrupt
    try:
        gk_bad = B.goalie_state(dates)
        tm_bad = B.team_state(B.game_dates())
    finally:
        B.pd.read_parquet = orig
    tm_full = B.team_state(dates)
    gcols = [c for c in gk_full.columns if c.startswith("gk_")]
    compare("goalie state", gk_full[["player_id", "game_id", "date"] + gcols],
            gk_bad[["player_id", "game_id", "date"] + gcols],
            ["player_id", "game_id"], "date", cut, res)
    tcols = [c for c in tm_full.columns if c.startswith("tm_")]
    compare("team state", tm_full[["team", "game_id", "date"] + tcols],
            tm_bad[["team", "game_id", "date"] + tcols],
            ["team", "game_id"], "date", cut, res)

    ok = all(v["identical"] for v in res.values())
    (ROOT / "configs" / "leakage_audit_g.json").write_text(
        json.dumps({"pass": ok, "checks": res}, indent=1))
    print("ALL CLEAN" if ok else "LEAK FOUND")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
