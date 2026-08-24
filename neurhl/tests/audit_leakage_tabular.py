"""NeurHL-3 — standing requirement Q: the A8 causality standard applied to
every tabular D-stage builder and the player-game feature frame.

Method (decisive, as A8): copy one season's source tensors into a sandbox,
CORRUPT every source row belonging to games after a cut date (values scrambled
by a fixed-seed rng — the rows stay present, so positional accidents are
caught too), rebuild the derived artifact from the sandbox, and require every
derived row for games at or before the cut to be BIT-IDENTICAL to the
full-corpus build. A builder that reads the future in any way moves.

Covers: build_absences, build_usage, build_goalie_games, and
models.player_game.build_frames (all feature columns). Writes
configs/leakage_audit_tabular.json; exits nonzero on any failure.

Run: uv run --no-project --python 3.12 --with numpy --with "pandas<3" \
     --with pyarrow python neurhl/tests/audit_leakage_tabular.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import TENSORS  # noqa: E402

SEASON = 2016
PREV = 2015
RNG = np.random.default_rng(20260824)

SRC_FILES = [f"player_games_{PREV}.parquet", f"player_games_{SEASON}.parquet",
             f"games_ctx_{PREV}.parquet", f"games_ctx_{SEASON}.parquet",
             f"stints_{PREV}.parquet", f"stints_{SEASON}.parquet",
             f"events_{PREV}.parquet", f"events_{SEASON}.parquet",
             f"xg_shots_{PREV}.parquet", f"xg_shots_{SEASON}.parquet",
             f"usage_{PREV}.parquet", f"usage_{SEASON}.parquet",
             f"absences_{PREV}.parquet", f"absences_{SEASON}.parquet",
             f"goalie_games_{PREV}.parquet", f"goalie_games_{SEASON}.parquet",
             "maps.json", "careers.parquet", "career_bios.parquet"]


def cut_date_and_games(sand: Path):
    gc = pd.read_parquet(sand / f"games_ctx_{SEASON}.parquet")
    gc = gc[gc.game_type == 2].copy()
    gc["date"] = pd.to_datetime(gc.date)
    cut = gc.date.quantile(0.5)
    future = set(gc[gc.date > cut].game_id.tolist())
    return cut, future


def corrupt(sand: Path, future: set):
    """Scramble every outcome-bearing source value in future games."""
    def scramble(f, cols):
        p = sand / f
        if not p.exists():
            return
        d = pd.read_parquet(p)
        m = d.game_id.isin(future)
        for c in cols:
            if c in d.columns and m.any():
                vals = d.loc[m, c].to_numpy()
                d.loc[m, c] = RNG.permutation(vals)
                if np.issubdtype(d[c].dtype, np.number):
                    noise = RNG.integers(0, 3, size=int(m.sum()))
                    d.loc[m, c] = (d.loc[m, c].to_numpy()
                                   + noise.astype(d[c].dtype)
                                   if np.issubdtype(d[c].dtype, np.integer)
                                   else d.loc[m, c].to_numpy() + noise)
        d.to_parquet(p, index=False)
    scramble(f"player_games_{SEASON}.parquet",
             ["toi_sec", "goals", "assists", "sog", "att", "goalie_start"])
    scramble(f"games_ctx_{SEASON}.parquet", ["home_g", "away_g", "outcome4"])
    scramble(f"stints_{SEASON}.parquet", ["dur_s", "strength_key"])
    scramble(f"events_{SEASON}.parquet", ["p1", "p2", "p3", "goalie"])
    scramble(f"xg_shots_{SEASON}.parquet", ["xg", "goal"])


def compare(name, full: pd.DataFrame, sand: pd.DataFrame, future: set,
            keys: list, results: list):
    f = full[~full.game_id.isin(future)].sort_values(keys, kind="stable") \
        .reset_index(drop=True)
    s = sand[~sand.game_id.isin(future)].sort_values(keys, kind="stable") \
        .reset_index(drop=True)
    if len(f) != len(s):
        results.append({"artifact": name, "pass": False,
                        "why": f"row count {len(f)} vs {len(s)}"})
        return
    bad = []
    for c in f.columns:
        a, b = f[c], s[c]
        same = (a.equals(b)
                or (np.issubdtype(a.dtype, np.floating)
                    and np.allclose(a.fillna(-9e9), b.fillna(-9e9),
                                    rtol=0, atol=0)))
        if not same:
            bad.append(c)
    results.append({"artifact": name, "n_rows": int(len(f)),
                    "moved_columns": bad, "pass": not bad})


def main():
    results = []
    with tempfile.TemporaryDirectory() as td:
        sand = Path(td) / "tensors"
        sand.mkdir()
        for f in SRC_FILES:
            src = TENSORS / f
            if src.exists():
                shutil.copy(src, sand / f)
        cut, future = cut_date_and_games(sand)
        print(f"cut {cut.date()}  future games corrupted: {len(future)}")
        corrupt(sand, future)

        import data.build_absences as BA
        import data.build_usage as BU
        import data.build_goalie_games as BG
        import models.player_game as PGM
        import sim.game_model as GM

        # rebuild derived artifacts from the corrupted sandbox
        for mod in (BA, BU, BG, PGM, GM):
            mod.TENSORS = sand
        ab_s = BA.build(SEASON)
        us_s = BU.build(SEASON)
        gg_s = BG.build(SEASON, {})
        fr_s = PGM.build_frames([PREV, SEASON])
        for mod in (BA, BU, BG, PGM, GM):
            mod.TENSORS = TENSORS
        ab_f = BA.build(SEASON)
        us_f = BU.build(SEASON)
        gg_f = BG.build(SEASON, {})
        fr_f = PGM.build_frames([PREV, SEASON])

        compare("absences", ab_f, ab_s, future,
                ["game_id", "team", "player_id"], results)
        compare("usage", us_f, us_s, future,
                ["game_id", "player_id"], results)
        gcols = [c for c in gg_f.columns]
        compare("goalie_games", gg_f[gcols], gg_s[gcols], future,
                ["game_id", "player_id"], results)
        feat = ["game_id", "player_id"] + [c for c in PGM.FEATURES
                                           if c in fr_f.columns]
        compare("player_game_frame_features",
                fr_f[fr_f.season_end == SEASON][feat],
                fr_s[fr_s.season_end == SEASON][feat], future,
                ["game_id", "player_id"], results)

    ok = all(r["pass"] for r in results)
    for r in results:
        print(f"  [{'PASS' if r['pass'] else 'FAIL'}] {r['artifact']}"
              + ("" if r["pass"] else f"  moved: {r.get('moved_columns') or r.get('why')}"))
    out = {"season": SEASON, "cut_rule": "median date", "results": results,
           "pass": ok}
    (ROOT / "configs" / "leakage_audit_tabular.json").write_text(
        json.dumps(out, indent=1, default=str))
    print(f"-> configs/leakage_audit_tabular.json  "
          f"{'ALL CLEAN' if ok else 'LEAKAGE DETECTED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
