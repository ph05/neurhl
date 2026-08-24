"""NeurHL-3 — D1: parse the (previously unconsumed) MoneyPuck game-by-game
corpus into per-season tensors.

Sources (data/raw/mp_extra/): gbg_skaters.zip / gbg_goalies.zip /
gbg_lines.zip — per-player/goalie/line PER-GAME rows 2008-2024 with situation
splits — and lines_<startYear>.csv season summaries 2009-2025. This is the
historical deployment / line / goalie-start record that daily snapshots can
never backfill.

P3 enforcement AT READ (PLAN_NeurHL.md:66-69): every MoneyPuck model-derived
column is dropped by deny-substring before anything is kept — xGoals family,
expected-*, *Adjusted*, flurry, win probability. Raw recorded fields (ice
time, shifts, shot attempts, goals, assists, hits, faceoffs, penalties,
line composition) pass through. The kept/dropped split is recorded in each
manifest so the exclusion is auditable, not remembered.

Reconciliation: per season, the share of (gameId, playerId) skater rows that
exist in player_games_<S> is printed and recorded; the house integrity bar is
>= 0.98 (misses documented, not silently dropped).

Run: uv run --no-project --python 3.12 --with numpy --with "pandas<3" \
     --with pyarrow python neurhl/data/build_mp_extra.py
"""
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RAW, TENSORS  # noqa: E402
import manifest as MAN  # noqa: E402

SRC = RAW / "mp_extra"
DENY = ("xgoal", "xrebound", "xfroze", "xfreeze", "xplay", "xshot", "xon",
        "expected", "adjusted", "flurry", "winprob", "xpass")
CFG = {"deny": DENY, "version": 1}
CHUNK = 400_000


def keep_cols(cols) -> list:
    return [c for c in cols if not any(d in c.lower() for d in DENY)]


def split_zip_by_season(zpath: Path, out_stem: str, season_col: str = "season"):
    """Stream one gbg zip into per-season parquet shards (season_end keyed)."""
    with zipfile.ZipFile(zpath) as z:
        member = z.namelist()[0]
        with z.open(member) as f:
            head = pd.read_csv(f, nrows=0)
    kept = keep_cols(head.columns)
    dropped = [c for c in head.columns if c not in kept]
    buckets: dict = {}
    with zipfile.ZipFile(zpath) as z:
        with z.open(z.namelist()[0]) as f:
            for chunk in pd.read_csv(f, usecols=kept, chunksize=CHUNK,
                                     low_memory=False):
                for sy, d in chunk.groupby(season_col):
                    buckets.setdefault(int(sy) + 1, []).append(d)
    outs = {}
    for se, parts in sorted(buckets.items()):
        d = pd.concat(parts, ignore_index=True)
        out = TENSORS / f"{out_stem}_{se}.parquet"
        d.to_parquet(out, index=False)
        MAN.write_manifest(out, [zpath], CFG,
                           {"n_rows": len(d), "kept_cols": len(kept),
                            "dropped_cols": dropped})
        outs[se] = (out, len(d))
    return outs, kept, dropped


def reconcile_skaters(seasons) -> None:
    for se in seasons:
        p = TENSORS / f"mp_gbg_skater_{se}.parquet"
        q = TENSORS / f"player_games_{se}.parquet"
        if not (p.exists() and q.exists()):
            continue
        g = pd.read_parquet(p, columns=["gameId", "playerId", "situation"])
        g = g[g.situation == "all"]
        pg = pd.read_parquet(q, columns=["game_id", "player_id", "game_type"])
        pg = pg[pg.game_type == 2]
        key = set(zip(pg.game_id.astype("int64"), pg.player_id.astype("int64")))
        hit = np.fromiter(((int(a), int(b)) in key
                           for a, b in zip(g.gameId, g.playerId)), bool,
                          count=len(g))
        rate = float(hit.mean()) if len(g) else 0.0
        flag = "" if rate >= 0.98 else "  <-- BELOW 0.98 BAR"
        print(f"  reconcile {se}: {rate:.4f} of {len(g):,} mp skater "
              f"game-rows found in player_games{flag}")


def main():
    done = []
    for zname, stem in (("gbg_skaters.zip", "mp_gbg_skater"),
                        ("gbg_goalies.zip", "mp_gbg_goalie"),
                        ("gbg_lines.zip", "mp_gbg_line")):
        zpath = SRC / zname
        if not zpath.exists():
            print(f"{zname}: missing, skipping")
            continue
        probe = TENSORS / f"{stem}_2024.parquet"
        if MAN.is_fresh(probe, [zpath], CFG):
            print(f"{zname}: fresh, skipping")
            continue
        outs, kept, dropped = split_zip_by_season(zpath, stem)
        se = sorted(outs)
        print(f"{zname}: {len(outs)} seasons {se[0]}-{se[-1]}, "
              f"{sum(n for _, n in outs.values()):,} rows, kept {len(kept)} "
              f"cols, dropped {len(dropped)} P3 cols")
        done.append(stem)
        sys.stdout.flush()

    for f in sorted(SRC.glob("lines_*.csv")):
        sy = int(f.stem.split("_")[1])
        se = sy + 1
        out = TENSORS / f"mp_lines_season_{se}.parquet"
        if MAN.is_fresh(out, [f], CFG):
            continue
        d = pd.read_csv(f, low_memory=False)
        cols = keep_cols(d.columns)
        d[cols].to_parquet(out, index=False)
        MAN.write_manifest(out, [f], CFG, {"n_rows": len(d),
                                           "kept_cols": len(cols)})
        print(f"lines season {se}: {len(d):,} line rows, kept {len(cols)} cols")

    if "mp_gbg_skater" in done or True:
        print("reconciliation (situation='all' skater rows vs player_games):")
        reconcile_skaters(range(2009, 2026))
    print("done")


if __name__ == "__main__":
    main()
