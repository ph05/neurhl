"""NeurHL-2 — unified shift table across both data eras.

Exact on-ice intervals are the substrate for stints, RAPM and the deployment
model. Two disjoint sources cover the corpus, and until now only one was used:

  * **native shift JSON** — `api.nhle.com/stats/rest/en/shiftcharts`, season_end
    **2011-2026**. Verified against the API: gameId 2009020001 returns 0 records,
    2010020001 returns 760, so 2010-11 is genuinely the first covered season.
  * **HTM TH/TV reports** — `data/raw/htm_reports/<season_end>/`, season_end
    **2008-2012**, 1,230 games/season, on disk and previously parsed only for
    the jersey->playerId map. `build_htm.parse_toi` already extracts exact shift
    times from them; `tensorize_htm` threw those times away.

Recovering TH/TV adds **season_end 2008, 2009 and 2010 (3,690 games)** of exact
on-ice data the JSON API cannot supply, closing the gap: the union of the two
sources covers **every season in the corpus, 2008-2026**. Seasons 2011-2012 are
covered by BOTH, which is a free parser validation rather than a merge conflict:
`--validate` compares per-player game TOI between the two sources over the
overlap. Native JSON wins where they disagree; `src` records provenance per row.

Note the directory convention, which is a live trap: `htm_reports/` is keyed by
**season_end** while gameIds are prefixed with season_end-1, so
`htm_reports/2009/PL020001.htm.gz` is game 2008020001. Reading it as season_start
silently resolves jerseys against the wrong season's roster — which fails softly,
dropping ~25% of players rather than raising.

Output `neurhl/data/tensors/shifts_<season_end>.parquet`:
    game_id, player_id, is_home, is_goalie, period, t0, t1, src(1=json, 2=htm)
with t0/t1 absolute game seconds, (period-1)*1200 + seconds_into_period.

Run: uv run --no-project --python 3.12 --with numpy --with "pandas<3" \
     --with pyarrow python neurhl/data/build_shifts.py [--seasons 2009 2010]
"""
import argparse
import gzip
import json
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RAW, TENSORS  # noqa: E402
from data.build_htm import NameResolver, onice_abbrevs, parse_pl, parse_toi  # noqa: E402
from data.build_onice import load_shifts  # noqa: E402
import manifest as MAN  # noqa: E402

HTM = RAW / "htm_reports"
SRC_JSON, SRC_HTM = 1, 2
JSON_SEASONS = range(2011, 2027)          # native shiftcharts coverage
HTM_SEASONS = range(2008, 2013)           # htm_reports/<season_end> on disk
CFG = {"version": 1, "max_period": 4}
COLS = ["game_id", "player_id", "is_home", "is_goalie", "period", "t0", "t1",
        "src"]

_RES = None
_GOALIES: set = set()


def _init(season_end, goalies, need_resolver):
    global _RES, _GOALIES
    _GOALIES = goalies
    _RES = NameResolver(season_end) if need_resolver else None


def season_goalies(season_end: int) -> set:
    """Goalie playerIds for a season, from the official goalie summary."""
    out = set()
    for tag in ("", "_po"):
        p = RAW / "nhl_player_reports" / f"goalie_summary_{season_end}{tag}.json.gz"
        if not p.exists():
            continue
        with gzip.open(p, "rt") as f:
            out |= {r["playerId"] for r in json.load(f)}
    return out


# ------------------------------------------------------------------ JSON era
def _json_game(args):
    """One game from native shift JSON -> shift rows."""
    gid, path, home_pids = args
    sh = load_shifts(path, _GOALIES)
    if sh is None:
        return None
    team, pid, is_g, t0, t1 = sh
    # Home side is resolved from the event shard's own home on-ice columns, not
    # from a re-read of raw PBP: whichever teamId most of the known home players
    # belong to. Robust to a stray mis-parsed record and costs no extra IO.
    if home_pids:
        c = Counter(int(t) for t, p in zip(team, pid) if int(p) in home_pids)
        home_team = c.most_common(1)[0][0] if c else int(team[0])
    else:
        home_team = int(team[0])
    is_home = team == home_team
    per = (t0 // 1200) + 1
    keep = per <= CFG["max_period"]
    n = int(keep.sum())
    if not n:
        return None
    return np.column_stack([
        np.full(n, gid, np.int64), pid[keep], is_home[keep].astype(np.int64),
        is_g[keep].astype(np.int64), per[keep], t0[keep], t1[keep],
        np.full(n, SRC_JSON, np.int64)])


# ------------------------------------------------------------------- HTM era
def _htm_game(args):
    """One game from TH/TV reports -> shift rows. TH is home, TV is visitor."""
    gid, pl_path = args
    th_p = pl_path.parent / pl_path.name.replace("PL", "TH")
    tv_p = pl_path.parent / pl_path.name.replace("PL", "TV")
    if not th_p.exists() or not tv_p.exists():
        return None
    try:
        html = _read(pl_path)
        evs = parse_pl(html)
        aab, hab = onice_abbrevs(html)
        th = parse_toi(_read(th_p))
        tv = parse_toi(_read(tv_p))
    except Exception:
        return None
    if not th and not tv:
        return None
    jm = _RES.jersey_map(evs, tv, th, aab, hab)

    rows = []
    for shifts, ab, home in ((th, hab, 1), (tv, aab, 0)):
        for s in shifts:
            if s["period"] > CFG["max_period"]:
                continue
            pid = jm.get((ab, s["jersey"]), 0)
            if not pid:
                continue
            rows.append((gid, pid, home, int(pid in _GOALIES), s["period"],
                         s["t0"], s["t1"], SRC_HTM))
    return np.array(rows, dtype=np.int64) if rows else None


def _read(path: Path) -> str:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        return f.read()


# ---------------------------------------------------------------------- main
def build(season_end: int, prefer: str, workers: int) -> pd.DataFrame:
    """Shift rows for one season from the preferred available source."""
    ev_p = TENSORS / f"events_{season_end}.parquet"
    if not ev_p.exists():
        return pd.DataFrame(columns=COLS)
    hcols = [f"h_on{i}" for i in range(7)]
    ev = pd.read_parquet(ev_p, columns=["game_id"] + hcols)
    home_by_game = {int(g): set(int(v) for v in d[hcols].to_numpy().ravel()
                                if v) for g, d in ev.groupby("game_id")}
    goalies = season_goalies(season_end)

    use_json = prefer == "json" and season_end in JSON_SEASONS
    use_htm = (not use_json) and season_end in HTM_SEASONS
    if not use_json and not use_htm:
        return pd.DataFrame(columns=COLS)

    if use_json:
        d = RAW / "shifts" / str(season_end)
        jobs = [(g, d / f"{g}.json.gz", home_by_game.get(g, set()))
                for g in sorted(home_by_game)
                if (d / f"{g}.json.gz").exists()]
        fn, need_res = _json_game, False
    else:
        d = HTM / str(season_end)
        jobs = [(g, d / f"PL{str(g)[4:]}.htm.gz") for g in sorted(home_by_game)
                if (d / f"PL{str(g)[4:]}.htm.gz").exists()]
        fn, need_res = _htm_game, True
    if not jobs:
        return pd.DataFrame(columns=COLS)

    out = []
    with Pool(workers, initializer=_init,
              initargs=(season_end, goalies, need_res)) as pool:
        for r in pool.imap_unordered(fn, jobs, chunksize=8):
            if r is not None and len(r):
                out.append(r)
    if not out:
        return pd.DataFrame(columns=COLS)
    arr = np.vstack(out)
    df = pd.DataFrame(arr, columns=COLS)
    return df.astype({"game_id": "uint32", "player_id": "uint32",
                      "is_home": "bool", "is_goalie": "bool",
                      "period": "uint8", "t0": "uint16", "t1": "uint16",
                      "src": "uint8"})


def toi_table(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["toi"] = d.t1.astype(int) - d.t0.astype(int)
    return d.groupby(["game_id", "player_id"], as_index=False).toi.sum()


def validate_overlap(season_end: int, workers: int) -> dict:
    """Compare HTM-derived TOI against native JSON on a season both cover.

    This is the only direct check available on the TH/TV parser, and it is a
    strong one: per-player game TOI is an end-to-end function of shift parsing,
    jersey resolution and home/away assignment together.
    """
    j = build(season_end, "json", workers)
    h = build(season_end, "htm", workers)
    if not len(j) or not len(h):
        return {"season": season_end, "ok": False, "why": "a source is empty"}
    tj, th = toi_table(j), toi_table(h)
    m = tj.merge(th, on=["game_id", "player_id"], suffixes=("_j", "_h"))
    both = set(map(tuple, tj[["game_id", "player_id"]].to_numpy())) & \
        set(map(tuple, th[["game_id", "player_id"]].to_numpy()))
    d = (m.toi_h - m.toi_j).abs()
    return {
        "season": season_end, "ok": True,
        "games_json": int(tj.game_id.nunique()), "games_htm": int(th.game_id.nunique()),
        "rows_json": len(tj), "rows_htm": len(th), "matched": len(m),
        "match_rate": round(len(both) / max(len(tj), 1), 4),
        "corr_toi": round(float(np.corrcoef(m.toi_j, m.toi_h)[0, 1]), 5),
        "mae_toi_s": round(float(d.mean()), 2),
        "median_abs_s": float(d.median()),
        "within_10s": round(float((d <= 10).mean()), 4),
        "within_60s": round(float((d <= 60).mean()), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--validate", type=int, nargs="*", default=None,
                    help="overlap seasons to cross-check HTM against JSON")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.validate is not None:
        res = [validate_overlap(s, args.workers)
               for s in (args.validate or [2011, 2012, 2013])]
        print(json.dumps(res, indent=1))
        (TENSORS / "shifts_overlap_validation.json").write_text(
            json.dumps(res, indent=1))
        return

    seasons = args.seasons or sorted(set(JSON_SEASONS) | set(HTM_SEASONS))
    for se in seasons:
        out = TENSORS / f"shifts_{se}.parquet"
        src = TENSORS / f"events_{se}.parquet"
        if not args.force and MAN.is_fresh(out, [src], CFG):
            print(f"{se}: fresh, skipping")
            continue
        prefer = "json" if se in JSON_SEASONS else "htm"
        df = build(se, prefer, args.workers)
        if not len(df):
            print(f"{se}: NO SHIFT SOURCE")
            continue
        df.to_parquet(out, index=False)
        MAN.write_manifest(out, [src], CFG,
                           {"n_shifts": len(df), "n_games": int(df.game_id.nunique()),
                            "source": prefer})
        toi = toi_table(df)
        per_team = (df.assign(toi=df.t1.astype(int) - df.t0.astype(int))
                    .groupby(["game_id", "is_home"]).toi.sum())
        print(f"{se}: {len(df):,} shifts / {df.game_id.nunique()} games "
              f"[{prefer}] | players/game {toi.game_id.value_counts().mean():.1f} "
              f"| team-TOI/game median {per_team.median():.0f}s "
              f"(6 skaters*3600=21600 nominal)")
        sys.stdout.flush()
    print("done")


if __name__ == "__main__":
    main()
