"""Parity and dry-run tests for neurhl/live/ingest_2027.py.

Everything runs in sandboxes under neurhl/data/tensors/_sandbox_ingest/
(gitignored). A sandbox holds SYMLINKS to the real history tables (seasons
< S, plus maps.json, career_bios, rapm priors) and real files for everything
the driver builds; the driver refuses to write through a symlink, so the real
tables are never touched. Raw files are read from data/raw/ (read only).

  parity   season 2026 built from the raw files (regular season + playoffs,
           as the historical tables were) and compared table by table with
           the real neurhl/data/tensors/*_2026.parquet.
  live     2026 treated as the live season: a results file with the first
           15 days, then the first 30 days (incremental), then an unchanged
           re-run (no-op). The 30-day tables are compared with the real 2026
           tables restricted to those games (walk-forward: they must match),
           then build_g_state runs in the sandbox and its 2026 state rows are
           compared with the real gst_*_2026 rows of those games.
  dryrun   the real 2027 path with an empty results file: must print
           "no completed 2027 games" and exit 0.
  fallback 2026 as the live season with MoneyPuck's file hidden (the 404
           case): --no-api-fallback defers, then degrades without xG; the
           default ingests the same games from the NHL play-by-play
           (nhl_api_shots); when the file appears they are re-ingested from
           MoneyPuck and every table equals the real one.

Usage: uv run --no-project --python 3.12 --with numpy --with "pandas<3" \
         --with pyarrow --with requests --with numba --with scikit-learn \
         --with scipy python neurhl/live/test_ingest_parity.py \
         {dryrun|parity|parity2013|live|deferral|fallback|all}
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

NRL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NRL))
from common import PROJ, TENSORS  # noqa: E402

SANDBOX = TENSORS / "_sandbox_ingest"
DRIVER = NRL / "live" / "ingest_2027.py"
XG_CKPT = SANDBOX / "xg_live_v2026.pkl"
HIST_FAMS = ["events", "games_ctx", "player_games", "usage", "onice_rates",
             "goalie_games", "stream", "pgx", "tgx", "xg_shots", "mp_shots",
             "shifts", "stints", "absences"]
KEYS = {"events": ["game_id", "event_idx"], "shifts": None,
        "stints": ["game_id", "period", "stint_idx"], "games_ctx": ["game_id"],
        "player_games": ["game_id", "player_id"], "mp_shots": ["game_id", "shotID"],
        "xg_shots": ["game_id", "period", "time", "shooter", "is_home", "goal"],
        "goalie_games": ["game_id", "player_id"], "stream": ["game_id", "seq"],
        "pgx": ["game_id", "player_id"], "tgx": ["game_id", "is_home"],
        "usage": ["game_id", "player_id"], "onice_rates": ["game_id", "player_id", "is_home"],
        "absences": ["game_id", "team", "player_id"],
        "gst_sk": ["game_id", "player_id"], "gst_gk": ["game_id", "player_id"],
        "gst_tm": ["game_id", "is_home"]}
ORDER = ["events", "shifts", "stints", "games_ctx", "player_games", "mp_shots",
         "xg_shots", "goalie_games", "stream", "pgx", "tgx", "usage", "onice_rates",
         "absences"]


def make_sandbox(name: str, S: int, later: bool = False) -> Path:
    """Symlink history (seasons < S; with later=True every season != S, so the
    xG freeze table can end in 2026 as train_xg's did)."""
    d = SANDBOX / name
    assert SANDBOX in d.parents and d != TENSORS
    if d.exists():
        shutil.rmtree(d)               # unlinks symlinks; never follows them
    d.mkdir(parents=True)
    n = 0
    for p in TENSORS.glob("*"):
        if p.is_dir():
            continue
        stem = p.name.split(".")[0]
        fam, _, yr = stem.rpartition("_")
        keep = (p.name in ("maps.json", "career_bios.parquet")
                or fam == "rapm_prior"
                or (fam in HIST_FAMS and yr.isdigit() and p.name.endswith(".parquet")
                    and (int(yr) < S or (later and int(yr) != S))))
        if keep:
            (d / p.name).symlink_to(p)
            n += 1
    print(f"sandbox {d.name}: {n} history symlinks (seasons < {S})")
    return d


def run_driver(args: list) -> tuple:
    t0 = time.time()
    r = subprocess.run([sys.executable, str(DRIVER)] + args, cwd=PROJ,
                       capture_output=True, text=True)
    out = r.stdout + r.stderr
    print(out.rstrip())
    return r.returncode, out, time.time() - t0


def compare(fam: str, real: pd.DataFrame, sb: pd.DataFrame, tol=1e-9) -> dict:
    res = {"table": fam, "rows_real": len(real), "rows_sandbox": len(sb)}
    res["cols_equal"] = list(real.columns) == list(sb.columns)
    res["extra_cols"] = {"real": [c for c in real.columns if c not in sb.columns],
                         "sandbox": [c for c in sb.columns if c not in real.columns]}
    common_cols = [c for c in real.columns if c in sb.columns]
    res["dtype_diff"] = {c: f"{real[c].dtype}->{sb[c].dtype}" for c in common_cols
                         if real[c].dtype != sb[c].dtype}
    key = KEYS.get(fam) or common_cols
    r = real[common_cols].sort_values(key, kind="stable").reset_index(drop=True)
    s = sb[common_cols].sort_values(key, kind="stable").reset_index(drop=True)
    if len(r) != len(s):
        kr = set(map(tuple, r[key].to_numpy().tolist()))
        ks = set(map(tuple, s[key].to_numpy().tolist()))
        res["only_real"] = len(kr - ks)
        res["only_sandbox"] = len(ks - kr)
        res["only_real_eg"] = sorted(kr - ks)[:3]
        res["only_sandbox_eg"] = sorted(ks - kr)[:3]
        both = r.merge(s[key], on=key)[key]
        r = r.merge(both, on=key).sort_values(key, kind="stable").reset_index(drop=True)
        s = s.merge(both, on=key).sort_values(key, kind="stable").reset_index(drop=True)
    bad, maxd = {}, 0.0
    for c in common_cols:
        a, b = r[c], s[c]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b) \
                and not pd.api.types.is_bool_dtype(a):
            av, bv = a.to_numpy(float), b.to_numpy(float)
            eq = np.isclose(av, bv, rtol=tol, atol=tol, equal_nan=True)
            if (~eq).any():
                d = np.nanmax(np.abs(av - bv)[~eq]) if np.isfinite(av - bv)[~eq].any() else np.inf
                bad[c] = (int((~eq).sum()), float(d))
                maxd = max(maxd, float(d))
        else:
            eq = (a.astype(str).to_numpy() == b.astype(str).to_numpy())
            if (~eq).any():
                bad[c] = (int((~eq).sum()), None)
    res["mismatch_cols"] = bad
    res["max_abs_diff"] = maxd
    res["match"] = (len(real) == len(sb) and res["cols_equal"] and not bad)
    return res


def report(rows: list) -> None:
    print(f"\n{'table':<14}{'rows real':>11}{'rows sbx':>11}  {'cols':<5}{'match':<7}details")
    for x in rows:
        det = []
        if x["extra_cols"]["real"] or x["extra_cols"]["sandbox"]:
            det.append(f"extra cols {x['extra_cols']}")
        if x["dtype_diff"]:
            det.append(f"dtypes {x['dtype_diff']}")
        if "only_real" in x:
            det.append(f"only_real {x['only_real']} {x['only_real_eg']} "
                       f"only_sbx {x['only_sandbox']} {x['only_sandbox_eg']}")
        if x["mismatch_cols"]:
            det.append("cells " + ", ".join(
                f"{c}:{n}" + (f"(max {d:.3g})" if d is not None else "")
                for c, (n, d) in list(x["mismatch_cols"].items())[:8]))
        print(f"{x['table']:<14}{x['rows_real']:>11,}{x['rows_sandbox']:>11,}  "
              f"{'ok' if x['cols_equal'] else 'DIFF':<5}{'YES' if x['match'] else 'no':<7}"
              + "; ".join(det))


def parity(S: int = 2026) -> list:
    """Season S from raw, compared with the real tables. The xG GBM is frozen
    in the sandbox with the covariate table ending in 2026, as train_xg.main
    built it for every historical vantage (see ingest_2027.freeze_era)."""
    d = make_sandbox(f"parity{S}", S, later=S < 2026)
    ck = SANDBOX / f"xg_live_v{S}.pkl"
    rc, out, sec = run_driver(["--season", str(S), "--tensors", str(d), "--results", "",
                               "--no-fetch", "--include-playoffs", "--xg-ckpt", str(ck),
                               "--xg-era-through", "2026", "--allow-late-freeze"])
    print(f"driver exit {rc} in {sec:.0f}s")
    assert rc == 0
    rows = []
    for fam in ORDER + ["rr"]:
        p = TENSORS / f"{fam}_{S}.parquet"
        if fam == "rr" and not p.exists():
            continue
        rows.append(compare(fam, pd.read_parquet(p), pd.read_parquet(d / p.name)))
    report(rows)
    (SANDBOX / f"parity{S}.json").write_text(json.dumps(rows, indent=1, default=str))
    return rows


def crash_recovery() -> None:
    """A run that dies after the per-game stage must be redone next run."""
    d = SANDBOX / "live2026"
    st = d / "_ingest_2026" / "state.json"
    ev = pd.read_parquet(d / "events_2026.parquet")
    last = int(ev.game_id.max())
    js = json.loads(st.read_text())
    js.update({"dirty": True, "inflight": [last]})
    st.write_text(json.dumps(js))
    before = pd.read_parquet(d / "pgx_2026.parquet")
    rc, out, _ = run_driver(["--season", "2026", "--tensors", str(d), "--no-fetch",
                             "--xg-ckpt", str(XG_CKPT), "--today",
                             "2026-09-27", "--results", str(live_results(30))])
    assert rc == 0 and "did not finish: redoing 1 games" in out
    assert pd.read_parquet(d / "pgx_2026.parquet").equals(before)
    assert not json.loads(st.read_text())["dirty"]
    print("crash recovery OK (dirty run redone, tables identical)")


def live_results(days: int) -> Path:
    gc = pd.read_parquet(TENSORS / "games_ctx_2026.parquet")
    gc = gc[gc.game_type == 2].copy()
    inv = {v: k for k, v in json.loads((TENSORS / "maps.json").read_text())["team"].items()}
    cut = (pd.Timestamp(gc.date.min()) + pd.Timedelta(days=int(days))).strftime("%Y-%m-%d")
    gc = gc[gc.date.astype(str) < cut]
    r = pd.DataFrame({"game_id": gc.game_id, "date": gc.date,
                      "home": gc.home_idx.map(inv), "away": gc.away_idx.map(inv),
                      "home_g": gc.home_g, "away_g": gc.away_g, "last_period": "REG"})
    p = SANDBOX / f"results_2026_first{days}d.csv"
    r.sort_values(["date", "game_id"]).to_csv(p, index=False)
    return p


def live() -> list:
    d = make_sandbox("live2026", 2026)
    # right rail on (files on disk, read only): tgx_2025 carries the official PP
    # counts, so the real tgx_2026 does too
    base = ["--season", "2026", "--tensors", str(d), "--no-fetch",
            "--xg-ckpt", str(XG_CKPT), "--today", "2026-09-27"]
    timings = {}
    for days in (15, 30, 30):
        print(f"\n=== live run: results through day {days} ===")
        rc, out, sec = run_driver(base + ["--results", str(live_results(days))])
        timings.setdefault(days, []).append(round(sec))
        assert rc == 0
    print(f"\nlive run wall times (s): {timings}")
    gids = set(pd.read_csv(live_results(30)).game_id)
    rows = []
    for fam in ORDER:
        real = pd.read_parquet(TENSORS / f"{fam}_2026.parquet")
        real = real[real.game_id.isin(list(gids))]
        sb = pd.read_parquet(d / f"{fam}_2026.parquet")
        rows.append(compare(fam, real, sb))
    report(rows)

    # build_g_state over the sandbox: 2008-2025 history + 30 days of 2026
    print("\n=== build_g_state in the sandbox ===")
    code = (f"import sys; from pathlib import Path; sys.path.insert(0, {str(NRL)!r}); "
            f"import data.build_g_state as BS; BS.TENSORS = Path({str(d)!r}); "
            f"BS.SEASONS = sorted(int(p.stem.split('_')[-1]) for p in "
            f"BS.TENSORS.glob('games_ctx_*.parquet') if p.stem.split('_')[-1].isdigit()); "
            f"print('SEASONS', BS.SEASONS[0], '..', BS.SEASONS[-1]); BS.main()")
    assert not any(p.is_symlink() for p in d.glob("gst_*"))
    t0 = time.time()
    r = subprocess.run([sys.executable, "-c", code], cwd=PROJ, capture_output=True, text=True)
    print(r.stdout.rstrip(), r.stderr.rstrip()[-2000:])
    print(f"build_g_state exit {r.returncode} in {time.time() - t0:.0f}s")
    assert r.returncode == 0
    for fam in ("gst_sk", "gst_gk", "gst_tm"):
        real = pd.read_parquet(TENSORS / f"{fam}_2026.parquet")
        real = real[real.game_id.isin(list(gids))]
        sb = pd.read_parquet(d / f"{fam}_2026.parquet")
        print(f"{fam}_2026 in sandbox: {len(sb):,} rows over "
              f"{sb.game_id.nunique()} games, dates {sb.date.min():%Y-%m-%d}.."
              f"{sb.date.max():%Y-%m-%d}")
        rows.append(compare(fam, real, sb, tol=1e-9))
    report(rows[-3:])
    for p in d.glob("gst_*"):          # keep only the live season's state files
        if not p.name.endswith("_2026.parquet"):
            p.unlink()
    (SANDBOX / "live2026.json").write_text(json.dumps(rows, indent=1, default=str))
    return rows


def deferral() -> None:
    """A game whose shift chart is missing is deferred, then ingested degraded
    once overdue, then re-ingested when the chart appears; the final tables
    equal the real ones for those games."""
    from common import RAW
    d = make_sandbox("defer2026", 2026)
    raw = SANDBOX / "defer2026_raw"
    if raw.exists():
        shutil.rmtree(raw)
    for sub in ("pbp", "right_rail"):     # right rail: tgx_2025 carries official PP counts
        (raw / sub).mkdir(parents=True)
        (raw / sub / "2026").symlink_to(RAW / sub / "2026")
    (raw / "mp_shots").symlink_to(RAW / "mp_shots")
    res = live_results(5)
    r = pd.read_csv(res)
    gx, dx = int(r.game_id.iloc[-1]), str(r.date.iloc[-1])
    (raw / "shifts" / "2026").mkdir(parents=True)
    for g in r.game_id:
        if int(g) != gx:
            (raw / "shifts" / "2026" / f"{g}.json.gz").symlink_to(
                RAW / "shifts" / "2026" / f"{g}.json.gz")
    day = lambda k: (pd.Timestamp(dx) + pd.Timedelta(days=k)).strftime("%Y-%m-%d")  # noqa
    base = ["--season", "2026", "--tensors", str(d), "--raw", str(raw), "--no-fetch",
            "--xg-ckpt", str(XG_CKPT), "--results", str(res)]
    rc, out, _ = run_driver(base + ["--today", day(1)])
    assert rc == 0 and f"deferred {gx}" in out
    assert gx not in set(pd.read_parquet(d / "games_ctx_2026.parquet").game_id)
    rc, out, _ = run_driver(base + ["--today", day(3)])
    assert rc == 0 and f"WARNING degraded ingest {gx}" in out
    pg = pd.read_parquet(d / "player_games_2026.parquet")
    assert pg[pg.game_id == gx].toi_sec.sum() == 0
    (raw / "shifts" / "2026" / f"{gx}.json.gz").symlink_to(
        RAW / "shifts" / "2026" / f"{gx}.json.gz")
    rc, out, _ = run_driver(base + ["--today", day(4)])
    assert rc == 0 and "0 degraded" in out
    gids = set(r.game_id)
    bad = []
    for fam in ORDER:
        real = pd.read_parquet(TENSORS / f"{fam}_2026.parquet")
        c = compare(fam, real[real.game_id.isin(list(gids))],
                    pd.read_parquet(d / f"{fam}_2026.parquet"))
        if not c["match"]:
            bad.append(fam)
    assert not bad, bad
    print(f"deferral OK (game {gx}: deferred, degraded, re-ingested; 14 tables match)")


SHOT_FREE = ["events", "shifts", "stints", "games_ctx", "player_games", "usage",
             "onice_rates", "absences"]


def fallback() -> None:
    """MoneyPuck's in-season file missing for the first 7 days of 2026. The old
    rule (--no-api-fallback) defers the last two days and degrades the rest
    without xG. The fallback ingests every game with shot rows from the
    play-by-play: the tables that do not read shots equal the real ones, the xG
    tables track them (team-game and skater-game xG, goalie xG faced). When the
    file appears every game is re-ingested from MoneyPuck and all 14 tables equal
    the real ones; then a re-run is a no-op."""
    from common import RAW
    d = make_sandbox("fallback2026", 2026)
    raw = SANDBOX / "fallback2026_raw"
    if raw.exists():
        shutil.rmtree(raw)
    for sub in ("pbp", "shifts", "right_rail"):     # no mp_shots: the file is "404"
        (raw / sub).mkdir(parents=True)
        (raw / sub / "2026").symlink_to(RAW / sub / "2026")
    res = live_results(7)
    gids = set(pd.read_csv(res).game_id)
    last = str(pd.read_csv(res).date.max())
    today = (pd.Timestamp(last) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    base = ["--season", "2026", "--tensors", str(d), "--raw", str(raw), "--no-fetch",
            "--xg-ckpt", str(XG_CKPT), "--results", str(res), "--today", today]
    st = d / "_ingest_2026" / "state.json"

    def real(fam):
        t = pd.read_parquet(TENSORS / f"{fam}_2026.parquet")
        return t[t.game_id.isin(list(gids))]

    # 1. old rule: the last two days deferred, older games degraded without xG
    rc, out, _ = run_driver(base + ["--no-api-fallback"])
    js = json.loads(st.read_text())
    assert rc == 0 and "pending:mp" in out and "WARNING degraded ingest" in out
    assert js["degraded"] and set(js["degraded"].values()) == {"mp"}
    assert not js["xg_source"] and not len(pd.read_parquet(d / "xg_shots_2026.parquet"))
    n_old = len(pd.read_parquet(d / "games_ctx_2026.parquet"))

    # 2. fallback: every game in, shot rows from the play-by-play
    rc, out, _ = run_driver(base)
    js = json.loads(st.read_text())
    assert rc == 0 and "NHL API shots for" in out and "0 degraded" in out
    assert set(js["xg_source"]) == {str(g) for g in gids}
    assert set(js["xg_source"].values()) == {"nhl_api"} and not js["pending"]
    assert len(pd.read_parquet(d / "games_ctx_2026.parquet")) == len(gids) > n_old
    bad = [f for f in SHOT_FREE if not compare(f, real(f),
                                               pd.read_parquet(d / f"{f}_2026.parquet"))["match"]]
    assert not bad, bad
    sb = pd.read_parquet(d / "stream_2026.parquet")
    shots = sb[(sb.game_type == 2) & sb.event_type.isin([7, 9, 14])]
    cov = float(shots.has_xg.mean())
    stats = {}
    for fam, key, col in (("tgx", ["game_id", "is_home"], "xgf_all"),
                          ("tgx", ["game_id", "is_home"], "xgf_ev"),
                          ("pgx", ["game_id", "player_id"], "ixg_all"),
                          ("goalie_games", ["game_id", "player_id"], "xgf")):
        j = real(fam)[key + [col]].merge(pd.read_parquet(d / f"{fam}_2026.parquet")[key + [col]],
                                         on=key, suffixes=("_r", "_a"))
        stats[f"{fam}.{col}"] = (float(np.corrcoef(j[col + "_r"], j[col + "_a"])[0, 1]),
                                 float(j[col + "_a"].sum() / j[col + "_r"].sum() - 1))
    print(f"fallback vs real ({len(gids)} games): shots with xG {cov:.2%}; "
          + ", ".join(f"{k} corr {c:.4f} bias {b:+.2%}" for k, (c, b) in stats.items()))
    assert cov > 0.999
    assert all(c >= 0.97 and abs(b) <= 0.03 for c, b in stats.values()), stats

    # 3. MoneyPuck's file appears: re-ingested from it, tables equal the real ones
    (raw / "mp_shots").symlink_to(RAW / "mp_shots")
    rc, out, _ = run_driver(base)
    js = json.loads(st.read_text())
    assert rc == 0 and out.count("from MoneyPuck (was NHL API)") == len(gids)
    assert set(js["xg_source"].values()) == {"moneypuck"} and not js["degraded"]
    bad = [f for f in ORDER if not compare(f, real(f),
                                           pd.read_parquet(d / f"{f}_2026.parquet"))["match"]]
    assert not bad, bad

    # 4. nothing changed: no-op
    rc, out, _ = run_driver(base)
    assert rc == 0 and "no new 2026 games" in out
    print(f"fallback OK ({len(gids)} games: old rule deferred/degraded; NHL API rows, "
          f"{len(SHOT_FREE)} shot-free tables equal; re-ingested from MoneyPuck, "
          f"{len(ORDER)} tables equal; no-op re-run)")


def dryrun() -> None:
    empty = SANDBOX / "results_2027_empty.csv"
    SANDBOX.mkdir(parents=True, exist_ok=True)
    empty.write_text("game_id,date,home,away,home_g,away_g,last_period\n")
    before = {p.name: p.stat().st_mtime_ns for p in TENSORS.glob("*_2027*")}
    rc, out, _ = run_driver(["--results", str(empty), "--no-fetch"])
    after = {p.name: p.stat().st_mtime_ns for p in TENSORS.glob("*_2027*")}
    assert rc == 0 and "no completed 2027 games" in out, (rc, out)
    assert before == after, "dry run touched a 2027 table"
    print(f"dryrun OK (exit {rc}; no 2027 table written)")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("dryrun", "all"):
        dryrun()
    if what in ("parity", "all"):
        parity(2026)
    if what in ("parity2013", "all"):
        parity(2013)                 # right-rail season, excluded game 2012020660
    if what in ("live", "all"):
        live()
        crash_recovery()
    if what in ("deferral", "all"):
        deferral()
    if what in ("fallback", "all"):
        fallback()
