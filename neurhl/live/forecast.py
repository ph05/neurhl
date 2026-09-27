"""NeurHL LIVE forecasts (PLAN_NeurHL4 LIVE): morning and pregame.

  --mode morning   every game today (ET), projected lineups, about 11:00 ET
  --mode pregame   each game starting 45-75 minutes from now that has no
                   pregame file yet, with the latest lineups and goalies
  --mode preview   the evening before a game day (descriptive, not scored)

For each game: lineups from lineup_resolver (NHL API > DailyFaceoff > fallback,
flagged), NeurHL-G inputs through sim/g_live.py (state after every played game),
the NeurHL-G bundle named in configs/live_models.json (weights, training
statistics and stack, all hash-checked), the Monte Carlo stat sheet, the
in-season house Elo reference and the frozen 1.0 preseason probability.

Writes neurhl/output/live/2027/<date>/{morning,pregame_<gid>}.csv, matching
*_players.csv stat lines and *_lineups.json, then publishes through
neurhl/live/publish.sh (bot clone; the push must land before puck drop).

  --date YYYY-MM-DD   override today (ET)     --dry-run   write, do not publish
  --now ISO-UTC       override the clock (rehearsals)
"""
import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJ = ROOT.parent
sys.path.insert(0, str(ROOT))
from common import NOUT  # noqa: E402

ET = ZoneInfo("America/New_York")
LIVE = NOUT / "live" / "2027"
CFG = ROOT / "configs" / "live_models.json"


def _resolver():
    import importlib.util
    p = ROOT / "live" / "lineup_resolver.py"
    spec = importlib.util.spec_from_file_location("lineup_resolver", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["lineup_resolver"] = m
    spec.loader.exec_module(m)
    return m


def names_map() -> dict:
    snaps = sorted((PROJ / "data" / "raw" / "rosters").glob("*/rosters.csv"))
    if not snaps:
        return {}
    r = pd.read_csv(snaps[-1])
    return {int(p): f"{f} {l}" for p, f, l in zip(r.player_id, r["first"], r["last"])}


def results_2027() -> pd.DataFrame:
    p = NOUT / "live" / "results_2027.csv"
    return pd.read_csv(p) if p.exists() else None


def run(games: list, date: str, tag: str, now: dt.datetime, dry: bool) -> list:
    from sim.g_forecast_core import forecast, sha
    import sim.g_live as GL
    R = _resolver()
    cfg = json.loads(CFG.read_text())
    bundle = cfg["neurhl_g"]
    lineups, rows_meta = {}, []
    for gid, home, away in games:
        lu = R.resolve(gid, date, home, away, as_of=now)
        lineups[gid] = lu
        rows_meta.append({"game_id": gid, "date": date, "home": home, "away": away,
                          "start_utc": lu.get("start_utc")})
    gdf = pd.DataFrame(rows_meta)
    ok = [g for g in gdf.game_id if lineups[g]["home"]["skaters"] and lineups[g]["away"]["skaters"]]
    gdf = gdf[gdf.game_id.isin(ok)].reset_index(drop=True)
    if not len(gdf):
        print(f"[forecast] {tag}: no resolvable games")
        return []
    lu_in = {g: {s: {"skaters": lineups[g][s]["skaters"], "goalie": lineups[g][s]["goalie"]}
                 for s in ("home", "away")} for g in gdf.game_id}
    hproj = None
    try:                     # NeurHL-H lineup projections (G input and H comparator)
        from sim.h_live import lineup_projection
        hproj = lineup_projection(gdf, lu_in)
    except Exception as e:  # noqa: BLE001
        print(f"[forecast] NeurHL-H projection failed: {type(e).__name__}: {e}")
    A, meta, _ = GL.build(gdf, lu_in, results_2027(), hproj)
    p_h = np.full(len(gdf), np.nan)
    try:                     # NeurHL-H; a failure here never blocks NeurHL-G
        from sim.h_live import forecast as h_forecast
        from sim.project_2027 import load_schedule
        from sim.schedule_context import build as sched_ctx
        sc = sched_ctx(load_schedule(), 2027)[["game_id", "home_rest", "away_rest"]]
        if hproj is not None:
            p_h = h_forecast(gdf, lu_in, A["CTX"][:, 0], sc, hproj)
    except Exception as e:  # noqa: BLE001
        print(f"[forecast] NeurHL-H comparator failed: {type(e).__name__}: {e}")
    try:                     # stat-sheet goal level (PLAN_NeurHL4 A1)
        from live.goal_calibration import current as goal_mult
    except Exception:  # noqa: BLE001
        import importlib.util
        _sp = importlib.util.spec_from_file_location("goal_calibration",
                                                     ROOT / "live" / "goal_calibration.py")
        _gc = importlib.util.module_from_spec(_sp)
        _sp.loader.exec_module(_gc)
        goal_mult = _gc.current
    try:
        gm = goal_mult()
    except Exception as e:  # noqa: BLE001
        print(f"[forecast] goal calibration unavailable ({e}); using 1.0")
        gm = 1.0
    rows, sheets = forecast(bundle, A, p_h=p_h, goal_mult=gm)
    frozen = pd.read_csv(NOUT / "games_2027.csv").set_index("game_id")
    code = subprocess.run(["git", "-C", str(PROJ), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    bsha = sha(ROOT / "checkpoints" / "g" / bundle / "bundle.json")[:16]
    created = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    nm = names_map()
    out_rows, pl_rows = [], []
    for i, r in gdf.iterrows():
        lu = lineups[r.game_id]
        elo = 1 / (1 + np.exp(-A["CTX"][i, 0]))
        out_rows.append({
            "game_id": int(r.game_id), "date": date, "start_utc": r.start_utc,
            "home": r.home, "away": r.away, "forecast": tag,
            "p_home_win_neurhl_g": round(rows[i]["p_home_win"], 5),
            "p_home_win_elo": round(float(elo), 5),
            "p_home_win_neurhl_h": round(float(p_h[i]), 5) if np.isfinite(p_h[i]) else None,
            "p_home_win_1p0": round(float(frozen.p_home_win.get(r.game_id, np.nan)), 5),
            "p_ot": round(rows[i]["p_ot"], 4), "stack_used": rows[i]["stack_used"],
            **{k: round(rows[i][k], 3) for k in ("goals_home", "goals_away", "goals_home_raw",
                                                   "goals_away_raw", "xgf_home", "xgf_away",
                                                   "sog_home", "sog_away")},
            "goal_mult": round(rows[i]["goal_mult"], 4),
            "lineup_home": lu["home"]["lineup_source"], "lineup_away": lu["away"]["lineup_source"],
            "goalie_home": lu["home"]["goalie"], "goalie_away": lu["away"]["goalie"],
            "goalie_src_home": lu["home"]["goalie_source"],
            "goalie_src_away": lu["away"]["goalie_source"],
            "bundle": bundle, "bundle_sha": bsha, "code": code, "created_utc": created})
        for p in sheets[i]["players"]:
            pl_rows.append({"game_id": int(r.game_id), "team": r.home if p["side"] == 0 else r.away,
                            "player_id": p["player_id"], "name": nm.get(p["player_id"], ""),
                            "toi_ev": round(p["toi_ev"], 2), "toi_pp": round(p["toi_pp"], 2),
                            "toi_sh": round(p["toi_sh"], 2),
                            **{f"{k}_{s}": round(v, 3) for k in ("sog", "goals", "assists", "points", "ixg")
                               for s, v in zip(("mean", "p10", "p90"), p[k])},
                            "p_goal": round(p["p_goal"], 4), "p_point": round(p["p_point"], 4)})
    d = LIVE / date
    d.mkdir(parents=True, exist_ok=True)
    stem = tag if tag in ("morning", "preview") else f"pregame_{int(gdf.game_id.iloc[0])}"
    files = [d / f"{stem}.csv", d / f"{stem}_players.csv", d / f"{stem}_lineups.json"]
    pd.DataFrame(out_rows).to_csv(files[0], index=False)
    pd.DataFrame(pl_rows).to_csv(files[1], index=False)
    files[2].write_text(json.dumps({str(k): v for k, v in lineups.items()}, indent=1, default=str))
    print(pd.DataFrame(out_rows)[["game_id", "home", "away", "p_home_win_neurhl_g",
                                  "p_home_win_elo", "lineup_home", "lineup_away"]].to_string())
    if not dry:
        rel = [str(f.relative_to(PROJ)) for f in files]
        import os
        env = {**os.environ, "PUBLISH_MSG": f"live: {stem} forecast {date}"}
        r = subprocess.run(["bash", str(ROOT / "live" / "publish.sh"), "-", *rel],
                           capture_output=True, text=True, env=env)
        print(r.stdout[-2000:], r.stderr[-2000:])
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("morning", "pregame", "preview", "nightly"), required=True)
    ap.add_argument("--date")
    ap.add_argument("--now")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    now = dt.datetime.fromisoformat(a.now) if a.now else dt.datetime.now(dt.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=dt.timezone.utc)
    date = a.date or now.astimezone(ET).strftime("%Y-%m-%d")
    if a.mode == "nightly":
        print("[forecast] nightly is handled by run_nightly.sh")
        return
    R = _resolver()
    games = R.games_on(date)
    if a.mode in ("morning", "preview"):
        # preview: the evening before, on post-deadline rosters; published but
        # descriptive only (PLAN_NeurHL4 LIVE scores morning and pregame)
        if (LIVE / date / f"{a.mode}.csv").exists():
            print(f"[forecast] {a.mode} {date} already written")
            return
        run(games, date, a.mode, now, a.dry_run)
        return
    for gid, home, away in games:
        if (LIVE / date / f"pregame_{gid}.csv").exists():
            continue
        lu = R.resolve(gid, date, home, away, as_of=now)
        st = lu.get("start_utc")
        if not st:
            continue
        mins = (dt.datetime.fromisoformat(st) - now).total_seconds() / 60
        if 45 <= mins <= 75:
            run([(gid, home, away)], date, "pregame", now, a.dry_run)


if __name__ == "__main__":
    main()
