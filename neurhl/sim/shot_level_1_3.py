"""NeurHL 1.3 season set: the 1.2 set with the season's league shot and attempt levels
anchored to last season's (PLAN_NeurHL_1_3 R1).

  m_sog = L_sog / M_sog, m_att = L_att / M_att
    L: 2025-26 league mean team shots / attempts per team-game (tgx_2026, the engine's
       target definition); M: the 1.2 set's mean over all 1,344 games (1.2 applies no
       shot ratio, so these are the engine's own levels)

Only shots, attempts and what is computed from them change: skater and team shots and
attempts, goalie shots against and saves, save and shooting percentages. Goals, xG, power
plays, win probabilities, points and playoff odds are the 1.2 values. The 1.2 files are
read, never written; the 1.3 set goes to output/neurhl_1_3/season/. The transformation is
exactly what sim/unified_2027.py --shot-level-last-season does to the same draws, applied to
the frozen set so that no input newer than 1.2's enters.
"""
import datetime as dt
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import NOUT, TENSORS  # noqa: E402

SRC = NOUT / "neurhl_1_2" / "season"
DST = NOUT / "neurhl_1_3" / "season"
PREV = 2026


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    lv = pd.read_parquet(TENSORS / f"tgx_{PREV}.parquet", columns=["sogf", "attf"]).mean()
    g = pd.read_csv(SRC / "games_2027.csv")
    M_sog = float(g[["sog_home", "sog_away"]].to_numpy().mean())
    M_att = float(g[["attempts_home", "attempts_away"]].to_numpy().mean())
    ms, ma = float(lv.sogf) / M_sog, float(lv.attf) / M_att
    print(f"shots {M_sog:.3f} -> {lv.sogf:.3f} (x{ms:.4f}); attempts {M_att:.3f} -> {lv.attf:.3f} (x{ma:.4f})")
    DST.mkdir(parents=True, exist_ok=True)

    for c in ("sog_home", "sog_away"):
        g[c] *= ms
    for c in ("attempts_home", "attempts_away"):
        g[c] *= ma
    g.to_csv(DST / "games_2027.csv", index=False, float_format="%.5f")

    t = pd.read_csv(SRC / "teams_2027.csv")
    for c in ("sog_for", "sog_against"):
        t[c] *= ms
    for c in ("attempts_for", "attempts_against"):
        t[c] *= ma
    t["shooting_pct"] = 100 * t.goals_for / t.sog_for
    t["save_pct"] = 1 - t.goals_against / t.sog_against
    t.to_csv(DST / "teams_2027.csv", index=False, float_format="%.4f")

    s = pd.read_csv(SRC / "skaters_2027.csv")
    s["sog"] *= ms
    s["att"] *= ma
    s["shooting_pct"] = 100 * s.goals / s.sog.clip(lower=1e-9)
    s.to_csv(DST / "skaters_2027.csv", index=False, float_format="%.4f")

    gl = pd.read_csv(SRC / "goalies_2027.csv")
    gl["sa"] *= ms
    gl["saves"] = gl.sa - gl.ga
    gl["sv_pct"] = 1 - gl.ga / gl.sa.clip(lower=1e-9)
    gl.to_csv(DST / "goalies_2027.csv", index=False, float_format="%.4f")

    pg = pd.read_csv(SRC / "player_games_2027.csv.gz")
    pg["sog"] *= ms
    pg["att"] *= ma
    pg.to_csv(DST / "player_games_2027.csv.gz", index=False, float_format="%.5f")

    shutil.copy2(SRC / "team_points_quantiles_2027.csv", DST / "team_points_quantiles_2027.csv")

    cons = json.loads((SRC / "consistency_2027.json").read_text())
    cons["team_sog_minus_player_sog_max"] = float(
        (t.set_index("team").sog_for - s.groupby("team").sog.sum()).abs().max())
    cons["league_sog_per_team_game"] = float(g[["sog_home", "sog_away"]].to_numpy().mean())
    cons["league_attempts_per_team_game"] = float(g[["attempts_home", "attempts_away"]].to_numpy().mean())
    (DST / "consistency_2027.json").write_text(json.dumps(cons, indent=1))

    run = json.loads((SRC / "run_2027.json").read_text())
    run.update({"derived_from": str(SRC.relative_to(ROOT.parent)),
                "derived_from_sha256": {f: sha(SRC / f) for f in ("games_2027.csv", "teams_2027.csv",
                                                                   "skaters_2027.csv", "goalies_2027.csv",
                                                                   "player_games_2027.csv.gz")},
                "shot_level": {"L_sog": float(lv.sogf), "M_sog": M_sog, "m_sog": ms,
                               "L_att": float(lv.attf), "M_att": M_att, "m_att": ma,
                               "rule": "PLAN_NeurHL_1_3 R1"},
                "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
    (DST / "run_2027.json").write_text(json.dumps(run, indent=1))
    print(f"-> {DST}")


if __name__ == "__main__":
    main()
