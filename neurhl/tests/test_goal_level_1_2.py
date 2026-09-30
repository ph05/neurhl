"""NeurHL 1.2 R3: goal-level prior and engine conversion in live/goal_calibration.current().

  1. without the 1.2 state, current() is the 1.1 formula (raw goals pooled as recorded)
  2. with it, the prior is L / M_full and a g2027_v2 forecast is converted by M_v3 / M_v2:
     the same m as if that forecast had been recorded pre-converted by the live engine
  3. a forecast by an engine with no recorded M is left out
Run: python neurhl/tests/test_goal_level_1_2.py
"""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

NRL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NRL))
FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        FAIL.append(name)


def load():
    spec = importlib.util.spec_from_file_location("gc_test", NRL / "live" / "goal_calibration.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def setup(tmp: Path, rows, with_12=True):
    G = load()
    cfg, out = tmp / "configs", tmp / "output"
    (out / "live" / "2027" / "2026-09-29").mkdir(parents=True)
    cfg.mkdir()
    G.CONFIGS, G.NOUT = cfg, out
    G.STATE, G.STATE_1_2 = cfg / "live_goal_calibration.json", cfg / "live_goal_calibration_1_2.json"
    G.STATE.write_text(json.dumps({"m0": 1.0107, "L": 3.0, "M": 2.97, "k": 300.0, "bundle": "g2027_v3"}))
    (cfg / "live_goal_calibration_g2027_v2.json").write_text(
        json.dumps({"m0": 0.97, "L": 3.0, "M": 3.09, "k": 300.0, "bundle": "g2027_v2"}))
    if with_12:
        G.STATE_1_2.write_text(json.dumps({"m0": 3.0 / 3.05, "L": 3.0, "M": 3.05, "k": 300.0,
                                           "bundle": "g2027_v3"}))
    res = pd.DataFrame({"game_id": [r[0] for r in rows], "home_g": [r[4] for r in rows],
                        "away_g": [r[5] for r in rows], "last_period": "REG"})
    res.to_csv(out / "live" / "results_2027.csv", index=False)
    for gid, b, gh, ga, *_ in rows:
        pd.DataFrame({"game_id": [gid], "forecast": ["pregame"], "goals_home_raw": [gh],
                      "goals_away_raw": [ga], "bundle": [b]}).to_csv(
            out / "live" / "2027" / "2026-09-29" / f"pregame_{gid}.csv", index=False)
    return G


def main():
    rows = [(1, "g2027_v2", 3.5, 3.3, 4, 2), (2, "g2027_v3", 2.9, 3.6, 3, 1)]
    with tempfile.TemporaryDirectory() as t:
        G = setup(Path(t), rows, with_12=False)
        m = G.current()
        want = (10 + 300 * 3.0) / (3.5 + 3.3 + 2.9 + 3.6 + 300 * 2.97)
        check("1.1 formula without the 1.2 state", abs(m - want) < 1e-12, f"{m:.6f} vs {want:.6f}")
    with tempfile.TemporaryDirectory() as t:
        G = setup(Path(t), rows)
        m = G.current()
        c = 2.97 / 3.09
        want = (10 + 300 * 3.0) / ((3.5 + 3.3) * c + 2.9 + 3.6 + 300 * 3.05)
        check("1.2 prior and v2 -> v3 conversion", abs(m - want) < 1e-12, f"{m:.6f} vs {want:.6f}")
    with tempfile.TemporaryDirectory() as t:
        G = setup(Path(t), [(1, "g2027_v3", 3.5 * 2.97 / 3.09, 3.3 * 2.97 / 3.09, 4, 2), rows[1]])
        check("same m as a pre-converted forecast", abs(G.current() - want) < 1e-12)
    with tempfile.TemporaryDirectory() as t:
        G = setup(Path(t), [(1, "g_unknown", 9.0, 9.0, 4, 2), rows[1]])
        want = (4 + 300 * 3.0) / (2.9 + 3.6 + 300 * 3.05)
        check("unknown engine left out", abs(G.current() - want) < 1e-12)
    print(f"{4 - len(FAIL)}/4 passed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
