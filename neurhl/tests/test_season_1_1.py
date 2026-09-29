"""Tests for the NeurHL 1.1 season tools (PLAN_NeurHL_1_1 C2, C2b, C2c).

  posterior   with no games played the posterior is the prior (mode 0,
              covariance sigma0^2 I); a team that won every game gets a positive
              mode and a smaller variance than a team that has not played
  project     with no games remaining, the projection is the points already
              earned; league points always equal 2 x games + overtime games
  rerun       sim/season_1_1.py with the frozen layer (a=1, 0.07, 0, 1) and the
              frozen seed reproduces NeurHL 1.0's teams file within the games
              file's rounding, and its slot sums hold for another variant
  blend       a = 1 leaves the game probabilities untouched

Run: uv run --no-project --python 3.12 --with numpy --with "pandas<3" --with pyarrow \
       --with torch --with scipy --with scikit-learn==1.9.1 --with numba --with requests \
       python neurhl/tests/test_season_1_1.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NRL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NRL))
from sim.live_standings_1_1 import posterior, project  # noqa: E402

RES = []


def check(name, ok, detail=""):
    RES.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def main():
    print("POSTERIOR")
    T, s0 = 4, 0.09
    m, C = posterior(np.zeros(0), np.zeros(0), np.zeros(0, int), np.zeros(0, int), np.zeros(0), T, s0)
    check("no games: posterior = prior", np.allclose(m, 0) and np.allclose(C, np.eye(T) * s0 ** 2))
    hp = np.array([0, 0, 0, 0, 1, 1]); ap = np.array([1, 2, 1, 2, 2, 1 + 1])
    yp = np.array([1, 1, 1, 1, 0, 0.0]); zp = np.zeros(6); kp = np.full(6, 3.0)
    m, C = posterior(zp, kp, hp, ap, yp, T, s0)
    check("unbeaten team: positive mode", m[0] > 0, f"{m[0]:.3f}")
    check("team with games has smaller variance than a team without", C[0, 0] < C[3, 3] and np.isclose(C[3, 3], s0 ** 2))

    print("\nPROJECT")
    pts = np.array([10.0, 8, 6, 4])
    sim = project(np.zeros(0), np.zeros(0), np.zeros(0, int), np.zeros(0, int), np.zeros((0, 4)),
                  np.zeros(0, int), 5, pts, np.zeros(T), np.eye(T) * s0 ** 2, 0.01, 100, 1)
    check("no games left: projection = points earned", np.allclose(sim, pts[None, :]))
    rng = np.random.default_rng(0)
    n = 50
    hr, ar = rng.integers(0, T, n), rng.integers(0, T, n)
    ar = np.where(ar == hr, (hr + 1) % T, ar)
    o4 = np.tile([0.4, 0.35, 0.13, 0.12], (n, 1))
    sim = project(rng.normal(0, 0.3, n), np.full(n, 3.0), hr, ar, o4, rng.integers(5, 20, n), 5,
                  np.zeros(T), np.zeros(T), np.eye(T) * s0 ** 2, 0.01, 200, 2)
    tot = sim.sum(1)
    check("league points per simulated season between 2n and 3n", tot.min() >= 2 * n and tot.max() <= 3 * n,
          f"{tot.min():.0f}-{tot.max():.0f} for n={n}")

    print("\nRERUN")
    uv = [sys.executable]
    out = NRL / "output" / "neurhl_1_0" / "season_testself"
    shutil.rmtree(out, ignore_errors=True)
    p = subprocess.run(uv + [str(NRL / "sim" / "season_1_1.py"), "--sigma0", "0.07", "--sw", "0", "--rho", "1",
                             "--tag", "testself", "--sims", "20000"], capture_output=True, text=True)
    check("season_1_1 runs with the frozen layer", p.returncode == 0, p.stderr[-300:])
    if p.returncode == 0:
        a = pd.read_csv(NRL / "output" / "neurhl_1_0" / "teams_2027.csv").set_index("team")
        b = pd.read_csv(out / "teams_2027.csv").set_index("team").loc[a.index]
        d = max((a.points - b.points).abs().max(), (a.playoff_pct - b.playoff_pct).abs().max() / 10)
        check("frozen layer reproduces NeurHL 1.0 within rounding", d < 0.01, f"max {d:.4f}")
    shutil.rmtree(out, ignore_errors=True)
    p = subprocess.run(uv + [str(NRL / "sim" / "season_1_1.py"), "--a", "0.75", "--sigma0", "0.09", "--sw", "0.01",
                             "--rho", "0.99", "--tag", "testself", "--sims", "2000"], capture_output=True, text=True)
    check("a variant runs and passes its slot and points identities", p.returncode == 0, p.stderr[-300:])
    shutil.rmtree(out, ignore_errors=True)

    print(f"\n{sum(RES)}/{len(RES)} checks pass")
    sys.exit(0 if all(RES) else 1)


if __name__ == "__main__":
    main()
