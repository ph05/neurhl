"""Tests for NeurHL 1.1 C1 (PLAN_NeurHL_1_1): the calibration fit and scorer.

  fit        configs/calibration_1_1.json matches a fresh fit (same source hash,
             same r_hat and b_hat); in-sample SOG coverage with r_hat is
             0.80 +- 0.03; the fit refuses rows from the sealed seasons
  identity   at b = 1 the calibrated goal mean is the frozen A1 mean exactly
  scorer     on the synthetic repository of eval/test_score_live_g.py (same
             forecast selection: pregame G1 and G3 count, G2 is late, G5 is
             uncommitted), the scorecard's paired goal and SOG log-score
             differences equal an independent computation, and degraded G3
             is left out of SOG

Run: uv run --no-project --python 3.12 --with numpy --with "pandas<3" --with pyarrow \
       --with scipy --with requests python neurhl/tests/test_calibration_1_1.py
"""
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import special, stats

NRL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NRL))
RES = []


def check(name, ok, detail=""):
    RES.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main():
    F = load("fit_cal", NRL / "eval" / "fit_calibration_1_1.py")
    cfg = json.loads((NRL / "configs" / "calibration_1_1.json").read_text())
    print("FIT")
    check("config built from the current source file",
          cfg["source_sha256"] == hashlib.sha256(F.SRC.read_bytes()).hexdigest())
    d = F.load()
    sd, gs = F.fit_dispersion(d), F.fit_slope(d)
    check("fresh fit reproduces r_hat and b_hat", sd["r_hat"] == cfg["sog_dispersion"]["r_hat"]
          and gs["b_hat"] == cfg["goal_slope"]["b_hat"], f"r {sd['r_hat']}, b {gs['b_hat']}")
    check("in-sample SOG coverage with r_hat within 0.80 +- 0.03",
          abs(sd["coverage_rhat"] - 0.80) <= 0.03, f"{sd['coverage_rhat']:.3f}")
    check("no sealed season in the fit", not set(cfg["seasons"]) & {2025, 2026}, str(cfg["seasons"]))
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "g.csv"
        pd.read_csv(F.SRC).head(10).assign(season_end=2025).to_csv(bad, index=False)
        src, F.SRC = F.SRC, bad
        try:
            F.load()
            refused = False
        except SystemExit:
            refused = True
        F.SRC = src
    check("fit refuses sealed-season rows", refused)

    print("\nIDENTITY")
    raw, m, M = np.array([2.1, 3.0, 4.4]), 0.92, 3.26
    check("b = 1 gives the frozen A1 goal mean", np.allclose(m * M * (raw / M) ** 1.0, m * raw, atol=1e-12))

    print("\nSCORER")
    T = load("tsl", NRL / "eval" / "test_score_live_g.py")
    with tempfile.TemporaryDirectory() as tmp:
        repo = T.Repo(Path(tmp) / "repo")
        T.build_main(repo)
        r_hat, b_hat, Mv = 95.0, 0.7, 3.2
        (repo.root / "neurhl/configs").mkdir(parents=True, exist_ok=True)
        (repo.root / "neurhl/configs/calibration_1_1.json").write_text(json.dumps(
            {"sog_dispersion": {"r_hat": r_hat}, "goal_slope": {"b_hat": b_hat}}))
        (repo.root / "neurhl/configs/live_goal_calibration.json").write_text(json.dumps({"M": Mv}))
        (repo.root / "neurhl/configs/calibration_1_1b.json").write_text(json.dumps({"k_hat": 12.0}))
        (repo.root / "neurhl/configs/calibration_1_1d.json").write_text(json.dumps({"b_x": 0.8, "k_x": 12.0}))
        p = subprocess.run([sys.executable, str(NRL / "eval" / "score_calibration_1_1.py"),
                            "--root", str(repo.root), "--offline", "--as-of", "2026-10-02"],
                           capture_output=True, text=True, env=repo.env)
        card_p = repo.root / "neurhl/output/live/scorecard_1_1_2027.json"
        check("scorer exits 0 and writes its card", p.returncode == 0 and card_p.exists(), p.stderr[-300:])
        card = json.loads(card_p.read_text())
        # independent: pregame G1 (goals 3.1/2.6, sog 31/27, reg 3-2, sogf 33/25) and
        # G3 (goals 2.9/2.8, reg 2-2 after the OT winner is removed; degraded for SOG)
        pois = lambda y, mu: y * np.log(mu) - mu - special.gammaln(y + 1)  # noqa: E731
        mult = 0.9
        mu_old = np.array([3.1, 2.6, 2.9, 2.8])
        y = np.array([3, 2, 2, 2.0])
        mu_new = mult * Mv * (mu_old / mult / Mv) ** b_hat
        dg = float(np.mean(pois(y, mu_new) - pois(y, mu_old)))
        check("goals: 2 games, 4 team-games", card["games"] == 2 and card["goals"]["n_team_games"] == 4)
        check("goals: paired log-score difference", abs(card["goals"]["diff"] - dg) < 1e-9,
              f"{card['goals']['diff']:.6f} vs {dg:.6f}")
        nb = lambda y, mu, r: stats.nbinom.logpmf(y, r, r / (r + mu))  # noqa: E731
        ys, mus = np.array([33.0, 25.0]), np.array([31.0, 27.0])
        ds = float(np.mean(nb(ys, mus, r_hat) - nb(ys, mus, 40.0)))
        check("SOG: degraded G3 left out", card["sog"]["n_team_games"] == 2)
        check("SOG: paired log-score difference", abs(card["sog"]["diff"] - ds) < 1e-9,
              f"{card['sog']['diff']:.6f} vs {ds:.6f}")
        gx = lambda y, mu, k: stats.gamma.logpdf(y, k, scale=mu / k)  # noqa: E731
        yx, mx = np.array([3.0, 2.0]), np.array([3.2, 2.5])
        dx = float(np.mean(gx(yx, mx, 12.0) - gx(yx, mx, 9.0)))
        check("xG (C1b): paired log-score difference, degraded G3 left out",
              card["xg"]["n_team_games"] == 2 and abs(card["xg"]["diff"] - dx) < 1e-9,
              f"{card['xg'].get('diff', float('nan')):.6f} vs {dx:.6f}")
        Mx = mx.mean()
        m2 = Mx * (mx / Mx) ** 0.8
        dd = float(np.mean(gx(yx, m2, 12.0) - gx(yx, mx, 9.0)))
        check("xG slope (C1d): paired log-score difference", abs(card["xg_slope"]["diff"] - dd) < 1e-9,
              f"{card['xg_slope'].get('diff', float('nan')):.6f} vs {dd:.6f}")
        check("skater shots (C1c): card present; the synthetic files carry no p10/p90, so unavailable",
              card.get("skater_sog", {}).get("available") is False)
        S = load("scal", NRL / "eval" / "score_calibration_1_1.py")
        v = S.interval_score(np.array([1.0, 1.0, 1.0]), np.array([3.0, 3.0, 3.0]), np.array([2.0, 0.0, 5.0]))
        check("interval score: width, plus 10 x the miss beyond each end", np.allclose(v, [2.0, 12.0, 22.0]), str(v))
        check("interim card makes no inference", card["status"] == "interim" and "holm" not in card)

    print(f"\n{sum(RES)}/{len(RES)} checks pass")
    sys.exit(0 if all(RES) else 1)


if __name__ == "__main__":
    main()
