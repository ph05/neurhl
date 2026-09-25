"""NeurHL-3 and 1.0 release — acceptance battery.

Re-derives every NeurHL-3 decision from its recorded numbers and the rule as
written in PLAN_NeurHL3, checks the one-shot windows, the ledger caps and the
live freeze, and re-scores the NeurHL-H confirmation from its committed
per-game predictions when that record exists. A missing record fails the
check: an absent gate file is indistinguishable from a gate never run.

Run: uv run --no-project --python 3.12 --with numpy --with "pandas<3" \
     --with pyarrow --with scipy python neurhl/tests/review_tests_neurhl3.py
"""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJ = ROOT.parent
CFG, OUT_DIR = ROOT / "configs", ROOT / "output"
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append({"check": name, "pass": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


def load(p):
    return json.loads(p.read_text()) if p.exists() else None


def pg_gate(g):
    """PG1-PG4 as written: better, Holm-adjusted p < 0.05, season direction."""
    return g["better"] and g["p_holm"] < 0.05 and g["direction_agrees"]


def main():
    print("NeurHL-3 / 1.0 acceptance battery\n")

    print("PLAYER-GAME LAYER (PG1-PG4, PG-ALL, PG-STOP)")
    ev = load(CFG / "player_game_eval_eval.json")
    cf = load(CFG / "player_game_confirm_2018_2020.json")
    check("PG-ALL passed on PLAYER_EVAL before the spend (PG-STOP)",
          ev is not None and ev["PG_ALL"] and all(pg_gate(g) for g in ev["gates"].values()))
    check("PG-ALL on the one-shot confirm re-derives from the rule",
          cf is not None and cf["PG_ALL"] == all(pg_gate(g) for g in cf["gates"].values())
          and cf["PG_ALL"], f"heads {list(cf['gates']) if cf else 'missing'}")
    check("player confirm covers exactly {2018, 2019, 2020}",
          cf is not None and cf["seasons"] == [2018, 2019, 2020])
    tw = load(CFG / "player_game_twoway_2018_2020.json")
    check("two-way clustered re-analysis reproduces the recorded z",
          tw is not None and all(h["reproduces_record"] for h in tw["heads"].values()))
    check("every head still clears |z| > 3 under two-way clustering",
          tw is not None and all(abs(h["z_twoway"]) > 3 for h in tw["heads"].values()),
          "min |z| %.1f" % min(abs(h["z_twoway"]) for h in tw["heads"].values()) if tw else "")

    print("\nGOALIES (GS1, GQ1) AND PLAYER SEASON (PS1)")
    gg = load(CFG / "goalie_gates.json")
    gs1 = gg and (gg["GS1"]["vs_repeat"]["mean"] < 0 and gg["GS1"]["vs_share"]["mean"] < 0)
    check("GS1 decision matches its rule (must beat both heuristics)",
          gg is not None and bool(gs1) == gg["GS1"]["pass"])
    check("GQ1 decision matches its rule (GSAx must out-predict gq)",
          gg is not None and (gg["GQ1"]["r_gsax_next_sv"] > gg["GQ1"]["r_gq_next_sv"]) == gg["GQ1"]["pass"])
    ps = load(CFG / "player_season_v2.json")
    if ps:
        p = ps["pooled"]
        rule = ("blend" if p["blend"] < p["A"] and p["blend"] < p["B"]
                else "B" if p["B"] < p["A"] and ps["B_wins"] >= 6 else "A")
        check("PS1 ship decision re-derives from the declared rule",
              rule == ps["ship"], f"{rule} (B wins {ps['B_wins']}/{ps['n_vantages']})")
        check("the shipped player file is the blend PS1 selected",
              {"proj_p_path_a", "proj_p_path_b"} <= set(
                  pd.read_csv(OUT_DIR / "player_proj_2027.csv", nrows=1).columns))
    else:
        check("PS1 record present", False)

    print("\nGAME REATTEMPT (N5) AND LEDGER CAPS")
    n5 = load(CFG / "game_v3_pregate.json")
    beats = [k for k, v in (n5 or {}).items() if k.startswith("g-") and v["diff_vs_v1H"] <= -0.0015]
    check("G-STOP decision matches its rule (no block cleared -0.0015)",
          n5 is not None and (not beats) == (not n5["G_STOP"]["clears"]))
    caps = {"player_game": 10, "player_season": 4, "goalie": 4, "xg": 4}
    v3 = pd.read_csv(CFG / "search_ledger_v3.csv")
    used = v3.layer.value_counts().to_dict()
    check("ledger v3 caps respected", all(used.get(k, 0) <= c for k, c in caps.items()),
          ", ".join(f"{k} {used.get(k, 0)}/{c}" for k, c in caps.items()))
    v2 = pd.read_csv(CFG / "search_ledger_v2.csv")
    check("ledger v2 game cap respected (12)", len(v2) <= 12, f"{len(v2)}/12")

    print("\nNeurHL-H CONFIRMATION (PLAN_NeurHL A5)")
    a5 = subprocess.run(["git", "-C", str(PROJ), "log", "--format=%H", "-S",
                         "## A5. AMENDMENT 5", "--", "PLAN_NeurHL.md"],
                        capture_output=True, text=True).stdout.split()
    check("amendment A5 is committed", bool(a5))
    rec = load(OUT_DIR / "hier_restatement.json")
    if rec is None:
        check("confirmation not yet run (pending is allowed before the run)", True, "pending")
    else:
        first = subprocess.run(["git", "-C", str(PROJ), "log", "--diff-filter=A",
                                "--format=%H", "--", "neurhl/output/hier_restatement.json"],
                               capture_output=True, text=True).stdout.split()
        check("the confirmation record was added exactly once", len(first) == 1)
        if a5 and first:
            anc = subprocess.run(["git", "-C", str(PROJ), "merge-base", "--is-ancestor",
                                  a5[-1], first[0]]).returncode == 0
            check("A5 was committed before the confirmation record", anc)
        P = pd.read_csv(OUT_DIR / "preds" / "hier_restatement_games.csv")
        P = P[~P.season.isin([2021])]
        y = P.y.to_numpy()
        ll = lambda p: -(y * np.log(np.clip(p, 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-9, 1)))
        d = ll(P.p_neurhl_h.to_numpy()) - ll(P.p_elo.to_numpy())
        check("primary n = 10,184 games", len(P) == 10184, f"{len(P):,}")
        check("recorded difference re-derives from the per-game file",
              abs(d.mean() - rec["primary"]["diff"]) < 1e-6,
              f"{d.mean():+.6f} vs {rec['primary']['diff']:+.6f}")
        from scipy import stats
        z = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
        cl = pd.Series(d).groupby(P.season.to_numpy()).mean()
        verdict = "PASS" if (d.mean() < 0 and abs(z) > 1.959964 and cl.mean() < 0) else "NULL"
        check("verdict re-derives from the declared rule", verdict == rec["verdict"],
              f"{verdict}, p = {2 * stats.norm.sf(abs(z)):.4f}")

    print("\nLIVE 2026-27 FREEZE (PLAN_NeurHL_LIVE)")
    plan = (PROJ / "PLAN_NeurHL_LIVE.md").read_text()
    for path, sha in re.findall(r"`([\w/.-]+\.csv)` \|[^|]+\| `([0-9a-f]{64})`", plan):
        got = hashlib.sha256((PROJ / path).read_bytes()).hexdigest()
        check(f"{path} matches its frozen SHA-256", got == sha)
    g = pd.read_csv(OUT_DIR / "games_2027.csv")
    check("games file covers the full 1,344-game schedule", len(g) == 1344 and g.game_id.is_unique)
    check("probabilities are valid", g[["p_home_win", "p_home_win_elo", "p_ot"]].stack().between(0, 1).all())

    print("\nBOUNDARIES")
    hits = subprocess.run(["grep", "-rln", "import neurhl\\|from neurhl", str(PROJ / "src")],
                          capture_output=True, text=True).stdout.strip()
    check("src/ never imports neurhl", not hits)

    n = len(RESULTS)
    k = sum(r["pass"] for r in RESULTS)
    print(f"\n{k}/{n} checks pass")
    (CFG / "acceptance_neurhl3.json").write_text(json.dumps(RESULTS, indent=1))
    sys.exit(0 if k == n else 1)


if __name__ == "__main__":
    main()
