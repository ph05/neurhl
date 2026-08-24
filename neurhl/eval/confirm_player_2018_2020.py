"""NeurHL-3 — the ONE-SHOT player-layer confirm on {2018, 2019, 2020}.

PLAN_NeurHL3 PG-STOP: this window is spent exactly once, and only after
PG-ALL passed on PLAYER_EVAL {2022-2026}. Guards, all mechanical:
  * refuses to run if its output already exists (spent once, ever);
  * refuses to run unless configs/player_game_eval_eval.json records
    PG_ALL == true (the pre-gate);
  * scores through the same instruments as the eval battery
    (backtest_player_game), seasons asserted via assert_scorable_player.

Run: uv run --no-project --python 3.12 --with numpy --with "pandas<3" \
     --with pyarrow --with scikit-learn --with scipy \
     python neurhl/eval/confirm_player_2018_2020.py
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import windows as W  # noqa: E402

CFG = ROOT / "configs"
OUT = CFG / "player_game_confirm_2018_2020.json"


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT.name} exists — the one-shot window was "
                         f"already spent and is never re-run.")
    pre = CFG / "player_game_eval_eval.json"
    if not pre.exists():
        raise SystemExit("PG-STOP: no PLAYER_EVAL record — run the eval "
                         "window first.")
    ev = json.loads(pre.read_text())
    if not ev.get("PG_ALL"):
        raise SystemExit("PG-STOP: PG-ALL did not pass on PLAYER_EVAL — the "
                         "confirm window stays unspent.")
    W.assert_scorable_player(W.PLAYER_CONFIRM, "confirm")

    uv = ["uv", "run", "--no-project", "--python", "3.12", "--with", "numpy",
          "--with", "pandas<3", "--with", "pyarrow"]
    r = subprocess.run(uv + ["--with", "scikit-learn", "python",
                             str(ROOT / "train" / "train_player_game.py"),
                             "--vantages"] + [str(s) for s in
                                              W.PLAYER_CONFIRM])
    if r.returncode != 0:
        raise SystemExit("confirm training failed; window NOT recorded as "
                         "spent")
    # score with the same instruments; backtest writes
    # player_game_eval_confirm.json which we re-label as the spend record
    r = subprocess.run(uv + ["--with", "scipy", "python",
                             str(ROOT / "eval" / "backtest_player_game.py"),
                             "--window", "confirm"])
    rec = json.loads((CFG / "player_game_eval_confirm.json").read_text())
    rec["spend"] = "one-shot {2018,2019,2020}; spent exactly once"
    OUT.write_text(json.dumps(rec, indent=1))
    print(f"-> {OUT}  PG-ALL(confirm) = {rec['PG_ALL']}")
    return 0 if rec["PG_ALL"] else 1


if __name__ == "__main__":
    sys.exit(main())
