"""NeurHL Tier-3c — season simulation provider (howe.rebuild_sim contract).

rebuild_sim() deterministically rebuilds the NeurHL 2026-27 season simulation
from the committed per-game probability artifact
(neurhl/output/preds/games_2027_neurhl.csv, written by project_2026_27.py) —
the exact analogue of src/howe.py: downstream market/portfolio tooling can
consume {"teams","pts","made_po","won_div","won_conf","won_cup"} unchanged.
CPU, seed 711, hash-stable (P9). Report-only alongside HOWE (PLAN_NeurHL S).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import NOUT  # noqa: E402
from sim.ratings_bridge import simulate  # noqa: E402

import backtest as B  # noqa: E402
import engine as E  # noqa: E402

NAME = "NeurHL"
SEED = 711
SEASON_END = 2027


def rebuild_sim(n_sims: int = 10_000, playoffs: bool = False) -> dict:
    games = pd.read_csv(NOUT / "preds" / "games_2027_neurhl.csv")
    import json
    v1 = json.loads((B.PROJ / "output" / "params.json").read_text())
    preds, _, _ = E.run_elo(B.g, K=v1["K"], H=v1["H"], phi_s=v1["phi_s"])
    om = E.fit_outcome(preds, list(range(2006, SEASON_END)))
    return simulate(games[["home", "away", "p_home"]], om, n_sims, SEED,
                    SEASON_END, playoffs=playoffs)
