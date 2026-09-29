"""NeurHL — shared paths, seeds, and repo-src import shim.

NeurHL is the repo's neural model track (prereg: PLAN_NeurHL.md at repo root).
Every NeurHL script imports this module first; it wires path constants and makes
the repo's src/ modules (engine, features, scoring, players, ...) importable.
The import is strictly one-way: nothing in src/ may import neurhl (enforced by
tests/review_tests_neurhl.py).

Branding rule: the model is "NeurHL" (exactly this capitalization) in all prose,
reports, and output model/column names; only filesystem artifacts are lowercase.
"""
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]           # repo root
SRC = PROJ / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))                     # engine, features, scoring, ...

NRL = PROJ / "neurhl"
RAW = PROJ / "data" / "raw"
PROC = PROJ / "data" / "processed"
MARKET = PROJ / "data" / "market"
TENSORS = NRL / "data" / "tensors"                   # gitignored
CKPT = NRL / "checkpoints"                           # gitignored
NOUT = NRL / "output"
EDA = NRL / "eda"
CONFIGS = NRL / "configs"

MODEL_NAME = "NeurHL"
SEED_H1, SEED_H2 = 711, 722    # house seed sequence (v4: 411/422 ... v6: 611/622)

UA = {"User-Agent":
      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}


def ensure_dirs() -> None:
    for d in (TENSORS, CKPT, NOUT / "preds", EDA / "figs", CONFIGS):
        d.mkdir(parents=True, exist_ok=True)


def refuse_if_frozen_1_0(what: str) -> None:
    """Stop before overwriting a NeurHL 1.0 file the FREEZE section hashes.
    Set NEURHL_ALLOW_REFREEZE=1 only for a deliberate, recorded rebuild."""
    import os
    plan = NRL.parent / "PLAN_NeurHL_1_0.md"
    if plan.exists() and "\n## FREEZE" in plan.read_text() and os.environ.get("NEURHL_ALLOW_REFREEZE") != "1":
        raise SystemExit(f"refusing to write {what}: neurhl/output/neurhl_1_0 is frozen (PLAN_NeurHL_1_0 FREEZE); "
                         "corrections are issued as new dated file sets")
