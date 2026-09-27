"""The only tensor reader NeurHL-G model, training and evaluation code may use.

PLAN_NeurHL4, section W. SEALED seasons (2025, 2026) are refused unless:
  * eval/seal_g.py has called windows.unseal() (the one-shot seal), or
  * the caller declares purpose="live_inputs": feature histories for live
    2026-27 inference, which never produce a score for a sealed season.
Every live_inputs access is appended to data/tensors/_g_loader_access.log.

Data builders (neurhl/data/build_*.py) are exempt: they construct per-season
tables and score nothing. tests/review_tests_neurhl4.py fails any NeurHL-G
module that reads TENSORS without going through this loader.
"""
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import TENSORS  # noqa: E402
import windows as W  # noqa: E402

PURPOSES = ("train", "score", "live_inputs")


def _check(season: int, purpose: str) -> None:
    if purpose not in PURPOSES:
        raise ValueError(f"purpose must be one of {PURPOSES}")
    if season in W.SEALED and not W.is_unsealed():
        if purpose != "live_inputs":
            raise PermissionError(
                f"season {season} is SEALED (PLAN_NeurHL4): only "
                f"eval/seal_g.py may read it for {purpose}")
        with open(TENSORS / "_g_loader_access.log", "a") as f:
            f.write(f"{dt.datetime.now().isoformat()} live_inputs {season}\n")


def path(name: str, season: int, purpose: str = "train",
         ext: str = "parquet") -> Path:
    _check(season, purpose)
    return TENSORS / f"{name}_{season}.{ext}"


def load(name: str, season: int, purpose: str = "train", **kw) -> pd.DataFrame:
    """Read TENSORS/<name>_<season>.parquet under the seal rules."""
    return pd.read_parquet(path(name, season, purpose), **kw)


def load_many(name: str, seasons, purpose: str = "train", **kw) -> pd.DataFrame:
    parts = []
    for s in seasons:
        p = path(name, s, purpose)
        if p.exists():
            d = pd.read_parquet(p, **kw)
            parts.append(d if "season_end" in d else d.assign(season_end=s))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
