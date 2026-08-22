"""NeurHL-2 — autoregressive rollout from S1, and its calibration check.

PLAN_NeurHL2 S4 is explicit that game probabilities must NOT come from rollout:
simulating ~300 sequential events compounds error and autoregressive samplers
drift, so the headline per-game number comes from semi-analytic integration
instead. This module exists for the other reason the plan gives — **rollout
calibration is mandatory before any Monte Carlo claim**, because S5 needs Monte
Carlo for joint quantities (player props, correlated standings, playoff paths)
and an uncalibrated sampler poisons all of them.

What is tested here is therefore drift, not accuracy: does the generative process,
run forward on its own output, still produce hockey? Simulated aggregates are
compared against the observed band for goals, shots, penalties and faceoffs, and
checked **per period**, because a sampler can hold the 60-minute total while
drifting badly within it.

Personnel are TEACHER-FORCED from the real game's on-ice timeline. Deployment is
a separate process (S3), so mixing an unvalidated deployment sampler into this
test would confound two failure modes: a drifting event process and a drifting
line-change process look identical in the aggregates. S3 gets its own check.

The state machine maintained during a rollout — clock, period, score, and the
strength implied by the sampled penalties — is what makes the sampled sequence
self-consistent. Without it the model would happily emit a power-play goal at
even strength.
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import TENSORS  # noqa: E402

PERIOD_LEN = 1200
N_PERIODS = 3


def sample_dt(o, k_dt, gen, device):
    """Draw from the predicted log-normal mixture, in seconds."""
    o = o.view(-1, k_dt, 3)
    w = torch.softmax(o[..., 0], -1)
    idx = torch.multinomial(w, 1, generator=gen).squeeze(-1)
    ar = torch.arange(o.shape[0], device=device)
    mu = o[ar, idx, 1]
    sd = o[ar, idx, 2].clamp(-5, 3).exp()
    y = mu + sd * torch.randn(mu.shape, device=device, generator=gen)
    return torch.expm1(y).clamp(min=0.0)


class RolloutState:
    """Clock, score and penalty box for one simulated game."""

    def __init__(self):
        self.t = 0.0
        self.period = 1
        self.score_h = 0
        self.score_a = 0
        self.pen_h = []          # expiry times
        self.pen_a = []
        self.counts = {}

    def strength(self):
        n_h = 5 - len([p for p in self.pen_h if p > self.t])
        n_a = 5 - len([p for p in self.pen_a if p > self.t])
        return max(n_h, 3), max(n_a, 3)

    def advance(self, dt):
        self.t += float(dt)
        while self.period < N_PERIODS and self.t >= self.period * PERIOD_LEN:
            self.period += 1

    def apply(self, etype, is_home):
        self.counts[etype] = self.counts.get(etype, 0) + 1
        key = (etype, is_home)
        self.counts[key] = self.counts.get(key, 0) + 1
        if etype == 7:                               # goal
            if is_home:
                self.score_h += 1
                if self.pen_a:
                    self.pen_a.pop(0)                # PP goal ends the penalty
            else:
                self.score_a += 1
                if self.pen_h:
                    self.pen_h.pop(0)
        elif etype == 10:                            # penalty
            (self.pen_h if is_home else self.pen_a).append(self.t + 120.0)


def observed_band(season: int, quantiles=(0.05, 0.95)) -> dict:
    """Per-game observed distribution of the aggregates a rollout must match."""
    import pandas as pd
    d = pd.read_parquet(TENSORS / f"stream_{season}.parquet",
                        columns=["game_id", "game_type", "event_type", "period"])
    d = d[d.game_type == 2]
    out = {}
    for name, ids in (("goal", [7]), ("shot-on-goal", [14]), ("penalty", [10]),
                      ("faceoff", [3]), ("hit", [8]), ("missed-shot", [9]),
                      ("blocked-shot", [1])):
        per_game = (d[d.event_type.isin(ids)].groupby("game_id").size()
                    .reindex(d.game_id.unique(), fill_value=0))
        out[name] = {"mean": float(per_game.mean()),
                     "lo": float(per_game.quantile(quantiles[0])),
                     "hi": float(per_game.quantile(quantiles[1]))}
    byp = {}
    for p in (1, 2, 3):
        s = d[(d.period == p) & (d.event_type == 7)].groupby("game_id").size()
        byp[p] = float(s.reindex(d.game_id.unique(), fill_value=0).mean())
    out["_goals_by_period"] = byp
    return out


def band_report(sim: dict, obs: dict, tol=0.15) -> dict:
    """Compare simulated per-game means to the observed band."""
    rows = []
    for k, o in obs.items():
        if k.startswith("_"):
            continue
        s = sim.get(k, 0.0)
        rel = (s - o["mean"]) / max(o["mean"], 1e-9)
        rows.append({"event": k, "sim": round(s, 3),
                     "obs": round(o["mean"], 3),
                     "obs_lo": o["lo"], "obs_hi": o["hi"],
                     "rel_err": round(rel, 4),
                     "in_band": bool(o["lo"] <= s <= o["hi"]),
                     "pass": bool(abs(rel) <= tol)})
    return {"rows": rows, "pass": bool(all(r["pass"] for r in rows))}
