"""NeurHL-2 — S4: game outcomes by semi-analytic integration of S1's hazards.

PLAN_NeurHL2 S4, implemented as written: **do not take game probabilities from
rollout.** Simulating ~300 sequential events compounds error and autoregressive
samplers drift. Instead S1 is queried for competing goal hazards as a function of
personnel and state, and those hazards are integrated forward over the game
clock with a Kolmogorov forward step, giving the full joint score distribution
and P(home win) in quadrature rather than by sampling.

Three things make this a genuine forecast rather than a replay:

  * **Personnel are PROJECTED, never actual.** Each team is represented by a
    TOI-weighted expected unit built from the walk-forward RAPM prior for that
    vantage (fit on seasons < V only). S1's on-ice encoder mean-pools its skaters
    anyway, so an expected unit is the right object to hand it — but the TOI
    weights must come from prior seasons, not from the game being predicted.
  * **State is enumerated, not observed.** Rates are queried on a grid of
    (score differential, period), so score effects — which E2 confirmed the model
    reproduces at ratio 1.04 — enter the integration properly. A model that
    ignored score state would misprice exactly the close games that decide the
    metric.
  * **History is marginalised.** S1 conditions on event history, which does not
    exist before puck drop. A bank of real histories is sampled from OTHER games,
    the target personnel and state are substituted into them, and the resulting
    hazards are averaged. That marginalises the nuisance history instead of
    inventing a single fake one, and it is why the query is Monte Carlo over
    contexts while the game itself is still integrated analytically.

The rate implied by a marked point process at a step is
`lambda_X = p(mark = X) / E[dt]`, which is what turns S1's next-event
distribution into a per-second hazard. E1 already established that summing those
probabilities reproduces observed goal counts to +0.53%, so the conversion is
anchored on a validated quantity.
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import TENSORS  # noqa: E402

MAX_GOALS = 12
PERIOD_LEN = 1200.0
REG_TIME = 3600.0
OT_LEN = 300.0
SCORES = list(range(-4, 5))


def expected_dt(o, k_dt):
    """E[dt] under the predicted log-normal mixture over log(dt+1)."""
    o = o.view(*o.shape[:-1], k_dt, 3)
    w = torch.softmax(o[..., 0], -1)
    mu, sd = o[..., 1], o[..., 2].clamp(-5, 3).exp()
    # E[expm1(Y)] for Y ~ N(mu, sd^2) is exp(mu + sd^2/2) - 1
    return (w * (torch.exp(mu + 0.5 * sd ** 2) - 1.0)).sum(-1).clamp(min=0.5)


def projected_unit(prior_df, roster_ids, pmap, n_sk=6, goalie=None):
    """TOI-weighted expected on-ice unit -> player indices for S1's encoder.

    The top `n_sk` skaters by prior TOI stand in for the team's average unit.
    This is deliberately an EXPECTED unit rather than a specific line: S1 pools
    its skaters, and a projection has no way to know which line is on the ice at
    an arbitrary future second.
    """
    d = prior_df[prior_df.player_id.isin(roster_ids)]
    d = d[~d.is_replacement.astype(bool)].nlargest(n_sk, "toi_s")
    idx = [pmap.get(int(p), 0) for p in d.player_id]
    idx = (idx + [0] * n_sk)[:n_sk]
    g = pmap.get(int(goalie), 0) if goalie else 0
    return [g] + idx


@torch.no_grad()
def hazard_grid(models, ctx_bank, home_unit, away_unit, era, rapm, dev,
                scores=SCORES, periods=(1, 2, 3)):
    """lambda_home, lambda_away (goals per second) on a (score, period) grid.

    `ctx_bank` is a batch of real event histories whose state fields are
    overwritten with the queried state and whose on-ice slots are overwritten
    with the two projected units. Averaging over the bank marginalises the
    history the forecast cannot know.
    """
    inv = ctx_bank["inv_tt"]
    gh = [j for j, k in inv.items() if k.split("|") == ["7", "1"]]
    ga = [j for j, k in inv.items() if k.split("|") == ["7", "0"]]
    on = torch.tensor(home_unit + away_unit, dtype=torch.long, device=dev)

    out = {}
    for per in periods:
        for sc in scores:
            b = {k: (v.clone() if torch.is_tensor(v) else v)
                 for k, v in ctx_bank["batch"].items()}
            B, T = b["etype"].shape
            b["on_ctx"] = on.view(1, 1, 14).expand(B, T, 14).contiguous()
            b["score_abs"] = torch.full((B, T), sc + 4, dtype=torch.long,
                                        device=dev)
            b["n_home"] = torch.full((B, T), 5, dtype=torch.long, device=dev)
            b["n_away"] = torch.full((B, T), 5, dtype=torch.long, device=dev)
            b["g_home"] = torch.ones(B, T, dtype=torch.long, device=dev)
            b["g_away"] = torch.ones(B, T, dtype=torch.long, device=dev)
            b["period"] = torch.full((B, T), per, dtype=torch.long, device=dev)
            b["era"] = era.view(1, -1).expand(B, -1).contiguous()

            ph = pa = edt = None
            for m in models:
                h, _ = m.encode(b, rapm)
                p = torch.softmax(m.h_tt(h), -1)
                e = expected_dt(m.h_dt(h), m.c.k_dt)
                ph = p[..., gh].sum(-1) if ph is None else ph + p[..., gh].sum(-1)
                pa = p[..., ga].sum(-1) if pa is None else pa + p[..., ga].sum(-1)
                edt = e if edt is None else edt + e
            n = len(models)
            ph, pa, edt = ph / n, pa / n, edt / n
            v = ctx_bank["valid"]
            lh = float((ph[v] / edt[v]).mean())
            la = float((pa[v] / edt[v]).mean())
            out[(per, sc)] = (lh, la)
    return out


def integrate(grid, periods=(1, 2, 3), step=10.0, max_goals=MAX_GOALS):
    """Forward Kolmogorov step over the joint score distribution.

    State is the full joint (goals_home, goals_away), not the differential,
    because the deliverable needs score lines and totals as well as P(win), and
    because the empty-net and OT logic downstream is defined on the actual score.
    """
    n = max_goals + 1
    P = np.zeros((n, n))
    P[0, 0] = 1.0
    gh = np.arange(n)[:, None] * np.ones((1, n))
    ga = np.ones((n, 1)) * np.arange(n)[None, :]
    diff = np.clip(gh - ga, -4, 4).astype(int)

    for per in periods:
        t = 0.0
        while t < PERIOD_LEN - 1e-9:
            dt = min(step, PERIOD_LEN - t)
            lh = np.vectorize(lambda d: grid[(per, int(d))][0])(diff)
            la = np.vectorize(lambda d: grid[(per, int(d))][1])(diff)
            ph = 1.0 - np.exp(-lh * dt)
            pa = 1.0 - np.exp(-la * dt)
            stay = P * (1 - ph) * (1 - pa)
            up_h = np.zeros_like(P)
            up_a = np.zeros_like(P)
            both = np.zeros_like(P)
            up_h[1:, :] = (P * ph * (1 - pa))[:-1, :]
            up_a[:, 1:] = (P * (1 - ph) * pa)[:, :-1]
            both[1:, 1:] = (P * ph * pa)[:-1, :-1]
            P = stay + up_h + up_a + both
            P /= P.sum()
            t += dt
    return P


def outcome_probs(P, ot_home_edge=0.5):
    """P(home win) in regulation + OT/SO, plus the regulation-tie mass.

    OT and the shootout are treated as a near-coin-flip conditional on reaching
    them, with a small home edge. That is deliberate: a 3v3 overtime and a
    skills competition are different processes from 5v5 hockey, and inventing
    structure for them from a 5v5-trained hazard model would be worse than
    admitting the uncertainty. S6 replaces this with an explicit 3v3 model.
    """
    n = P.shape[0]
    gh = np.arange(n)[:, None]
    ga = np.arange(n)[None, :]
    p_reg_home = float(P[gh > ga].sum())
    p_reg_away = float(P[gh < ga].sum())
    p_tie = float(P[gh == ga].sum())
    p_home = p_reg_home + p_tie * ot_home_edge
    return {"p_home_win": p_home, "p_reg_home": p_reg_home,
            "p_reg_away": p_reg_away, "p_tie": p_tie,
            "exp_goals_home": float((P * gh).sum()),
            "exp_goals_away": float((P * ga).sum())}
