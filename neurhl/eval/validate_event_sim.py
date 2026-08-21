"""NeurHL-2 — S1 validation: gates E1 (event realism) and E2 (score effects).

Both gates are computed **analytically from the model's predictive
distributions**, not by rolling the simulator forward. That is deliberate and
follows PLAN_NeurHL2 S4: an autoregressive rollout over ~300 sequential events
compounds its own error, so a rollout that looks wrong tells you nothing about
whether the one-step process is wrong. Summing predicted probabilities over
held-out steps isolates the event model itself. Rollout calibration is a separate,
later obligation attached to S5's Monte Carlo path.

  **E1 — event realism.** For every held-out game, expected counts are
  `sum_t p(type, team | history_t)` per period. A correct process reproduces
  observed goals, shots, penalties and faceoffs per game AND per period. Per
  period matters: a model can get 60 minutes right while putting the goals in
  the wrong ones, and the game-outcome integration in S4 is period-aware.

  **E2 — score effects.** Leading teams suppress and trailing teams push; this is
  one of the largest documented effects in hockey. The gate is not "the model
  knows the average shot rate" but "the model reproduces the GRADIENT of shot
  rate across score states". A model that flattens that gradient will misprice
  every close game.

Also reported, from PLAN_NeurHL2 validation level 1:
  * dt calibration by PIT — the observed inter-event time evaluated under its own
    predicted log-normal mixture CDF should be Uniform(0,1). Non-uniformity means
    the simulator will sample the game clock wrong even if event mixes are right.
  * per-head held-out log likelihood, never just the aggregate. v1's event-LM
    blowup was only diagnosable once heads were reported separately.

Run: uv run --no-project --python 3.12 --with numpy --with torch \
     --with "pandas<3" --with pyarrow python neurhl/eval/validate_event_sim.py
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import TENSORS, CKPT  # noqa: E402
from models.event_sim import Cfg, EventSim  # noqa: E402
from train.train_event_sim import (Season, era_vectors, make_batch,  # noqa: E402
                                   device, TRAIN, VAL)

# Bands are wide on purpose: they catch a broken process, not a hockey opinion.
E1_TOL = 0.10          # relative error on per-game expected counts
E2_MIN_GRAD = 0.5      # shots/60 per goal of score deficit, observed direction
NAMES = {7: "goal", 14: "shot-on-goal", 9: "missed-shot", 1: "blocked-shot",
         10: "penalty", 3: "faceoff", 8: "hit", 15: "stoppage"}


def load_model(path: Path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    c = Cfg(**{k: v for k, v in ck["cfg"].items()
               if k in Cfg.__init__.__code__.co_varnames})
    m = EventSim(c).to(dev)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck


def mixture_cdf_at(o, y, k_dt):
    """CDF of the predicted log-normal mixture at log-time y."""
    o = o.view(*o.shape[:2], k_dt, 3)
    w = torch.softmax(o[..., 0], -1)
    mu, sd = o[..., 1], o[..., 2].clamp(-5, 3).exp()
    z = (y.unsqueeze(-1) - mu) / (sd * math.sqrt(2.0))
    return (w * 0.5 * (1 + torch.erf(z))).sum(-1)


def randomised_pit(o, dt, k_dt, gen=None):
    """PIT for INTEGER-second dt under a continuous density.

    Inter-event times are recorded in whole seconds, so a continuous model puts
    an atom's worth of probability across each integer's bin and the ordinary
    PIT cannot be uniform no matter how good the model is -- it piles up wherever
    the common dt values sit. That is a defect of the diagnostic, not of the
    simulator. The randomised (Dunn-Smyth) version treats each observation as
    the interval [dt-0.5, dt+0.5] and draws uniformly inside its probability
    mass, which IS uniform under a correctly specified model.
    """
    lo = torch.log1p((dt - 0.5).clamp(min=0.0))
    hi = torch.log1p(dt + 0.5)
    c_lo = mixture_cdf_at(o, lo, k_dt)
    c_hi = mixture_cdf_at(o, hi, k_dt)
    u = torch.rand(c_lo.shape, device=c_lo.device, generator=gen)
    return c_lo + u * (c_hi - c_lo).clamp(min=0)


@torch.no_grad()
def collect(model, season, eras, dev, rapm, vocab, max_games=None, bs=8):
    """Predicted distributions and observed outcomes over every held-out step."""
    inv_tt = {v: k for k, v in vocab["type_team"].items()}
    n_games = season.n_games if max_games is None else min(max_games,
                                                           season.n_games)
    acc = {"p_tt": [], "obs_tt": [], "period": [], "score": [], "pit": [],
           "strength": [], "team_next": [], "raw_pit": [], "dt": [],
           "score_abs": []}
    for start in range(0, n_games, bs):
        picks = [(season, g) for g in range(start, min(start + bs, n_games))]
        b = make_batch([season], picks, eras, dev)
        h, _ = model.encode(b, rapm)
        p = torch.softmax(model.h_tt(h), -1)
        y = torch.log1p(b["tgt_dt"])
        g = torch.Generator(device=h.device); g.manual_seed(7)
        pit = randomised_pit(model.h_dt(h), b["tgt_dt"], model.c.k_dt, g)
        raw_pit = mixture_cdf_at(model.h_dt(h), y, model.c.k_dt)
        v = b["tgt_valid"]
        acc["p_tt"].append(p[v].float().cpu().numpy())
        acc["obs_tt"].append(b["tgt_tt"][v].cpu().numpy())
        acc["period"].append(b["period"][v].cpu().numpy())
        acc["score"].append(b["score"][v].cpu().numpy() - 4)
        acc["score_abs"].append(b["score_abs"][v].cpu().numpy() - 4)
        acc["strength"].append(b["strength"][v].cpu().numpy())
        acc["pit"].append(pit[v].float().cpu().numpy())
        acc["raw_pit"].append(raw_pit[v].float().cpu().numpy())
        acc["dt"].append(b["tgt_dt"][v].float().cpu().numpy())
    out = {k: np.concatenate(v) for k, v in acc.items() if v}
    out["n_games"] = n_games
    out["inv_tt"] = inv_tt
    return out


def e1(c: dict) -> dict:
    """Expected vs observed counts per game, overall and per period."""
    p, obs, per = c["p_tt"], c["obs_tt"], c["period"]
    n_g = c["n_games"]
    inv = c["inv_tt"]
    cols = {}
    for j, key in inv.items():
        et, team = key.split("|")
        cols.setdefault(int(et), []).append(j)
    rows = []
    for et, js in sorted(cols.items()):
        if et not in NAMES:
            continue
        exp = float(p[:, js].sum()) / n_g
        act = float(np.isin(obs, js).sum()) / n_g
        rel = (exp - act) / max(act, 1e-9)
        rows.append({"event": NAMES[et], "expected": round(exp, 3),
                     "observed": round(act, 3), "rel_err": round(rel, 4),
                     "pass": bool(abs(rel) <= E1_TOL)})
    byper = []
    for pd_ in (1, 2, 3):
        m = per == pd_
        if not m.any():
            continue
        for et, js in sorted(cols.items()):
            if et not in ("", 7, 14):
                continue
        for et in (7, 14):
            js = cols.get(et, [])
            if not js:
                continue
            exp = float(p[m][:, js].sum()) / n_g
            act = float(np.isin(obs[m], js).sum()) / n_g
            byper.append({"period": pd_, "event": NAMES[et],
                          "expected": round(exp, 3), "observed": round(act, 3),
                          "rel_err": round((exp - act) / max(act, 1e-9), 4)})
    ok = all(r["pass"] for r in rows)
    per_ok = all(abs(r["rel_err"]) <= E1_TOL for r in byper)
    return {"per_game": rows, "per_period": byper,
            "pass": bool(ok and per_ok), "pass_overall": bool(ok),
            "pass_per_period": bool(per_ok)}


def e2(c: dict, only_5v5=False) -> dict:
    """Score effects: does each side's attempt rate move with ITS OWN lead?

    Measured per SIDE, not per game state. Grouping by the home lead alone nets
    the two effects against each other -- when the home team leads it suppresses
    and the away team pushes, so the total attempt share barely moves and the
    effect looks absent even when the model has it exactly right. Each step
    therefore contributes two rows, one per side, keyed on that side's own
    differential.

    Reported both all-strength and 5v5-only. Score effects are a property of
    even-strength play; the all-strength curve is confounded because trailing
    teams pull goalies and leading teams take more penalties, so the strength MIX
    changes with the score independently of any behavioural effect. Both are
    reported so neither framing can be cherry-picked.
    """
    p, obs, lead = c["p_tt"], c["obs_tt"], c["score_abs"]
    if only_5v5:
        m5 = c["strength"] == 0
        p, obs, lead = p[m5], obs[m5], lead[m5]
    inv = c["inv_tt"]
    js = {"1": [], "0": []}
    for j, key in inv.items():
        et, team = key.split("|")
        if int(et) in (14, 7, 9) and team in js:
            js[team].append(j)

    own_lead = np.concatenate([lead, -lead])
    pred = np.concatenate([p[:, js["1"]].sum(1), p[:, js["0"]].sum(1)])
    ob = np.concatenate([np.isin(obs, js["1"]).astype(float),
                         np.isin(obs, js["0"]).astype(float)])
    rows = []
    for L in range(-3, 4):
        m = own_lead == L
        if m.sum() < 500:
            continue
        rows.append({"own_lead": int(L), "n": int(m.sum()),
                     "pred_share": round(float(pred[m].mean()), 5),
                     "obs_share": round(float(ob[m].mean()), 5)})
    if len(rows) < 3:
        return {"rows": rows, "pass": False, "why": "too few score states"}
    x = np.array([r["own_lead"] for r in rows], float)
    yp = np.array([r["pred_share"] for r in rows])
    yo = np.array([r["obs_share"] for r in rows])
    gp = float(np.polyfit(x, yp, 1)[0])
    go = float(np.polyfit(x, yo, 1)[0])
    ratio = gp / go if go != 0 else float("nan")
    return {"rows": rows, "grad_pred": round(gp, 6), "grad_obs": round(go, 6),
            "ratio": round(ratio, 3), "same_direction": bool(gp * go > 0),
            "corr": round(float(np.corrcoef(yp, yo)[0, 1]), 4),
            "pass": bool(gp * go > 0 and 0.5 <= ratio <= 1.5)}


def pit_report(c: dict) -> dict:
    u = c["pit"]
    u = u[np.isfinite(u)]
    edges = np.linspace(0, 1, 11)
    cnt, _ = np.histogram(u, bins=edges)
    frac = cnt / max(len(u), 1)
    dev = float(np.abs(frac - 0.1).max())
    ks = float(np.max(np.abs(np.sort(u) - np.linspace(0, 1, len(u)))))
    return {"n": int(len(u)), "decile_fracs": [round(float(x), 4) for x in frac],
            "max_decile_dev": round(dev, 4), "ks": round(ks, 4),
            "mean": round(float(u.mean()), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--season", type=int, default=VAL)
    ap.add_argument("--max-games", type=int, default=400)
    args = ap.parse_args()

    dev = device()
    path = Path(args.ckpt) if args.ckpt else sorted(CKPT.glob("s1_seed*.pt"))[0]
    model, ck = load_model(path, dev)
    vocab = json.loads((TENSORS / "seq_vocab.json").read_text())
    s = Season(args.season)
    eras, _ = era_vectors(TRAIN + [VAL])
    rapm = s.rapm.to(dev)
    print(f"checkpoint {path.name}  step {ck['step']}  val {ck['val']['loss']:.4f}")
    print(f"scoring season {args.season} ({min(args.max_games, s.n_games)} games)\n")

    c = collect(model, s, eras, dev, rapm, vocab, max_games=args.max_games)

    r1 = e1(c)
    print("E1 event realism — expected vs observed per game")
    print(f"  {'event':<16}{'expected':>10}{'observed':>10}{'rel err':>10}")
    for r in r1["per_game"]:
        print(f"  {r['event']:<16}{r['expected']:>10.3f}{r['observed']:>10.3f}"
              f"{r['rel_err']:>+10.2%}  {'ok' if r['pass'] else 'FAIL'}")
    print(f"  per-period goals/shots: "
          f"{'ok' if r1['pass_per_period'] else 'FAIL'}")
    for r in r1["per_period"]:
        print(f"    P{r['period']} {r['event']:<14}{r['expected']:>8.3f}"
              f"{r['observed']:>8.3f}{r['rel_err']:>+9.2%}")
    print(f"  E1 -> {'PASS' if r1['pass'] else 'FAIL'}\n")

    r2_all = e2(c, only_5v5=False)
    r2 = e2(c, only_5v5=True)
    print("E2 score effects — a side's unblocked-attempt rate vs its OWN lead (5v5)")
    print(f"  {'lead':>6}{'n':>9}{'pred':>10}{'obs':>10}")
    for r in r2["rows"]:
        print(f"  {r['own_lead']:>+6}{r['n']:>9,}{r['pred_share']:>10.5f}"
              f"{r['obs_share']:>10.5f}")
    print(f"  5v5        gradient pred {r2.get('grad_pred')} vs obs "
          f"{r2.get('grad_obs')} (ratio {r2.get('ratio')}, corr {r2.get('corr')})")
    print(f"  allstrength gradient pred {r2_all.get('grad_pred')} vs obs "
          f"{r2_all.get('grad_obs')} (ratio {r2_all.get('ratio')}, "
          f"corr {r2_all.get('corr')})")
    print(f"  E2 -> {'PASS' if r2['pass'] else 'FAIL'} (5v5 is the gate; "
          f"all-strength {'passes' if r2_all['pass'] else 'fails'})\n")

    pr = pit_report(c)
    prr = pit_report({"pit": c["raw_pit"]})
    print(f"dt calibration, RANDOMISED PIT (n={pr['n']:,}): max decile deviation "
          f"{pr['max_decile_dev']:.4f}, mean {pr['mean']:.4f}")
    print(f"  deciles {pr['decile_fracs']}")
    print(f"  [naive continuous PIT, invalid for integer-second dt: "
          f"max dev {prr['max_decile_dev']:.4f}]")
    dt_o = c["dt"]
    qs = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    print(f"  observed dt quantiles {[round(float(np.quantile(dt_o,q)),1) for q in qs]}"
          f" at {qs}")

    out = {"checkpoint": path.name, "step": ck["step"], "season": args.season,
           "val_loss": ck["val"], "E1": r1, "E2": r2, "E2_all_strength": r2_all, "dt_pit": pr,
           "dt_pit_naive": prr}
    p = Path(__file__).resolve().parents[1] / "configs" / "event_sim_gates.json"
    p.write_text(json.dumps(out, indent=1, default=float))
    print(f"\n-> {p}")
    return 0 if (r1["pass"] and r2["pass"]) else 1


if __name__ == "__main__":
    sys.exit(main())
