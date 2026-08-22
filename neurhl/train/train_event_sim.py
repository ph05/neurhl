"""NeurHL-2 — S1 training: the event simulator.

Protocol, fixed before training:

  * **Train 2008-2016, validate and gate on 2017.** CONFIRM (2018-2026) is not
    touched. TUNE is spent and therefore usable as development signal, which is
    exactly what a validation set is. Walk-forward models for the confirmatory
    game-level test come later and separately.
  * **Splits are by GAME, never by event.** Events inside a game are massively
    correlated — a random event split would put a shot and its own rebound on
    opposite sides of the split and report a fantasy validation number.
  * **Era conditioning is PRIOR-only.** The FiLM vector is built from the
    PREVIOUS season's league rates, so it is available before season V is played
    and extrapolates to 2027. A season index would not: the projection season has
    no learned embedding.

The reported metric is held-out next-event log likelihood, decomposed per head.
Aggregate loss hides which part of the model is broken — v1's event-LM blowup was
only diagnosable once the heads were reported separately.

Run: uv run --no-project --python 3.12 --with numpy --with torch \
     --with "pandas<3" --with pyarrow python neurhl/train/train_event_sim.py
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import TENSORS, CKPT  # noqa: E402
from models.event_sim import Cfg, EventSim, count_params  # noqa: E402
import windows as W  # noqa: E402

MAX_LEN = 384
TRAIN = list(range(2008, 2017))
VAL = 2017
SEED = 20260821
ERA_KEYS = ["g60", "sog60", "pen60", "fo60", "hit60", "blk60", "dt_med",
            "sh5v5"]


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Season:
    """One season's flat arrays plus its game offsets."""

    def __init__(self, se: int):
        z = np.load(TENSORS / f"seq_{se}.npz")
        self.se = se
        self.d = {k: z[k] for k in z.files}
        self.off = self.d["offsets"]
        self.n_games = len(self.off) - 1
        self.rapm = torch.from_numpy(self.d["rapm"])


def era_vectors(seasons) -> dict:
    """League rates per season, then SHIFTED so season V sees V-1."""
    raw = {}
    for se in seasons:
        p = TENSORS / f"seq_{se}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        et, dt = z["etype"], z["dt"]
        n_g = len(z["offsets"]) - 1
        per = lambda ids: float(np.isin(et, ids).sum()) / n_g  # noqa: E731
        raw[se] = np.array([
            per([7]), per([14, 7]), per([10]), per([3]), per([8]), per([1]),
            float(np.median(dt)), float((z["strength"] == 0).mean())],
            dtype=np.float32)
    out = {}
    keys = sorted(raw)
    for se in keys:
        prev = se - 1 if (se - 1) in raw else se
        out[se] = raw[prev]
    M = np.stack([out[k] for k in keys])
    mu, sd = M.mean(0), M.std(0) + 1e-6
    return {k: torch.from_numpy((out[k] - mu) / sd) for k in keys}, (mu, sd)


def make_batch(seasons, picks, eras, dev):
    """picks: list of (season_obj, game_index)."""
    B = len(picks)
    T = min(MAX_LEN, max(int(s.off[g + 1] - s.off[g]) - 1 for s, g in picks))
    T = max(T, 2)
    ints = {}
    I = ["etype", "team", "zone", "stype", "strength", "score", "period",
         "score_abs", "n_home", "n_away", "g_home", "g_away"]
    F = ["dt", "t_rem", "xa", "ya", "xg"]
    Bm = ["has_xy", "has_xg"]
    for k in I + Bm + ["n_for", "n_against"]:
        ints[k] = np.zeros((B, T), np.int64)
    for k in F:
        ints[k] = np.zeros((B, T), np.float32)
    on_next = np.zeros((B, T, 14), np.int64)
    valid = np.zeros((B, T), bool)
    tgt = {k: np.zeros((B, T), np.int64) for k in
           ["tgt_tt", "tgt_zone", "tgt_actor"]}
    tgt.update({k: np.zeros((B, T), np.float32) for k in
                ["tgt_dt", "tgt_xa", "tgt_ya"]})
    tgt["tgt_has_xy"] = np.zeros((B, T), np.int64)
    tgt_valid = np.zeros((B, T), bool)
    era = np.zeros((B, len(ERA_KEYS)), np.float32)

    for i, (s, g) in enumerate(picks):
        a, b = int(s.off[g]), int(s.off[g + 1])
        n = min(b - a - 1, T)
        if n <= 0:
            continue
        d = s.d
        sl = slice(a, a + n)
        nx = slice(a + 1, a + 1 + n)
        for k in I:
            ints[k][i, :n] = d[k][sl]
        for k in F:
            ints[k][i, :n] = d[k][sl]
        for k in Bm + ["n_for", "n_against"]:
            ints[k][i, :n] = d[k][sl]
        on_next[i, :n] = d["on"][nx]
        valid[i, :n] = True
        tgt_valid[i, :n] = True
        tgt["tgt_tt"][i, :n] = d["tt"][nx]
        tgt["tgt_zone"][i, :n] = d["zone"][nx]
        tgt["tgt_actor"][i, :n] = d["actor"][nx]
        tgt["tgt_dt"][i, :n] = d["dt"][nx]
        tgt["tgt_xa"][i, :n] = d["xa"][nx]
        tgt["tgt_ya"][i, :n] = d["ya"][nx]
        tgt["tgt_has_xy"][i, :n] = d["has_xy"][nx]
        era[i] = eras[s.se].numpy()

    out = {k: torch.from_numpy(v).to(dev) for k, v in ints.items()}
    out.update({k: torch.from_numpy(v).to(dev) for k, v in tgt.items()})
    out["on_next"] = torch.from_numpy(on_next).to(dev)
    out["valid"] = torch.from_numpy(valid).to(dev)
    out["tgt_valid"] = torch.from_numpy(tgt_valid).to(dev)
    out["era"] = torch.from_numpy(era).to(dev)
    out["tgt_actor"] = out["tgt_actor"].clamp(min=-1)
    return out


def evaluate(model, seasons, eras, dev, rapm_by, max_batches=60, bs=8):
    model.eval()
    rng = np.random.default_rng(0)
    agg = {}
    n = 0
    with torch.no_grad():
        for _ in range(max_batches):
            s = seasons[rng.integers(len(seasons))]
            gs = rng.integers(0, s.n_games, size=bs)
            b = make_batch(seasons, [(s, int(g)) for g in gs], eras, dev)
            o = model(b, rapm_by[s.se])
            for k, v in o.items():
                agg[k] = agg.get(k, 0.0) + float(v)
            n += 1
    model.train()
    return {k: v / max(n, 1) for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--tag", type=str, default="s1")
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = device()
    for s in TRAIN + [VAL]:
        if s in W.CONFIRM:
            raise SystemExit(f"season {s} is in CONFIRM and must not be used here")

    vocab = json.loads((TENSORS / "seq_vocab.json").read_text())
    tr = [Season(s) for s in TRAIN]
    va = [Season(VAL)]
    eras, _ = era_vectors(TRAIN + [VAL])
    rapm_by = {s.se: s.rapm.to(dev) for s in tr + va}

    c = Cfg(n_players=vocab["n_players"], n_tt=vocab["n_type_team"],
            d_model=args.d_model, n_layer=args.layers, n_era=len(ERA_KEYS))
    model = EventSim(c).to(dev)
    print(f"device {dev} | params {count_params(model):,} | "
          f"train {sum(s.n_games for s in tr):,} games "
          f"({sum(len(s.d['etype']) for s in tr):,} events) | val {VAL}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01,
                            betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    hist = []
    best = float("inf")
    CKPT.mkdir(parents=True, exist_ok=True)

    for step in range(1, args.steps + 1):
        s = tr[rng.integers(len(tr))]
        gs = rng.integers(0, s.n_games, size=args.bs)
        b = make_batch(tr, [(s, int(g)) for g in gs], eras, dev)
        o = model(b, rapm_by[s.se])
        opt.zero_grad(set_to_none=True)
        o["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate(model, va, eras, dev, rapm_by)
            hist.append({"step": step, "train_loss": o["loss"].detach().item(), **ev})
            print(f"step {step:>6} | train {o['loss'].detach().item():7.4f} | val "
                  f"{ev['loss']:7.4f}  dt {ev['dt']:6.3f}  tt {ev['tt']:6.3f}  "
                  f"loc {ev['loc']:6.3f}  zone {ev['zone']:6.3f}  "
                  f"actor {ev['actor']:6.3f} | {time.time()-t0:5.0f}s "
                  f"| lr {sched.get_last_lr()[0]:.2e}")
            sys.stdout.flush()
            if ev["loss"] < best:
                best = ev["loss"]
                torch.save({"model": model.state_dict(), "cfg": c.__dict__,
                            "step": step, "val": ev, "seed": args.seed},
                           CKPT / f"{args.tag}_seed{args.seed}.pt")
    (CKPT / f"{args.tag}_seed{args.seed}_hist.json").write_text(
        json.dumps(hist, indent=1))
    print(f"\nbest val loss {best:.4f} -> {CKPT}/{args.tag}_seed{args.seed}.pt")


if __name__ == "__main__":
    main()
