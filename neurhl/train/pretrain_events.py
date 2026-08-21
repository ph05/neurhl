"""NeurHL — event-LM pretraining, one snapshot per vantage (PLAN_NeurHL P2).

Usage:
  uv run --no-project --python 3.12 --with torch --with numpy --with "pandas<3" \
    --with pyarrow python neurhl/train/pretrain_events.py --vantage 2017 \
    [--seasons-from 2008|2012] [--init-from CKPT] [--smoke]

Data: event shards for seasons <= vantage-1 (P1), starting at --seasons-from
(2008 — the A1/HTM corpus passed integrity). Early stopping on the LATEST
in-window season (still <= vantage-1). Checkpoint:
neurhl/checkpoints/event_lm_v<V>.pt (+ .sha256). Deterministic seed
20140 + vantage. Training on MPS; all downstream inference on CPU (P9).

Mechanical note (ledger pt-opt-1): batches are sliced from a fully
pre-transformed padded tensor bank (all bucketings computed once at load) —
identical objectives/architecture/hyperparameters to the original loader; the
only behavioral difference is that the rare >512-event game (playoff multi-OT)
is cropped to its first 512 events instead of a random window.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CKPT, PROC, TENSORS  # noqa: E402
from models.event_lm import (EventLM, coord_bucket, dt_bucket,  # noqa: E402
                             lm_loss, strength_class)

MAX_LEN = 512
EV_COLS = ["game_id", "period", "t", "dt", "event_type", "zone",
           "shot_type", "strength", "home_event", "xn", "yn", "has_coord",
           "score_h", "score_a", "p1", "p2", "venue"] \
    + [f"h_on{i}" for i in range(7)] + [f"a_on{i}" for i in range(7)]
PLAYER_COLS = ["p1", "p2"] + [f"h_on{i}" for i in range(7)] \
    + [f"a_on{i}" for i in range(7)]
BANK_INT = ["event_type", "zone", "shot_type", "home_event", "period",
            "venue", "p1", "p2", "dtb", "coord", "strength_cls", "score_cls"]


def era_lookup() -> dict:
    games = pd.read_csv(PROC / "games.csv")
    ts = pd.read_csv(PROC / "team_seasons.csv")
    r = games[games.game_type == "R"]
    per = r.groupby("season_end").apply(lambda d: pd.Series({
        "gpg": (d.home_g + d.away_g).mean(),
        "ot": d.went_ot.astype(bool).mean(),
        "so": d.went_so.astype(bool).mean(),
        "mabs": d.margin.abs().mean()}), include_groups=False)
    per["parity"] = ts.groupby("season_end").pts_pct.std()
    out = {}
    for se in per.index:
        if se - 1 in per.index:
            p = per.loc[se - 1]
            out[se] = np.array([p.gpg, p.ot, p.so, p.mabs, p.parity,
                                float(se >= 2016), float(se in (2020, 2021)),
                                (se - 2006) / 20.0], dtype=np.float32)
    return out


def build_bank(seasons, vocab: dict, era: dict, smoke: bool = False) -> dict:
    """Fully pre-transformed padded tensors for all games in `seasons`."""
    vmap = np.zeros(8_500_000, dtype=np.int32)
    for pid, idx in vocab.items():
        vmap[int(pid)] = idx
    games = []
    for se in seasons:
        p = TENSORS / f"events_{se}.parquet"
        if not p.exists():
            continue
        ev = pd.read_parquet(p, columns=EV_COLS)
        ev = ev[ev.period <= 7]
        if smoke:
            keep = ev.game_id.unique()[:60]
            ev = ev[ev.game_id.isin(keep)]
        for c in PLAYER_COLS:
            ev[c] = vmap[ev[c].to_numpy(np.int64).clip(0, 8_499_999)]
        for gid, gdf in ev.groupby("game_id", sort=True):
            games.append((se, {c: gdf[c].to_numpy()[:MAX_LEN]
                               for c in EV_COLS if c != "game_id"}))
    N = len(games)
    bank = {c: np.zeros((N, MAX_LEN), np.int32) for c in
            ["event_type", "zone", "shot_type", "home_event", "period",
             "venue", "p1", "p2", "strength", "dt", "xn", "yn", "has_coord",
             "score_h", "score_a"]}
    for i in range(7):
        bank[f"h_on{i}"] = np.zeros((N, MAX_LEN), np.int32)
        bank[f"a_on{i}"] = np.zeros((N, MAX_LEN), np.int32)
    bank["era"] = np.zeros((N, 8), np.float32)
    bank["dressed"] = np.zeros((N, 46), np.int32)
    bank["length"] = np.zeros(N, np.int32)
    for i, (se, arr) in enumerate(games):
        L = len(arr["period"])
        bank["length"][i] = L
        for c, a in arr.items():
            if c in bank:
                bank[c][i, :L] = a
        dressed = np.unique(np.concatenate([arr[c] for c in PLAYER_COLS]))
        dressed = dressed[dressed > 0][:46]
        bank["dressed"][i, :len(dressed)] = dressed
        bank["era"][i] = era.get(se, np.zeros(8, np.float32))
    t = {k: torch.as_tensor(v) for k, v in bank.items()}
    t["h_on"] = torch.stack([t.pop(f"h_on{i}") for i in range(7)], dim=-1)
    t["a_on"] = torch.stack([t.pop(f"a_on{i}") for i in range(7)], dim=-1)
    t["dtb"] = dt_bucket(t.pop("dt").long()).to(torch.int32)
    t["coord"] = coord_bucket(t.pop("xn"), t.pop("yn"),
                              t.pop("has_coord")).to(torch.int32)
    t["strength_cls"] = strength_class(t.pop("strength").long()).to(torch.int32)
    t["score_cls"] = ((t.pop("score_h") - t.pop("score_a"))
                      .clamp(-3, 3) + 3).to(torch.int32)
    t["period"] = t["period"].clamp(0, 8)
    eq = t["p1"].unsqueeze(-1) == t["dressed"].unsqueeze(1)
    t["p1_slot"] = torch.where(eq.any(-1) & (t["p1"] > 0),
                               eq.float().argmax(-1),
                               torch.full_like(t["p1"], -1).long()).long()
    return t


def get_batch(bank, idx, device):
    L = int(bank["length"][idx].max())
    b = {}
    for k in BANK_INT:
        b[k] = bank[k][idx, :L].long().to(device)
    b["h_on"] = bank["h_on"][idx, :L].long().to(device)
    b["a_on"] = bank["a_on"][idx, :L].long().to(device)
    b["dressed"] = bank["dressed"][idx].long().to(device)
    b["p1_slot"] = bank["p1_slot"][idx, :L].to(device)
    b["era"] = bank["era"][idx].to(device)
    return b


def run_epoch(model, bank, device, cfg, opt=None, gen=None):
    tr = opt is not None
    model.train(tr)
    n = len(bank["era"])
    order = torch.randperm(n, generator=gen) if tr else torch.arange(n)
    tot, cnt = 0.0, 0
    for i in range(0, n, cfg["batch_size"]):
        idx = order[i:i + cfg["batch_size"]]
        b = get_batch(bank, idx, device)
        mask = (torch.rand(b["p1"].shape, device=device)
                < cfg["mask_ratio"]) & (b["p1"] > 0)
        with torch.set_grad_enabled(tr):
            out = model(b, mask)
            losses = lm_loss(out, b, mask)
        if tr:
            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        tot += float(losses["total"].detach()) * len(idx)
        cnt += len(idx)
    return tot / max(cnt, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vantage", type=int, required=True)
    ap.add_argument("--seasons-from", type=int, default=2008)
    ap.add_argument("--init-from", type=str, default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = json.loads((TENSORS.parents[1] / "configs" / "pretrain.json")
                     .read_text())
    seed = cfg["seed_base"] + args.vantage
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed)

    vocab = json.loads((TENSORS / f"vocab_{args.vantage}.json").read_text())
    era = era_lookup()
    seasons = list(range(args.seasons_from, args.vantage))
    val_season = seasons[-1]
    t0 = time.time()
    bank_tr = build_bank(seasons[:-1], vocab, era, args.smoke)
    bank_va = build_bank([val_season], vocab, era, args.smoke)
    print(f"train games {len(bank_tr['era'])}, val games "
          f"{len(bank_va['era'])} [load {time.time() - t0:.0f}s]")
    device = ("mps" if torch.backends.mps.is_available() and not args.smoke
              else "cpu")
    model = EventLM(n_players=len(vocab), d_model=cfg["d_model"],
                    n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
                    d_player=cfg["d_player_embed"],
                    dropout=cfg["dropout"]).to(device)
    if args.init_from:
        state = torch.load(args.init_from, map_location=device)
        own = model.state_dict()
        for k, v in state.items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
            elif k == "player_emb.weight":       # vocab grew: copy prefix
                own[k][:v.shape[0]] = v
        model.load_state_dict(own)
        print(f"initialized from {args.init_from}")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    best, best_ep, patience = float("inf"), -1, 3
    ck = CKPT / f"event_lm_v{args.vantage}.pt"
    epochs = 2 if args.smoke else cfg["epochs_max"]
    for ep in range(epochs):
        t1 = time.time()
        trl = run_epoch(model, bank_tr, device, cfg, opt, gen)
        val = run_epoch(model, bank_va, device, cfg)
        print(f"epoch {ep}: train {trl:.4f}  val({val_season}) {val:.4f} "
              f"[{time.time() - t1:.0f}s]")
        sys.stdout.flush()
        if val < best - 1e-4:
            best, best_ep = val, ep
            torch.save(model.state_dict(), ck)
        elif ep - best_ep >= patience:
            print(f"early stop at {ep} (best {best:.4f} @ {best_ep})")
            break
    sha = hashlib.sha256(ck.read_bytes()).hexdigest()
    (ck.with_suffix(".sha256")).write_text(sha)
    meta = {"vantage": args.vantage, "seasons": seasons, "val": val_season,
            "seed": seed, "best_val": best, "sha256": sha,
            "seasons_from": args.seasons_from}
    (ck.with_suffix(".json")).write_text(json.dumps(meta, indent=1))
    print(f"saved {ck.name} val={best:.4f} sha={sha[:12]}")


if __name__ == "__main__":
    main()
