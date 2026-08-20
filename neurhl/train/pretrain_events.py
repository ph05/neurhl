"""NeurHL — event-LM pretraining, one snapshot per vantage (PLAN_NeurHL P2).

Usage:
  uv run --no-project --python 3.12 --with torch --with numpy --with "pandas<3" \
    --with pyarrow python neurhl/train/pretrain_events.py --vantage 2017 \
    [--seasons-from 2008|2012] [--init-from CKPT] [--smoke]

Data: event shards for seasons <= vantage-1 (P1), starting at --seasons-from
(2008 iff the A1/HTM corpus passed integrity, else 2012). Early stopping on the
LATEST in-window season (still <= vantage-1). Checkpoint:
neurhl/checkpoints/event_lm_v<V>.pt (+ .sha256). Deterministic seed
20140 + vantage. Training on MPS; all downstream inference on CPU (P9).
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
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CKPT, PROC, TENSORS  # noqa: E402
from models.event_lm import (EventLM, coord_bucket, dt_bucket,  # noqa: E402
                             lm_loss, strength_class)

MAX_LEN = 512
EV_COLS = ["game_id", "game_type", "period", "t", "dt", "event_type", "zone",
           "shot_type", "strength", "home_event", "xn", "yn", "has_coord",
           "score_h", "score_a", "p1", "p2", "venue"] \
    + [f"h_on{i}" for i in range(7)] + [f"a_on{i}" for i in range(7)]
PLAYER_COLS = ["p1", "p2"] + [f"h_on{i}" for i in range(7)] \
    + [f"a_on{i}" for i in range(7)]


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


class GameDataset(Dataset):
    def __init__(self, seasons, vocab: dict, era: dict, smoke: bool = False):
        self.games = []
        vmap = np.zeros(8_500_000, dtype=np.int32)
        for pid, idx in vocab.items():
            vmap[int(pid)] = idx
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
                arr = {c: gdf[c].to_numpy() for c in EV_COLS if c != "game_id"}
                dressed = np.unique(np.concatenate(
                    [arr[c] for c in PLAYER_COLS]))
                dressed = dressed[dressed > 0][:46]
                pad = np.zeros(46 - len(dressed), dtype=dressed.dtype)
                self.games.append((se, gid, arr,
                                   np.concatenate([dressed, pad])))
        self.era = era

    def __len__(self):
        return len(self.games)

    def __getitem__(self, i):
        se, gid, arr, dressed = self.games[i]
        L = len(arr["period"])
        s = 0
        if L > MAX_LEN:
            s = np.random.randint(0, L - MAX_LEN + 1)
        sl = slice(s, s + MAX_LEN)
        out = {c: torch.as_tensor(np.ascontiguousarray(a[sl]).astype(np.int64))
               for c, a in arr.items()}
        out["era"] = torch.as_tensor(self.era.get(
            se, np.zeros(8, dtype=np.float32)))
        out["dressed"] = torch.as_tensor(dressed.astype(np.int64))
        # slot index of p1 among dressed (-1 if none)
        d = out["dressed"]
        p1 = out["p1"]
        eq = p1.unsqueeze(1) == d.unsqueeze(0)
        slot = torch.where(eq.any(1) & (p1 > 0), eq.float().argmax(1),
                           torch.full_like(p1, -1))
        out["p1_slot"] = slot
        return out


def collate(batch):
    L = max(len(b["period"]) for b in batch)
    out = {}
    for k in batch[0]:
        if k in ("era", "dressed"):
            out[k] = torch.stack([b[k] for b in batch])
        else:
            out[k] = torch.stack([
                torch.nn.functional.pad(b[k], (0, L - len(b[k])),
                                        value=-1 if k == "p1_slot" else 0)
                for b in batch])
    out["h_on"] = torch.stack([out.pop(f"h_on{i}") for i in range(7)], dim=-1)
    out["a_on"] = torch.stack([out.pop(f"a_on{i}") for i in range(7)], dim=-1)
    out["dtb"] = dt_bucket(out["dt"])
    out["coord"] = coord_bucket(out["xn"], out["yn"], out["has_coord"])
    out["strength_cls"] = strength_class(out["strength"])
    out["score_cls"] = (out["score_h"] - out["score_a"]).clamp(-3, 3) + 3
    out["period"] = out["period"].clamp(0, 8)
    return out


def run_epoch(model, loader, device, cfg, opt=None, rng=None):
    tr = opt is not None
    model.train(tr)
    tot, n = 0.0, 0
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        mask = (torch.rand(b["p1"].shape, device=device,
                           generator=None) < cfg["mask_ratio"]) & (b["p1"] > 0)
        with torch.set_grad_enabled(tr):
            out = model(b, mask)
            losses = lm_loss(out, b, mask)
        if tr:
            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        tot += float(losses["total"].detach()) * len(b["p1"])
        n += len(b["p1"])
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vantage", type=int, required=True)
    ap.add_argument("--seasons-from", type=int, default=2012)
    ap.add_argument("--init-from", type=str, default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = json.loads((TENSORS.parents[1] / "configs" / "pretrain.json")
                     .read_text())
    seed = cfg["seed_base"] + args.vantage
    torch.manual_seed(seed)
    np.random.seed(seed)

    vocab = json.loads((TENSORS / f"vocab_{args.vantage}.json").read_text())
    era = era_lookup()
    seasons = list(range(args.seasons_from, args.vantage))
    val_season = seasons[-1]
    t0 = time.time()
    train_ds = GameDataset(seasons[:-1], vocab, era, args.smoke)
    val_ds = GameDataset([val_season], vocab, era, args.smoke)
    print(f"train games {len(train_ds)}, val games {len(val_ds)} "
          f"[load {time.time() - t0:.0f}s]")
    device = ("mps" if torch.backends.mps.is_available() and not args.smoke
              else "cpu")
    model = EventLM(n_players=len(vocab), d_model=cfg["d_model"],
                    n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
                    d_player=cfg["d_player_embed"],
                    dropout=cfg["dropout"]).to(device)
    if args.init_from:
        model.load_state_dict(torch.load(args.init_from,
                                         map_location=device))
        print(f"initialized from {args.init_from}")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg["epochs_max"] * max(len(train_ds) // cfg["batch_size"], 1))
    tl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                    collate_fn=collate, num_workers=0)
    vl = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                    collate_fn=collate, num_workers=0)
    best, best_ep, patience = float("inf"), -1, 3
    ck = CKPT / f"event_lm_v{args.vantage}.pt"
    epochs = 2 if args.smoke else cfg["epochs_max"]
    for ep in range(epochs):
        t1 = time.time()
        tr = run_epoch(model, tl, device, cfg, opt)
        sched.step()
        va = run_epoch(model, vl, device, cfg)
        print(f"epoch {ep}: train {tr:.4f}  val({val_season}) {va:.4f} "
              f"[{time.time() - t1:.0f}s]")
        sys.stdout.flush()
        if va < best - 1e-4:
            best, best_ep = va, ep
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
