"""Train one NeurHL-G snapshot and predict a season (PLAN_NeurHL4 M, L).

snapshot(T, cfg) trains on every game of seasons < T (from 2009) and predicts
every game of season T. Standardisation statistics come from the training rows
only. Reads the master tensor through data/g_loader.py, so SEALED seasons are
invisible unless the seal script has unsealed them.

Losses (masked where a target is missing):
  skaters  Huber on ice time by strength (minutes); Poisson deviance, scaled by
           the target mean, on SOG, attempts, ixG, goals, assists, on-ice xGF/xGA
  teams    Poisson deviance on xGF (EV, PP), SOG, attempts, PP opportunities,
           regulation goals; Huber on PP minutes
  outcome  cross-entropy on outcome4, weight w_out
Phase 1 trains with w_out = cfg.w_out1; phase 2 (if epochs2 > 0) fine-tunes with
w_out = 1 at a lower learning rate and an L2-SP pull toward the phase-1 weights.
"""
import copy
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.g_loader import load_master  # noqa: E402
from models.neurhl_g import TEAM_RAW, NeurHLG  # noqa: E402

DEFAULT = {"d": 64, "dropout": 0.2, "attn": False, "freeze_heads": False,
           "epochs1": 10, "epochs2": 4, "lr1": 2e-3, "lr2": 3e-4, "wd": 1e-2,
           "l2sp": 1e-3, "w_out1": 0.3, "batch": 256, "seed": 0,
           "train_from": 2009, "threads": 8}

SK_T = ["toi_ev", "toi_pp", "toi_sh", "isog", "iatt", "ixg_all", "g", "a",
        "oi_xgf_all", "oi_xga_all"]
SK_OUT = ["toi_ev", "toi_pp", "toi_sh", "isog", "iatt", "ixg", "g", "a",
          "oi_xgf", "oi_xga"]
TM_T = {"xgf_ev": "xgf_ev", "xgf_pp": "xgf_pp", "sogf": "sogf", "attf": "attf",
        "gf_reg": "goals", "pp_opps": "pp_opps"}


class Data:
    """Master arrays, standardised for a given training mask."""

    def __init__(self, purpose="train"):
        self.A, self.meta, self.names = load_master(purpose)
        n = self.names
        self.tm_raw_idx = [n["tm_feat"].index(c) for c in TEAM_RAW]
        self.gk_raw_idx = n["gk_feat"].index("gk_gsax_shrunk")

    def prepare(self, train_mask):
        A = self.A
        out = {}
        self.stats = {}
        for key, arr, dims in (("SK", A["SK"], (0, 1, 2)), ("GK", A["GK"], (0, 1)),
                               ("TM", A["TM"], (0, 1)), ("CTX", A["CTX"], (0,))):
            x = arr[train_mask]
            if key == "SK":
                x = x[A["SKM"][train_mask] > 0]
            else:
                x = x.reshape(-1, arr.shape[-1])
            mu = np.nanmean(x, 0)
            sd = np.nanstd(x, 0)
            mu = np.where(np.isfinite(mu), mu, 0.0)
            sd = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0)
            self.stats[key] = (mu, sd)
            z = (arr - mu) / sd
            out[key] = np.nan_to_num(np.clip(z, -8, 8), nan=0.0).astype(np.float32)
        out["SKB"] = np.nan_to_num(A["SKB"], nan=0.0).astype(np.float32)
        out["SKM"], out["SKP"] = A["SKM"], A["SKP"]
        out["TMR"] = A["TM"][..., self.tm_raw_idx].astype(np.float32)
        out["GKR"] = A["GK"][..., self.gk_raw_idx].astype(np.float32)
        out["SKY"], out["TMY"] = A["SKY"], A["TMY"]
        out["O4"] = self.meta.outcome4.to_numpy().astype(np.int64)
        out["ELO"] = np.nan_to_num(A["CTX"][:, 0]).astype(np.float32)
        ri = [self.names["sk_feat"].index(c) for c in ("rapm_cf_off", "rapm_cf_def")]
        out["RAPM"] = np.nan_to_num(A["SK"][..., ri]).astype(np.float32)
        tf = self.names["tm_feat"]
        hi = [tf.index(c) for c in ("h_cfpct", "h_clshare")] if "h_cfpct" in tf else []
        out["HR"] = (np.nan_to_num(A["TM"][..., hi], nan=0.5).astype(np.float32) if hi
                     else np.full(A["TM"].shape[:2] + (2,), 0.5, np.float32))
        return out


def batch(P, idx):
    return {k: torch.as_tensor(v[idx]) for k, v in P.items()}


def pois(mu, y, mask):
    """Poisson deviance-style loss scaled by the target mean."""
    mu = mu.clamp(min=1e-6)
    y = torch.nan_to_num(y, nan=0.0)
    ll = (mu - y * torch.log(mu)) - (y - torch.xlogy(y, y))   # >= 0
    s = (ll * mask).sum() / mask.sum().clamp(min=1)
    return s / ((y * mask).sum() / mask.sum().clamp(min=1)).clamp(min=1e-3)


def huber(mu, y, mask, delta=1.0):
    y = torch.nan_to_num(y, nan=0.0)
    return (F.huber_loss(mu, y, reduction="none", delta=delta) * mask).sum() \
        / mask.sum().clamp(min=1)


def losses(out, bt, names, w_out):
    SKY, TMY, m = bt["SKY"], bt["TMY"], bt["SKM"]
    L = {}
    for j, (t, o) in enumerate(zip(SK_T, SK_OUT)):
        y = SKY[..., j]
        mk = m * torch.isfinite(y).float()
        L[f"sk_{t}"] = huber(out[o], y, mk) if t.startswith("toi") else pois(out[o], y, mk)
    tmt = names["tm_tgt"]
    for t, o in TM_T.items():
        y = TMY[..., tmt.index(t)]
        mk = torch.isfinite(y).float()
        L[f"tm_{t}"] = pois(out[o], y, mk)
    y = TMY[..., tmt.index("pp_m")]
    L["tm_pp_m"] = huber(out["pp_m"], y, torch.isfinite(y).float())
    L["out"] = F.nll_loss(torch.log(out["o4"].clamp(min=1e-9)), bt["O4"])
    total = sum(v for k, v in L.items() if k != "out") / 10.0 + w_out * L["out"]
    return total, L


def train_snapshot(D, T, cfg):
    cfg = {**DEFAULT, **cfg}
    torch.manual_seed(cfg["seed"] * 1000 + T)
    np.random.seed(cfg["seed"] * 1000 + T)
    torch.set_num_threads(cfg["threads"])
    s = D.meta.season_end.to_numpy()
    tr = (s < T) & (s >= cfg["train_from"])
    P = D.prepare(tr)
    n = D.names
    model = NeurHLG(len(n["sk_feat"]), len(n["gk_feat"]), len(n["tm_feat"]),
                    len(n["ctx"]), d=cfg["d"], p=cfg["dropout"], attn=cfg["attn"],
                    freeze_heads=cfg["freeze_heads"],
                    elo_anchor=cfg.get("elo_anchor", False),
                    lineup_terms=cfg.get("lineup_terms", False),
                    h_terms=cfg.get("h_terms", False))
    idx_tr = np.where(tr)[0]
    rng = np.random.default_rng(cfg["seed"] * 7919 + T)

    def run(epochs, lr, w_out, anchor=None):
        params = [q for q in model.parameters() if q.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=cfg["wd"])
        steps = max(1, epochs * math.ceil(len(idx_tr) / cfg["batch"]))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
        model.train()
        for ep in range(epochs):
            perm = rng.permutation(idx_tr)
            for i in range(0, len(perm), cfg["batch"]):
                bt = batch(P, perm[i:i + cfg["batch"]])
                out = model(bt)
                loss, _ = losses(out, bt, n, w_out)
                if anchor is not None:
                    loss = loss + cfg["l2sp"] * sum(((q - a) ** 2).sum() for q, a in
                                                   zip(params, anchor))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                opt.step()
                sched.step()

    run(cfg["epochs1"], cfg["lr1"], cfg["w_out1"])
    if cfg["epochs2"] > 0:
        anchor = [q.detach().clone() for q in model.parameters() if q.requires_grad]
        run(cfg["epochs2"], cfg["lr2"], 1.0, anchor)
    model.eval()
    model.stats = dict(D.stats)
    return model, P


@torch.no_grad()
def predict(model, P, idx, batch_size=1024):
    outs = []
    for i in range(0, len(idx), batch_size):
        bt = batch(P, idx[i:i + batch_size])
        o = model(bt)
        outs.append({k: v.numpy() for k, v in o.items() if k != "joint"})
    return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}
