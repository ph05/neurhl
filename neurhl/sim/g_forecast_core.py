"""NeurHL-G forecast core: saved snapshot + tonight's inputs -> forecasts.

A snapshot bundle (neurhl/checkpoints/g/<name>/) holds, per seed, the network
weights; plus the standardisation statistics of its training rows, the
configuration, the walk-forward stack coefficients, and SHA-256 hashes of every
file (bundle.json). forecast() standardises live arrays exactly as training
did, averages the seeds, applies the stack, and runs the Monte Carlo stat sheet.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import CKPT  # noqa: E402
from models.neurhl_g import TEAM_RAW, NeurHLG  # noqa: E402
from sim.boxscore_mc import simulate  # noqa: E402

BUNDLES = CKPT / "g"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save_bundle(name, models, stats, cfg, names, stack):
    """models: list of trained NeurHLG (one per seed); stats: {key: (mu, sd)};
    stack: {"cols": [...], "coef": [...], "intercept": float}."""
    d = BUNDLES / name
    d.mkdir(parents=True, exist_ok=True)
    files = {}
    for k, m in enumerate(models):
        p = d / f"seed{k}.pt"
        torch.save(m.state_dict(), p)
        files[p.name] = sha(p)
    np.savez(d / "stats.npz", **{f"{k}_mu": v[0] for k, v in stats.items()},
             **{f"{k}_sd": v[1] for k, v in stats.items()})
    files["stats.npz"] = sha(d / "stats.npz")
    meta = {"cfg": cfg, "names": names, "stack": stack, "files": files}
    (d / "bundle.json").write_text(json.dumps(meta, indent=1, default=float))
    return sha(d / "bundle.json")


def load_bundle(name):
    d = BUNDLES / name
    meta = json.loads((d / "bundle.json").read_text())
    for f, h in meta["files"].items():
        if sha(d / f) != h:
            raise RuntimeError(f"bundle {name}: {f} hash mismatch")
    z = np.load(d / "stats.npz")
    stats = {k[:-3]: (z[k], z[k[:-3] + "_sd"]) for k in z.files if k.endswith("_mu")}
    n, cfg = meta["names"], meta["cfg"]
    models = []
    for f in sorted(x for x in meta["files"] if x.endswith(".pt")):
        m = NeurHLG(len(n["sk_feat"]), len(n["gk_feat"]), len(n["tm_feat"]),
                    len(n["ctx"]), d=cfg.get("d", 64), p=cfg.get("dropout", 0.2),
                    attn=cfg.get("attn", False), freeze_heads=cfg.get("freeze_heads", False),
                    elo_anchor=cfg.get("elo_anchor", False),
                    lineup_terms=cfg.get("lineup_terms", False),
                    h_terms=cfg.get("h_terms", False))
        m.load_state_dict(torch.load(d / f, map_location="cpu"))
        m.eval()
        models.append(m)
    return models, stats, meta


def standardise(A, stats, names):
    P = {}
    for key in ("SK", "GK", "TM", "CTX"):
        mu, sd = stats[key]
        z = (A[key] - mu) / sd
        P[key] = np.nan_to_num(np.clip(z, -8, 8), nan=0.0).astype(np.float32)
    P["SKB"] = np.nan_to_num(A["SKB"], nan=0.0).astype(np.float32)
    P["SKM"], P["SKP"] = A["SKM"], A["SKP"]
    P["TMR"] = A["TM"][..., [names["tm_feat"].index(c) for c in TEAM_RAW]].astype(np.float32)
    P["GKR"] = A["GK"][..., names["gk_feat"].index("gk_gsax_shrunk")].astype(np.float32)
    P["ELO"] = np.nan_to_num(A["CTX"][:, 0]).astype(np.float32)
    ri = [names["sk_feat"].index(c) for c in ("rapm_cf_off", "rapm_cf_def")]
    P["RAPM"] = np.nan_to_num(A["SK"][..., ri]).astype(np.float32)
    tf = names["tm_feat"]
    hi = [tf.index(c) for c in ("h_cfpct", "h_clshare")] if "h_cfpct" in tf else []
    P["HR"] = (np.nan_to_num(A["TM"][..., hi], nan=0.5).astype(np.float32) if hi
               else np.full(A["TM"].shape[:2] + (2,), 0.5, np.float32))
    return P


@torch.no_grad()
def forecast(name, A, n_sims=10_000, seed=711):
    """Returns (per-game rows, per-game stat sheets)."""
    torch.set_num_threads(4)
    models, stats, meta = load_bundle(name)
    names = meta["names"]
    P = standardise(A, stats, names)
    bt = {k: torch.as_tensor(v) for k, v in P.items()}
    outs = [{k: v.numpy() for k, v in m(bt).items()} for m in models]
    o = {k: np.mean([x[k] for x in outs], 0) for k in outs[0]}
    sigma = float(np.mean([m.log_sigma.exp().item() for m in models]))
    st = meta["stack"]
    lg = np.log(np.clip(o["p_home_win"], 1e-6, 1 - 1e-6) /
                np.clip(1 - o["p_home_win"], 1e-6, 1))
    cols = {"elo_logit": A["CTX"][:, 0], "lg": lg}
    z = st["intercept"] + sum(c * cols[k] for k, c in zip(st["cols"], st["coef"]))
    p_final = 1 / (1 + np.exp(-z))
    rows, sheets = [], []
    for i in range(len(p_final)):
        # rescale outcome4 so its home-win mass equals the stacked probability
        o4 = o["o4"][i].copy()
        ph = o4[0] + o4[2]
        o4[[0, 2]] *= p_final[i] / max(ph, 1e-9)
        o4[[1, 3]] *= (1 - p_final[i]) / max(1 - ph, 1e-9)
        g = {"o4": o4, "goals": o["goals"][i], "sogf": o["sogf"][i], "xgf": o["xgf"][i],
             "pp_opps": o["pp_opps"][i], "sigma": sigma, "mask": A["SKM"][i],
             "player_id": A["SKID"][i],
             **{k: o[k][i] for k in ("isog", "g", "a", "ixg", "toi_ev", "toi_pp", "toi_sh")}}
        sh = simulate(g, n=n_sims, seed=seed + i)
        rows.append({"p_home_win": float(p_final[i]), "p_home_win_raw": float(o["p_home_win"][i]),
                     "p_ot": float(o4[2] + o4[3]),
                     "xgf_home": float(o["xgf"][i, 0]), "xgf_away": float(o["xgf"][i, 1]),
                     "goals_home": float(o["goals"][i, 0]), "goals_away": float(o["goals"][i, 1]),
                     "sog_home": float(o["sogf"][i, 0]), "sog_away": float(o["sogf"][i, 1])})
        sheets.append(sh)
    return rows, sheets
