"""NeurHL — walk-forward game-model training for one predict-season.

Usage:
  ... python neurhl/train/train_game.py --predict-season 2015 [--seeds 0 1 2 3 4]
      [--smoke]

Protocol (PLAN_NeurHL P1/P7): train on regular-season games with season_end in
[2009, T-1] (2008 serves as form/era history), early-stop on T-1, then predict
season T's games. Player embeddings from embeddings_v<T>.npz (players <= T-1);
per-player form features are EWMAs shifted strictly pre-game (within T this is
legitimate in-season h2-style information). Per-seed checkpoints + a per-game
ensemble prediction CSV: neurhl/output/preds/game_preds_<T>.csv.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CKPT, NOUT, TENSORS  # noqa: E402
from models.game_model import GameModel, game_loss  # noqa: E402

EWMA_ALPHA = 0.1
ERA_COLS = ["prior_gpg", "prior_ot_share", "prior_so_share", "prior_margin_abs",
            "prior_parity", "flag_3v3", "flag_covid", "season_scaled"]
ERA_SCALE = np.array([1 / 6, 1, 1, 1 / 2.3, 10, 1, 1, 1], dtype=np.float32)


FORM_N = 25          # rolling window, games
FORM_COLS = ["h_gf", "h_ga", "h_pts", "a_gf", "a_ga", "a_pts"]
FORM_PRIOR = {"gf": 2.7, "ga": 2.7, "pts": 1.1}    # league-average cold start


def team_form(gc: pd.DataFrame) -> pd.DataFrame:
    """Rolling pre-game team form (PLAN_NeurHL amendment A1).

    For every game, each team's goals-for/against per game and points per game
    over its previous FORM_N games. Strictly pre-game (the current result is
    appended only after the row is emitted), so this is P1-clean and mirrors
    how the house Elo updates within a season. Returns a frame indexed by
    game_id with FORM_COLS.
    """
    g = gc.sort_values("date").reset_index(drop=True)
    hist: dict = {}
    rows = []
    for r in g.itertuples(index=False):
        rec = {}
        for side, tid in (("h", r.home_idx), ("a", r.away_idx)):
            last = hist.get(tid, [])[-FORM_N:]
            if last:
                rec[f"{side}_gf"] = float(np.mean([x[0] for x in last]))
                rec[f"{side}_ga"] = float(np.mean([x[1] for x in last]))
                rec[f"{side}_pts"] = float(np.mean([x[2] for x in last]))
            else:
                rec[f"{side}_gf"] = FORM_PRIOR["gf"]
                rec[f"{side}_ga"] = FORM_PRIOR["ga"]
                rec[f"{side}_pts"] = FORM_PRIOR["pts"]
        rows.append(rec)
        home_win = r.home_g > r.away_g
        extra = r.outcome4 >= 2
        hist.setdefault(r.home_idx, []).append(
            (r.home_g, r.away_g, 2 if home_win else (1 if extra else 0)))
        hist.setdefault(r.away_idx, []).append(
            (r.away_g, r.home_g, 2 if not home_win else (1 if extra else 0)))
    out = pd.DataFrame(rows)
    out["game_id"] = g.game_id.values
    return out.set_index("game_id")


def form_vector(rec) -> list:
    """Scaled form features for the ctx slots (order = FORM_COLS)."""
    return [rec["h_gf"] / 3.0, rec["h_ga"] / 3.0, rec["h_pts"] / 2.0,
            rec["a_gf"] / 3.0, rec["a_ga"] / 3.0, rec["a_pts"] / 2.0]


def era_table_row(T: int) -> np.ndarray:
    """Scaled walk-forward era vector for season T (shared with sim/)."""
    from data.tensorize_games import era_table
    era = era_table()
    v = np.array([era.loc[T, c] for c in ERA_COLS], np.float32) * ERA_SCALE
    return np.nan_to_num(v)


def load_frames(seasons):
    gc, pg = [], []
    for se in seasons:
        g = pd.read_parquet(TENSORS / f"games_ctx_{se}.parquet")
        g["season_end"] = se
        gc.append(g[g.game_type == 2])
        p = pd.read_parquet(TENSORS / f"player_games_{se}.parquet")
        p = p[p.game_type == 2].copy()
        p["season_end"] = se
        pg.append(p)
    gc = pd.concat(gc, ignore_index=True)
    pg = pd.concat(pg, ignore_index=True)
    dates = gc.set_index("game_id").date.astype(str)
    pg["date"] = pg.game_id.map(dates)
    return gc, pg


def add_form(pg: pd.DataFrame) -> pd.DataFrame:
    pg = pg.sort_values(["player_id", "date", "game_id"]).copy()
    grp = pg.groupby("player_id")
    pg["ewma_toi"] = grp.toi_sec.transform(
        lambda s: s.ewm(alpha=EWMA_ALPHA).mean().shift(1)).fillna(0) / 60.0
    pg["gp_todate"] = grp.cumcount()
    return pg


def build_tensors(gc, pg, emb_npz, smoke=False, form=None):
    ids = emb_npz["ids"]
    emb = emb_npz["emb"]
    row = {int(p): i for i, p in enumerate(ids)}
    d = emb.shape[1]
    pos_mean = np.zeros((3, d), dtype=np.float32)
    bios = pd.read_parquet(TENSORS / "career_bios.parquet")
    for pgi in range(3):
        sel = [row[p] for p in bios[bios.pos_group == pgi].player_id
               if p in row]
        if sel:
            pos_mean[pgi] = emb[sel].mean(0)
    if smoke:
        gc = gc.head(400)
    pg = pg[pg.game_id.isin(set(gc.game_id))]
    # games with no player rows have no usable inputs (2 unparseable 2009 HTM
    # reports); drop rather than feed empty rosters (bugfix gm-fix-1)
    n_before = len(gc)
    gc = gc[gc.game_id.isin(set(pg.game_id))]
    if len(gc) < n_before:
        print(f"  dropped {n_before - len(gc)} games with no roster data")
    by_game = {k: v for k, v in pg.groupby("game_id")}
    N = len(gc)
    out = {
        "emb": np.zeros((N, 2, 20, d), np.float32),
        "pctx": np.zeros((N, 2, 20, 6), np.float32),
        "pad": np.ones((N, 2, 20), bool),
        "ctx": np.zeros((N, 16), np.float32),
        "era": np.zeros((N, 8), np.float32),
        "outcome4": np.zeros(N, np.int64),
        "goals_h": np.zeros(N, np.int64), "goals_a": np.zeros(N, np.int64),
        "sat5_h": np.zeros(N, np.int64), "sat5_a": np.zeros(N, np.int64),
    }
    for side in ("h", "a"):
        out[f"skater_mask_{side}"] = np.zeros((N, 20), bool)
    for f in ("toi", "shots", "goals_p", "assists_p"):
        out[f"{f}_h"] = np.zeros((N, 20), np.float32)
        out[f"{f}_a"] = np.zeros((N, 20), np.float32)
    # player id per roster slot: not a model input, but lets evaluation join
    # each slot back to that player's own pre-game baseline
    out["pids"] = np.zeros((N, 2, 20), np.int64)
    n_coldstart = 0
    game_ids = np.zeros(N, np.int64)
    for i, g in enumerate(gc.itertuples(index=False)):
        game_ids[i] = g.game_id
        rows = by_game.get(g.game_id)
        if rows is None:
            continue
        for t, is_home in ((0, True), (1, False)):
            side = rows[rows.is_home == is_home]
            gk = side[side.pos_group == 2].sort_values(
                ["goalie_start", "ewma_toi"], ascending=False)
            sk = side[side.pos_group < 2].sort_values("ewma_toi",
                                                     ascending=False)
            sel = pd.concat([gk.head(2), sk.head(18)])
            suf = "h" if is_home else "a"
            for j, r in enumerate(sel.itertuples(index=False)):
                if j >= 20:
                    break
                ridx = row.get(r.player_id)
                if ridx is None:
                    out["emb"][i, t, j] = pos_mean[r.pos_group]
                    n_coldstart += 1
                else:
                    out["emb"][i, t, j] = emb[ridx]
                pos1h = np.zeros(3, np.float32)
                pos1h[r.pos_group] = 1
                out["pctx"][i, t, j] = [*pos1h, r.ewma_toi / 20.0,
                                        np.log1p(r.gp_todate) / 5.0,
                                        r.goalie_start]
                out["pad"][i, t, j] = False
                out["pids"][i, t, j] = r.player_id
                if r.pos_group < 2 and j >= 2:
                    k = j
                    out[f"skater_mask_{suf}"][i, k] = True
                    out[f"toi_{suf}"][i, k] = r.toi_sec
                    out[f"shots_{suf}"][i, k] = r.sog
                    out[f"goals_p_{suf}"][i, k] = r.goals
                    out[f"assists_p_{suf}"][i, k] = r.assists
        b2b_h = float(getattr(g, "home_rest", 9) <= 1)
        b2b_a = float(getattr(g, "away_rest", 9) <= 1)
        out["ctx"][i, :9] = [min(g.home_rest, 7) / 7, min(g.away_rest, 7) / 7,
                             b2b_h, b2b_a, g.home_km3d / 1000,
                             g.away_km3d / 1000, g.home_dtz / 3,
                             g.away_dtz / 3, g.days_in / 200]
        if form is not None and g.game_id in form.index:
            out["ctx"][i, 9:15] = form_vector(form.loc[g.game_id])
        out["era"][i] = np.array([getattr(g, c) for c in ERA_COLS],
                                 np.float32) * ERA_SCALE
        out["outcome4"][i] = g.outcome4
        out["goals_h"][i] = g.home_g
        out["goals_a"][i] = g.away_g
        out["sat5_h"][i] = g.sat5_h
        out["sat5_a"][i] = g.sat5_a
    out["ctx"] = np.nan_to_num(out["ctx"])
    out["era"] = np.nan_to_num(out["era"])
    tensors = {k: torch.as_tensor(v) for k, v in out.items()}
    return tensors, game_ids, n_coldstart


def slice_batch(tensors, idx):
    return {k: v[idx] for k, v in tensors.items()}


def eval_probs(model, tensors, device, bs=512):
    model.eval()
    ps = []
    with torch.no_grad():
        n = len(tensors["outcome4"])
        for i in range(0, n, bs):
            b = {k: v[i:i + bs].to(device) for k, v in tensors.items()}
            out = model(b)
            ps.append(torch.softmax(out["out4"], -1).cpu())
    return torch.cat(ps)


def logloss4(p, y):
    return float(-torch.log(p.gather(1, y.unsqueeze(1)).clamp(1e-9)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict-season", type=int, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--train-from", type=int, default=2009)
    ap.add_argument("--no-test", action="store_true",
                    help="predict season not yet played (T=2027): train+val "
                         "only, still writes game_val_<T>.csv for calibration")
    args = ap.parse_args()
    T = args.predict_season
    cfg = json.loads((TENSORS.parents[1] / "configs" / "game_model.json")
                     .read_text())
    seasons = list(range(args.train_from, (T if args.no_test else T) +
                         (0 if args.no_test else 1)))
    gc, pg = load_frames(seasons)
    pg = add_form(pg)
    emb_npz = np.load(TENSORS / f"embeddings_v{T}.npz")
    # in-season rolling form over the whole ordered history (amendment A1);
    # each row uses only games before it, matching how house Elo updates
    form = team_form(gc) if cfg.get("use_team_form") else None
    if cfg.get("val_mode") == "iid_holdout":
        # Selecting epochs on the single adjacent season made every seed fit
        # that season's idiosyncrasies: at T=2012 the val season implied
        # temperature 1.06 while the test season needed 4.90. Instead hold out
        # a seeded random slice of ALL past seasons (i.i.d. with training, and
        # strictly < T so still P1-clean) and train on season T-1 as well,
        # which also recovers ~1,230 games of scarce training data.
        all_gc = gc[gc.season_end < T]
        tens_all, _, cold_tr = build_tensors(all_gc, pg, emb_npz, args.smoke,
                                             form)
        n_all = len(tens_all["outcome4"])
        rng = np.random.default_rng(9000 + T)
        perm = rng.permutation(n_all)
        n_val = int(round(cfg.get("val_frac", 0.15) * n_all))
        vi = torch.as_tensor(perm[:n_val].copy())
        ti = torch.as_tensor(perm[n_val:].copy())
        tens_tr = {k: v[ti] for k, v in tens_all.items()}
        tens_va = {k: v[vi] for k, v in tens_all.items()}
    else:
        tr_gc = gc[gc.season_end < T - 1]
        va_gc = gc[gc.season_end == T - 1]
        tens_tr, _, cold_tr = build_tensors(tr_gc, pg, emb_npz, args.smoke,
                                            form)
        tens_va, _, _ = build_tensors(va_gc, pg, emb_npz, args.smoke, form)
    if args.no_test:
        tens_te, te_ids, cold_te = tens_va, np.zeros(0, np.int64), 0
    else:
        te_gc = gc[gc.season_end == T]
        tens_te, te_ids, cold_te = build_tensors(te_gc, pg, emb_npz,
                                                 args.smoke, form)
    print(f"T={T}: train {len(tens_tr['outcome4'])}, val "
          f"{len(tens_va['outcome4'])}, test {len(tens_te['outcome4'])} "
          f"(cold-start slots: train {cold_tr}, test {cold_te})")
    device = "mps" if torch.backends.mps.is_available() and not args.smoke \
        else "cpu"
    W = cfg["loss_weights"]
    ens, ens_val = [], []
    for seed in (args.seeds[:1] if args.smoke else args.seeds):
        torch.manual_seed(1000 * T + seed)
        np.random.seed(1000 * T + seed)
        arch = cfg.get("arch", {})
        model = GameModel(d=arch.get("d", 128),
                          trunk_dim=arch.get("trunk_dim", 512),
                          dropout=arch.get("dropout", 0.1),
                          enc_layers=arch.get("enc_layers", 2),
                          n_heads=arch.get("n_heads", 4)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                weight_decay=cfg.get("weight_decay", 0.01))
        n = len(tens_tr["outcome4"])
        best, best_ep, patience = float("inf"), -1, 10
        ck = CKPT / f"game_T{T}_s{seed}.pt"
        epochs = 2 if args.smoke else cfg["epochs_max"]
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(n)
            for i in range(0, n, cfg["batch_size"]):
                idx = perm[i:i + cfg["batch_size"]]
                b = {k: v[idx].to(device) for k, v in tens_tr.items()}
                opt.zero_grad(set_to_none=True)
                losses = game_loss(model(b), b, W,
                                   cfg["label_smoothing_outcome4"])
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            pv = eval_probs(model, tens_va, device)
            vll = logloss4(pv, tens_va["outcome4"])
            if ep % 5 == 0 or args.smoke:
                print(f"  seed {seed} ep {ep}: val out4 ll {vll:.4f}")
                sys.stdout.flush()
            if vll < best - 1e-4:
                best, best_ep = vll, ep
                torch.save(model.state_dict(), ck)
            elif ep - best_ep >= patience:
                break
        model.load_state_dict(torch.load(ck, map_location=device))
        pt = eval_probs(model, tens_te, device)
        ens.append(pt)
        ens_val.append(eval_probs(model, tens_va, device))
        (ck.with_suffix(".sha256")).write_text(
            hashlib.sha256(ck.read_bytes()).hexdigest())
        print(f"  seed {seed}: best val {best:.4f}; test out4 ll "
              f"{logloss4(pt, tens_te['outcome4']):.4f}")
    (NOUT / "preds").mkdir(parents=True, exist_ok=True)
    if not args.no_test:
        P = torch.stack(ens).mean(0)
        y = tens_te["outcome4"]
        p_home = (P[:, 0] + P[:, 2]).clamp(1e-9, 1 - 1e-9)
        y_home = ((y == 0) | (y == 2)).float()
        ll_home = float(-(y_home * torch.log(p_home)
                          + (1 - y_home) * torch.log(1 - p_home)).mean())
        df = pd.DataFrame({"game_id": te_ids, "p_home_reg": P[:, 0],
                           "p_away_reg": P[:, 1], "p_home_extra": P[:, 2],
                           "p_away_extra": P[:, 3], "outcome4": y})
        df.to_csv(NOUT / "preds" / f"game_preds_{T}.csv", index=False)
    # val-season ensemble preds: the ONLY data calibrate.py may fit on (P1/P7)
    Pv = torch.stack(ens_val).mean(0)
    pd.DataFrame({"p_home_reg": Pv[:, 0], "p_away_reg": Pv[:, 1],
                  "p_home_extra": Pv[:, 2], "p_away_extra": Pv[:, 3],
                  "outcome4": tens_va["outcome4"]}).to_csv(
        NOUT / "preds" / f"game_val_{T}.csv", index=False)
    if args.no_test:
        print(f"T={T}: no-test mode — checkpoints + game_val_{T}.csv written")
    else:
        print(f"T={T} ENSEMBLE: out4 ll {logloss4(P, y):.5f}, "
              f"home-win ll {ll_home:.5f} -> preds/game_preds_{T}.csv")


if __name__ == "__main__":
    main()
