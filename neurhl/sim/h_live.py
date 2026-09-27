"""NeurHL-H live forecasts: the confirmed game model on tonight's lineups.

NeurHL-H (confirmed against Elo, PLAN_NeurHL A5) = Layer 1 (per-player neural
residual on on-ice shot rates and ice time, train/train_player.py) aggregated
over the dressed skaters, then a thin logistic head on [Elo logit, projected
shot-share difference, projected close-shot-share difference, rest difference]
fit on all earlier seasons (eval/hier_core.py). This module runs it for tonight:

  * tonight's lineup rows are appended to season 2027 inside NeurHL-H's own
    frame builder (reads of the season-2027 tables are intercepted and extended;
    real 2026-27 games are included once the nightly ingest has built them), so
    every EWMA is read after the last played game, exactly as in backtests;
  * Layer 1 for predict-season 2027 is trained once on seasons <= 2026 and cached
    (checkpoints/player_T2027.pt, SHA-256 in its sidecar);
  * the head is refit on seasons 2011-2026 with the frozen C-selection rule.
"""
import contextlib
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import CKPT, TENSORS  # noqa: E402

SEASON = 2027


@contextlib.contextmanager
def tonight_rows(extra_pg: pd.DataFrame, extra_gc: pd.DataFrame):
    real = pd.read_parquet

    def rp(path, *a, columns=None, **k):
        name = Path(path).name
        add = {"player_games_2027.parquet": extra_pg,
               "games_ctx_2027.parquet": extra_gc,
               "onice_rates_2027.parquet": None}.get(name, "no")
        if isinstance(add, str):
            return real(path, *a, columns=columns, **k)
        base = real(path, *a, **k) if Path(path).exists() else pd.DataFrame()
        df = base if add is None else pd.concat([base, add], ignore_index=True)
        if name == "onice_rates_2027.parquet" and not len(df):
            df = pd.DataFrame(columns=["game_id", "player_id", "is_home", "cf", "ca",
                                       "cf5", "ca5", "clf", "cla"])
        return df[columns] if columns is not None else df
    pd.read_parquet = rp
    try:
        yield
    finally:
        pd.read_parquet = real


def layer1_model(d, emb, X, Y, BASE, is_tr):
    """Train (or load) Layer 1 for predict-season 2027 on seasons <= 2026."""
    from models.player_model import PlayerModel, player_loss
    ck = CKPT / f"player_T{SEASON}.pt"
    model = PlayerModel(d_player=emb.shape[1])
    if ck.exists():
        model.load_state_dict(torch.load(ck, map_location="cpu"))
        model.eval()
        return model
    torch.manual_seed(50000 + SEASON)
    np.random.seed(50000 + SEASON)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    Etr, Xtr = torch.as_tensor(emb[is_tr]), torch.as_tensor(X[is_tr])
    Ytr, Btr = torch.as_tensor(Y[is_tr]), torch.as_tensor(BASE[is_tr])
    n = len(Xtr)
    nv = max(n // 10, 1)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(7))
    vi, ti = perm[:nv], perm[nv:]
    best, best_ep = float("inf"), -1
    for ep in range(12):
        model.train()
        order = ti[torch.randperm(len(ti))]
        for i in range(0, len(order), 4096):
            b = order[i:i + 4096]
            opt.zero_grad(set_to_none=True)
            player_loss(model(Etr[b], Xtr[b], Btr[b]), Ytr[b])["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(player_loss(model(Etr[vi], Xtr[vi], Btr[vi]), Ytr[vi])["total"])
        if v < best - 1e-4:
            best, best_ep = v, ep
            torch.save(model.state_dict(), ck)
        if ep - best_ep >= 3:
            break
    ck.with_suffix(".json").write_text(json.dumps(
        {"predict_season": SEASON, "val": best, "n_train": int(is_tr.sum()),
         "sha256": hashlib.sha256(ck.read_bytes()).hexdigest()}, indent=1))
    model.load_state_dict(torch.load(ck, map_location="cpu"))
    model.eval()
    return model


def head():
    """Thin head refit on 2011-2026 exactly as hier_core.predict fits it."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from eval.backtest_hier import COLS_DEFAULT, CS, nll
    from eval.hier_core import game_frame
    d = game_frame(TENSORS, SEASON - 1)
    tr = d[d.season_end < SEASON]
    sc = StandardScaler().fit(tr[COLS_DEFAULT])
    best = (9.0, None)
    for C in CS:
        m = LogisticRegression(C=C, max_iter=4000).fit(sc.transform(tr[COLS_DEFAULT]), tr.y)
        l_ = nll(m.predict_proba(sc.transform(tr[COLS_DEFAULT]))[:, 1], tr.y.to_numpy()).mean()
        if l_ < best[0]:
            best = (l_, m)
    return sc, best[1], COLS_DEFAULT


def lineup_projection(games: pd.DataFrame, lineups: dict) -> pd.DataFrame:
    """NeurHL-H Layer-1 projections aggregated over tonight's dressed skaters:
    per game_id, cfpct_h/a and close-shot share clsh_h/a (also NeurHL-G inputs)."""
    import train.train_player as TP
    tidx = json.loads((TENSORS / "maps.json").read_text())["team"]
    bios = pd.read_parquet(TENSORS / "career_bios.parquet").set_index("player_id")
    pg_rows, gc_rows = [], []
    for r in games.itertuples():
        gc_rows.append({"game_id": r.game_id, "game_type": 2, "date": r.date,
                        "home_idx": tidx[r.home], "away_idx": tidx[r.away]})
        for side, is_home in (("home", 1), ("away", 0)):
            for pid in lineups[r.game_id][side]["skaters"]:
                pg_rows.append({"game_id": r.game_id, "game_type": 2, "player_id": int(pid),
                                "is_home": is_home, "pos_group": int(bios.pos_group.get(int(pid), 0)),
                                "toi_sec": 0})
    with tonight_rows(pd.DataFrame(pg_rows), pd.DataFrame(gc_rows)):
        d = TP.build_player_frame(list(range(2008, SEASON + 1)))
    emb_npz = np.load(TENSORS / f"embeddings_v{SEASON}.npz")
    emb, X, Y, BASE = TP.featurize(d, emb_npz)
    ids = set(games.game_id)
    tonight = d.game_id.isin(ids).to_numpy()
    is_tr = (d.season_end < SEASON).to_numpy()
    model = layer1_model(d, emb, X, Y, BASE, is_tr)
    with torch.no_grad():
        P = model(torch.as_tensor(emb[tonight]), torch.as_tensor(X[tonight]),
                  torch.as_tensor(BASE[tonight])).numpy()
    proj = d.loc[tonight, ["game_id", "is_home", "pos_group"]].copy()
    proj["p_toi"] = P[:, 4]
    for i, nm in enumerate(("cf", "ca", "clf", "cla")):
        proj[f"p_{nm}"] = P[:, i] * P[:, 4] / 3600.0
    agg = proj[proj.pos_group < 2].groupby(["game_id", "is_home"])[
        ["p_cf", "p_ca", "p_clf", "p_cla"]].sum()
    agg["cfpct"] = agg.p_cf / (agg.p_cf + agg.p_ca).clip(lower=1e-6)
    piv = agg.reset_index().pivot(index="game_id", columns="is_home")
    piv.columns = [f"{a}_{'h' if b else 'a'}" for a, b in piv.columns]
    piv["clsh_h"] = piv.p_clf_h / (piv.p_clf_h + piv.p_cla_h).clip(lower=1e-6)
    piv["clsh_a"] = piv.p_clf_a / (piv.p_clf_a + piv.p_cla_a).clip(lower=1e-6)
    return piv


def forecast(games: pd.DataFrame, lineups: dict, elo_logit: np.ndarray,
             rest: pd.DataFrame, piv: pd.DataFrame = None) -> np.ndarray:
    """games: game_id, date, home, away; lineups as for sim/g_live.build;
    elo_logit per game (sim/g_live.elo_logits); rest: game_id, home_rest, away_rest."""
    if piv is None:
        piv = lineup_projection(games, lineups)
    sc, m, cols = head()
    rs = rest.set_index("game_id")
    out = []
    for k, r in enumerate(games.itertuples()):
        if r.game_id not in piv.index:
            out.append(np.nan)
            continue
        q = piv.loc[r.game_id]
        clh, cla = q.clsh_h, q.clsh_a
        rd = (min(rs.home_rest.get(r.game_id, 3), 7) - min(rs.away_rest.get(r.game_id, 3), 7)) / 7
        x = pd.DataFrame([{"elo_logit": elo_logit[k], "proj_diff": q.cfpct_h - q.cfpct_a,
                           "proj_cl_diff": clh - cla, "rest_diff": rd}])[cols]
        out.append(float(m.predict_proba(sc.transform(x))[:, 1][0]))
    return np.array(out)
