"""L3 (ORR 1.1 plan): player-specific games-played projections.

The current skater pipeline (orr.players.pipeline, strict protocol) sets each
rostered skater's expected games in four steps:

  1. structural games from the depth-chart recursion (orr.deploy.deploy:
     health rate q from injury spells, dressing competition, call-up cover);
  2. a walk-forward OLS stack on [1, structural, GP(V-1), GP(V-2)] for
     players with NHL history (orr.deploy.gp_stack / apply_stack);
  3. fixed multipliers for rookies and thin records (Params.rk_gp_mult,
     thin_gp_mult);
  4. a logit shift so each team-position's roster games equal the walk-forward
     coverage share of its dressed slots (deploy with gp_override + cover),
     after which ice time is conserved within team.

This module replaces steps 2-3 with a walk-forward model of the season games
SHARE (actual GP in V for any team / schedule), trained on the same kind of
roster in seasons before V and using each player's own availability history:
games played, injury-spell games and healthy-scratch games in each of the
last three seasons, age, position and usage tier (projected minutes, depth
rank on the roster, last season's minutes per game), plus the structural
share from step 1. Step 4 (coverage and ice-time conservation) is kept, so
team totals are unchanged and only the split within a team moves.

Nothing else in the pipeline changes. `pipeline_l3` and `run_season_l3` are
copies of orr.players.pipeline and orr.backtest.players_bt.run_season (strict
protocol, means only) with the games step swapped; with model='none' they
reproduce the current pipeline exactly (checked by `verify`).
"""
from __future__ import annotations

import functools
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from orr import config as C
from orr import data as D
from orr import deploy as DP
from orr import players as PL
from orr.backtest import players_bt as BT

GAMES = BT.GAMES            # 82 (players_bt scores every season on 82 games)
ABS_FIRST = 2012            # first season of the absences file
FIRST_TRAIN = 2011          # first season with opening-roster proxies (boxes)


@dataclass(frozen=True)
class GPConfig:
    model: str = "gbm"          # 'gbm' | 'glm' | 'none' (= current pipeline)
    feats: str = "full"         # 'full' | 'nostruct' (drop the structural share)
    rookies: str = "model"      # 'model': rookies too; 'old': rookies keep steps 1+3
    window: int = 8             # training seasons before V (walk-forward)
    lr: float = 0.05            # gbm learning rate
    iters: int = 200            # gbm boosting iterations
    leaves: int = 15            # gbm max leaf nodes
    min_leaf: int = 40          # gbm min samples per leaf
    l2: float = 1.0             # gbm l2 regularisation
    ridge: float = 1.0          # glm ridge (on standardised coefficients)
    paths: str = "orr"          # forecast averaging: 'orr' = only ORR's path uses
                                # the new games; 'all' = Marcel's path too

    def key(self) -> tuple:
        return tuple(sorted(asdict(self).items()))


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _abs_by_season() -> pd.DataFrame:
    a = D.absences()
    return a.groupby(["season_end", "player_id"])[
        ["dressed", "window_games", "injury_spell_games", "scratch_games"]].sum()


def _sched(y: int) -> float:
    return DP._sched(y)


def history_features(V: int, ids) -> pd.DataFrame:
    """Availability history before V (82-game basis). NaN where the source
    does not cover the season (panel before 2009, absences before 2012)."""
    P = PL.panel()
    ids = np.asarray(ids)
    rows = pd.Series(ids).map(P.row)
    ok = rows.notna().to_numpy()
    ri = rows.fillna(0).astype(int).to_numpy()
    out = pd.DataFrame({"player_id": ids})
    ab = _abs_by_season()
    for k in (1, 2, 3):
        y = V - k
        if P.seasons[0] <= y <= P.seasons[-1]:
            j = P.j(y)
            g = np.where(ok, P.x["gp"][ri, j], 0.0)
            out[f"gp{k}"] = g * 82.0 / _sched(y)
            if k == 1:
                toi = np.where(ok, P.x["toi_all"][ri, j], 0.0)
                out["toipg1"] = np.where(g > 0, toi / np.maximum(g, 1), np.nan)
        else:
            out[f"gp{k}"] = np.nan
            if k == 1:
                out["toipg1"] = np.nan
        if y >= ABS_FIRST and y in ab.index.get_level_values(0):
            s = ab.xs(y, level="season_end")
            f = 82.0 / _sched(y)
            out[f"inj{k}"] = pd.Series(ids).map(s.injury_spell_games).fillna(0).to_numpy() * f
            out[f"scr{k}"] = pd.Series(ids).map(s.scratch_games).fillna(0).to_numpy() * f
            if k == 1:
                out["win1"] = pd.Series(ids).map(s.window_games).fillna(0).to_numpy() * f
        else:
            out[f"inj{k}"] = np.nan
            out[f"scr{k}"] = np.nan
            if k == 1:
                out["win1"] = np.nan
    # previous main team (for 'same club' flag)
    if P.seasons[0] <= V - 1 <= P.seasons[-1]:
        out["team1"] = np.where(ok, P.team[ri, P.j(V - 1)], None)
    else:
        out["team1"] = None
    return out


FEATS_FULL = ["struct", "gp1", "gp2", "gp3", "inj1", "inj2", "inj3", "scr1", "scr2",
              "scr3", "win1", "toipg1", "age", "is_d", "score", "rank", "rank_rel",
              "n_pos", "q", "log_nhl_gp", "log_pick", "same_team", "has_hist"]


def features(V: int, dep: pd.DataFrame, proj: pd.DataFrame, games: int) -> pd.DataFrame:
    """One row per deployed (rostered) skater: dep = structural deploy output."""
    d = dep[["player_id", "team", "pos", "score", "q", "has_hist", "gp", "games_out", "age"]].copy()
    avail = (games - d.games_out.to_numpy(float)) / games
    d["avail"] = avail
    d["struct"] = d.gp.to_numpy(float) / np.maximum(avail, 1e-6) / games
    d["rank"] = DP.depth_rank(d).to_numpy(float)
    d["n_pos"] = d.groupby(["team", "pos"]).player_id.transform("size").astype(float)
    d["rank_rel"] = d["rank"] / d.pos.map(DP.N_DRESS).astype(float)
    d["is_d"] = (d.pos == "D").astype(float)
    pj = proj.set_index("player_id")
    d["log_nhl_gp"] = np.log1p(d.player_id.map(pj.nhl_gp).fillna(0.0).to_numpy(float))
    names = d.player_id.map(pj.name) if "name" in pj else None
    d["log_pick"] = np.log(PL.draft_pick(d.player_id.to_numpy(),
                                         None if names is None else names.to_numpy()))
    h = history_features(V, d.player_id.to_numpy())
    d = d.merge(h, on="player_id", how="left")
    d["same_team"] = (d.team1.astype(object) == d.team.astype(object)).astype(float)
    d["has_hist"] = d.has_hist.astype(bool)
    return d


def season_target(s: int, ids) -> np.ndarray:
    """Games share of season s (any team), clipped to [0, 1]."""
    P = PL.panel()
    g = pd.Series(ids).map(dict(zip(P.ids, P.x["gp"][:, P.j(s)]))).fillna(0.0).to_numpy(float)
    return np.clip(g / _sched(s), 0.0, 1.0)


def structural(V: int, prm: PL.Params, roster: pd.DataFrame, kind: str, games: int,
               proj: pd.DataFrame):
    gmap = DP.league_gp_map(V, prm) if kind == "expost" else None
    cover = DP.coverage(V, "opening") if kind == "opening" else None
    ros = roster[roster.player_id.isin(proj.player_id)]
    dep = DP.deploy(proj, ros, V, games, prm.dress_noise, prm.q_scale, gp_map=gmap,
                    cover=cover)
    return dep, ros, gmap, cover


_FRAMES: dict = {}


def season_frame(s: int, prm: PL.Params, kind: str) -> pd.DataFrame | None:
    """Training rows for season s: this kind of roster ('opening': first-10
    proxy; 'expost': season-team), features as of the start of s, target."""
    key = (s, kind, prm.key(), prm.dress_noise, prm.q_scale)
    if key in _FRAMES:
        return _FRAMES[key]
    ros, flag = DP.historical_roster(s, kind)
    if kind == "opening" and flag != "first10":
        _FRAMES[key] = None
        return None
    proj = PL.calibrate_thin_rates(PL.project(s, prm, ids=ros.player_id), prm)
    dep, _, _, _ = structural(s, prm, ros, kind, GAMES, proj)
    F = features(s, dep, proj, GAMES)
    F["y"] = season_target(s, F.player_id.to_numpy())
    F["season"] = s
    _FRAMES[key] = F
    return F


def train_frame(V: int, prm: PL.Params, kind: str, window: int) -> pd.DataFrame | None:
    fr = []
    for s in range(max(FIRST_TRAIN, V - window), V):
        if s in C.BROKEN_SEASONS or s > PL.panel().seasons[-1]:
            continue
        f = season_frame(s, prm, kind)
        if f is not None and len(f):
            fr.append(f)
    return pd.concat(fr, ignore_index=True) if fr else None


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def _cols(cfg: GPConfig) -> list[str]:
    c = list(FEATS_FULL)
    if cfg.feats == "nostruct":
        c.remove("struct")
    return c


class GLM:
    """Fractional logit (binomial quasi-likelihood: the mean model of a
    beta-binomial with logit link), ridge on standardised coefficients.
    Missing blocks are imputed at 0 with an indicator."""

    def __init__(self, ridge: float):
        self.ridge = ridge

    def _design(self, X: pd.DataFrame, fit: bool) -> np.ndarray:
        A = X.astype(float).copy()
        miss = A.isna()
        if fit:
            self.miss_cols = [c for c in A.columns if miss[c].any()]
        for c in self.miss_cols:
            A[f"m_{c}"] = miss[c].astype(float)
        A = A.fillna(0.0)
        A["age2"] = (A.age - 27.0) ** 2
        A["struct2"] = A.struct ** 2 if "struct" in A else 0.0
        M = A.to_numpy(float)
        if fit:
            self.mu, self.sd = M.mean(0), M.std(0) + 1e-9
        return np.column_stack([np.ones(len(M)), (M - self.mu) / self.sd])

    def fit(self, X, y):
        Z = self._design(X, True)
        lam = self.ridge

        def f(b):
            p = np.clip(expit(Z @ b), 1e-9, 1 - 1e-9)
            ll = -(y * np.log(p) + (1 - y) * np.log(1 - p)).sum()
            g = Z.T @ (p - y)
            ll += 0.5 * lam * (b[1:] ** 2).sum()
            g[1:] += lam * b[1:]
            return ll, g
        b0 = np.zeros(Z.shape[1])
        b0[0] = np.log(max(y.mean(), 1e-3) / max(1 - y.mean(), 1e-3))
        self.b = minimize(f, b0, jac=True, method="L-BFGS-B",
                          options={"maxiter": 2000}).x
        return self

    def predict(self, X):
        return expit(self._design(X, False) @ self.b)


def make_model(cfg: GPConfig):
    if cfg.model == "glm":
        return GLM(cfg.ridge)
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=cfg.lr, max_iter=cfg.iters,
        max_leaf_nodes=cfg.leaves, min_samples_leaf=cfg.min_leaf,
        l2_regularization=cfg.l2, early_stopping=False, random_state=0)


_MODELS: dict = {}


def fitted_model(V: int, prm: PL.Params, kind: str, cfg: GPConfig):
    key = (V, kind, prm.key(), prm.dress_noise, prm.q_scale, cfg.model, cfg.feats,
           cfg.window, cfg.lr, cfg.iters, cfg.leaves, cfg.min_leaf, cfg.l2, cfg.ridge)
    if key in _MODELS:
        return _MODELS[key]
    tr = train_frame(V, prm, kind, cfg.window)
    if tr is None or len(tr) < 200:
        _MODELS[key] = None
        return None
    # features the training seasons do not cover at all (e.g. injury spells
    # before 2012 when V = 2012) carry no information: drop them
    cols = [c for c in _cols(cfg) if tr[c].notna().any()]
    X = tr[cols].astype(float)
    m = make_model(cfg).fit(X, tr.y.to_numpy(float))
    _MODELS[key] = (m, cols, int(len(tr)), sorted(tr.season.unique().tolist()))
    return _MODELS[key]


def predict_share(V, prm, kind, cfg, F: pd.DataFrame) -> np.ndarray | None:
    fm = fitted_model(V, prm, kind, cfg)
    if fm is None:
        return None
    m, cols, _, _ = fm
    return np.clip(m.predict(F[cols].astype(float)), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Pipeline copy (orr.players.pipeline, means only) with the games step swapped
# ---------------------------------------------------------------------------
def blend_paths_all(tot: pd.DataFrame, V: int, prm: PL.Params, games: int) -> pd.DataFrame:
    """Copy of orr.players.blend_paths in which every path uses ORR's games:
    Marcel's per-game goals/assists x ORR games, and the last-season-games
    path (ORR per game x last season's games) becomes ORR's own path."""
    wm, wl = prm.blend_marcel, prm.blend_lastgp
    if wm <= 0 and wl <= 0:
        return tot
    t = tot.copy()
    m = marcel_pg(V)
    ids = t.player_id
    mg_pg, ma_pg = ids.map(m.g_pg), ids.map(m.a_pg)
    has_m = mg_pg.notna()
    wm_i = np.where(has_m, wm, 0.0)
    wh = 1.0 - wm_i
    gp = t.gp.to_numpy(float)
    new_g = wh * t.g + wm_i * mg_pg.fillna(0) * gp
    new_a = wh * t.a + wm_i * ma_pg.fillna(0) * gp
    t["g_hattrick"], t["a_hattrick"] = t.g, t.a
    t["g"], t["a"] = new_g, new_a
    t["p"] = t.g + t.a
    return t


@functools.lru_cache(maxsize=None)
def marcel_pg(V: int) -> pd.DataFrame:
    m = PL.marcel_baseline(V).set_index("player_id")
    return pd.DataFrame({"g_pg": m.g / m.gp, "a_pg": m.a / m.gp})


def pipeline_l3(V: int, prm: PL.Params, roster: pd.DataFrame, games: int, kind: str,
                cfg: GPConfig, eval_ids=(), stack: bool = True, diag: dict | None = None):
    ids = set(roster.player_id) | set(eval_ids)
    proj = PL.calibrate_thin_rates(PL.project(V, prm, ids=ids), prm)
    dep, ros, gmap, cover = structural(V, prm, roster, kind, games, proj)
    gs = DP.apply_stack(dep, V, prm, kind, games) if stack else \
        dep.set_index("player_id").gp.to_dict()
    grp = dict(zip(proj.player_id, PL.thin_group(proj)))
    gm = PL.group_mults(prm, "gp")
    cap = dict(zip(dep.player_id, games - dep.games_out))
    gs = {k: min(v * gm[grp.get(k, "est")], cap.get(k, games)) for k, v in gs.items()}
    used = False
    if cfg.model != "none":
        F = features(V, dep, proj, games)
        sh = predict_share(V, prm, kind, cfg, F)
        if sh is not None:
            used = True
            g = sh * games * F.avail.to_numpy(float)
            take = np.ones(len(F), bool) if cfg.rookies == "model" else F.has_hist.to_numpy(bool)
            for pid, v, t in zip(F.player_id, g, take):
                if t:
                    gs[pid] = min(float(v), cap.get(pid, games))
    if diag is not None:
        diag["model_used"] = used
    dep = DP.deploy(proj, ros, V, games, prm.dress_noise, prm.q_scale, gp_override=gs,
                    cover=cover)
    tot = DP.add_totals(dep, proj)
    if cfg.paths == "all" and used:
        tot = blend_paths_all(tot, V, prm, games)
    else:
        tot = PL.blend_paths(tot, V, prm, games)
    thin = tot.player_id.map(dict(zip(proj.player_id, PL.thin_group(proj)))) == "thin"
    if prm.thin_rate_mult != 1.0 and thin.any():
        tot = tot.copy()
        cols = [c for c in tot.columns if c in ("g", "a", "p", "a1", "a2", "ppg", "ppa")
                or c.startswith(("g_p", "a_p", "p_p")) or c in ("g_sd", "a_sd", "p_sd")]
        tot.loc[thin, cols] = tot.loc[thin, cols] * prm.thin_rate_mult
    return tot.copy(), proj


def run_season_l3(V: int, prm: PL.Params, cfg: GPConfig, diag: dict | None = None) -> pd.DataFrame:
    """Copy of players_bt.run_season (strict protocol, sims=0)."""
    act = BT.actuals(V)
    roster, flag = DP.historical_roster(V, "opening")
    P = PL.panel()
    hist_ids = set(P.ids[PL._hist_gp(P, V) > 0])
    ids = set(roster.player_id) | (set(act.player_id) & hist_ids)
    kind = "opening" if flag == "first10" else "expost"
    tot, proj = pipeline_l3(V, prm, roster, GAMES, kind, cfg, eval_ids=ids,
                            stack=BT.STACK_GP, diag=diag)
    tot = tot.copy()
    tot["on_roster"] = True
    res = proj[~proj.player_id.isin(tot.player_id)].copy()
    if len(res):
        gp = BT.reserve_gp(V)
        res["gp"] = gp
        for k in PL.SITS:
            res[f"toi_{k}"] = gp * res[f"tpg_{k}"]
        rt = DP.add_totals(res[["player_id", "gp", *[f"toi_{k}" for k in PL.SITS]]], proj)
        rt["on_roster"] = False
        rt["team"] = None
        tot = pd.concat([tot, rt], ignore_index=True)
    tot = tot.drop(columns=["has_hist", "name", "age"], errors="ignore")
    tot = tot.merge(proj[["player_id", "has_hist", "name", "age", "pos"]]
                    .rename(columns={"pos": "pos_p"}), on="player_id", how="left")
    tot = tot.merge(act, on="player_id", how="left").copy()
    tot["roster_proxy"] = flag
    tot["V"] = V
    return tot


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def gp_rows(tot: pd.DataFrame) -> pd.DataFrame:
    """All rostered skaters (with or without NHL history); actual GP 0 if
    he did not play."""
    r = tot[tot.on_roster].copy()
    r["act_gp0"] = r.act_gp.fillna(0.0)
    return r


def season_metrics(tot: pd.DataFrame) -> dict:
    r = gp_rows(tot)
    e = r.gp - r.act_gp0
    mm = BT.metrics(tot)
    m, _ = BT.samples(tot)
    a = tot[m["all"]]
    return {"n_rostered": int(len(r)), "gp_mae": float(e.abs().mean()),
            "gp_bias": float(e.mean()),
            "gp_mae_hist": float(e[r.has_hist.fillna(False).astype(bool)].abs().mean()),
            "gp_mae_all": float((a.gp - a.act_gp).abs().mean()),
            "n_all": mm["all"]["n"], "p_mae_all": mm["all"]["mae"], "p_bias_all": mm["all"]["bias"],
            "n_A": mm["A"]["n"], "p_mae_A": mm["A"]["mae"],
            "n_B": mm["B"]["n"], "p_mae_B": mm["B"]["mae"]}


def pool(per: dict, seasons) -> dict:
    out = {}
    rows = [per[V] for V in seasons]
    for k, nk in (("gp_mae", "n_rostered"), ("gp_bias", "n_rostered"),
                  ("gp_mae_all", "n_all"), ("p_mae_all", "n_all"), ("p_bias_all", "n_all"),
                  ("p_mae_A", "n_A"), ("p_mae_B", "n_B")):
        n = sum(r[nk] for r in rows)
        out[k] = sum(r[k] * r[nk] for r in rows) / n
    out["n_rostered"] = sum(r["n_rostered"] for r in rows)
    out["n_all"] = sum(r["n_all"] for r in rows)
    return out
