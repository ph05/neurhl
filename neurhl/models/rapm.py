"""NeurHL-2 — stint-level ridge RAPM (Macdonald 2011 JQAS; Gramacy et al. 2013).

The quantity every downstream player model is anchored on. Raw on-ice rates are
a TEAM property: a fourth-liner buried against top opposition and a first-liner
starting in the offensive zone are measured on incomparable scales. RAPM
deconfounds them by regressing stint outcomes on WHO WAS ON THE ICE, so each
player's coefficient is identified by the many different teammate/opponent
combinations he appears in.

Design, per 5v5 stint, two observations (one per attacking side):

    y   = events_for / (duration / 3600)          rate per 60
    X   = +1 on each attacking skater's OFFENCE column
          +1 on each defending skater's DEFENCE column
          +1 home indicator, +1 intercept          (both unpenalised)
    w   = duration in seconds

Weighting by duration while the target is a per-60 rate is deliberate: the
contribution to X'Wy is dur * (count * 3600 / dur) = 3600 * count, so a 1-second
stint containing a goal contributes proportional to the goal, not an exploded
rate. The design is stable at any stint length without an arbitrary minimum.

Solved as ridge with an unpenalised intercept/home block:

    (X'WX + lam*P) b = X'Wy,   P = diag(0,0,1,1,...,1)

by Cholesky, so the fit is DETERMINISTIC and bit-reproducible (no seeds, no
iteration count, no early stopping). Posterior SEs come from the sandwich
diagonal and are RETAINED: S5 samples player effects from them, so a 20-game
rookie is not simulated with a 500-game veteran's confidence.

Players below `min_toi_s` in the window are POOLED into a per-position
replacement column rather than given their own coefficient. This both
conditions the system and produces the replacement-level baseline S6 needs for
cold-start players, instead of leaving it implicit.

Vantage discipline: `fit()` takes an explicit season list. To predict season V,
callers pass seasons <= V-1 (registry vantage PRIOR). Nothing here reads V.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import TENSORS  # noqa: E402

N_SK = 6
H_COLS = [f"h_s{i}" for i in range(1, N_SK + 1)]
A_COLS = [f"a_s{i}" for i in range(1, N_SK + 1)]
TARGETS = {"cf": ("cf_h", "cf_a"), "sog": ("sog_h", "sog_a"),
           "g": ("g_h", "g_a")}
POS_F, POS_D, POS_G = 0, 1, 2
N_FIXED = 2                       # intercept, home
TEAM_PEN = 10.0                   # identifies the collinear team-FE block only


def load_stints(seasons, game_type=2, only_5v5=True,
                with_teams=True) -> pd.DataFrame:
    """Stints for the given seasons.

    `with_teams` attaches TEAM-SEASON ids (team identity crossed with season)
    for the A6 fixed effects: a franchise is a different object in 2009 than in
    2011, so a single team id would pool three different rosters and defeat the
    purpose of absorbing team level.
    """
    parts = []
    for s in seasons:
        p = TENSORS / f"stints_{s}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        d = d[d.game_type == game_type]
        if only_5v5:
            d = d[d.is_5v5]
        if with_teams:
            gc = TENSORS / f"games_ctx_{s}.parquet"
            if gc.exists():
                g = pd.read_parquet(gc, columns=["game_id", "home_idx",
                                                 "away_idx"]).set_index("game_id")
                d = d.copy()
                # home_idx is uint8; cast before crossing with season or the
                # team-season id silently overflows
                d["team_off"] = (d.game_id.map(g.home_idx).astype("float64")
                                 + 100 * s)
                d["team_def"] = (d.game_id.map(g.away_idx).astype("float64")
                                 + 100 * s)
        parts.append(d)
    if not parts:
        raise FileNotFoundError(f"no stint shards for seasons {list(seasons)}")
    return pd.concat(parts, ignore_index=True)


def player_positions(seasons) -> dict:
    """playerId -> pos_group, from the official dressed-roster tables.

    Never inferred from on-ice slot index: slots are sorted by playerId, so slot
    order carries no positional meaning at all.
    """
    pos = {}
    for s in seasons:
        p = TENSORS / f"player_games_{s}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p, columns=["player_id", "pos_group"])
        for pid, pg in d.drop_duplicates("player_id").itertuples(index=False):
            pos.setdefault(int(pid), int(pg))
    return pos


class RAPMDesign:
    """Sparse design over stints; column layout is fixed and introspectable."""

    def __init__(self, st: pd.DataFrame, min_toi_s: int = 12000,
                 positions: dict | None = None, team_fe: bool = True):
        self.min_toi_s = min_toi_s
        self.team_fe = team_fe
        h = st[H_COLS].to_numpy(np.int64)
        a = st[A_COLS].to_numpy(np.int64)
        dur = st.dur_s.to_numpy(np.float64)

        # h/a are (n_stints, N_SK) row-major, so element i*N_SK+c belongs to
        # stint i -> repeat (not tile) the durations to line them up.
        flat = np.concatenate([h.ravel(), a.ravel()])
        wflat = np.tile(np.repeat(dur, N_SK), 2)
        m = flat > 0
        tot = np.bincount(flat[m], weights=wflat[m])
        self.toi = {int(p): float(tot[p]) for p in np.flatnonzero(tot)}
        toi = self.toi
        positions = positions or {}

        keep = sorted(p for p, v in toi.items() if v >= min_toi_s and p > 0)
        self.players = keep
        self.pid_to_slot = {p: i for i, p in enumerate(keep)}
        n = len(keep)
        # pooled replacement columns, one per position group
        self.repl_slot = {POS_F: n, POS_D: n + 1}
        self.n_eff = n + 2
        self.positions = positions
        # Team-season fixed effects (A6). Within a season a player's column is
        # nearly collinear with his team's roster -- he plays essentially every
        # shift for one club -- so plain ridge cannot separate "this player is
        # good" from "his team is good" and the coefficients retain team level.
        # With team-season effects absorbed (unpenalised, like intercept/home),
        # player coefficients estimate performance RELATIVE TO their own team,
        # which is the identified quantity and the one that transfers when a
        # player moves.
        if self.team_fe and "team_off" in st.columns:
            ts = pd.unique(pd.concat([st.team_off, st.team_def]).dropna())
            self.team_slot = {int(t): i for i, t in enumerate(sorted(ts))}
        else:
            self.team_slot = {}
            self.team_fe = False
        self.n_team = len(self.team_slot)
        self.tm_off0 = N_FIXED
        self.tm_def0 = N_FIXED + self.n_team
        self.off0 = N_FIXED + 2 * self.n_team
        self.def0 = self.off0 + self.n_eff
        self.n_cols = self.def0 + self.n_eff
        self.n_unpen = N_FIXED + 2 * self.n_team

        allp = np.unique(np.concatenate([h.ravel(), a.ravel()]))
        allp = allp[allp > 0]
        self.lut = np.full(int(allp.max()) + 2, -1, np.int64)
        for pid in allp:
            self.lut[int(pid)] = self._slot(int(pid))
        self.X, self.w, self.rows_meta = self._build(h, a, dur, st)

    def _slot(self, pid: int) -> int:
        s = self.pid_to_slot.get(pid)
        if s is not None:
            return s
        return self.repl_slot.get(self.positions.get(pid, POS_F),
                                  self.repl_slot[POS_F])

    def transform(self, st: pd.DataFrame):
        """Design for UNSEEN stints under this fit's column mapping.

        Players absent from the training window resolve to their position's
        pooled replacement column, which is what makes out-of-sample scoring
        (gate R2) and the 2026-27 projection well-defined for newcomers instead
        of silently dropping them.
        """
        h = st[H_COLS].to_numpy(np.int64)
        a = st[A_COLS].to_numpy(np.int64)
        dur = st.dur_s.to_numpy(np.float64)
        need = int(max(h.max(initial=0), a.max(initial=0))) + 2
        if need > len(self.lut):
            grow = np.full(need, -1, np.int64)
            grow[:len(self.lut)] = self.lut
            self.lut = grow
        unseen = np.flatnonzero(self.lut == -1)
        if len(unseen):
            self.lut[unseen] = self.repl_slot[POS_F]
        for pid in np.unique(np.concatenate([h.ravel(), a.ravel()])):
            if pid > 0 and self.lut[int(pid)] < 0:
                self.lut[int(pid)] = self._slot(int(pid))
        X, w, _ = self._build(h, a, dur, st)
        return X, w

    def _build(self, h, a, dur, st):
        n_st = len(dur)
        lut = self.lut
        rows, cols = [], []
        if self.team_fe:
            t_h = st.team_off.map(self.team_slot).to_numpy(dtype="float64")
            t_a = st.team_def.map(self.team_slot).to_numpy(dtype="float64")
        for attack_home in (True, False):
            base = np.arange(n_st) + (0 if attack_home else n_st)
            off_side, def_side = (h, a) if attack_home else (a, h)
            for side, col0 in ((off_side, self.off0), (def_side, self.def0)):
                for c in range(side.shape[1]):
                    pid = side[:, c]
                    m = pid > 0
                    rows.append(base[m])
                    cols.append(col0 + lut[pid[m]])
            rows.append(base)                       # intercept
            cols.append(np.zeros(n_st, np.int64))
            if attack_home:
                rows.append(base)                   # home indicator
                cols.append(np.ones(n_st, np.int64))
            if self.team_fe:
                ta, td = (t_h, t_a) if attack_home else (t_a, t_h)
                for tv, col0 in ((ta, self.tm_off0), (td, self.tm_def0)):
                    m = np.isfinite(tv)
                    rows.append(base[m])
                    cols.append(col0 + tv[m].astype(np.int64))
        r = np.concatenate(rows)
        c = np.concatenate(cols)
        X = sparse.csr_matrix(
            (np.ones(len(r), np.float64), (r, c)),
            shape=(2 * n_st, self.n_cols))
        w = np.concatenate([dur, dur])
        meta = {"n_stints": n_st, "n_rows": 2 * n_st,
                "n_players": len(self.players), "n_cols": self.n_cols}
        return X, w, meta

    def targets(self, st: pd.DataFrame, names) -> dict:
        dur = st.dur_s.to_numpy(np.float64)
        per60 = 3600.0 / np.maximum(dur, 1.0)
        out = {}
        for nm in names:
            fh, fa = TARGETS[nm]
            out[nm] = np.concatenate([st[fh].to_numpy(np.float64) * per60,
                                      st[fa].to_numpy(np.float64) * per60])
        return out


def normal_equations(design: RAPMDesign, ys: dict) -> tuple:
    """X'WX and X'Wy, formed ONCE. The whole ridge path is then nearly free,
    which is what makes an honest lambda search cheap enough to actually run."""
    X, w = design.X, design.w
    Xw = X.multiply(w[:, None]).tocsr()
    XtWX = np.asarray((X.T @ Xw).todense())
    XtWy = {nm: np.asarray(Xw.T @ y).ravel() for nm, y in ys.items()}
    return XtWX, XtWy


def solve(design: RAPMDesign, ys: dict, lam: float, want_se=True,
          normal=None) -> dict:
    """Weighted ridge with unpenalised intercept/home, by Cholesky."""
    X, w = design.X, design.w
    XtWX, XtWy = normal if normal is not None else normal_equations(design, ys)
    pen = np.full(design.n_cols, lam)
    n_unpen = getattr(design, "n_unpen", N_FIXED)
    pen[:n_unpen] = 0.0
    # The team-season dummies sum to 1 on every row, so together with the
    # intercept they are EXACTLY collinear and the normal matrix is singular.
    # A token ridge on that block alone breaks the tie. Team diagonals are the
    # summed TOI of a whole roster (order 1e6-1e7 seconds), so TEAM_PEN is
    # ~1e-6 relative -- it identifies the system without shrinking team effects
    # in any meaningful sense, and player columns keep the full lambda.
    if n_unpen > N_FIXED:
        pen[N_FIXED:n_unpen] = TEAM_PEN
    cf = cho_factor(XtWX + np.diag(pen), lower=True, check_finite=False)

    out = {"lam": lam, "n_cols": design.n_cols, **design.rows_meta}
    coefs = {}
    for nm, y in ys.items():
        b = cho_solve(cf, XtWy[nm], check_finite=False)
        coefs[nm] = b
        resid = y - X @ b
        dof = max(X.shape[0] - design.n_cols, 1)
        out[f"sigma2_{nm}"] = float((w * resid ** 2).sum() / dof)
    out["coef"] = coefs

    if want_se:
        Ainv = cho_solve(cf, np.eye(design.n_cols), check_finite=False)
        sand = np.einsum("ij,jk,ki->i", Ainv, XtWX, Ainv)
        out["se_unit"] = np.sqrt(np.maximum(sand, 0.0))
    return out


def wmse(y, pred, w) -> float:
    """Duration-weighted MSE, the quantity gate R2 is scored on."""
    return float((w * (y - pred) ** 2).sum() / w.sum())


def to_frame(design: RAPMDesign, res: dict, target: str) -> pd.DataFrame:
    b = res["coef"][target]
    o, d = design.off0, design.def0
    n = design.n_eff
    se = res.get("se_unit")
    sig = np.sqrt(res.get(f"sigma2_{target}", 1.0))
    rows = []
    for pid, s in design.pid_to_slot.items():
        rows.append((pid, float(b[o + s]), float(b[d + s]),
                     float(se[o + s] * sig) if se is not None else np.nan,
                     float(se[d + s] * sig) if se is not None else np.nan,
                     design.toi.get(pid, 0.0), 0))
    for pg, s in design.repl_slot.items():
        rows.append((-1 - pg, float(b[o + s]), float(b[d + s]),
                     float(se[o + s] * sig) if se is not None else np.nan,
                     float(se[d + s] * sig) if se is not None else np.nan,
                     0.0, 1))
    df = pd.DataFrame(rows, columns=["player_id", f"{target}_off",
                                     f"{target}_def", f"{target}_off_se",
                                     f"{target}_def_se", "toi_s",
                                     "is_replacement"])
    df.attrs["intercept"] = float(b[0])
    df.attrs["home"] = float(b[1])
    return df


def fingerprint(res: dict, target: str) -> str:
    """Bit-level identity of a fit, so reruns are provably identical."""
    b = np.ascontiguousarray(res["coef"][target], dtype=np.float64)
    return hashlib.sha256(b.tobytes()).hexdigest()[:16]


def fit(seasons, targets=("cf", "g"), lam=200.0, min_toi_s=12000,
        only_5v5=True) -> tuple:
    st = load_stints(seasons, only_5v5=only_5v5)
    pos = player_positions(seasons)
    d = RAPMDesign(st, min_toi_s=min_toi_s, positions=pos)
    ys = d.targets(st, targets)
    res = solve(d, ys, lam)
    return d, res, st
