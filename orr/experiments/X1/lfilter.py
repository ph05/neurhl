"""X1 task 4: the in-season filter with a lineup offset.

``run_filter_lineup`` is a copy of ``orr.ratings.run_filter`` (unchanged
except where marked ``# X1``): the lineup offsets lo_h / lo_a (log
multipliers on home / away regulation goals, ``lineups.offsets``) are added
to the goal context offsets, so they enter BOTH the pregame prediction and
the measurement update; with ``shot_mult`` > 0 the same offsets (times
shot_mult) also enter the shots rows. Games without a box score get 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from orr import ratings as R
from orr import structural as S
from orr.ratings import NG, NT, NSTATE, SLOT, HP, _ctx, _prior, _proc, _season_setup, split_prior


def run_filter_lineup(hp: HP, seasons, use_goalie: bool = False, first: int = 2008,
                      pre_override: dict | None = None, lineup: pd.DataFrame | None = None,
                      shot_mult: float = 0.0) -> pd.DataFrame:
    g = S.game_frame()
    gt = S.goalie_game_talent(*hp.goalie) if use_goalie else None
    seasons = sorted(seasons)
    last = max(seasons)
    Q = _proc(hp)
    rows = []
    # X1: lineup offsets by gid (0 where no box score)
    if lineup is not None:
        lu = g[["gid"]].merge(lineup[["gid", "lo_h", "lo_a"]], on="gid", how="left").fillna(0.0)
        LO_H = dict(zip(lu.gid, lu.lo_h))
        LO_A = dict(zip(lu.gid, lu.lo_a))
    for V in range(first, last + 1):
        gv = g[g.season_end == V].reset_index(drop=True)
        setup = _season_setup(V, hp)
        st = setup["st"]
        if pre_override and V in pre_override:
            r_V, mu_V = pre_override[V]
            setup["pre"] = split_prior(V, hp, r_V)
        x, P = _prior(V, hp, setup)
        if pre_override and V in pre_override:
            x[0] = mu_V
        x0, P0 = x.copy(), P.copy()
        cg_h, cg_a = _ctx(st["ctx"], gv, "h"), _ctx(st["ctx"], gv, "a")
        if use_goalie and st.get("ctx_gk") is not None:
            gm = gv[["gid"]].merge(gt, on="gid", how="left")
            known = (gm.gdiff_h.notna() & gm.gdiff_a.notna()).to_numpy()
            b = st["beta_gk"] * hp.gk_scale
            kh = _ctx(st["ctx_gk"], gv, "h") + b * np.nan_to_num(gm.gdiff_a.to_numpy())
            ka = _ctx(st["ctx_gk"], gv, "a") + b * np.nan_to_num(gm.gdiff_h.to_numpy())
            cg_h = np.where(known, kh, cg_h)
            cg_a = np.where(known, ka, cg_a)
        # X1: lineup offsets
        if lineup is not None:
            lh = gv.gid.map(LO_H).fillna(0.0).to_numpy()
            la = gv.gid.map(LO_A).fillna(0.0).to_numpy()
        else:
            lh = la = np.zeros(len(gv))
        cg_h = cg_h + lh
        cg_a = cg_a + la
        sh = st.get("shots") if hp.use_shots else None
        if sh is not None:
            cs_h = _ctx(sh["beta"], gv, "h") + sh["beta"]["margin"] * np.clip(gv.margin, -3, 3) \
                + sh["beta"]["extra"] * gv.went_extra + shot_mult * lh          # X1
            cs_a = _ctx(sh["beta"], gv, "a") + sh["beta"]["margin"] * np.clip(-gv.margin, -3, 3) \
                + sh["beta"]["extra"] * gv.went_extra + shot_mult * la          # X1
        hi = gv.home.map(SLOT).to_numpy()
        ai = gv.away.map(SLOT).to_numpy()
        disp_g = st["disp_g"] * hp.phi_g
        disp_s = (sh["dispersion"] * hp.phi_s) if sh is not None else None
        prev = None
        d0 = gv.date.iloc[0]
        for d, idx in gv.groupby("date").indices.items():
            if prev is not None:
                P += Q * (d - prev).days
            prev = d
            k = len(idx)
            h_, a_ = hi[idx], ai[idx]
            Hg = np.zeros((2 * k, NSTATE))
            r = np.arange(k)
            Hg[r, 0] = 1
            Hg[r, 1] = 1
            Hg[r, NG + NT * h_] = 1
            Hg[r, NG + NT * h_ + 2] = 1
            Hg[r, NG + NT * a_ + 1] = 1
            Hg[r, NG + NT * a_ + 3] = 1
            Hg[k + r, 0] = 1
            Hg[k + r, NG + NT * a_] = 1
            Hg[k + r, NG + NT * a_ + 2] = 1
            Hg[k + r, NG + NT * h_ + 1] = 1
            Hg[k + r, NG + NT * h_ + 3] = 1
            cg = np.r_[cg_h[idx], cg_a[idx]]
            eta = Hg @ x + cg
            HP_ = Hg @ P
            V2 = np.einsum("ij,ij->i", HP_, Hg)
            cha = np.einsum("ij,ij->i", HP_[:k], Hg[k:])
            days = (d - d0).days
            feta = Hg @ x0 + cg
            HP0 = Hg @ P0
            HQ = Hg @ Q
            fV = np.einsum("ij,ij->i", HP0 + HQ * days, Hg)
            fc = np.einsum("ij,ij->i", HP0[:k] + HQ[:k] * days, Hg[k:])
            rows.append(np.column_stack([
                gv.gid.to_numpy()[idx], eta[:k], eta[k:], V2[:k], V2[k:], cha,
                feta[:k], feta[k:], fV[:k], fV[k:], fc]))
            y = np.r_[gv.reg_h.to_numpy()[idx], gv.reg_a.to_numpy()[idx]].astype(float)
            m = np.exp(eta)
            H, z, Rm = [Hg], [(y - m) / m], [disp_g / m]
            if sh is not None:
                sy = np.r_[gv.sh_h.to_numpy()[idx], gv.sh_a.to_numpy()[idx]]
                ok = ~np.isnan(sy)
                if ok.any():
                    Hs = np.zeros((2 * k, NSTATE))
                    Hs[r, 2] = 1
                    Hs[r, 3] = 1
                    Hs[r, NG + NT * h_] = 1
                    Hs[r, NG + NT * a_ + 1] = 1
                    Hs[k + r, 2] = 1
                    Hs[k + r, NG + NT * a_] = 1
                    Hs[k + r, NG + NT * h_ + 1] = 1
                    es = Hs @ x + np.r_[cs_h[idx], cs_a[idx]]
                    ms = np.exp(es)
                    H.append(Hs[ok])
                    z.append(((sy - ms) / ms)[ok])
                    Rm.append((disp_s / ms)[ok])
            H = np.vstack(H)
            z = np.concatenate(z)
            Rm = np.concatenate(Rm)
            PHt = P @ H.T
            Sm = H @ PHt
            Sm[np.diag_indices(len(z))] += Rm
            K = np.linalg.solve(Sm, PHt.T).T
            x = x + K @ z
            P = P - K @ PHt.T
            P = (P + P.T) / 2
    out = pd.DataFrame(np.vstack(rows), columns=[
        "gid", "eta_h", "eta_a", "v_h", "v_a", "c_ha",
        "feta_h", "feta_a", "fv_h", "fv_a", "fc_ha"])
    out["gid"] = out.gid.astype(int)
    out = out.merge(g[["gid", "season_end"]], on="gid")
    return out


def probs(hp: HP, seasons, use_goalie=False, lineup=None, shot_mult=0.0,
          pre_override=None, ot_params=None) -> pd.DataFrame:
    """run_filter_lineup + ratings.predict_probs (OT/SO walk-forward from the
    same run unless ``ot_params`` given), merged with outcomes."""
    pred = run_filter_lineup(hp, seasons, use_goalie=use_goalie, lineup=lineup,
                             shot_mult=shot_mult, pre_override=pre_override)
    pred = pred[pred.season_end >= 2010]
    pp = R.predict_probs(pred, hp, ot_params=ot_params if ot_params is not None else {})
    g = S.game_frame()[["gid", "game_id", "season_end", "home_win"]]
    return g.merge(pp[["gid", "p_home_win"]], on="gid")
