"""L1 task 4: the in-season filter with predicted starting-goalie offsets.

``run_filter_gk`` is a copy of ``orr.ratings.run_filter`` (unchanged except
where marked ``# L1``). The goal offsets of each game come from one of three
sources, chosen separately for the pregame prediction (``pred_src``) and the
measurement update (``upd_src``, default = ``pred_src``):

  'mix'     the start-share mix: no-goalie rest/travel coefficients ctx, no
            starter offset (= run_filter(use_goalie=False))
  'actual'  the actual starters: ctx_gk + beta_gk * gdiff (= run_filter(
            use_goalie=True)); falls back to the mix where a starter is unknown
  'pred'    the choice model's expected starter: ctx_gk (or ctx, with
            ``ctx_kind='ctx'``) + lam * beta_gk * E[gdiff]; falls back to the
            mix on exactly the games where 'actual' does (both starters
            recorded, goalie GLM available) or where E[gdiff] is missing

``egd``: DataFrame gid, egd_h, egd_a (expected talent - reference of the
home / away starter, ``starters.expected_gdiff``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from orr import structural as S
from orr.ratings import NG, NT, NSTATE, SLOT, HP, _ctx, _prior, _proc, _season_setup, split_prior


def run_filter_gk(hp: HP, seasons, egd: pd.DataFrame | None = None, pred_src: str = "mix",
                  upd_src: str | None = None, ctx_kind: str = "gk", lam: float = 1.0,
                  first: int = 2008, pre_override: dict | None = None) -> pd.DataFrame:
    upd_src = upd_src or pred_src
    g = S.game_frame()
    gt = S.goalie_game_talent(*hp.goalie)
    seasons = sorted(seasons)
    last = max(seasons)
    Q = _proc(hp)
    rows = []
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
        # per-game offsets
        cm_h, cm_a = _ctx(st["ctx"], gv, "h"), _ctx(st["ctx"], gv, "a")

        # L1: offsets by source (mix / actual starters / predicted starters)
        def offsets(src):
            if src == "mix" or st.get("ctx_gk") is None:
                return cm_h, cm_a
            gm = gv[["gid"]].merge(gt, on="gid", how="left")
            known = (gm.gdiff_h.notna() & gm.gdiff_a.notna()).to_numpy()
            b = st["beta_gk"] * hp.gk_scale
            if src == "actual":
                dh, da, cx, lv = gm.gdiff_h.to_numpy(), gm.gdiff_a.to_numpy(), st["ctx_gk"], 1.0
            elif src == "pred":
                e = gv[["gid"]].merge(egd, on="gid", how="left")
                dh, da = e.egd_h.to_numpy(), e.egd_a.to_numpy()
                known = known & ~np.isnan(dh) & ~np.isnan(da)
                cx = st["ctx_gk"] if ctx_kind == "gk" else st["ctx"]
                lv = lam
            else:
                raise ValueError(src)
            kh = _ctx(cx, gv, "h") + lv * b * np.nan_to_num(da)
            ka = _ctx(cx, gv, "a") + lv * b * np.nan_to_num(dh)
            return np.where(known, kh, cm_h), np.where(known, ka, cm_a)

        cg_h, cg_a = offsets(pred_src)
        cu_h, cu_a = (cg_h, cg_a) if upd_src == pred_src else offsets(upd_src)
        sh = st.get("shots") if hp.use_shots else None
        if sh is not None:
            cs_h = _ctx(sh["beta"], gv, "h") + sh["beta"]["margin"] * np.clip(gv.margin, -3, 3) \
                + sh["beta"]["extra"] * gv.went_extra
            cs_a = _ctx(sh["beta"], gv, "a") + sh["beta"]["margin"] * np.clip(-gv.margin, -3, 3) \
                + sh["beta"]["extra"] * gv.went_extra
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
            # ---- measurement update (L1: with the update's offsets)
            y = np.r_[gv.reg_h.to_numpy()[idx], gv.reg_a.to_numpy()[idx]].astype(float)
            eta_u = eta if upd_src == pred_src else Hg @ x + np.r_[cu_h[idx], cu_a[idx]]
            m = np.exp(eta_u)
            H, z, R = [Hg], [(y - m) / m], [disp_g / m]
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
                    R.append((disp_s / ms)[ok])
            H = np.vstack(H)
            z = np.concatenate(z)
            R = np.concatenate(R)
            PHt = P @ H.T
            Sm = H @ PHt
            Sm[np.diag_indices(len(z))] += R
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
