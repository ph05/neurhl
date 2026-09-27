"""NeurHL-G: a lineup-conditioned single-game simulation network (PLAN_NeurHL4 M).

Structure, per game and side s (opponent o):

  player encoder   h_i = MLP(state_i, position)                 (shared)
  player heads     residual deltas, zero-initialised, so the untrained model
                   reproduces the shrunk-history baselines exactly
  team layer       deterministic and conservation-preserving:
    PP opportunities   lam_pp_s = log5(team drew, opponent took) x player terms
    time               T_pp_s = tau * lam_pp_s,  T_sh_s = T_pp_o,
                       T_ev = 60 - T_pp_s - T_pp_o
    ice time           TOI_i,str = n_str * T_str * softmax within position group
                       (EV: 3 F + 2 D; PP: 5 of all; SH: 4 of all), so team ice
                       time is exact by construction
    xG rates           log r_ev_s = log log5(team xGF60, opp xGA60)
                                    + TOI-weighted sum of own off_i
                                    - TOI-weighted sum of opponent def_j + deltas
    allocations        player ixG, SOG, attempts, goals, assists and on-ice
                       xGF/xGA are shares of team totals, so they sum exactly
    goals              G_s = xGF_s x exp(finishing - opposing goalie effect)
  outcome          regulation goals ~ Poisson with a shared lognormal pace shock
                   (Gauss-Hermite, 5 nodes) supplying tie mass; overtime winner
                   from a logistic in log(lam_h / lam_a)

With freeze_heads=True only the global scalars train: that is rung R4, the
non-neural engine. Rates are per 60 minutes; ice time is in minutes.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# raw columns the team layer reads (names from g_names.json)
TEAM_RAW = ["tm_xgf_ev60_d95", "tm_xga_ev60_d95", "tm_xgf_pp60_d95",
            "tm_xga_sh60_d95", "tm_pp_opps_pg_d95", "tm_pk_opps_pg_d95",
            "tm_sogf_pg_d95", "tm_soga_pg_d95", "tm_neff_d95"]
GH_X, GH_W = [float(v) for v in (-2.0201828705, -0.9585724646, 0.0, 0.9585724646,
                                 2.0201828705)], \
             [float(v) for v in (0.0199532421, 0.3936193232, 0.9453087205,
                                 0.3936193232, 0.0199532421)]
MAXG = 12
STEP = 2          # integration step, minutes


def mlp(i, h, o, p=0.1, out_zero=False):
    last = nn.Linear(h, o)
    if out_zero:
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.LayerNorm(h),
                         nn.Dropout(p), last)


class NeurHLG(nn.Module):
    def __init__(self, n_sk, n_gk, n_tm, n_ctx, d=64, p=0.2, attn=False,
                 freeze_heads=False, elo_anchor=False, lineup_terms=False):
        super().__init__()
        self.freeze_heads = freeze_heads
        self.elo_anchor = elo_anchor
        self.lineup_terms = lineup_terms
        # lineup vs usual: tonight's TOI-weighted on-ice xGF60 / xGA60 relative to
        # the team's history, and TOI-weighted RAPM; coefficients start at zero
        self.kappa = nn.Parameter(torch.zeros(3))
        self.sk_enc = mlp(n_sk + 2, 128, d, p)
        self.tm_enc = mlp(n_tm, 64, 32, p)
        self.gk_enc = mlp(n_gk, 32, 16, p)
        self.ctx_enc = mlp(n_ctx, 32, 16, p)
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True, dropout=p) \
            if attn else None
        self.attn_gate = nn.Parameter(torch.zeros(1)) if attn else None
        # player deltas: toi(3) sog att xg fin ast off def pdraw ptake = 12
        self.sk_head = mlp(d + 32 + 16, 64, 12, p, out_zero=True)
        # team deltas: ev_rate pp_rate sog pen fin sh_rate = 6
        self.tm_head = mlp(32 * 2 + 16 * 2 + 16, 64, 6, p, out_zero=True)
        # goalie save effect delta
        self.gk_head = mlp(16 + 16, 32, 1, p, out_zero=True)
        # global scalars (log space where positive)
        g = {"L_ev": 2.25, "L_pp": 6.5, "L_sh": 0.7, "L_ppo": 3.2, "L_sog": 30.0,
             "tau": 1.55, "apg": 1.7, "att_ratio": 1.8}
        self.logs = nn.ParameterDict({k: nn.Parameter(torch.tensor(math.log(v)))
                                      for k, v in g.items()})
        self.home_ev = nn.Parameter(torch.tensor(0.03))
        # Elo anchor: log-rate shift +/- beta * elo_logit / 2 on the two sides'
        # scoring rates, so team strength starts at Elo's level and the network
        # learns what the lineup adds (a logit of 0.4 ~ a 0.1 goal edge)
        self.elo_beta = nn.Parameter(torch.tensor(0.25))
        self.fin0 = nn.Parameter(torch.tensor(0.05))
        self.gk_coef = nn.Parameter(torch.tensor(1.0))
        self.log_sigma = nn.Parameter(torch.tensor(math.log(0.12)))
        self.ot_a = nn.Parameter(torch.tensor(0.1))
        self.ot_b = nn.Parameter(torch.tensor(0.6))
        # score-and-time hazard table: log rate multiplier by own lead (-3..3)
        # and phase; initialised from the audited S1 score-effect curve
        # (configs/event_sim_gates.json: trailing +5-10%, leading -6-11%)
        s1 = torch.tensor([0.083, 0.096, 0.053, 0.0, -0.066, -0.121, -0.115])
        self.hz = nn.Parameter(s1[:, None].repeat(1, 5).clone())
        self.shrink_k = 5.0
        if freeze_heads:
            for m in (self.sk_enc, self.tm_enc, self.gk_enc, self.ctx_enc,
                      self.sk_head, self.tm_head, self.gk_head):
                for q in m.parameters():
                    q.requires_grad_(False)

    # ----------------------------------------------------------- helpers
    def L(self, k):
        return self.logs[k].exp()

    def team_base(self, raw, own, opp):
        """Shrunk log5 team rates. raw: (B,2,len(TEAM_RAW)) unstandardised."""
        n = torch.nan_to_num(raw[..., 8], nan=0.0).clamp(min=0)

        def sh(x, L):
            x = torch.where(torch.isfinite(x), x, L.expand_as(x))
            return (n * x + self.shrink_k * L) / (n + self.shrink_k)

        L_ev, L_pp, L_ppo, L_sog = self.L("L_ev"), self.L("L_pp"), self.L("L_ppo"), self.L("L_sog")
        xf_ev, xa_ev = sh(raw[..., 0], L_ev), sh(raw[..., 1], L_ev)
        xf_pp, xa_sh = sh(raw[..., 2], L_pp), sh(raw[..., 3], L_pp)
        ppo, pko = sh(raw[..., 4], L_ppo), sh(raw[..., 5], L_ppo)
        sf, sa = sh(raw[..., 6], L_sog), sh(raw[..., 7], L_sog)
        opp_i = [1, 0]
        b = {}
        b["xf_ev"], b["xa_ev"] = xf_ev, xa_ev
        b["ev"] = xf_ev * xa_ev[:, opp_i] / L_ev
        b["pp"] = xf_pp * xa_sh[:, opp_i] / L_pp
        b["ppo"] = ppo * pko[:, opp_i] / L_ppo
        b["sog"] = sf * sa[:, opp_i] / L_sog
        return b

    # ----------------------------------------------------------- forward
    def forward(self, bt):
        SK, SKB, SKM, SKP = bt["SK"], bt["SKB"], bt["SKM"], bt["SKP"]
        B = SK.shape[0]
        pos = F.one_hot(SKP.long(), 2).float()
        h = self.sk_enc(torch.cat([SK, pos], -1))                 # (B,2,20,d)
        if self.attn is not None:
            flat = h.reshape(B, 2 * 20, -1)
            mask = (SKM.reshape(B, 40) == 0)
            a, _ = self.attn(flat, flat, flat, key_padding_mask=mask)
            h = h + self.attn_gate * a.reshape(h.shape)
        t = self.tm_enc(bt["TM"])                                 # (B,2,32)
        g = self.gk_enc(bt["GK"])                                 # (B,2,16)
        c = self.ctx_enc(bt["CTX"])                               # (B,16)
        opp = [1, 0]
        c2 = c[:, None, :].expand(B, 2, 16)
        sk_in = torch.cat([h, t[:, :, None, :].expand(-1, -1, 20, -1),
                           c2[:, :, None, :].expand(-1, -1, 20, -1)], -1)
        dsk = self.sk_head(sk_in) * (0.0 if self.freeze_heads else 1.0)
        dtm = self.tm_head(torch.cat([t, t[:, opp], g, g[:, opp], c2], -1)) \
            * (0.0 if self.freeze_heads else 1.0)
        dgk = self.gk_head(torch.cat([g, c2], -1))[..., 0] \
            * (0.0 if self.freeze_heads else 1.0)

        m = SKM
        neg = torch.finfo(SK.dtype).min / 4
        eps = 1e-6
        base = self.team_base(bt["TMR"], None, None)
        b_toi_ev, b_toi_pp, b_toi_sh = SKB[..., 0], SKB[..., 1], SKB[..., 2]
        b_sog, b_att, b_xg = SKB[..., 3], SKB[..., 4], SKB[..., 5]
        b_a = SKB[..., 7]

        def logit_share(logb, delta, group_mask):
            z = torch.log(logb.clamp(min=eps)) + delta
            z = torch.where(group_mask > 0, z, torch.full_like(z, neg))
            return torch.softmax(z, -1)

        # ---- special teams and time
        pdraw, ptake = dsk[..., 10], dsk[..., 11]
        wsum = lambda x: (x * m).sum(-1) / m.sum(-1).clamp(min=1)
        lam_ppo = base["ppo"] * torch.exp(wsum(pdraw) + wsum(ptake)[:, opp] + dtm[..., 3])
        T_pp = self.L("tau") * lam_ppo                           # (B,2) minutes
        T_sh = T_pp[:, opp]
        T_ev = (60.0 - T_pp - T_sh).clamp(min=30.0)

        # ---- ice time allocation (exact budgets)
        isF = (SKP == 0).float() * m
        isD = (SKP == 1).float() * m
        shF = logit_share(b_toi_ev, dsk[..., 0], isF)
        shD = logit_share(b_toi_ev, dsk[..., 0], isD)
        toi_ev = (3.0 * shF + 2.0 * shD) * T_ev[..., None]
        toi_pp = 5.0 * logit_share(b_toi_pp + 0.05, dsk[..., 1], m) * T_pp[..., None]
        toi_sh = 4.0 * logit_share(b_toi_sh + 0.05, dsk[..., 2], m) * T_sh[..., None]
        toi_all = toi_ev + toi_pp + toi_sh

        # ---- xG rates with lineup effects
        off, dfn = dsk[..., 8], dsk[..., 9]
        w_ev = toi_ev / (5.0 * T_ev[..., None])                   # sums to 1 per side
        lineup = (w_ev * off).sum(-1) - (w_ev[:, opp] * dfn[:, opp]).sum(-1)
        sgn = torch.tensor([1.0, -1.0])
        elo_sh = 0.5 * self.elo_beta * bt["ELO"][:, None] * sgn if self.elo_anchor \
            else torch.zeros_like(lineup)
        if self.lineup_terms:
            lin_f = (w_ev * SKB[..., 8]).sum(-1).clamp(min=0.3)
            lin_a = (w_ev * SKB[..., 9]).sum(-1).clamp(min=0.3)
            rp = torch.nan_to_num(bt["RAPM"], nan=0.0)
            r_off, r_def = (w_ev * rp[..., 0]).sum(-1), (w_ev * rp[..., 1]).sum(-1)
            lineup = lineup + self.kappa[0] * torch.log(lin_f / base["xf_ev"]) \
                + self.kappa[1] * torch.log(lin_a[:, opp] / base["xa_ev"][:, opp]) \
                + self.kappa[2] * (r_off - r_def[:, opp])
        r_ev = base["ev"] * torch.exp(lineup + dtm[..., 0] + elo_sh
                                      + self.home_ev * sgn)
        w_pp = toi_pp / (5.0 * T_pp[..., None]).clamp(min=eps)
        w_sh = toi_sh / (4.0 * T_sh[..., None]).clamp(min=eps)
        lineup_pp = (w_pp * off).sum(-1) - (w_sh[:, opp] * dfn[:, opp]).sum(-1)
        r_pp = base["pp"] * torch.exp(lineup_pp + dtm[..., 1] + elo_sh)
        r_sh = self.L("L_sh") * torch.exp(dtm[..., 5])
        xgf_ev = r_ev * T_ev / 60.0
        xgf_pp = r_pp * T_pp / 60.0
        xgf_sh = r_sh * T_sh / 60.0
        xgf = xgf_ev + xgf_pp + xgf_sh

        # ---- player allocations (sum exactly to team totals)
        def alloc(total, logw, delta):
            z = torch.log(logw.clamp(min=eps)) + delta
            z = torch.where(m > 0, z, torch.full_like(z, neg))
            return total[..., None] * torch.softmax(z, -1)

        ixg = alloc(xgf_ev, toi_ev * b_xg, dsk[..., 5]) \
            + alloc(xgf_pp, toi_pp * (b_xg + 0.3), dsk[..., 5]) \
            + alloc(xgf_sh, toi_sh * (b_xg + 0.05), dsk[..., 5])
        sog_team = base["sog"] * torch.exp(dtm[..., 2]) * (T_ev + 1.3 * T_pp + 0.4 * T_sh) / 60.0
        sog = alloc(sog_team, toi_all * b_sog, dsk[..., 3])
        att_team = sog_team * self.L("att_ratio")
        att = alloc(att_team, toi_all * b_att, dsk[..., 4])

        # ---- goals: finishing and the opposing goalie
        fin_share = ixg / xgf[..., None].clamp(min=eps)
        fin = self.fin0 + (fin_share * dsk[..., 6]).sum(-1) + dtm[..., 4]
        gk_raw = torch.nan_to_num(bt["GKR"], nan=0.0)
        gk_eff = self.gk_coef * gk_raw + dgk                    # save effect
        goals = xgf * torch.exp(fin - gk_eff[:, opp])          # regulation mean
        g_i = alloc(goals, ixg, dsk[..., 6])
        a_i = alloc(goals * self.L("apg"), toi_all * b_a, dsk[..., 7])
        oi_xgf = xgf_ev[..., None] * toi_ev / T_ev[..., None] \
            + xgf_pp[..., None] * toi_pp / T_pp[..., None].clamp(min=eps) \
            + xgf_sh[..., None] * toi_sh / T_sh[..., None].clamp(min=eps)
        oi_xga = xgf_ev[:, opp][..., None] * toi_ev / T_ev[..., None] \
            + xgf_pp[:, opp][..., None] * toi_sh / T_sh[..., None].clamp(min=eps) \
            + xgf_sh[:, opp][..., None] * toi_pp / T_pp[..., None].clamp(min=eps)

        out = {"toi_ev": toi_ev, "toi_pp": toi_pp, "toi_sh": toi_sh,
               "isog": sog, "iatt": att, "ixg": ixg, "g": g_i, "a": a_i,
               "oi_xgf": oi_xgf, "oi_xga": oi_xga,
               "xgf_ev": xgf_ev, "xgf_pp": xgf_pp, "xgf_sh": xgf_sh, "xgf": xgf,
               "sogf": sog_team, "attf": att_team, "goals": goals,
               "pp_opps": lam_ppo, "pp_m": T_pp}
        out.update(self.outcome(goals[:, 0], goals[:, 1]))
        return out

    def outcome(self, lh, la):
        """outcome4 probabilities: [home reg, away reg, home OT/SO, away OT/SO].

        Integration over the goal differential in STEP-minute steps (home minus
        away, -D..D). Each team's scoring rate is its mean rate times a learned
        multiplier indexed by its own lead (-3..3) and the game phase (minutes
        0-40, 40-50, 50-56, 56-58, 58-60): trailing teams push, leaders sit
        back, tied teams turn cautious late, pulled goalies at the end. This is
        the score-and-time hazard table; it supplies tie mass by mechanism
        rather than by a scalar. A shared lognormal pace shock (5-node
        Gauss-Hermite) carries game-to-game tempo variation.
        """
        sig = self.log_sigma.exp()
        D = 8
        n = 2 * D + 1
        diffs = torch.arange(-D, D + 1)
        lead_h = diffs.clamp(-3, 3) + 3                      # index 0..6
        lead_a = (-diffs).clamp(-3, 3) + 3
        B = lh.shape[0]
        p_home = p_away = p_tie = 0.0
        for x, w in zip(GH_X, GH_W):
            s = math.sqrt(2.0) * sig * x - 0.5 * sig ** 2
            rh = (lh * torch.exp(s))[:, None] * (STEP / 60.0)    # per step
            ra = (la * torch.exp(s))[:, None] * (STEP / 60.0)
            dist = torch.zeros(B, n)
            dist[:, D] = 1.0
            for t in range(0, 60, STEP):
                ph_ = 0 if t < 40 else 1 if t < 50 else 2 if t < 56 else 3 if t < 58 else 4
                mh = torch.exp(self.hz[lead_h, ph_])[None, :]
                ma = torch.exp(self.hz[lead_a, ph_])[None, :]
                qh = 1 - torch.exp(-rh * mh)
                qa = 1 - torch.exp(-ra * ma)
                up = dist * qh * (1 - qa)
                dn = dist * qa * (1 - qh)
                stay = dist - up - dn
                new = stay.clone()
                new[:, 1:] = new[:, 1:] + up[:, :-1]
                new[:, :-1] = new[:, :-1] + dn[:, 1:]
                new[:, -1] = new[:, -1] + up[:, -1]
                new[:, 0] = new[:, 0] + dn[:, 0]
                dist = new
            wt = w / math.sqrt(math.pi)
            p_home = p_home + wt * dist[:, D + 1:].sum(-1)
            p_away = p_away + wt * dist[:, :D].sum(-1)
            p_tie = p_tie + wt * dist[:, D]
        tot = (p_home + p_away + p_tie).clamp(min=1e-9)
        p_home, p_away, p_tie = p_home / tot, p_away / tot, p_tie / tot
        q = torch.sigmoid(self.ot_a + self.ot_b * torch.log(lh / la))
        o4 = torch.stack([p_home, p_away, p_tie * q, p_tie * (1 - q)], -1)
        return {"o4": o4, "p_home_win": p_home + p_tie * q}
