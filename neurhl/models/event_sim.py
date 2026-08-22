"""NeurHL-2 — S1: neural marked temporal point process over the event stream.

Transformer-Hawkes family (Zuo et al., ICML 2020): causal self-attention over
event history parameterises the conditional intensity, and the "marks" are the
things that make a hockey event an event — who did what, where, and to whom.
Factorisation per step:

    p(dt | H, personnel) . p(type,team | H, personnel)
        . p(location | type, H) . p(actor | on-ice)

Three design commitments, each from a measured failure rather than taste:

  * **Player effects are ANCHORED, not free.** The on-ice representation is the
    pooled RAPM prior plus `s * tanh(residual)` with the residual projection
    ZERO-INITIALISED, so at step 0 the network reproduces the validated ridge
    solution exactly and can only depart from it by earning that on held-out
    event likelihood. v1's freely-learned player layer regressed every player to
    the league mean (sd 0.006 against an EWMA baseline's 0.042); the same
    identity-init pattern is what rescued it.

  * **Conditioning on personnel is the interface S4 needs.** Each step predicts
    the NEXT event given the on-ice set at that moment. Deployment is a separate
    process (S3), so S1 is a conditional hazard model — exactly the object the
    semi-analytic game integration consumes, rather than a rollout that has to
    guess who is on the ice.

  * **Missing location is masked, never zero-filled.** Non-shot events have no
    coordinates before 2012 and zero is a real place on the ice (centre of the
    attacking blue line region), so a zero-fill would teach the model that every
    pre-2012 hit happened at the same spot. `has_xy` gates the location loss and
    a learned "absent" vector replaces the coordinate input.

Continuous time is modelled as a MIXTURE of log-normals over log(dt+1), not a
bucketed categorical: the simulator must sample real inter-event times and
integrate hazards over them, and buckets would quantise the game clock.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_SLOT = 14
N_RAPM = 5


class Cfg:
    def __init__(self, n_players, n_tt, d_model=256, n_layer=6, n_head=8,
                 d_ff=1024, d_player=32, k_dt=8, k_loc=6, dropout=0.1,
                 n_etype=17, n_zone=5, n_stype=16, n_strength=11, n_score=9,
                 n_period=6, n_era=8):
        self.__dict__.update(locals())
        del self.__dict__["self"]


class OnIceEncoder(nn.Module):
    """Pooled on-ice representation: RAPM prior + zero-init learned residual."""

    def __init__(self, c: Cfg):
        super().__init__()
        self.emb = nn.Embedding(c.n_players, c.d_player, padding_idx=0)
        nn.init.normal_(self.emb.weight, std=0.02)
        with torch.no_grad():
            self.emb.weight[0].zero_()
        # 4 pooled groups: for-skaters, for-goalie, against-skaters, against-goalie
        self.rapm_proj = nn.Linear(4 * N_RAPM, c.d_model // 2)
        self.res_proj = nn.Linear(4 * c.d_player, c.d_model // 2)
        nn.init.zeros_(self.res_proj.weight)
        nn.init.zeros_(self.res_proj.bias)
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, on, rapm_tab):
        """on: [B,T,14] CURRENT on-ice indices, home slots 0-6, away 7-13.

        Orientation is fixed to home/away and is deliberately NOT keyed to the
        next event's owner. An earlier version oriented 'for'/'against' by a
        `home_next` flag equal to the owning team of the NEXT event -- verified
        at 100% agreement with the target's team component -- which handed the
        model half its own (type, team) label. That information does not exist
        at simulation time: the deployment process decides who is ON THE ICE,
        never who will touch the puck next. With the leak removed the model must
        infer possession from history, which is what S4 actually needs.
        """
        B, T, _ = on.shape
        r = rapm_tab[on]                                   # [B,T,14,5]
        e = self.emb(on)                                   # [B,T,14,dp]
        mask = (on > 0).unsqueeze(-1).float()

        def pool(x, lo, hi):
            m = mask[..., lo:hi, :]
            return (x[..., lo:hi, :] * m).sum(-2) / m.sum(-2).clamp(min=1)

        hg, hs = pool(r, 0, 1), pool(r, 1, 7)
        ag, as_ = pool(r, 7, 8), pool(r, 8, 14)
        hge, hse = pool(e, 0, 1), pool(e, 1, 7)
        age, ase = pool(e, 7, 8), pool(e, 8, 14)

        base = self.rapm_proj(torch.cat([hs, hg, as_, ag], -1))
        res = self.res_proj(torch.cat([hse, hge, ase, age], -1))
        return torch.cat([base, self.scale * torch.tanh(res)], -1), e


class EventSim(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.c = c
        d = c.d_model
        self.e_type = nn.Embedding(c.n_etype, 48)
        self.e_team = nn.Embedding(4, 12)
        self.e_zone = nn.Embedding(c.n_zone, 12)
        self.e_stype = nn.Embedding(c.n_stype, 12)
        self.e_str = nn.Embedding(c.n_strength, 24)
        self.e_score = nn.Embedding(c.n_score, 16)
        self.e_score_abs = nn.Embedding(c.n_score, 16)
        self.e_per = nn.Embedding(c.n_period, 12)
        self.absent_xy = nn.Parameter(torch.zeros(2))
        n_scalar = 14
        self.in_proj = nn.Linear(48 + 12 + 12 + 12 + 24 + 16 + 16 + 12 + n_scalar, d)
        self.onice = OnIceEncoder(c)
        self.ctx_proj = nn.Linear(d, d)
        self.era_proj = nn.Linear(c.n_era, 2 * d)          # FiLM
        self.pos = nn.Parameter(torch.zeros(1, 1024, d))
        nn.init.normal_(self.pos, std=0.01)

        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=c.n_head, dim_feedforward=c.d_ff,
            dropout=c.dropout, batch_first=True, norm_first=True,
            activation="gelu")
        self.enc = nn.TransformerEncoder(layer, c.n_layer)
        self.norm = nn.LayerNorm(d)

        self.h_dt = nn.Linear(d, 3 * c.k_dt)
        self.h_tt = nn.Linear(d, c.n_tt)
        self.h_loc = nn.Linear(d, 6 * c.k_loc)
        self.h_zone = nn.Linear(d, c.n_zone)
        self.h_actor = nn.Linear(d, c.d_player)

    def encode(self, b, rapm_tab):
        c = self.c
        xy = torch.stack([b["xa"] / 100.0, b["ya"] / 50.0], -1)
        m = b["has_xy"].unsqueeze(-1).float()
        xy = m * xy + (1 - m) * self.absent_xy
        scal = torch.cat([
            torch.log1p(b["dt"]).unsqueeze(-1) / 3.0,
            (b["t_rem"] / 3600.0).unsqueeze(-1),
            xy, b["has_xy"].unsqueeze(-1).float(),
            b["xg"].unsqueeze(-1), b["has_xg"].unsqueeze(-1).float(),
            (b["n_for"].float() / 6.0).unsqueeze(-1),
            (b["n_against"].float() / 6.0).unsqueeze(-1),
            # absolute home-relative state, and the score x time-remaining
            # interaction: score effects intensify as time runs out, and a model
            # given only the main effects has to rediscover that from scratch
            (b["n_home"].float() / 6.0).unsqueeze(-1),
            (b["n_away"].float() / 6.0).unsqueeze(-1),
            b["g_home"].float().unsqueeze(-1),
            b["g_away"].float().unsqueeze(-1),
            (((b["score_abs"].float() - 4.0) / 4.0)
             * (b["t_rem"] / 3600.0)).unsqueeze(-1)], -1)
        tok = torch.cat([
            self.e_type(b["etype"]), self.e_team(b["team"]),
            self.e_zone(b["zone"]), self.e_stype(b["stype"]),
            self.e_str(b["strength"]), self.e_score(b["score"]),
            self.e_per(b["period"]), self.e_score_abs(b["score_abs"]),
            scal], -1)
        h = self.in_proj(tok)

        on_ctx, pemb = self.onice(b["on_ctx"], rapm_tab)
        h = h + self.ctx_proj(on_ctx)

        gamma, beta = self.era_proj(b["era"]).chunk(2, -1)
        h = h * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

        T = h.shape[1]
        h = h + self.pos[:, :T]
        causal = torch.triu(torch.ones(T, T, device=h.device, dtype=torch.bool),
                            diagonal=1)
        h = self.enc(h, mask=causal, src_key_padding_mask=~b["valid"])
        return self.norm(h), pemb

    # ---------------------------------------------------------------- losses
    def dt_nll(self, h, target, valid):
        c = self.c
        o = self.h_dt(h).view(*h.shape[:2], c.k_dt, 3)
        logw = F.log_softmax(o[..., 0], -1)
        mu = o[..., 1]
        logs = o[..., 2].clamp(-5, 3)
        y = torch.log1p(target).unsqueeze(-1)
        comp = (-0.5 * ((y - mu) / logs.exp()) ** 2 - logs
                - 0.5 * math.log(2 * math.pi))
        nll = -torch.logsumexp(logw + comp, -1)
        return (nll * valid).sum() / valid.sum().clamp(min=1)

    def loc_nll(self, h, xa, ya, mask):
        c = self.c
        o = self.h_loc(h).view(*h.shape[:2], c.k_loc, 6)
        logw = F.log_softmax(o[..., 0], -1)
        mx, my = o[..., 1], o[..., 2]
        lsx = o[..., 3].clamp(-3, 3)
        lsy = o[..., 4].clamp(-3, 3)
        tx = (xa / 100.0).unsqueeze(-1)
        ty = (ya / 50.0).unsqueeze(-1)
        comp = (-0.5 * ((tx - mx) / lsx.exp()) ** 2 - lsx
                - 0.5 * ((ty - my) / lsy.exp()) ** 2 - lsy
                - math.log(2 * math.pi))
        nll = -torch.logsumexp(logw + comp, -1)
        return (nll * mask).sum() / mask.sum().clamp(min=1)

    def actor_nll(self, h, pemb, on_ctx, target, mask):
        q = self.h_actor(h).unsqueeze(-2)                   # [B,T,1,dp]
        logits = (q * pemb).sum(-1) / math.sqrt(self.c.d_player)
        logits = logits.masked_fill(on_ctx == 0, -1e9)
        tgt = target.clamp(min=0).long()
        nll = F.cross_entropy(logits.flatten(0, 1), tgt.flatten(),
                              reduction="none").view_as(target)
        return (nll * mask).sum() / mask.sum().clamp(min=1)

    def forward(self, b, rapm_tab, w=None):
        w = w or {"dt": 1.0, "tt": 1.0, "loc": 0.3, "zone": 0.2, "actor": 0.3}
        h, pemb = self.encode(b, rapm_tab)
        v = b["tgt_valid"].float()
        out = {}
        out["dt"] = self.dt_nll(h, b["tgt_dt"], v)
        out["tt"] = (F.cross_entropy(
            self.h_tt(h).flatten(0, 1), b["tgt_tt"].flatten(),
            reduction="none").view_as(v) * v).sum() / v.sum().clamp(min=1)
        lm = v * b["tgt_has_xy"].float()
        out["loc"] = self.loc_nll(h, b["tgt_xa"], b["tgt_ya"], lm)
        out["zone"] = (F.cross_entropy(
            self.h_zone(h).flatten(0, 1), b["tgt_zone"].flatten(),
            reduction="none").view_as(v) * v).sum() / v.sum().clamp(min=1)
        am = v * (b["tgt_actor"] >= 0).float()
        out["actor"] = self.actor_nll(h, pemb, b["on_ctx"], b["tgt_actor"], am)
        out["loss"] = sum(w[k] * out[k] for k in w)
        return out


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
