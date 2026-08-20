"""NeurHL Tier-2 — roster-aware multi-task game model (PLAN_NeurHL A).

Inputs per game: two 20-player rosters as (embedding ⊕ player-context) token
sets encoded by a small transformer with three learned position-group queries
(F/D/G attention pooling); a context vector (rest/b2b/travel/days-in); and the
walk-forward era vector applied as FiLM conditioning on the trunk (the model
shifts its scoring baseline across metas instead of averaging them).

Heads:
  H1 outcome4  categorical (home-reg, away-reg, home-extra, away-extra)
  H2 score     bivariate Poisson (λh, λa, λ12)
  H3 sat5      aux Poisson rates for 5v5 shot attempts for/against
  H4 player    per-skater TOI share (softmax), shot rate (Poisson),
               P(>=1 goal), P(>=1 assist)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

D_PLAYER_CTX = 6      # pos onehot(3), ewma_toi_min, log1p(gp_todate), starter
D_CTX = 16
D_ERA = 8


class GameModel(nn.Module):
    def __init__(self, d_player: int = 64, d: int = 128, trunk_dim: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.p_proj = nn.Linear(d_player + D_PLAYER_CTX, d)
        layer = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=dropout,
                                           batch_first=True, norm_first=True,
                                           activation="gelu")
        self.team_enc = nn.TransformerEncoder(layer, 2)
        self.queries = nn.Parameter(torch.randn(3, d) * 0.02)
        self.pool = nn.MultiheadAttention(d, 4, batch_first=True,
                                          dropout=dropout)
        self.ctx_proj = nn.Sequential(nn.Linear(D_CTX, 64), nn.GELU())
        self.fc1 = nn.Linear(6 * d + 64, trunk_dim)
        self.film = nn.Linear(D_ERA, 2 * trunk_dim)
        self.fc2 = nn.Linear(trunk_dim, trunk_dim)
        self.drop = nn.Dropout(dropout)
        self.h_out = nn.Linear(trunk_dim, 4)
        self.h_score = nn.Linear(trunk_dim, 3)
        self.h_sat = nn.Linear(trunk_dim, 2)
        self.h4 = nn.Sequential(nn.Linear(d + trunk_dim, 64), nn.GELU(),
                                nn.Linear(64, 4))

    def encode_team(self, emb, pctx, pad):
        tok = self.p_proj(torch.cat([emb, pctx], dim=-1))       # B,20,d
        h = self.team_enc(tok, src_key_padding_mask=pad)
        q = self.queries.unsqueeze(0).expand(h.shape[0], -1, -1)
        pooled, _ = self.pool(q, h, h, key_padding_mask=pad)    # B,3,d
        return h, pooled.flatten(1)                             # B,20,d / B,3d

    def forward(self, b: dict):
        """b: emb (B,2,20,dp), pctx (B,2,20,6), pad (B,2,20) bool,
        ctx (B,16), era (B,8)."""
        hh, ph = self.encode_team(b["emb"][:, 0], b["pctx"][:, 0],
                                  b["pad"][:, 0])
        ha, pa = self.encode_team(b["emb"][:, 1], b["pctx"][:, 1],
                                  b["pad"][:, 1])
        z = torch.cat([ph, pa, self.ctx_proj(b["ctx"])], dim=-1)
        t = F.gelu(self.fc1(z))
        gamma, beta = self.film(b["era"]).chunk(2, dim=-1)
        t = t * (1 + gamma) + beta
        t = self.drop(F.gelu(self.fc2(t)) + t)
        lam = F.softplus(self.h_score(t)) + 1e-4                # λh, λa, λ12
        sat = F.softplus(self.h_sat(t)) * 40 + 1.0
        tb = t.unsqueeze(1).expand(-1, 20, -1)
        p4 = {"home": self.h4(torch.cat([hh, tb], -1)),
              "away": self.h4(torch.cat([ha, tb], -1))}
        return {"out4": self.h_out(t), "lam": lam, "sat": sat, "p4": p4}


def bivpois_nll(lam: torch.Tensor, x: torch.Tensor, y: torch.Tensor,
                kmax: int = 11) -> torch.Tensor:
    """-log P(x,y) under bivariate Poisson (Karlis-Ntzoufras), stable."""
    l1, l2, l12 = lam[:, 0], lam[:, 1], lam[:, 2]
    x = x.clamp(0, kmax).float()
    y = y.clamp(0, kmax).float()
    base = (-(l1 + l2 + l12) + x * torch.log(l1) - torch.lgamma(x + 1)
            + y * torch.log(l2) - torch.lgamma(y + 1))
    ks = torch.arange(kmax + 1, device=lam.device).float()      # K+1
    kx = x.unsqueeze(1)
    ky = y.unsqueeze(1)
    k = ks.unsqueeze(0)
    valid = (k <= torch.minimum(kx, ky))
    # C(x,k) C(y,k) k! with arguments clamped so masked entries never hit
    # lgamma at non-positive integers (NaN grads leak through where() otherwise)
    dxk = (kx - k).clamp(min=0)
    dyk = (ky - k).clamp(min=0)
    logterm = (torch.lgamma(kx + 1) - torch.lgamma(k + 1)
               - torch.lgamma(dxk + 1)
               + torch.lgamma(ky + 1) - torch.lgamma(dyk + 1)
               - torch.lgamma(k + 1)
               + k * (torch.log(l12.unsqueeze(1))
                      - torch.log(l1.unsqueeze(1))
                      - torch.log(l2.unsqueeze(1))))
    logterm = torch.where(valid, logterm,
                          torch.full_like(logterm, -1e30))
    return -(base + torch.logsumexp(logterm, dim=1))


def game_loss(out: dict, b: dict, weights: dict,
              label_smoothing: float = 0.02) -> dict:
    losses = {"out4": F.cross_entropy(out["out4"], b["outcome4"],
                                      label_smoothing=label_smoothing)}
    losses["score"] = bivpois_nll(out["lam"], b["goals_h"], b["goals_a"]).mean()
    sat_t = torch.stack([b["sat5_h"], b["sat5_a"]], -1).float()
    losses["sat"] = (out["sat"] - sat_t * torch.log(out["sat"])).mean()
    p4l = []
    for side, gh in (("home", "h"), ("away", "a")):
        o = out["p4"][side]                                     # B,20,4
        sk = b[f"skater_mask_{gh}"]                             # B,20 bool
        toi = b[f"toi_{gh}"]                                    # B,20 (secs)
        share = toi / toi.sum(-1, keepdim=True).clamp(min=1)
        logit = o[..., 0].masked_fill(~sk, -1e30)
        p4l.append(-(share * F.log_softmax(logit, -1)).sum(-1).mean())
        rate = F.softplus(o[..., 1]) + 1e-4
        shots = b[f"shots_{gh}"].float()
        p4l.append(((rate - shots * torch.log(rate))[sk]).mean())
        p4l.append(F.binary_cross_entropy_with_logits(
            o[..., 2][sk], (b[f"goals_p_{gh}"][sk] > 0).float()))
        p4l.append(F.binary_cross_entropy_with_logits(
            o[..., 3][sk], (b[f"assists_p_{gh}"][sk] > 0).float()))
    losses["player"] = sum(p4l) / len(p4l)
    losses["total"] = (weights["outcome4"] * losses["out4"]
                       + weights["score"] * losses["score"]
                       + weights["sat5"] * losses["sat"]
                       + weights["player_rates"] * losses["player"])
    return losses
