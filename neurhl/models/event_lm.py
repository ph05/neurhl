"""NeurHL Tier-1a — event-LM: causal transformer over PBP event sequences.

Pretraining objective (configs/pretrain.json): multi-task next-event prediction
(event type, actor-among-dressed, coordinate bucket, time-delta bucket) plus
denoising of masked input factors (actor identity masked at mask_ratio and
reconstructed at the same position). The product is the player-embedding table
(64-d), exported per vantage snapshot; the transformer body is retained on disk
for probes only.

Token = sum of factor embeddings: event type, strength class, zone, coord
bucket, dt bucket, score-state, period, home/away, venue, actor embedding,
secondary embedding, mean-pooled on-ice embeddings, and a linear projection of
the walk-forward era vector. All ids are per-vantage vocab indices (0 = PAD).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# frozen bucket definitions (changing these voids a snapshot)
DT_EDGES = [1, 3, 6, 12, 25, 50, 100]                # 8 buckets
XBINS, YBINS = 13, 6                                  # +1 missing = 79 coord ids
N_COORD = XBINS * YBINS + 1
N_STRENGTH = 10
N_SCORE = 7                                           # home-away diff clamp ±3


def dt_bucket(dt: torch.Tensor) -> torch.Tensor:
    b = torch.zeros_like(dt)
    for i, e in enumerate(DT_EDGES):
        b = b + (dt >= e).long()
    return b


def coord_bucket(xn: torch.Tensor, yn: torch.Tensor,
                 has: torch.Tensor) -> torch.Tensor:
    xb = ((xn.float() + 100) / 200 * XBINS).clamp(0, XBINS - 1).long()
    yb = ((yn.float() + 43) / 86 * YBINS).clamp(0, YBINS - 1).long()
    return torch.where(has.bool(), xb * YBINS + yb + 1, torch.zeros_like(xb))


def strength_class(code: torch.Tensor) -> torch.Tensor:
    ag, ask = code // 1000, (code // 100) % 10
    hsk, hg = (code // 10) % 10, code % 10
    d = hsk - ask
    cls = torch.full_like(code, 9)                    # other
    cls = torch.where((ask == 5) & (hsk == 5) & (ag == 1) & (hg == 1), 0, cls)
    cls = torch.where((d == 1) & (ag == 1) & (hg == 1), 1, cls)   # h pp1
    cls = torch.where((d >= 2) & (ag == 1) & (hg == 1), 2, cls)
    cls = torch.where((d == -1) & (ag == 1) & (hg == 1), 3, cls)  # a pp1
    cls = torch.where((d <= -2) & (ag == 1) & (hg == 1), 4, cls)
    cls = torch.where((ask == 4) & (hsk == 4) & (ag == 1) & (hg == 1), 5, cls)
    cls = torch.where((ask == 3) & (hsk == 3) & (ag == 1) & (hg == 1), 6, cls)
    cls = torch.where(hg == 0, 7, cls)                # home net empty
    cls = torch.where(ag == 0, 8, cls)                # away net empty
    return cls


class EventLM(nn.Module):
    def __init__(self, n_players: int, n_event_types: int = 20,
                 n_shot_types: int = 16, n_venues: int = 48,
                 d_model: int = 192, d_player: int = 64, n_layers: int = 4,
                 n_heads: int = 6, dropout: float = 0.2, era_dim: int = 8):
        super().__init__()
        self.player_emb = nn.Embedding(n_players + 1, d_player, padding_idx=0)
        self.p_actor = nn.Linear(d_player, d_model, bias=False)
        self.p_second = nn.Linear(d_player, d_model, bias=False)
        self.p_onice = nn.Linear(d_player, d_model, bias=False)
        self.e_type = nn.Embedding(n_event_types + 1, d_model, padding_idx=0)
        self.e_zone = nn.Embedding(4, d_model, padding_idx=0)
        self.e_shot = nn.Embedding(n_shot_types + 1, d_model, padding_idx=0)
        self.e_dt = nn.Embedding(8, d_model)
        self.e_str = nn.Embedding(N_STRENGTH, d_model)
        self.e_score = nn.Embedding(N_SCORE, d_model)
        self.e_period = nn.Embedding(9, d_model)
        self.e_home = nn.Embedding(3, d_model)        # away/none/home (0/1/2)
        self.e_venue = nn.Embedding(n_venues, d_model, padding_idx=0)
        self.p_era = nn.Linear(era_dim, d_model)
        self.mask_actor = nn.Parameter(torch.zeros(d_model))
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model, dropout=dropout,
            batch_first=True, norm_first=True, activation="gelu")
        self.body = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.h_type = nn.Linear(d_model, n_event_types + 1)
        self.h_coord = nn.Linear(d_model, N_COORD)
        self.h_dt = nn.Linear(d_model, 8)
        self.h_actor_q = nn.Linear(d_model, d_player)   # dot vs dressed embeds
        self.h_mask_q = nn.Linear(d_model, d_player)

    def token(self, b: dict, actor_masked: torch.Tensor) -> torch.Tensor:
        pe = self.player_emb
        actor = self.p_actor(pe(b["p1"]))
        actor = torch.where(actor_masked.unsqueeze(-1), self.mask_actor, actor)
        onice = torch.cat([b["h_on"], b["a_on"]], dim=-1)     # B,L,14
        oe = pe(onice)
        cnt = (onice > 0).sum(-1, keepdim=True).clamp(min=1)
        pooled = self.p_onice(oe.sum(-2) / cnt)
        # coord is TARGET-ONLY (ledger pt-fix-3): the HTM era has no recorded
        # coordinates, so a coord input factor splits the corpus into two
        # incompatible input domains; zone (present in both eras) carries the
        # location input instead, and h_coord still learns location as output.
        x = (self.e_type(b["event_type"]) + self.e_zone(b["zone"])
             + self.e_shot(b["shot_type"])
             + self.e_dt(b["dtb"]) + self.e_str(b["strength_cls"])
             + self.e_score(b["score_cls"]) + self.e_period(b["period"])
             + self.e_home(b["home_event"] + 1) + self.e_venue(b["venue"])
             + actor + self.p_second(pe(b["p2"])) + pooled
             + self.p_era(b["era"]).unsqueeze(1))
        return self.drop(x)

    def forward(self, b: dict, actor_masked: torch.Tensor):
        x = self.token(b, actor_masked)
        L = x.shape[1]
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool,
                                       device=x.device), 1)
        h = self.norm(self.body(x, mask=causal))
        dressed = self.player_emb(b["dressed"])               # B,40,dp
        out = {
            "type": self.h_type(h), "coord": self.h_coord(h),
            "dt": self.h_dt(h),
            "actor": torch.einsum("bld,bnd->bln", self.h_actor_q(h), dressed),
            "masked_actor": torch.einsum("bld,bnd->bln",
                                         self.h_mask_q(h), dressed),
        }
        return out


def lm_loss(out: dict, b: dict, actor_masked: torch.Tensor) -> dict:
    """Next-event losses (shift by one) + same-position masked-actor loss."""
    pad = b["event_type"] == 0
    valid_next = (~pad)[:, 1:]

    def nce(logits, target, extra_mask=None):
        m = valid_next if extra_mask is None else (valid_next & extra_mask)
        if not m.any():
            return torch.zeros((), device=logits.device)
        return F.cross_entropy(logits[:, :-1][m], target[:, 1:][m])

    # coord scored ONLY where the target event has a recorded coordinate —
    # HTM-era events (2008-2011) carry none, and training the head to predict
    # "missing" poisons cross-domain validation (ledger pt-fix-2)
    losses = {"type": nce(out["type"], b["event_type"]),
              "coord": nce(out["coord"], b["coord"],
                           extra_mask=(b["coord"][:, 1:] > 0)),
              "dt": nce(out["dt"], b["dtb"])}
    # actor: only where next event has a real actor among dressed
    tgt = b["p1_slot"]                                # index into dressed, -1 none
    m = valid_next & (tgt[:, 1:] >= 0)
    if m.any():
        losses["actor"] = F.cross_entropy(out["actor"][:, :-1][m],
                                          tgt[:, 1:][m])
    mm = actor_masked & (b["p1_slot"] >= 0) & ~pad
    if mm.any():
        losses["masked_actor"] = F.cross_entropy(out["masked_actor"][mm],
                                                 b["p1_slot"][mm])
    losses["total"] = sum(losses.values())
    return losses
