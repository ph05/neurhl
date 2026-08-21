"""NeurHL Layer 1 — player performance model (hierarchical architecture).

Predicts what a player will DO in a given game — on-ice shot attempts for and
against, close-range attempts, and ice time — rather than an abstract
embedding. Two reasons this is the right place to put network capacity:

  * supervision volume: ~750,000 player-games across 2008-2026, versus ~7,000
    team-games. Layer 1 can afford to be a real model; Layer 2 cannot.
  * the output is physically meaningful and low-dimensional, so Layer 2 becomes
    a thin, roster-aware aggregation (sum of predicted player contributions)
    instead of having to rediscover team strength from raw roster composition —
    which EDA-09 proved it cannot do.

Inputs per player-game (all strictly pre-game, P1-clean). STATIC_COLS carries
the career-context terms that let the layer learn effects a team-level model
cannot represent at all:
  * age and age^2 — aging curves (rise to ~24-27, decline after ~30)
  * career games — experience / establishment
  * games with CURRENT team and a recent-move flag — trade and call-up
    acclimation, where a player's own history overstates his immediate output
  * days rest, home flag, position group
plus the player embedding (64) and his own shifted EWMAs of on-ice CF/CA/
close-range/TOI.

Outputs: cf60, ca60, clf60, cla60 (per-60 on-ice rates) and toi (seconds).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

STATIC_COLS = ["pos_f", "pos_d", "pos_g", "age", "age_sq", "career_gp",
               "team_gp", "recent_move", "home", "rest"]
EWMA_COLS = ["e_cf", "e_ca", "e_clf", "e_cla", "e_toi", "e_cfpct"]
N_STATIC = len(STATIC_COLS)
N_EWMA = len(EWMA_COLS)
D_IN_EXTRA = N_STATIC + N_EWMA


class PlayerModel(nn.Module):
    """RESIDUAL form: start from the player's own EWMA baseline and learn a
    multiplicative correction.

    Predicting the LEVEL directly fails badly (measured): a player's on-ice
    shot differential is mostly a property of his TEAM, which player-only
    features cannot recover, so a level-predicting network regresses everyone
    to the league mean — 7x too little spread, and it destroyed the
    between-team variance that carries the signal. Anchoring on the EWMA keeps
    that variance and confines the network to what it can actually learn:
    how age, team tenure, a recent move, rest and role SHIFT a player away
    from his own recent form. Initialised at the identity (delta ~ 0), so the
    model starts exactly at the baseline and can only improve on it.
    """

    def __init__(self, d_player: int = 64, hidden: int = 128,
                 dropout: float = 0.1, max_log_adj: float = 0.7):
        super().__init__()
        self.max_log_adj = max_log_adj
        self.net = nn.Sequential(
            nn.Linear(d_player + D_IN_EXTRA, hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 5))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, emb: torch.Tensor, feats: torch.Tensor,
                baseline: torch.Tensor) -> torch.Tensor:
        """baseline (B,5) = EWMA [cf60, ca60, clf60, cla60, toi_sec].
        -> (B,5) adjusted, strictly positive."""
        delta = self.net(torch.cat([emb, feats], dim=-1))
        adj = torch.tanh(delta) * self.max_log_adj
        return baseline.clamp(min=1e-3) * torch.exp(adj)


def player_loss(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Poisson deviance on counts implied by predicted per-60 rates x actual
    TOI (so rate errors are scored where the ice time actually was), plus a
    Poisson term on ice time itself."""
    toi_true = target[:, 4].clamp(min=1.0)
    hours = toi_true / 3600.0
    losses = {}
    for i, name in enumerate(("cf", "ca", "clf", "cla")):
        lam = (pred[:, i] * hours).clamp(min=1e-6)
        k = target[:, i]
        losses[name] = (lam - k * torch.log(lam)).mean()
    lam_t = pred[:, 4].clamp(min=1.0)
    losses["toi"] = (lam_t / 60.0 - (toi_true / 60.0)
                     * torch.log(lam_t / 60.0)).mean()
    losses["total"] = (losses["cf"] + losses["ca"] + 0.5 * losses["clf"]
                       + 0.5 * losses["cla"] + losses["toi"])
    return losses
