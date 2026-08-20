"""NeurHL Tier-1b — career encoder: pre-NHL career -> event-LM embedding space.

A GRU over a player's PRE-NHL career season rows (junior/NCAA/AHL/European;
NHL rows excluded so training matches the rookie-inference distribution
exactly) plus static bio/draft features, distilled onto the frozen event-LM
player embeddings (train/train_career.py). At inference it supplies embeddings
for players unseen in the PBP corpus at a vantage; the rookie blend
w = gp/(gp+40) then mixes it with the PBP embedding as games accrue.
"""
import torch
import torch.nn as nn

N_LEAGUES = 30
D_SEQ = 16 + 5          # league embedding + [age, gp, g/gp, a/gp, p/gp]
D_STATIC = 9            # pos one-hot(3), height, weight, shoots, round, overall, undrafted


class CareerEncoder(nn.Module):
    def __init__(self, d_out: int = 64, d_hidden: int = 64):
        super().__init__()
        self.league_emb = nn.Embedding(N_LEAGUES, 16)
        self.gru = nn.GRU(D_SEQ, d_hidden, num_layers=2, batch_first=True,
                          dropout=0.1)
        self.head = nn.Sequential(
            nn.Linear(d_hidden + D_STATIC, 128), nn.GELU(),
            nn.Linear(128, d_out))

    def forward(self, league: torch.Tensor, feats: torch.Tensor,
                lengths: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        """league (B,L) int, feats (B,L,5), lengths (B,), static (B,9)."""
        x = torch.cat([self.league_emb(league), feats], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu().clamp(min=1), batch_first=True,
            enforce_sorted=False)
        _, h = self.gru(packed)
        return self.head(torch.cat([h[-1], static], dim=-1))


def career_features(rows, bio, nhl_league_idx: int, max_len: int = 20):
    """Numpy career rows (season-sorted, <= vantage-1, pre-NHL only) -> tensors.

    rows: iterable of (league_idx, age, gp, g, a, p); bio: dict-like with
    pos_group, height_in, weight_lb, shoots_left, draft_round, draft_overall.
    """
    import numpy as np
    rows = [r for r in rows if r[0] != nhl_league_idx][-max_len:]
    L = len(rows)
    league = np.zeros(max_len, dtype=np.int64)
    feats = np.zeros((max_len, 5), dtype=np.float32)
    for i, (lg, age, gp, g, a, p) in enumerate(rows):
        league[i] = lg
        gp = max(gp, 1)
        feats[i] = [(age - 18) / 10.0, gp / 82.0, g / gp, a / gp, p / gp]
    pos = np.zeros(3, dtype=np.float32)
    pos[int(bio["pos_group"])] = 1
    static = np.array([*pos, (bio["height_in"] - 72) / 3.0,
                       (bio["weight_lb"] - 195) / 20.0, bio["shoots_left"],
                       bio["draft_round"] / 7.0, bio["draft_overall"] / 224.0,
                       float(bio["draft_overall"] == 0)], dtype=np.float32)
    return league, feats, max(L, 1), static
