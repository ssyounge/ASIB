import torch
import torch.nn as nn
from typing import List

class LightweightAttnMBM(nn.Module):
    """Lightweight attention-based MBM.

    Projects teacher features to key/value tokens and computes a
    single fusion embedding using multi-head attention.
    """
    def __init__(self, feat_dims: List[int], out_dim: int, r: int = 4,
                 n_head: int = 1, learnable_q: bool = False) -> None:
        super().__init__()
        self.learnable_q = learnable_q
        self.embed_dim = max(1, out_dim // r)
        self.n_tokens = len(feat_dims)

        # query projection or learnable query
        if learnable_q:
            self.q = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        else:
            self.q_proj = nn.Linear(sum(feat_dims), self.embed_dim)

        # per-teacher key/value projections
        self.k_proj = nn.ModuleList([nn.Linear(d, self.embed_dim) for d in feat_dims])
        self.v_proj = nn.ModuleList([nn.Linear(d, self.embed_dim) for d in feat_dims])

        self.attn = nn.MultiheadAttention(self.embed_dim, n_head, batch_first=True)
        self.out_proj = nn.Linear(self.embed_dim, out_dim)

    def forward(self, feats_2d: List[torch.Tensor], feats_4d=None) -> torch.Tensor:
        batch_size = feats_2d[0].size(0)

        if self.learnable_q:
            q = self.q.expand(batch_size, -1, -1)
        else:
            cat = torch.cat(feats_2d, dim=1)
            q = self.q_proj(cat).unsqueeze(1)

        keys = [proj(f).unsqueeze(1) for f, proj in zip(feats_2d, self.k_proj)]
        values = [proj(f).unsqueeze(1) for f, proj in zip(feats_2d, self.v_proj)]
        k = torch.cat(keys, dim=1)
        v = torch.cat(values, dim=1)

        attn_out, _ = self.attn(q, k, v)
        out = self.out_proj(attn_out.squeeze(1))
        return out
