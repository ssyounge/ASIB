import torch
import torch.nn as nn
from typing import List, Optional, Union, Tuple

class LightweightAttnMBM(nn.Module):
    """Lightweight attention-based MBM.

    Projects teacher features to key/value tokens and computes a
    single fusion embedding using multi-head attention.
    If a query tensor is provided, it is used for the attention query;
    otherwise the original behaviour (query from concatenated teacher
    features or learnable vector) is kept for backward compatibility.
    The forward now returns both the fused feature and attention map.
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

    def forward(
        self,
        query_or_feats: Union[torch.Tensor, List[torch.Tensor]],
        feats_2d: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        query_or_feats : Tensor or List[Tensor]
            If Tensor, treated as the query features from the student. If a
            list, the original behaviour is used and `feats_2d` is ignored.
        feats_2d : List[Tensor], optional
            Teacher features used as key/value tokens when query is provided.

        Returns
        -------
        syn_feat : Tensor
            The fused feature.
        attn : Tensor
            Attention weights from the multi-head attention layer.
        """

        if isinstance(query_or_feats, list):
            # backward compatible: original API mbm(feats_2d)
            feats = query_or_feats
            batch_size = feats[0].size(0)
            if self.learnable_q:
                q = self.q.expand(batch_size, -1, -1)
            else:
                q = self.q_proj(torch.cat(feats, dim=1)).unsqueeze(1)
        else:
            assert feats_2d is not None, "Teacher features must be provided"
            batch_size = query_or_feats.size(0)
            if self.learnable_q:
                q = self.q.expand(batch_size, -1, -1)
            else:
                q = self.q_proj(query_or_feats).unsqueeze(1)
            feats = feats_2d

        keys = [proj(f).unsqueeze(1) for f, proj in zip(feats, self.k_proj)]
        values = [proj(f).unsqueeze(1) for f, proj in zip(feats, self.v_proj)]
        k = torch.cat(keys, dim=1)
        v = torch.cat(values, dim=1)

        attn_out, attn = self.attn(q, k, v)
        out = self.out_proj(attn_out.squeeze(1))
        if isinstance(query_or_feats, list):
            return out
        return out, attn
