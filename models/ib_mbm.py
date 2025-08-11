# models/ib_mbm.py

import torch
import torch.nn as nn
from typing import List, Optional, Tuple


class IB_MBM(nn.Module):
    """Information‑Bottleneck Manifold‑Bridging Module."""

    def __init__(
        self,
        q_dim: int,
        kv_dim: int,
        d_emb: int,
        beta: float = 1e-2,
        n_head: int = 1,
        logvar_clip: float = 10.0,
        min_std: float = 1e-4,
    ):
        n_head = int(n_head or 1)
        d_emb = int(d_emb)
        if d_emb % n_head != 0:
            raise ValueError("d_emb must be divisible by n_head")
        super().__init__()
        self.q_proj = nn.Linear(q_dim, d_emb)
        self.kv_proj = nn.Linear(kv_dim, d_emb)
        self.attn = nn.MultiheadAttention(d_emb, n_head, batch_first=True)
        self.mu = nn.Linear(d_emb, d_emb)
        self.logvar = nn.Linear(d_emb, d_emb)
        self.beta = beta
        self.logvar_clip = float(logvar_clip)
        self.min_std = float(min_std)

    def forward(self, q_feat: torch.Tensor, kv_feats: torch.Tensor):
        if q_feat.dim() == 2:
            q = self.q_proj(q_feat).unsqueeze(1)
        else:
            q = self.q_proj(q_feat)

        if kv_feats.dim() == 4:
            batch_size = kv_feats.shape[0]
            kv_feats = kv_feats.view(batch_size, -1)
            kv = self.kv_proj(kv_feats).unsqueeze(1)
        elif kv_feats.dim() == 3:
            # Support both per-token projection (in_features == feat_dim)
            # and flattened projection (in_features == tokens * feat_dim)
            bsz, tokens, feat_dim = kv_feats.shape
            in_features = self.kv_proj.in_features
            if in_features == feat_dim:
                kv = self.kv_proj(kv_feats)  # (B, tokens, d_emb)
            elif in_features == tokens * feat_dim:
                kv = self.kv_proj(kv_feats.reshape(bsz, tokens * feat_dim)).unsqueeze(1)
            else:
                raise RuntimeError(
                    f"KV projection in_features={in_features} mismatches KV shape (tokens={tokens}, feat={feat_dim})."
                )
        else:
            kv = self.kv_proj(kv_feats).unsqueeze(1)

        syn, _ = self.attn(q, kv, kv)
        syn = syn.squeeze(1)
        mu, logvar = self.mu(syn), torch.clamp(self.logvar(syn), -self.logvar_clip, self.logvar_clip)
        std = torch.exp(0.5 * logvar).clamp_min(self.min_std)
        z = mu + std * torch.randn_like(std)
        return z, mu, logvar


class SynergyHead(nn.Sequential):
    def __init__(self, in_dim: int, num_classes: int = 100, p: float = 0.0) -> None:
        super().__init__(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(in_dim, num_classes),
        )


def build_ib_mbm_from_teachers(
    teachers: List[nn.Module],
    cfg: dict,
    query_dim: Optional[int] = None,
) -> Tuple[IB_MBM, SynergyHead]:
    use_da = bool(cfg.get("use_distillation_adapter", False))
    feat_dims = [
        (t.distill_dim if use_da and hasattr(t, "distill_dim") else t.get_feat_dim())
        for t in teachers
    ]
    if not use_da:
        unique_dims = set(int(d) for d in feat_dims)
        if len(unique_dims) > 1:
            raise ValueError(
                "Teacher feature dims differ. Enable use_distillation_adapter to align dimensions."
            )

    qdim = cfg.get("ib_mbm_query_dim") or query_dim
    if not qdim:
        raise ValueError("`ib_mbm_query_dim` must be specified for IB‑MBM.")

    mbm = IB_MBM(
        q_dim=qdim,
        kv_dim=max(feat_dims),
        d_emb=cfg.get("ib_mbm_out_dim", 512),
        beta=cfg.get("ib_beta", 1e-2),
        n_head=cfg.get("ib_mbm_n_head", 1),
        logvar_clip=cfg.get("ib_mbm_logvar_clip", 10),
        min_std=cfg.get("ib_mbm_min_std", 1e-4),
    )

    head = SynergyHead(
        in_dim=cfg.get("ib_mbm_out_dim", 512),
        num_classes=cfg.get("num_classes", 100),
        p=cfg.get("synergy_head_dropout", cfg.get("ib_mbm_dropout", 0.0)),
    )
    return mbm, head


