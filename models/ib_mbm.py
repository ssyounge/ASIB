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
        self.q_norm = nn.LayerNorm(d_emb)
        self.kv_norm = nn.LayerNorm(d_emb)
        self.attn = nn.MultiheadAttention(d_emb, n_head, batch_first=True)
        self.out_norm = nn.LayerNorm(d_emb)
        self.mu = nn.Linear(d_emb, d_emb)
        self.logvar = nn.Linear(d_emb, d_emb)
        self.beta = beta
        self.logvar_clip = float(logvar_clip)
        self.min_std = float(min_std)

    def forward(self, q_feat: torch.Tensor, kv_feats: torch.Tensor, sample: bool = True):
        if q_feat.dim() == 2:
            q = self.q_norm(self.q_proj(q_feat)).unsqueeze(1)
        else:
            q = self.q_norm(self.q_proj(q_feat))

        if kv_feats.dim() == 4:
            # Explicit flatten for clarity: (B, T, H, W) → (B, T*H*W)
            batch_size = kv_feats.shape[0]
            kv_feats = kv_feats.flatten(1)
            kv = self.kv_norm(self.kv_proj(kv_feats)).unsqueeze(1)
        elif kv_feats.dim() == 3:
            # Support both per-token projection (in_features == feat_dim)
            # and flattened projection (in_features == tokens * feat_dim)
            bsz, tokens, feat_dim = kv_feats.shape
            in_features = self.kv_proj.in_features
            if in_features == feat_dim:
                kv = self.kv_norm(self.kv_proj(kv_feats))  # (B, tokens, d_emb)
            elif in_features == tokens * feat_dim:
                kv = self.kv_norm(self.kv_proj(kv_feats.reshape(bsz, tokens * feat_dim))).unsqueeze(1)
            else:
                raise RuntimeError(
                    f"KV projection in_features={in_features} mismatches KV shape (tokens={tokens}, feat={feat_dim})."
                )
        else:
            kv = self.kv_norm(self.kv_proj(kv_feats)).unsqueeze(1)

        syn_raw, _ = self.attn(q, kv, kv)
        # Residual connection with pre-normed q
        syn = self.out_norm(syn_raw + q).squeeze(1)
        mu, logvar = self.mu(syn), torch.clamp(self.logvar(syn), -self.logvar_clip, self.logvar_clip)
        std = torch.exp(0.5 * logvar).clamp_min(self.min_std)
        if self.training and sample:
            z = mu + std * torch.randn_like(std)
        else:
            z = mu
        return z, mu, logvar


class SynergyHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int = 100,
        p: float = 0.0,
        learnable_temp: bool = False,
        temp_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(in_dim, num_classes),
        )
        self.learnable_temp = bool(learnable_temp)
        if self.learnable_temp:
            # log_temp=0 -> temperature=1.0
            self.log_temp = nn.Parameter(torch.tensor(float(temp_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if getattr(self, "learnable_temp", False):
            out = out / torch.exp(self.log_temp).clamp_min(1e-4)
        return out


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
        raise ValueError("`ib_mbm_query_dim` must be specified for IB_MBM.")

    ib_mbm = IB_MBM(
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
        learnable_temp=bool(cfg.get("synergy_temp_learnable", False)),
        temp_init=float(cfg.get("synergy_temp_init", 0.0)),
    )
    return ib_mbm, head


