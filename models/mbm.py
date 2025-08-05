# models/mbm.py

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

# ────────────────────────────── IB‑MBM ──────────────────────────────
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
        # handle None or string inputs from config
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
        q = self.q_proj(q_feat).unsqueeze(1)  # (batch_size, 1, d_emb)
        kv = self.kv_proj(kv_feats).unsqueeze(1)  # (batch_size, 1, d_emb)
        syn, _ = self.attn(q, kv, kv)
        syn = syn.squeeze(1)
        mu, logvar = self.mu(syn), torch.clamp(
            self.logvar(syn), -self.logvar_clip, self.logvar_clip
        )
        # AMP 환경 under‑flow 방지
        std = torch.exp(0.5 * logvar).clamp_min(self.min_std)
        z = mu + std * torch.randn_like(std)
        return z, mu, logvar

    # legacy `loss()` 는 사용처가 사라져 제거했습니다.


class SynergyHead(nn.Sequential):
    """2‑Layer MLP head mapping IB‑MBM embedding → logits."""

    def __init__(self, in_dim: int, num_classes: int = 100, p: float = 0.0) -> None:
        super().__init__(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(in_dim, num_classes),
        )

def build_from_teachers(
    teachers: List[nn.Module],
    cfg: dict,
    query_dim: Optional[int] = None,
) -> Tuple[IB_MBM, SynergyHead]:
    """Returns ``(IB_MBM, SynergyHead)``. ``mbm_type`` is ignored."""

    # 1) collect feature dimensions from teachers
    use_da = bool(cfg.get("use_distillation_adapter", False))
    feat_dims = [
        (t.distill_dim if use_da and hasattr(t, "distill_dim") else t.get_feat_dim())
        for t in teachers
    ]

    qdim = cfg.get("mbm_query_dim") or query_dim
    if not qdim:
        raise ValueError("`mbm_query_dim` must be specified for IB‑MBM.")

    mbm = IB_MBM(
        q_dim=qdim,
        kv_dim=max(feat_dims),
        d_emb=cfg.get("mbm_out_dim", 512),
        beta=cfg.get("ib_beta", 1e-2),
        n_head=cfg.get("mbm_n_head", 1),
        logvar_clip=cfg.get("mbm_logvar_clip", 10),
        min_std=cfg.get("mbm_min_std", 1e-4),
    )

    head = SynergyHead(
        in_dim=cfg.get("mbm_out_dim", 512),
        num_classes=cfg.get("num_classes", 100),
        p=cfg.get("synergy_head_dropout", cfg.get("mbm_dropout", 0.0)),
    )
    return mbm, head
