# modules/ib_mbm.py
"""Information Bottleneck variant of the Manifold Bridging Module."""

import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence


class IB_MBM(nn.Module):
    """Information-Bottleneck Manifold-Bridging-Module."""

    def __init__(
        self,
        q_dim: int,
        kv_dim: int,
        d_emb: int,
        beta: float = 1e-2,
    ) -> None:
        super().__init__()
        self.q_proj = nn.Linear(q_dim, d_emb)
        self.kv_proj = nn.Linear(kv_dim, d_emb)
        self.attn = nn.MultiheadAttention(d_emb, 1, batch_first=True)
        self.mu = nn.Linear(d_emb, d_emb)
        self.logvar = nn.Linear(d_emb, d_emb)
        self.beta = beta

    def forward(
        self, query_feat: torch.Tensor, teacher_feats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return bottleneck sample and statistics."""

        q = self.q_proj(query_feat).unsqueeze(1)
        kv = self.kv_proj(teacher_feats)
        syn, _ = self.attn(q, kv, kv)
        syn = syn.squeeze(1)

        mu, logvar = self.mu(syn), self.logvar(syn)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return z, mu, logvar

    def loss(self, z, mu, logvar, labels, decoder):
        ce = nn.CrossEntropyLoss()(decoder(z), labels)
        q = Normal(mu, torch.exp(0.5 * logvar))
        p = Normal(torch.zeros_like(mu), torch.ones_like(mu))
        kl = kl_divergence(q, p).mean()
        return ce + self.beta * kl

