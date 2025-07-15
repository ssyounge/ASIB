# models/vib_head.py

from __future__ import annotations

import torch
import torch.nn as nn

class VIBHead(nn.Module):
    """Simple VIB head producing mean and log-variance."""

    def __init__(self, in_dim: int, z_dim: int):
        super().__init__()
        self.fc_mu = nn.Linear(in_dim, z_dim)
        self.fc_logvar = nn.Linear(in_dim, z_dim)

    def forward(self, x):
        x = x.flatten(1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def kl_beta(self, global_step: int, cfg: dict):
        """Return KL-scaling factor with warm-up."""
        student_iters = cfg.get("student_iters", 1)
        beta = cfg.get("beta_bottleneck", 1e-3)
        warm_len = 0.3 * student_iters
        if global_step < warm_len:
            return beta * (global_step / warm_len)
        return beta
