# models/ib/gate_mbm.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class GateMBM(nn.Module):
    """
    GateMBM  (Adaptive-Gate 옵션 포함)
    -----------------------------------
    · *adaptive* == False  → 기존 fixed-gate (채널별 학습 파라미터)
    · *adaptive* == True   → **샘플별 스칼라 gate \u03b1(x)** 를 작은 MLP 로 예측

    beta  : KL 항 가중치 (외부에서 재-scale 하지 않음)
    """
    def __init__(
        self,
        c_in1: int,
        c_in2: int,
        n_cls: int = 100,
        z_dim: int = 512,
        beta: float = 1e-3,
        *,
        clamp: tuple[float, float] = (-6.0, 2.0),
        dropout_p: float = 0.1,
        *,
        adaptive: bool = False,
        gate_hidden: int = 128,
    ):
        super().__init__()
        # ensure scalar value to avoid list * Tensor errors
        self.beta = float(beta)
        self.cmin, self.cmax = clamp
        c = max(c_in1, c_in2)                        # 정보 보존
        self.proj1 = nn.Conv2d(c_in1, c, 1)          # 업/다운 자동 해결
        self.proj2 = nn.Conv2d(c_in2, c, 1)
        self.adaptive = adaptive
        if adaptive:
            self.gate_mlp = nn.Sequential(
                nn.Linear(2 * c, gate_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(gate_hidden, 1),
                nn.Sigmoid(),
            )
        else:
            self.gate = nn.Parameter(torch.zeros(1, c, 1, 1))  # 0 -> 0.5 after sigmoid
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_p)
        self.mu = nn.Linear(c, z_dim)
        self.log = nn.Linear(c, z_dim)
        self.head = nn.Linear(z_dim, n_cls)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor, log_kl: bool = False):
        # ── ① (N,C) → (N,C,1,1) 로 변환해 Conv 1×1 입력 보장 ──────────
        if f1.dim() == 2:
            f1 = f1.unsqueeze(-1).unsqueeze(-1)
        if f2.dim() == 2:
            f2 = f2.unsqueeze(-1).unsqueeze(-1)

        # ── ② 1×1 Conv 로 channel 맞추기 ─────────────────────────────
        f1 = self.proj1(f1)
        f2 = self.proj2(f2)
        if self.adaptive:
            v1 = self.pool(f1).flatten(1)
            v2 = self.pool(f2).flatten(1)
            alpha = self.gate_mlp(torch.cat([v1, v2], dim=1))
            g = alpha.view(-1, 1, 1, 1)
        else:
            g = torch.sigmoid(self.gate)
        fused = g * f1 + (1 - g) * f2
        fused = self.dropout(fused)
        v = self.pool(fused).flatten(1)
        mu = self.mu(v)
        log = self.log(v).clamp(self.cmin, self.cmax)
        std = torch.exp(0.5 * log)
        z = mu + torch.randn_like(mu) * std
        # KL per-sample  → mean
        kl = -0.5 * (1 + log - mu.pow(2) - log.exp()).mean()
        kl_scaled = self.beta * kl               # <-- 가중치 적용
        out = self.head(z)
        # (z, logits, scaled_KL, raw_KL, mu, log_var)
        return z, out, kl_scaled, kl, mu, log
