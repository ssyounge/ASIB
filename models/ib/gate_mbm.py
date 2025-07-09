# models/ib/gate_mbm.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class GateMBM(nn.Module):
    """
    *beta*  : KL 항에 곱해지는 가중치 (teacher_vib_update 등 외부에서 다시
              scale 하지 않아도 되도록 내부에서 적용)
    """
    def __init__(self, c_in1: int, c_in2: int, z_dim: int = 512, n_cls: int = 100,
                 beta: float = 1e-3, dropout_p: float = 0.1):
        super().__init__()
        # ensure scalar value to avoid list * Tensor errors
        self.beta = float(beta)
        c = max(c_in1, c_in2)                        # 정보 보존
        self.proj1 = nn.Conv2d(c_in1, c, 1)          # 업/다운 자동 해결
        self.proj2 = nn.Conv2d(c_in2, c, 1)
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
        g = torch.sigmoid(self.gate)
        fused = g * f1 + (1 - g) * f2
        fused = self.dropout(fused)
        v = self.pool(fused).flatten(1)
        mu = self.mu(v)
        log = self.log(v)
        z = mu + torch.randn_like(mu) * (0.5 * log).exp()
        # KL per-sample  → mean
        kl = -0.5 * (1 + log - mu.pow(2) - log.exp()).mean()
        kl_scaled = self.beta * kl               # <-- 가중치 적용
        out = self.head(z)
        # (z, logits, scaled_KL, raw_KL, mu, log_var)
        return z, out, kl_scaled, kl, mu, log
