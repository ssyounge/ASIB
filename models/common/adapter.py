# models/common/adapter.py

import torch.nn as nn

class BottleneckAdapter(nn.Module):
    """
    [Student-side] Partial-Freeze 단계에서 얼린 블록 바로 뒤에 삽입되는 어댑터.

    단순한 2-layer bottleneck 구조로 입력과 동일한 차원을 출력하며
    잔차 연결(residual)을 사용한다.
    """

    def __init__(self, dim: int, r: int = 4):
        super().__init__()
        mid = max(1, dim // r)
        self.adapter = nn.Sequential(
            nn.Linear(dim, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, dim, bias=False),
        )

    def forward(self, x):
        # DEBUG
        # print(f"[BottleneckAdapter] in  shape={tuple(x.shape)}")
        if x.dim() > 2:                # (N,C,H,W) -> (N,C)
            x = x.flatten(1)
            # print(f"[BottleneckAdapter] GAP/flatten -> {tuple(x.shape)}")
        out = self.adapter(x)
        # print(f"[BottleneckAdapter] out shape={tuple(out.shape)}")
        #
        return x + out                 # residual


# ---------------------------------------------------------------------------
# Channel-wise 2D Adapter
# ---------------------------------------------------------------------------
class ChannelAdapter2D(nn.Module):
    """1×1 Conv – GN/ReLU – 1×1 Conv with residual."""

    def __init__(self, in_ch: int, out_ch: int | None = None, groups: int = 32):
        super().__init__()
        out_ch = in_ch if out_ch is None else out_ch
        gn = min(groups, out_ch)   # groups ≤ channels
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(gn, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.GroupNorm(gn, out_ch),
        )

    def forward(self, x):
        return x + self.net(x)


# ---------------------------------------------------------------------------
# Bottleneck MLP Adapter (2D tokens)
# ---------------------------------------------------------------------------
class BottleneckMLP(nn.Module):
    """Linear-ReLU-Linear bottleneck with residual."""

    def __init__(self, dim: int, r: int = 4):
        super().__init__()
        mid = max(1, dim // r)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, dim, bias=False),
        )

    def forward(self, x):
        return x + self.mlp(x)


# ---------------------------------------------------------------------------
# Token Adapter 1-D (Transformer token)
# ---------------------------------------------------------------------------
class TokenAdapter1D(nn.Module):
    """Simple 2-layer MLP for transformer tokens."""

    def __init__(self, dim: int, hidden: int | None = None):
        super().__init__()
        hidden = dim if hidden is None else hidden
        self.proj = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return x + self.proj(x)

# ────────────────────────────────────────────────────────────────
#  Teacher-side   DistillationAdapter  (기존 teachers/adapters.py)
# ────────────────────────────────────────────────────────────────
class DistillationAdapter(nn.Module):
    """[Teacher]  학생 feature dim에 맞추는 작은 MLP.
    `cfg['train_distill_adapter_only']` 때만 업데이트 됩니다."""

    def __init__(self, in_dim: int,
                 hidden_dim: int | None = None,
                 out_dim: int | None = None,
                 cfg: dict | None = None):
        super().__init__()
        cfg = cfg or {}
        hidden_dim = cfg.get("distill_hidden_dim", hidden_dim or in_dim // 2)
        out_dim    = cfg.get("distill_out_dim",    out_dim  or in_dim // 4)

        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, x):
        if x.dim() > 2:                       # (N,C,H,W) → (N,C)
            x = x.flatten(1)
        return self.proj(x)

