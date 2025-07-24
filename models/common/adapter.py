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
        print(f"[BottleneckAdapter] in  shape={tuple(x.shape)}")
        if x.dim() > 2:                # (N,C,H,W) -> (N,C)
            x = x.flatten(1)
            print(f"[BottleneckAdapter] GAP/flatten -> {tuple(x.shape)}")
        out = self.adapter(x)
        print(f"[BottleneckAdapter] out shape={tuple(out.shape)}")
        #
        return x + out                 # residual
