# mbm.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ManifoldBridgingModule(nn.Module):
    """
    - 2D 입력([N,d]) => MLP 융합 => (N, out_dim) 텐서
    - 4D 입력([N,C,H,W]) => 1x1 Conv 융합 => (N, out_channels_4d, H, W) 텐서
    """
    def __init__(
        # 2D MLP 용 파라미터
        in_dim=None,      # 기존 in_dim_2d 대신
        hidden_dim=None,  # 기존 hidden_dim_2d 대신
        out_dim=None,     # 기존 out_dim_2d 대신

        # 4D Conv 용 파라미터
        in_ch_4d=None,
        out_ch_4d=None
    ):
        super().__init__()

        # ----- 2D MLP -----
        if in_dim is not None and hidden_dim is not None and out_dim is not None:
            self.use_mlp = True
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, out_dim)
            )
        else:
            self.use_mlp = False

        # ----- 4D Conv -----
        if in_ch_4d is not None and out_ch_4d is not None:
            self.use_conv = True
            self.conv1 = nn.Conv2d(in_ch_4d, out_ch_4d, kernel_size=1, bias=False)
            self.bn1   = nn.BatchNorm2d(out_ch_4d)
            self.relu  = nn.ReLU(inplace=True)
        else:
            self.use_conv = False

    def forward(self, feat1, feat2):
        """
        feat1, feat2: 2D or 4D 텐서.
        => 2D면 MLP path, 4D면 Conv path.
        => 최종적으로 텐서(2D 또는 4D) 반환.
        """
        # 2D 입력
        if feat1.dim() == 2 and feat2.dim() == 2:
            if not self.use_mlp:
                raise ValueError("MBM: got 2D input but MLP not configured! Check in_dim/hidden_dim/out_dim.")
            x = torch.cat([feat1, feat2], dim=1)  # => (N, d1 + d2)
            synergy_2d = self.mlp(x)             # => (N, out_dim)
            return synergy_2d

        # 4D 입력
        elif feat1.dim() == 4 and feat2.dim() == 4:
            if not self.use_conv:
                raise ValueError("MBM: got 4D input but Conv not configured! Check in_ch_4d/out_ch_4d.")
            x = torch.cat([feat1, feat2], dim=1)  # => (N, c1 + c2, H, W)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            # => (N, out_ch_4d, H, W)
            return x

        else:
            raise ValueError(
                f"MBM forward mismatch: "
                f"feat1.shape={feat1.shape}, feat2.shape={feat2.shape}"
            )

class SynergyHead(nn.Module):
    """
    2D 전용 Head: (N, in_dim) -> (N, num_classes)
    """
    def __init__(self, in_dim, num_classes=100):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, synergy_emb):
        # synergy_emb: (N, in_dim)
        return self.fc(synergy_emb)
