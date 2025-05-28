# models/mbm.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ManifoldBridgingModule(nn.Module):
    """
    통합 MBM:
      - 2D 입력([N,d]) => MLP 융합 (self.use_mlp)
      - 4D 입력([N,C,H,W]) => 1x1 Conv 융합 (self.use_conv)
    """

    def __init__(
        self,
        # 2D MLP 설정
        in_dim_2d=None,
        hidden_dim_2d=None,
        out_dim_2d=None,
        # 4D Conv 설정
        c1_4d=None,
        c2_4d=None,
        out_channels_4d=None
    ):
        super().__init__()

        # (A) 2D용 MLP 구성
        self.use_mlp = False
        if (in_dim_2d is not None) and (hidden_dim_2d is not None) and (out_dim_2d is not None):
            self.use_mlp = True
            self.mlp = nn.Sequential(
                nn.Linear(in_dim_2d, hidden_dim_2d),
                nn.ReLU(),
                nn.Linear(hidden_dim_2d, hidden_dim_2d),
                nn.ReLU(),
                nn.Linear(hidden_dim_2d, out_dim_2d)  # synergy embedding
            )

        # (B) 4D용 Conv 구성
        self.use_conv = False
        if (c1_4d is not None) and (c2_4d is not None) and (out_channels_4d is not None):
            self.use_conv = True
            in_ch = c1_4d + c2_4d
            self.conv1 = nn.Conv2d(in_ch, out_channels_4d, kernel_size=1, bias=False)
            self.bn1   = nn.BatchNorm2d(out_channels_4d)
            self.relu  = nn.ReLU(inplace=True)

    def forward(self, input1, input2, feat_key="feat_2d"):
        """
        input1, input2: 
         - 딕셔너리(Teacher 출력)일 수도 있고, 
         - 이미 Tensor(2D/4D)일 수도 있음.
         
        feat_key: dict일 때 "feat_2d" or "feat_4d" 등에서 텐서를 꺼낼 키.
        """

        # 1) 만약 inputX가 dict이면, feat_key를 꺼내 텐서로 만든다.
        if isinstance(input1, dict):
            feat1 = input1[feat_key]
        else:
            feat1 = input1  # 이미 텐서라고 가정

        if isinstance(input2, dict):
            feat2 = input2[feat_key]
        else:
            feat2 = input2

        # 2) 이제 feat1, feat2는 모두 텐서여야 한다.
        if feat1.dim() == 2 and feat2.dim() == 2:
            # 2D path
            if not self.use_mlp:
                raise ValueError("MBM: 2D input detected but MLP is not configured!")
            x = torch.cat([feat1, feat2], dim=1)  # [N, d1+d2]
            synergy_emb = self.mlp(x)  # [N, out_dim_2d]
            return {"feat_2d": synergy_emb}

        elif feat1.dim() == 4 and feat2.dim() == 4:
            # 4D path
            if not self.use_conv:
                raise ValueError("MBM: 4D input detected but Conv is not configured!")
            x = torch.cat([feat1, feat2], dim=1)  # => [N, c1+c2, H, W]
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            return {"feat_4d": x}

        else:
            raise ValueError(
                f"MBM forward: feat1/feat2 dimension mismatch or unsupported shape.\n"
                f" -> feat1.shape={feat1.shape}, feat2.shape={feat2.shape}"
            )

class SynergyHead(nn.Module):
    """
    2D 전용 SynergyHead (in_dim => num_classes).
    4D를 바로 로짓으로 연결하고 싶다면, 
    별도의 SynergyHead4D를 구현하거나 global pooling 등을 추가해야 함.
    """
    def __init__(self, in_dim, num_classes=100):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, synergy_emb):
        # synergy_emb: [N, in_dim]
        return self.fc(synergy_emb)
