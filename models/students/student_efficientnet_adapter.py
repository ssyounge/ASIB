# models/students/student_efficientnet_adapter.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

class ExtendedAdapterEffNetB2(nn.Module):
    """
    EfficientNet-B2 + Adapter
    1) self.backbone.features(x) => 4D
    2) adapter => (4D -> 4D)
    3) global pool => flatten => classifier => logit
    => 최종 (feature_dict, logit, ce_loss) 반환
    """
    def __init__(self, base_model, num_classes=100):
        super().__init__()
        self.backbone = base_model
        self.num_classes = num_classes
        self.criterion_ce = nn.CrossEntropyLoss()

        # adapter
        self.adapter_conv1 = nn.Conv2d(1408, 512, kernel_size=1, bias=False)
        self.adapter_gn1   = nn.GroupNorm(32, 512)
        self.adapter_conv2 = nn.Conv2d(512, 1408, kernel_size=1, bias=False)
        self.adapter_gn2   = nn.GroupNorm(32, 1408)
        self.adapter_relu  = nn.ReLU(inplace=True)

    def forward(self, x, y=None):
        # 1) backbone 4D feature
        fx = self.backbone.features(x)  # shape: (N,1408,H,W)

        # 2) adapter
        xa = self.adapter_conv1(fx)
        xa = self.adapter_gn1(xa)
        xa = self.adapter_relu(xa)
        xa = self.adapter_conv2(xa)
        xa = self.adapter_gn2(xa)

        fx = fx + xa
        fx = self.adapter_relu(fx)  # shape: [N,1408,H,W]

        # 3) global pool => flatten => classifier => logit
        feat_2d = F.adaptive_avg_pool2d(fx, (1,1)).flatten(1)  # [N,1408]
        logit = self.backbone.classifier(feat_2d)              # [N,num_classes]

        # optional CE loss
        ce_loss = None
        if y is not None:
            ce_loss = self.criterion_ce(logit, y)

        # feature dict
        feature_dict = {
            "feat_4d": fx,        # [N,1408,H,W]  (adapter 후)
            "feat_2d": feat_2d,   # [N,1408]
        }
        return feature_dict, logit, ce_loss


def create_efficientnet_b2_with_adapter(pretrained=True):
    """
    1) efficientnet_b2 로드
    2) classifier => (1408->100)
    3) ExtendedAdapterEffNetB2 래핑 => (feature_dict, logit, ce_loss) 반환
    """
    if pretrained:
        model = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
    else:
        model = efficientnet_b2(weights=None)

    in_feats = model.classifier[1].in_features  # 1408
    model.classifier[1] = nn.Linear(in_feats, 100)

    student_model = ExtendedAdapterEffNetB2(model, num_classes=100)
    return student_model
