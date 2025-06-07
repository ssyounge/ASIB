# models/teachers/teacher_efficientnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

class TeacherEfficientNetWrapper(nn.Module):
    """
    Teacher 모델(EfficientNet-B2) forward:
     => (feature_dict, logit, ce_loss) 반환
    feature_dict 예시:
      {
        "feat_4d": [N, 1408, H, W],   # backbone.features(x)
        "feat_2d": [N, 1408],         # global pooled
      }
    """
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.criterion_ce = nn.CrossEntropyLoss()

        # 추가: EffNet-B2의 글로벌 피처 차원(1408)
        self.feat_dim = 1408
    
    def forward(self, x, y=None):
        # 1) 4D feature from backbone.features
        # compute features with gradient support so that the teacher can be
        # fine-tuned during adaptive updates
        f4d = self.backbone.features(x)  # shape: [N, 1408, h, w]

        # 2) 최종 로짓(이미지 x 그대로 -> self.backbone(x))
        logit = self.backbone(x)

        # 3) feat_2d: f4d를 adaptive pooling => flatten
        fpool = F.adaptive_avg_pool2d(f4d, (1,1)).flatten(1)  # [N, 1408]

        # (optional) CE loss
        ce_loss = None
        if y is not None:
            ce_loss = self.criterion_ce(logit, y)

        # Dict로 묶어서 반환
        feature_dict = {
            "feat_4d": f4d,       # [N, 1408, h, w]
            "feat_2d": fpool,     # [N, 1408]
        }
        return feature_dict, logit, ce_loss

    def get_feat_dim(self):
        """
        Returns the dimension of the 2D feature (feat_2d).
        EffNet-B2 => 1408
        """
        return self.feat_dim

def create_efficientnet_b2(num_classes=100, pretrained=True):
    """
    EfficientNet-B2를 로드한 뒤, (in_feats->num_classes) 교체
    => TeacherEfficientNetWrapper로 감싸서 반환
    """
    if pretrained:
        model = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
    else:
        model = efficientnet_b2(weights=None)

    in_feats = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_feats, num_classes)

    teacher_model = TeacherEfficientNetWrapper(model)
    return teacher_model
