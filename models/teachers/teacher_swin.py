# models/teachers/teacher_swin.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import swin_t, Swin_T_Weights

class TeacherSwinWrapper(nn.Module):
    """
    Teacher 모델(Swin Tiny) forward 래퍼.
    => (feat, logit, ce_loss) 반환
    """
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.criterion_ce = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        with torch.no_grad():
            feat_map = self.backbone.forward_features(x) # feat_map = (N, C, H, W)
            feat = F.adaptive_avg_pool2d(feat_map, (1,1)).flatten(1) # shape = (N, C)

        # 3) head => logit
        logit = self.backbone.head(feat)

        ce_loss = None
        if y is not None:
            ce_loss = self.criterion_ce(logit, y)

        return feat, logit, ce_loss

def create_swin_t(num_classes=100, pretrained=True):
    """
    Creates Swin Tiny model with `num_classes` out_features in the head.
    e.g. CIFAR-100 => num_classes=100
         ImageNet100 => num_classes=100
    """
    if pretrained:
        model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
    else:
        model = swin_t(weights=None)

    # Replace final head
    in_ch = model.head.in_features
    model.head = nn.Linear(in_ch, num_classes)

    teacher_model = TeacherSwinWrapper(model)
    return teacher_model
