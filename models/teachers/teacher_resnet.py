import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet101, ResNet101_Weights

class TeacherResNetWrapper(nn.Module):
    """
    Teacher 모델(ResNet101) forward 래퍼:
     (feat, logit, ce_loss) 반환
    """
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.criterion_ce = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        # Stem
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        # Layers
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # Global pool
        x = self.backbone.avgpool(x) # shape: (N, 2048, 1, 1)
        feat = torch.flatten(x, 1)  # shape: (N, 2048)

        # FC => logit
        logit = self.backbone.fc(feat)

        ce_loss = None
        if y is not None:
            ce_loss = self.criterion_ce(logit, y)

        return feat, logit, ce_loss


def create_resnet101(num_classes=100, pretrained=True):
    """
    Loads ResNet101 (optionally pretrained on ImageNet-1K).
    Then replaces the final FC (in_features -> num_classes).
    e.g. for CIFAR-100, ImageNet100, etc.
    """
    if pretrained:
        model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
    else:
        model = resnet101(weights=None)

    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    return model
