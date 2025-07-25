# models/students/resnet152_student.py

import torch
import torch.nn as nn
from torchvision.models import resnet152, ResNet152_Weights

class ExtendedAdapterResNet152(nn.Module):
    """
    ResNet152 + Adapter (layer3 뒤 etc.)
    => forward(x,y=None)->(feature_dict, logit, ce_loss)
    feature_dict:
      {
        "feat_4d": [N, 1024(or 2048?), H, W]  # adapter 이후 최종 feature
        "feat_2d": [N, ???],                # global pool
        "feat_4d_layer1": [N, C1, H1, W1],  # layer1 출력
        "feat_4d_layer2": [N, C2, H2, W2],  # layer2 출력
        "feat_4d_layer3": [N, C3, H3, W3],  # layer3 출력
      }
    """
    def __init__(self, base_model):
        super().__init__()
        self.backbone = base_model
        self.criterion_ce = nn.CrossEntropyLoss()
        self.feat_dim = 2048

        # 기존 resnet structure references
        self.conv1   = self.backbone.conv1
        self.bn1     = self.backbone.bn1
        self.relu    = self.backbone.relu
        self.maxpool = self.backbone.maxpool
        self.layer1  = self.backbone.layer1
        self.layer2  = self.backbone.layer2
        self.layer3  = self.backbone.layer3
        self.layer4  = self.backbone.layer4
        self.avgpool = self.backbone.avgpool
        self.fc      = self.backbone.fc

        # adapter
        self.adapter_conv1 = nn.Conv2d(1024, 512, kernel_size=1, bias=False)
        self.adapter_gn1   = nn.GroupNorm(32, 512)
        self.adapter_conv2 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.adapter_gn2   = nn.GroupNorm(32, 512)
        self.adapter_conv3 = nn.Conv2d(512, 1024, kernel_size=1, bias=False)
        self.adapter_gn3   = nn.GroupNorm(32, 1024)
        self.adapter_relu  = nn.ReLU(inplace=True)

    def get_feat_dim(self) -> int:
        """Return the dimension of the 2D feature (feat_2d)."""
        return self.feat_dim

    def forward(self, x, y=None):
        # 1) stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 2) layer1,2,3
        x = self.layer1(x)
        feat_layer1 = x
        x = self.layer2(x)
        feat_layer2 = x
        x = self.layer3(x)
        feat_layer3 = x

        # 3) adapter
        xa = self.adapter_conv1(x)
        xa = self.adapter_gn1(xa)
        xa = self.adapter_relu(xa)
        xa = self.adapter_conv2(xa)
        xa = self.adapter_gn2(xa)
        xa = self.adapter_relu(xa)
        xa = self.adapter_conv3(xa)
        xa = self.adapter_gn3(xa)

        x = x + xa
        x = self.adapter_relu(x)  # => shape: [N, 1024, H, W]

        # 4) layer4
        f4 = self.layer4(x)  # [N,2048,H',W']

        # 5) global pool => flatten => fc => logit
        gp   = self.avgpool(f4)     # [N,2048,1,1]
        feat_2d = torch.flatten(gp, 1)  # [N,2048]
        logit   = self.fc(feat_2d)      # [N,100]

        # optional CE
        ce_loss = None
        if y is not None:
            ce_loss = self.criterion_ce(logit, y)

        # dict
        # adapter output x => 4D(1024) or final f4 => 4D(2048), 취사선택
        # 여기서는 "feat_4d"=adapter 마지막 => f4
        feature_dict = {
            "feat_4d": f4,         # [N,2048,H',W']
            "feat_2d": feat_2d,    # [N,2048]
            # intermediate features for KD baselines
            "feat_4d_layer1": feat_layer1,
            "feat_4d_layer2": feat_layer2,
            "feat_4d_layer3": feat_layer3,
        }
        return feature_dict, logit, ce_loss


def create_resnet152_with_extended_adapter(
    pretrained: bool = True,
    num_classes: int = 100,
    small_input: bool = False,
):
    """
    ResNet152 load => last FC => 100
    => ExtendedAdapterResNet152 => (dict, logit, ce_loss)
    """
    if pretrained:
        base = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
    else:
        base = resnet152(weights=None)

    if small_input:
        base.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()

    num_ftrs = base.fc.in_features
    base.fc = nn.Linear(num_ftrs, num_classes)

    model = ExtendedAdapterResNet152(base)
    return model
