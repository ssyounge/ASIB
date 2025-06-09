# models/students/student_swin_adapter.py

import torch
import torch.nn as nn
import timm
import torch.nn.functional as F

class StudentSwinAdapter(nn.Module):
    """
    Example Student model:
      1) Swin => (N, in_features)
      2) adapter => residual
      3) final fc => logit
      => returns (feature_dict, logit, ce_loss)
    """
    def __init__(self, pretrained=True, adapter_dim=64, num_classes=100,
                 small_input: bool = False):
        super().__init__()
        self.criterion_ce = nn.CrossEntropyLoss()

        # 1) load Swin Tiny from timm
        img_sz = 32 if small_input else 224
        self.swin = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            img_size=img_sz,
        )
        
        # 2) remove default classifier => (N, in_features)
        in_features = self.swin.head.in_features
        self.swin.head = nn.Identity()
        self.feat_dim = in_features

        # 3) small linear adapter => residual
        self.adapter = nn.Sequential(
            nn.Linear(in_features, adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, in_features),
        )

        # 4) final linear => (N, num_classes)
        self.fc = nn.Linear(in_features, num_classes)

    def get_feat_dim(self) -> int:
        """Return the dimension of the 2D feature (feat_2d)."""
        return self.feat_dim

    def forward(self, x, y=None):
        """
        => (feature_dict, logit, ce_loss)

        1) Swin => feat_2d (N, in_features)
        2) adapter + residual => feat_2d
        3) final fc => logit
        """
        # 1) base Swin => 2D embedding
        feat_2d = self.swin(x)  # shape: [N, in_features]

        # (만약 swin forward_features(...) => 4D를 보고 싶다면, 
        #  teacher_swin 처럼 f4d = self.swin.forward_features(x)을 활용)

        # 2) adapter residual
        adapter_out = feat_2d + self.adapter(feat_2d)  # [N,in_features]

        # 3) final classification
        logit = self.fc(adapter_out)  # [N, num_classes]

        # optional CE
        ce_loss = None
        if y is not None:
            ce_loss = self.criterion_ce(logit, y)

        # feature_dict
        # 만약 4D feat가 필요하다면 => f4d = self.swin.forward_features(x) etc.
        # 여기서는 2D만 반환
        feature_dict = {
            "feat_2d": adapter_out,  # [N, in_features]
            # "feat_4d": ??? => Swin forward_features(x)로 추출 가능
        }
        return feature_dict, logit, ce_loss


def create_swin_adapter_student(pretrained=True, adapter_dim=64, num_classes=100,
                                small_input: bool = False):
    """
    Creates the Student Swin model w/ adapter => (dict, logit, ce_loss)
    """
    return StudentSwinAdapter(
        pretrained=pretrained,
        adapter_dim=adapter_dim,
        num_classes=num_classes,
        small_input=small_input,
    )
