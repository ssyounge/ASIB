"""
models/students/student_swin_adapter.py

Student model: Swin Transformer with partial freeze + Adapter layer.
Using timm. Minimal example returning a single `out` => shape (N, num_classes).
"""

import torch
import torch.nn as nn
import timm

class StudentSwinAdapter(nn.Module):
    """
    Example Student model based on Swin Transformer (tiny).
    We'll freeze all layers except an adapter near the end, then do final classification.
    (Similar to ResNet/EfficientNet adapters that just return `out`)
    """
    def __init__(self, pretrained=True, adapter_dim=64, num_classes=100):
        super().__init__()
        # 1) load Swin Tiny from timm
        self.swin = timm.create_model(
            "swin_tiny_patch4_window7_224", 
            pretrained=pretrained
        )
        
        # 2) remove default classifier => (N, in_features)
        in_features = self.swin.head.in_features
        self.swin.head = nn.Identity()

        # 3) Insert a small linear adapter => residual
        self.adapter = nn.Sequential(
            nn.Linear(in_features, adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, in_features),
        )

        # 4) final linear => (N, num_classes)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        1) Swin => (N, in_features)
        2) pass adapter w/ skip => (N, in_features)
        3) final fc => out(N, num_classes)
        returns `out` only, consistent with student_resnet_adapter, student_efficientnet_adapter
        """
        # base Swin embedding => shape (N, in_features)
        feat = self.swin(x)

        # adapter residual
        adapter_out = feat + self.adapter(feat)

        # final classification
        out = self.fc(adapter_out)  # shape (N, num_classes)
        return out


def create_swin_adapter_student(pretrained=True, adapter_dim=64, num_classes=100):
    """
    Creates the Student Swin model with the adapter. 
    final forward(x)-> out(N, num_classes)
    """
    return StudentSwinAdapter(
        pretrained=pretrained,
        adapter_dim=adapter_dim,
        num_classes=num_classes
    )
