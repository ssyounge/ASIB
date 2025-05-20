"""
models/students/student_swin_adapter.py

Student model: Swin Transformer with partial freeze + Adapter layer.
Using timm or huggingface. Minimal example with partial freeze.
"""

import torch
import torch.nn as nn
import timm

class StudentSwinAdapter(nn.Module):
    """
    Example Student model based on Swin Transformer.
    We'll assume Swin-T (tiny) for demonstration. 
    We'll freeze all layers except an adapter near the end, then do final classification.
    """
    def __init__(self, pretrained=True, adapter_dim=64, num_classes=100):
        super().__init__()
        self.swin = timm.create_model("swin_tiny_patch4_window7_224", pretrained=pretrained)
        in_features = self.swin.head.in_features
        self.swin.head = nn.Identity()  # remove default classifier

        # Insert a small adapter after the final stage
        self.adapter = nn.Sequential(
            nn.Linear(in_features, adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, in_features),
        )

        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        1) forward Swin => [N, in_features]
        2) pass adapter => same dimension
        3) final fc
        returns (feat, logit, None)
        """
        feat = self.swin(x)  # shape [N, in_features]

        # pass adapter
        adapter_out = feat + self.adapter(feat)

        logit = self.fc(adapter_out)

        return adapter_out, logit, None
