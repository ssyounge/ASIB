"""
models/student_vit_adapter.py

Student model: Vision Transformer (ViT) with partial freeze + Adapter layer.
Using timm or torchvision if available, or a custom huggingface model, etc.
This is a minimal example.
"""

import torch
import torch.nn as nn
import timm  # for demonstration, can use huggingface or other libs

class StudentViTAdapter(nn.Module):
    """
    Example Student model based on ViT + Adapter layer.
    We'll freeze the lower transformer blocks, 
    then insert a small adapter in the last block or so.
    """
    def __init__(self, pretrained=True, adapter_dim=64, num_classes=100):
        super().__init__()
        # e.g. use timm: "vit_base_patch16_224" or smaller 
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=pretrained)
        
        # Replace final head => for 100 classes or CIFAR, etc.
        # depends on the dimension (768 for base ViT)
        in_features = self.vit.head.in_features
        self.vit.head = nn.Identity()  # remove default classifier

        # Adapter: e.g. a small MLP or bottleneck to partially freeze
        # We'll demonstrate a naive approach: a single linear 
        self.adapter = nn.Sequential(
            nn.Linear(in_features, adapter_dim),
            nn.ReLU(inplace=True),
            nn.Linear(adapter_dim, in_features),
        )
        # final classifier
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        1) forward vit => [N, in_features]
        2) pass adapter => same dimension
        3) final fc => [N, num_classes]
        returns: (feat, logit, None)
         - feat: could be adapter output
         - logit: final classification
        """
        # vit forward => patch embedding => [N, in_features]
        feat = self.vit(x)  # shape [N, 768] for base vit

        # pass through adapter
        adapter_out = feat + self.adapter(feat)  # residual style if you want

        # final fc
        logit = self.fc(adapter_out)

        return adapter_out, logit, None
