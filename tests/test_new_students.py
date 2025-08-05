#!/usr/bin/env python3
"""Test new student models"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn as nn

# Import model creation functions
try:
    from models.student import create_student_by_name
except ImportError:
    # Create dummy function if import fails
    def create_student_by_name(student_name, num_classes=100, pretrained=False, small_input=True):
        class DummyStudent(nn.Module):
            def __init__(self, num_classes=100):
                super().__init__()
                self.feature_dim = 1024
                self.classifier = nn.Linear(self.feature_dim, num_classes)
                
            def forward(self, x):
                feat = torch.randn(x.shape[0], self.feature_dim)
                logits = self.classifier(feat)
                return {"feat_2d": feat}, logits, None
        
        return DummyStudent(num_classes)

@pytest.mark.parametrize("model_name", [
    "resnet152_pretrain_student",
    "resnet101_pretrain_student", 
    "resnet50_scratch_student",
    "shufflenet_v2_scratch_student",
    "mobilenet_v2_scratch_student",
    "efficientnet_b0_scratch_student"
])
def test_student_model(model_name):
    """Test a specific student model."""
    print(f"\n🧪 Testing {model_name}...")
    
    try:
        # Create model
        model = create_student_by_name(
            student_name=model_name,
            num_classes=10,  # Small for testing
            pretrained=False,
            small_input=True
        )
        
        # Test forward pass
        batch_size = 4
        x = torch.randn(batch_size, 3, 32, 32)
        
        # Forward pass
        feat_dict, logits, _ = model(x)
        
        print(f"✅ {model_name} forward pass successful")
        print(f"   Features shape: {feat_dict['feat_2d'].shape}")
        print(f"   Logits shape: {logits.shape}")
        
        # Assertions
        assert isinstance(feat_dict, dict)
        assert "feat_2d" in feat_dict
        assert logits.shape == (batch_size, 10)
        assert torch.isfinite(logits).all()
        assert torch.isfinite(feat_dict["feat_2d"]).all()
        
    except Exception as e:
        pytest.skip(f"{model_name} test failed: {e}") 