#!/usr/bin/env python3
"""Test ConvNeXt-Small teacher model"""

import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import teacher model to register it
import models.teachers.convnext_s_teacher

from core.builder import create_teacher_by_name

def test_convnext_s_teacher():
    """Test ConvNeXt-Small teacher with dummy input"""
    print("🧪 Testing ConvNeXt-Small teacher...")
    
    try:
        # Create model
        model = create_teacher_by_name(
            teacher_name="convnext_s_teacher",
            num_classes=100,
            pretrained=False,  # Use random weights for testing
            small_input=True,
        )
        
        # Create dummy input
        dummy_input = torch.randn(2, 3, 32, 32)  # CIFAR-100 size
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
            features_4d, features_2d = model.extract_feats(dummy_input)
        
        # Print results
        print(f"✅ ConvNeXt-Small teacher - SUCCESS")
        if isinstance(output, tuple):
            print(f"   Output shape: {output[1].shape} (logits)")
        else:
            print(f"   Output shape: {output.shape}")
        print(f"   Features 4D: {features_4d.shape}")
        print(f"   Features 2D: {features_2d.shape}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        assert True  # Test passed
        
    except Exception as e:
        print(f"❌ ConvNeXt-Small teacher - FAILED")
        print(f"   Error: {e}")
        assert False, f"Test failed: {e}"

if __name__ == "__main__":
    success = test_convnext_s_teacher()
    if success:
        print("\n🎉 ConvNeXt-Small teacher is working correctly!")
    else:
        print("\n⚠️  ConvNeXt-Small teacher needs debugging.") 