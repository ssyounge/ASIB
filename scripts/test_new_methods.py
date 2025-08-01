#!/usr/bin/env python3
"""
test_new_methods.py

Test script for newly added KD methods:
- SimKD (2022)
- ReviewKD (2020) 
- SSKD (2020)
- AB (Attention Branch, 2019)
- FT (Factor Transfer, 2019)
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Any

from methods import (
    SimKDDistiller,
    ReviewKDDistiller,
    SSKDDistiller,
    ABDistiller,
    FTDistiller,
)

def create_dummy_models():
    """Create dummy teacher and student models for testing."""
    
    class DummyTeacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(64, 100)
            
        def forward(self, x):
            feat = self.conv(x)
            pooled = self.pool(feat)
            logits = self.fc(pooled.flatten(1))
            return {
                "logit": logits,
                "feat_2d": feat,
                "attention": feat,  # For AB
                "factor": feat,     # For FT
            }
    
    class DummyStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 32, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(32, 100)
            
        def forward(self, x):
            feat = self.conv(x)
            pooled = self.pool(feat)
            logits = self.fc(pooled.flatten(1))
            return {
                "feat_2d": feat,
                "attention": feat,  # For AB
                "factor": feat,     # For FT
            }, logits, None
    
    return DummyTeacher(), DummyStudent()

def test_method(method_class, method_name: str, config: Dict[str, Any]):
    """Test a specific KD method."""
    print(f"\n🧪 Testing {method_name}...")
    
    # Create models
    teacher, student = create_dummy_models()
    
    # Create distiller
    distiller = method_class(teacher, student, config=config)
    
    # Create dummy data
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32)
    y = torch.randint(0, 100, (batch_size,))
    
    # Test forward pass
    try:
        loss, logits = distiller(x, y)
        print(f"✅ {method_name} forward pass successful")
        print(f"   Loss: {loss.item():.4f}")
        print(f"   Logits shape: {logits.shape}")
        return True
    except Exception as e:
        print(f"❌ {method_name} forward pass failed: {e}")
        return False

def main():
    """Test all new KD methods."""
    print("🚀 Testing New KD Methods")
    print("=" * 50)
    
    # Test configurations
    test_configs = {
        "simkd": {
            "ce_alpha": 0.3,
            "tau_start": 4.0,
            "feature_weight": 1.0,
        },
        "reviewkd": {
            "ce_alpha": 0.3,
            "tau_start": 4.0,
            "feature_weight": 1.0,
            "attention_weight": 0.5,
        },
        "sskd": {
            "ce_alpha": 0.3,
            "tau_start": 4.0,
            "contrastive_weight": 1.0,
            "feature_weight": 0.5,
        },
        "ab": {
            "ce_alpha": 0.3,
            "tau_start": 4.0,
            "attention_weight": 1.0,
            "feature_weight": 0.5,
        },
        "ft": {
            "ce_alpha": 0.3,
            "tau_start": 4.0,
            "factor_weight": 1.0,
            "feature_weight": 0.5,
        },
    }
    
    # Test methods
    methods = [
        (SimKDDistiller, "SimKD"),
        (ReviewKDDistiller, "ReviewKD"),
        (SSKDDistiller, "SSKD"),
        (ABDistiller, "AB"),
        (FTDistiller, "FT"),
    ]
    
    results = {}
    for method_class, method_name in methods:
        config = test_configs.get(method_name.lower(), {})
        success = test_method(method_class, method_name, config)
        results[method_name] = success
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 30)
    for method_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{method_name:<12} | {status}")
    
    passed = sum(results.values())
    total = len(results)
    print(f"\n🎯 Overall: {passed}/{total} methods passed")

if __name__ == "__main__":
    main() 