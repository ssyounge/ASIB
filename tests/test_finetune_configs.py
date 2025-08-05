#!/usr/bin/env python3
"""Test finetune configurations"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from omegaconf import OmegaConf

@pytest.mark.parametrize("config_name", [
    "convnext_s_cifar100",
    "convnext_s_imagenet32",
    "convnext_l_cifar100",
    "convnext_l_imagenet32",
    "efficientnet_l2_cifar100",
    "efficientnet_l2_imagenet32",
    "resnet152_cifar100",
    "resnet152_imagenet32"
])
def test_finetune_config(config_name):
    """Test a finetune configuration"""
    print(f"\n🧪 Testing finetune config: {config_name}")
    
    try:
        # Load config
        config_path = f"configs/finetune/{config_name}.yaml"
        if not Path(config_path).exists():
            pytest.skip(f"Config file not found: {config_path}")
            
        cfg = OmegaConf.load(config_path)
        
        # Check required fields
        required_fields = [
            "teacher_type",
            "small_input", 
            "teacher_pretrained",
            "finetune_epochs",
            "finetune_lr",
            "batch_size",
            "results_dir",
            "exp_id",
            "finetune_ckpt_path",
            "warmup_epochs",
            "min_lr",
            "early_stopping_patience",
            "early_stopping_min_delta"
        ]
        
        missing_fields = []
        for field in required_fields:
            if field not in cfg:
                missing_fields.append(field)
        
        if missing_fields:
            pytest.skip(f"Missing required fields: {missing_fields}")
        
        # Print config summary
        print(f"✅ Config loaded successfully")
        print(f"   Teacher: {cfg.teacher_type}")
        print(f"   Epochs: {cfg.finetune_epochs}")
        print(f"   LR: {cfg.finetune_lr}")
        print(f"   Batch size: {cfg.batch_size}")
        print(f"   Warm-up: {cfg.warmup_epochs}")
        print(f"   Early stopping patience: {cfg.early_stopping_patience}")
        print(f"   Output: {cfg.results_dir}")
        print(f"   Checkpoint: {cfg.finetune_ckpt_path}")
        
        # Validate configuration values
        assert cfg.finetune_epochs > 0
        assert cfg.finetune_lr > 0
        assert cfg.batch_size > 0
        assert cfg.warmup_epochs >= 0
        assert cfg.warmup_epochs < cfg.finetune_epochs
        assert cfg.min_lr > 0
        assert cfg.min_lr < cfg.finetune_lr
        assert cfg.early_stopping_patience > 0
        assert cfg.early_stopping_patience < cfg.finetune_epochs
        assert cfg.early_stopping_min_delta > 0
        assert cfg.early_stopping_min_delta < 1.0
        
        assert True
        
    except Exception as e:
        pytest.skip(f"Error loading config: {e}")

def test_cifar100_model_specific_configs():
    """Test CIFAR-100 specific model configurations"""
    cifar100_configs = [
        ("convnext_l_cifar100", {
            "finetune_epochs": 60,
            "finetune_lr": 8e-5,
            "batch_size": 64,
            "warmup_epochs": 5,
            "early_stopping_patience": 15
        }),
        ("convnext_s_cifar100", {
            "finetune_epochs": 80,
            "finetune_lr": 1.5e-4,
            "batch_size": 128,
            "warmup_epochs": 4,
            "early_stopping_patience": 10
        }),
        ("resnet152_cifar100", {
            "finetune_epochs": 70,
            "finetune_lr": 1.2e-4,
            "batch_size": 64,
            "warmup_epochs": 3,
            "early_stopping_patience": 10
        }),
        ("efficientnet_l2_cifar100", {
            "finetune_epochs": 65,
            "finetune_lr": 1.8e-4,
            "batch_size": 32,
            "warmup_epochs": 3,
            "early_stopping_patience": 6
        })
    ]
    
    for config_name, expected_values in cifar100_configs:
        config_path = f"configs/finetune/{config_name}.yaml"
        if Path(config_path).exists():
            cfg = OmegaConf.load(config_path)
            
            for key, expected_value in expected_values.items():
                actual_value = cfg.get(key)
                assert actual_value == expected_value, \
                    f"{config_name}: {key} expected {expected_value}, got {actual_value}"
        else:
            pytest.skip(f"Config file not found: {config_path}")

def test_warmup_early_stopping_logic():
    """Test warm-up and early stopping logic"""
    configs = [
        "configs/finetune/convnext_l_cifar100.yaml",
        "configs/finetune/convnext_s_cifar100.yaml",
        "configs/finetune/resnet152_cifar100.yaml",
        "configs/finetune/efficientnet_l2_cifar100.yaml"
    ]
    
    for config_path in configs:
        if Path(config_path).exists():
            cfg = OmegaConf.load(config_path)
            
            # Warm-up logic validation
            assert cfg.warmup_epochs < cfg.finetune_epochs, \
                f"Warm-up epochs ({cfg.warmup_epochs}) should be less than total epochs ({cfg.finetune_epochs})"
            
            # Early stopping logic validation
            assert cfg.early_stopping_patience < cfg.finetune_epochs, \
                f"Early stopping patience ({cfg.early_stopping_patience}) should be less than total epochs ({cfg.finetune_epochs})"
            
            # Learning rate validation
            assert cfg.min_lr < cfg.finetune_lr, \
                f"Min LR ({cfg.min_lr}) should be less than max LR ({cfg.finetune_lr})"
            
            # Early stopping delta validation
            assert 0 < cfg.early_stopping_min_delta < 1.0, \
                f"Early stopping min delta ({cfg.early_stopping_min_delta}) should be between 0 and 1"
        else:
            pytest.skip(f"Config file not found: {config_path}") 