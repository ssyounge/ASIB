#!/usr/bin/env python3
"""Test core.utils functions used in main.py"""

import torch
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from core.utils import (
    _renorm_ce_kd,
    renorm_ce_kd,
    setup_partial_freeze_schedule,
    setup_partial_freeze_schedule_with_cfg,
    setup_safety_switches,
    setup_safety_switches_with_cfg,
    auto_set_mbm_query_dim,
    auto_set_mbm_query_dim_with_model,
    cast_numeric_configs,
)


class TestRenormCeKd:
    """Test renorm_ce_kd functions"""
    
    def test_renorm_ce_kd_basic(self):
        """Test basic renorm_ce_kd functionality"""
        config = {
            "ce_alpha": 0.5,
            "kd_alpha": 0.3
        }
        
        renorm_ce_kd(config)
        
        # Should normalize to sum = 1.0
        assert abs(config["ce_alpha"] + config["kd_alpha"] - 1.0) < 1e-5
    
    def test_renorm_ce_kd_zero_values(self):
        """Test with zero values"""
        config = {
            "ce_alpha": 0.0,
            "kd_alpha": 0.0
        }
        
        renorm_ce_kd(config)
        
        # Should set equal values
        assert abs(config["ce_alpha"] - 0.5) < 1e-5
        assert abs(config["kd_alpha"] - 0.5) < 1e-5
    
    def test_renorm_ce_kd_one_zero(self):
        """Test with one zero value"""
        config = {
            "ce_alpha": 1.0,
            "kd_alpha": 0.0
        }
        
        renorm_ce_kd(config)
        
        # Should normalize
        assert abs(config["ce_alpha"] + config["kd_alpha"] - 1.0) < 1e-5
    
    def test_renorm_ce_kd_already_normalized(self):
        """Test with already normalized values"""
        config = {
            "ce_alpha": 0.7,
            "kd_alpha": 0.3
        }
        
        original_ce = config["ce_alpha"]
        original_kd = config["kd_alpha"]
        
        renorm_ce_kd(config)
        
        # Should remain the same
        assert abs(config["ce_alpha"] - original_ce) < 1e-5
        assert abs(config["kd_alpha"] - original_kd) < 1e-5


class TestSetupPartialFreezeSchedule:
    """Test setup_partial_freeze_schedule functions"""
    
    def test_setup_partial_freeze_schedule_basic(self):
        """Test basic setup_partial_freeze_schedule"""
        result = setup_partial_freeze_schedule(3)
        
        # Should return list of -1s
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(x == -1 for x in result)
    
    def test_setup_partial_freeze_schedule_with_cfg(self):
        """Test setup_partial_freeze_schedule_with_cfg"""
        cfg = {
            "experiment": {
                "num_stages": 3,
                "student_freeze_level": -1
            }
        }
        
        setup_partial_freeze_schedule_with_cfg(cfg, 3)
        
        # Should set schedule in experiment section
        assert "student_freeze_schedule" in cfg["experiment"]
        assert len(cfg["experiment"]["student_freeze_schedule"]) == 3
    
    def test_setup_partial_freeze_schedule_custom(self):
        """Test with custom schedule"""
        result = setup_partial_freeze_schedule(5)
        
        # Should return list of -1s with correct length
        assert isinstance(result, list)
        assert len(result) == 5
        assert all(x == -1 for x in result)


class TestSetupSafetySwitches:
    """Test setup_safety_switches functions"""
    
    def test_setup_safety_switches_basic(self):
        """Test basic setup_safety_switches"""
        result = setup_safety_switches(3)
        
        # Should return dict with safety switches
        assert isinstance(result, dict)
        assert "student_freeze_level" in result
        assert "teacher1_freeze_level" in result
        assert "teacher2_freeze_level" in result
        assert "student_freeze_schedule" in result
        assert len(result["student_freeze_schedule"]) == 3
    
    def test_setup_safety_switches_with_cfg(self):
        """Test setup_safety_switches_with_cfg"""
        cfg = {
            "experiment": {
                "num_stages": 3,
                "use_partial_freeze": False
            }
        }
        
        setup_safety_switches_with_cfg(cfg, 3)
        
        # Should set safety switches in experiment section when use_partial_freeze is False
        assert "student_freeze_level" in cfg["experiment"]
        assert "teacher1_freeze_level" in cfg["experiment"]
        assert "teacher2_freeze_level" in cfg["experiment"]
        assert "student_freeze_schedule" in cfg["experiment"]
        assert cfg["experiment"]["student_freeze_level"] == -1
    
    def test_setup_safety_switches_custom(self):
        """Test with custom safety switches"""
        result = setup_safety_switches(4)
        
        # Should return dict with correct number of stages
        assert isinstance(result, dict)
        assert len(result["student_freeze_schedule"]) == 4
        assert all(x == -1 for x in result["student_freeze_schedule"])


class TestAutoSetMbmQueryDim:
    """Test auto_set_mbm_query_dim functions"""
    
    def test_auto_set_mbm_query_dim_basic(self):
        """Test basic auto_set_mbm_query_dim"""
        cfg = {
            "ib_mbm_query_dim": None
        }
        
        auto_set_mbm_query_dim(cfg)
        
        # Should set default query_dim
        assert cfg.get("ib_mbm_query_dim") == 512
    
    def test_auto_set_mbm_query_dim_with_model(self):
        """Test auto_set_mbm_query_dim_with_model"""
        cfg = {
            "device": "cuda",
            "ib_mbm_query_dim": None,
        }
        
        # Mock model with feature dimension
        class MockModel:
            def __init__(self):
                self.feature_dim = 1024
            
            def __call__(self, x):
                return {
                    "distill_feat": torch.randn(1, 1024),
                    "feat_2d": torch.randn(1, 1024)
                }, None, None
        
        model = MockModel()
        
        auto_set_mbm_query_dim_with_model(model, cfg)
        
        # Should set top-level ib_mbm_query_dim
        assert cfg["ib_mbm_query_dim"] == 1024
    
    def test_auto_set_mbm_query_dim_already_set(self):
        """Test when query_dim is already set"""
        cfg = {
            "ib_mbm_query_dim": 256
        }
        
        auto_set_mbm_query_dim(cfg)
        
        # Should not change existing value
        assert cfg.get("ib_mbm_query_dim") == 256
    
    def test_auto_set_mbm_query_dim_no_feature_dim(self):
        """Test with model that has no feature_dim"""
        cfg = {
            "ib_mbm_query_dim": None
        }
        
        # Should handle gracefully
        auto_set_mbm_query_dim(cfg)
        
        # Should set default value
        assert cfg.get("ib_mbm_query_dim") == 512
        assert (cfg.get("ib_mbm_query_dim")) is None or (cfg.get("ib_mbm_query_dim")) > 0


class TestCastNumericConfigs:
    """Test cast_numeric_configs function"""
    
    def test_cast_numeric_configs_basic(self):
        """Test basic numeric casting"""
        cfg = {
            "student_lr": "0.1",
            "teacher_lr": "0.01",
            "student_weight_decay": "0.0001",
            "teacher_weight_decay": "0.0001",
            "momentum": "0.9",
            "nesterov": "true",
            "num_stages": "4",
            "student_epochs_per_stage": "15",
            "teacher_adapt_epochs": "0",
        }
        
        cast_numeric_configs(cfg)
        
        # Check that values were converted
        assert isinstance(cfg["student_lr"], float)
        assert isinstance(cfg["teacher_lr"], float)
        assert isinstance(cfg["student_weight_decay"], float)
        assert isinstance(cfg["teacher_weight_decay"], float)
        assert isinstance(cfg["momentum"], float)
        assert isinstance(cfg["nesterov"], bool)
        assert isinstance(cfg["num_stages"], int)
        assert isinstance(cfg["student_epochs_per_stage"], int)
        assert isinstance(cfg["teacher_adapt_epochs"], int)
    
    def test_cast_numeric_configs_already_correct(self):
        """Test with already correct types"""
        cfg = {
            "student_lr": 0.1,
            "teacher_lr": 0.01,
            "num_stages": 4,
            "nesterov": True,
        }
        
        original_values = cfg.copy()
        
        cast_numeric_configs(cfg)
        
        # Should remain the same
        assert cfg["student_lr"] == original_values["student_lr"]
        assert cfg["teacher_lr"] == original_values["teacher_lr"]
        assert cfg["num_stages"] == original_values["num_stages"]
        assert cfg["nesterov"] == original_values["nesterov"]
    
    def test_cast_numeric_configs_mixed_types(self):
        """Test with mixed string and numeric types"""
        cfg = {
            "student_lr": "0.1",
            "teacher_lr": 0.01,
            "num_stages": "4",
            "batch_size": 128,
            "nesterov": "false",
            "use_amp": True,
        }
        
        cast_numeric_configs(cfg)
        
        # Check conversions
        assert isinstance(cfg["student_lr"], float)
        assert isinstance(cfg["teacher_lr"], float)
        assert isinstance(cfg["num_stages"], int)
        assert isinstance(cfg["batch_size"], int)
        assert isinstance(cfg["nesterov"], bool)
        assert isinstance(cfg["use_amp"], bool)
    
    def test_cast_numeric_configs_nested(self):
        """Test with nested configuration"""
        cfg = {
            "experiment": {
                "student_lr": "0.1",
                "teacher_lr": "0.01",
                "num_stages": "4",
            },
            "global_lr": "0.001",
        }
        
        cast_numeric_configs(cfg)
        
        # Check nested conversions
        assert isinstance(cfg["experiment"]["student_lr"], float)
        assert isinstance(cfg["experiment"]["teacher_lr"], float)
        assert isinstance(cfg["experiment"]["num_stages"], int)
        assert isinstance(cfg["global_lr"], float)


class TestIntegration:
    """Test integration scenarios"""
    
    def test_main_py_core_utils_flow(self):
        """Test the flow of core utils functions as used in main.py"""
        # Simulate main.py config processing
        cfg = {
            "experiment": {
                "student_lr": "0.1",
                "teacher_lr": "0.01",
                "num_stages": "3",
                "ce_alpha": "0.7",
                "kd_alpha": "0.3",
                "ib_mbm_query_dim": None,
            }
        }
        
        # Cast numeric configs
        cast_numeric_configs(cfg)
        
        # Setup partial freeze schedule
        setup_partial_freeze_schedule_with_cfg(cfg, 3)
        
        # Setup safety switches
        setup_safety_switches_with_cfg(cfg, 3)
        
        # Renorm ce_kd
        renorm_ce_kd(cfg["experiment"])
        
        # Auto-set ib_mbm query dim (fallback from absence of legacy key)
        class MockModel:
            def __init__(self):
                self.feature_dim = 512
            
            def __call__(self, x):
                return {
                    "distill_feat": torch.randn(1, 512),
                    "feat_2d": torch.randn(1, 512)
                }, None, None
        
        model = MockModel()
        cfg["device"] = "cuda"
        auto_set_mbm_query_dim_with_model(model, cfg)
        
        # Verify all functions worked correctly
        assert isinstance(cfg["experiment"]["student_lr"], float)
        assert isinstance(cfg["experiment"]["teacher_lr"], float)
        assert isinstance(cfg["experiment"]["num_stages"], int)
        assert "student_freeze_schedule" in cfg["experiment"]
        assert "student_freeze_level" in cfg["experiment"]
        assert abs(cfg["experiment"]["ce_alpha"] + cfg["experiment"]["kd_alpha"] - 1.0) < 1e-5
        assert cfg["ib_mbm_query_dim"] == 512
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Test with empty config
        cfg = {}
        cast_numeric_configs(cfg)
        result1 = setup_partial_freeze_schedule(1)
        result2 = setup_safety_switches(1)
        assert isinstance(result1, list)
        assert isinstance(result2, dict)
        
        # Test with None values
        cfg = {
            "student_lr": None,
            "teacher_lr": None,
        }
        cast_numeric_configs(cfg)
        
        # Test with invalid string values
        cfg = {
            "student_lr": "invalid",
            "num_stages": "invalid",
        }
        # Should handle gracefully or raise appropriate error
        try:
            cast_numeric_configs(cfg)
        except (ValueError, TypeError):
            pass  # Expected behavior
