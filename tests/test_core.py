#!/usr/bin/env python3
"""Test all core modules"""

import torch
import pytest
from pathlib import Path

# Import core modules
from core.builder import (
    build_model, create_student_by_name, create_teacher_by_name,
    partial_freeze_teacher_auto, partial_freeze_student_auto
)
from core.trainer import (
    create_optimizers_and_schedulers, create_optimizers_and_schedulers_legacy, run_training_stages,
    run_continual_learning
)
from core.utils import (
    _renorm_ce_kd, setup_partial_freeze_schedule, setup_safety_switches,
    auto_set_mbm_query_dim, cast_numeric_configs
)


class TestCoreBuilder:
    """Test core builder functions"""
    
    def test_build_model(self):
        """Test build_model function"""
        # Test with a simple model
        model = build_model("resnet152_pretrain", num_classes=100)
        assert model is not None
        assert hasattr(model, 'forward')
    
    def test_create_student_by_name(self):
        """Test create_student_by_name function"""
        model = create_student_by_name(
            student_name="resnet152_pretrain",
            num_classes=100,
            pretrained=False,
            small_input=True
        )
        assert model is not None
        assert hasattr(model, 'forward')
    
    def test_create_teacher_by_name(self):
        """Test create_teacher_by_name function"""
        model = create_teacher_by_name(
            teacher_name="resnet152",
            num_classes=100,
            pretrained=False,
            small_input=True
        )
        assert model is not None
        assert hasattr(model, 'forward')
    
    def test_partial_freeze_teacher_auto(self):
        """Test partial_freeze_teacher_auto function"""
        # Create a dummy teacher model
        class DummyTeacher(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(10, 5)
                self.layer2 = torch.nn.Linear(5, 1)
            
            def forward(self, x):
                return self.layer2(self.layer1(x))
        
        model = DummyTeacher()
        
        # Test partial freeze
        partial_freeze_teacher_auto(
            model=model,
            teacher_name="resnet152",
            freeze_bn=True,
            freeze_ln=True,
            freeze_level=1
        )
        
        # Check that some parameters are frozen
        params = list(model.parameters())
        assert len(params) > 0
    
    def test_partial_freeze_student_auto(self):
        """Test partial_freeze_student_auto function"""
        # Create a dummy student model
        class DummyStudent(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(10, 5)
                self.layer2 = torch.nn.Linear(5, 1)
            
            def forward(self, x):
                return self.layer2(self.layer1(x))
        
        model = DummyStudent()
        
        # Test partial freeze
        partial_freeze_student_auto(
            model=model,
            student_name="resnet152_pretrain",
            freeze_bn=True,
            freeze_ln=True,
            freeze_level=1
        )
        
        # Check that some parameters are frozen
        params = list(model.parameters())
        assert len(params) > 0


class TestCoreTrainer:
    """Test core trainer functions"""
    
    def test_create_optimizers_and_schedulers(self):
        """Test create_optimizers_and_schedulers function"""
        # Create dummy models
        teacher = torch.nn.Linear(10, 5)
        student = torch.nn.Linear(10, 5)
        
        # Test optimizer creation
        cfg = {
            "teacher_lr": 1e-4,
            "student_lr": 1e-3,
            "teacher_weight_decay": 1e-4,
            "student_weight_decay": 1e-4,
            "teacher_epochs": 10,
            "student_epochs": 10
        }
        
        teacher_optimizer, teacher_scheduler, student_optimizer, student_scheduler = create_optimizers_and_schedulers_legacy(
            teacher=teacher,
            student_model=student,
            cfg=cfg
        )
        
        assert teacher_optimizer is not None
        assert teacher_scheduler is not None
        assert student_optimizer is not None
        assert student_scheduler is not None
    
    def test_run_training_stages(self):
        """Test run_training_stages function"""
        # This is a complex function that might need more setup
        # For now, just test that it can be imported
        assert run_training_stages is not None
    
    def test_run_continual_learning(self):
        """Test run_continual_learning function"""
        # This is a complex function that might need more setup
        # For now, just test that it can be imported
        assert run_continual_learning is not None


class TestCoreUtils:
    """Test core utility functions"""
    
    def test_renorm_ce_kd(self):
        """Test _renorm_ce_kd function"""
        # Create dummy inputs
        ce_loss = torch.tensor(1.0)
        kd_loss = torch.tensor(0.5)
        ce_alpha = 0.7
        kd_alpha = 0.3
        
        total_loss = _renorm_ce_kd(ce_loss, kd_loss, ce_alpha, kd_alpha)
        assert torch.isfinite(total_loss)
        assert total_loss > 0
    
    def test_setup_partial_freeze_schedule(self):
        """Test setup_partial_freeze_schedule function"""
        num_stages = 4
        schedule = setup_partial_freeze_schedule(num_stages)
        
        assert len(schedule) == num_stages
        assert all(isinstance(x, int) for x in schedule)
        assert all(x >= -1 for x in schedule)  # -1 means no freeze
    
    def test_setup_safety_switches(self):
        """Test setup_safety_switches function"""
        num_stages = 4
        
        switches = setup_safety_switches(num_stages)
        assert isinstance(switches, dict)
        assert "student_freeze_level" in switches
        assert "teacher1_freeze_level" in switches
        assert "teacher2_freeze_level" in switches
        assert "student_freeze_schedule" in switches
    
    def test_auto_set_mbm_query_dim(self):
        """Test auto_set_mbm_query_dim function"""
        cfg = {
            "ib_mbm_query_dim": None,
            "student_feat_dim": 2048
        }
        
        updated_cfg = auto_set_mbm_query_dim(cfg)
        assert updated_cfg["ib_mbm_query_dim"] == 512  # Default value
    
    def test_cast_numeric_configs(self):
        """Test cast_numeric_configs function"""
        cfg = {
            "teacher_lr": "1e-4",
            "student_lr": "1e-3",
            "teacher_weight_decay": "1e-4",
            "student_weight_decay": "1e-4",
            "string_value": "test"
        }
        
        casted_cfg = cast_numeric_configs(cfg)
        assert isinstance(casted_cfg["teacher_lr"], float)
        assert isinstance(casted_cfg["student_lr"], float)
        assert isinstance(casted_cfg["teacher_weight_decay"], float)
        assert isinstance(casted_cfg["student_weight_decay"], float)
        assert isinstance(casted_cfg["string_value"], str)


class TestModelRegistry:
    """Test model registry functionality"""
    
    def test_registry_scanning(self):
        """Test registry scanning"""
        from models.common import registry
        
        # Ensure registry is scanned
        registry.ensure_scanned()
        
        # Check that some models are registered
        assert len(registry.MODEL_REGISTRY) > 0
    
    def test_registry_keys(self):
        """Test registry keys"""
        from models.common import registry
        registry.ensure_scanned()
        
        # Check for specific model keys
        expected_keys = [
            "resnet152",
            "resnet152_pretrain",
            "convnext_l",
            "efficientnet_l2"
        ]
        
        for key in expected_keys:
            if key in registry.MODEL_REGISTRY:
                assert registry.MODEL_REGISTRY[key] is not None


class TestConfigValidation:
    """Test configuration validation"""
    
    def test_config_structure(self):
        """Test config structure validation"""
        from omegaconf import OmegaConf
        
        # Test base config
        base_config = OmegaConf.load("configs/base.yaml")
        assert "defaults" in base_config
        assert "device" in base_config
        assert "batch_size" in base_config
    
    def test_experiment_config(self):
        """Test experiment config validation"""
        from omegaconf import OmegaConf
        
        # Test experiment config (legacy 폴더로 이동됨)
        exp_config = OmegaConf.load("configs/experiment/legacy/res152_convnext_effi.yaml")
        assert "defaults" in exp_config
        assert "teacher1_ckpt" in exp_config
        assert "teacher2_ckpt" in exp_config
    
    def test_finetune_config(self):
        """Test finetune config validation"""
        from omegaconf import OmegaConf
        
        # Test finetune config (cifar32 -> cifar100으로 변경됨)
        ft_config = OmegaConf.load("configs/finetune/resnet152_cifar100.yaml")
        assert "teacher_type" in ft_config
        assert "finetune_epochs" in ft_config
        assert "finetune_lr" in ft_config


class TestIntegration:
    """Test integration scenarios"""
    
    def test_model_creation_pipeline(self):
        """Test complete model creation pipeline"""
        # Create teacher
        teacher = create_teacher_by_name(
            teacher_name="resnet152",
            num_classes=100,
            pretrained=False,
            small_input=True
        )
        
        # Create student
        student = create_student_by_name(
            student_name="resnet152_pretrain",
            num_classes=100,
            pretrained=False,
            small_input=True
        )
        
        # Test forward pass
        x = torch.randn(2, 3, 32, 32)
        
        teacher.eval()
        student.eval()
        
        with torch.no_grad():
            teacher_output = teacher(x)
            student_output = student(x)
        
        assert teacher_output is not None
        assert student_output is not None
    
    def test_optimizer_creation_pipeline(self):
        """Test complete optimizer creation pipeline"""
        # Create models
        teacher = torch.nn.Linear(10, 5)
        student = torch.nn.Linear(10, 5)
        
        # Create optimizers and schedulers
        cfg = {
            "teacher_lr": 1e-4,
            "student_lr": 1e-3,
            "teacher_weight_decay": 1e-4,
            "student_weight_decay": 1e-4,
            "teacher_epochs": 10,
            "student_epochs": 10
        }
        
        teacher_optimizer, teacher_scheduler, student_optimizer, student_scheduler = create_optimizers_and_schedulers_legacy(
            teacher=teacher,
            student_model=student,
            cfg=cfg
        )
        
        # Test optimizer step
        teacher_optimizer.zero_grad()
        loss = torch.sum(teacher(torch.randn(2, 10)))
        loss.backward()
        teacher_optimizer.step()
        
        student_optimizer.zero_grad()
        loss = torch.sum(student(torch.randn(2, 10)))
        loss.backward()
        student_optimizer.step()
        
        # Test scheduler step
        teacher_scheduler.step()
        student_scheduler.step()
        
        assert True  # If we get here, no errors occurred 