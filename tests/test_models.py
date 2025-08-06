#!/usr/bin/env python3
"""Test all teacher and student models"""

import torch
import pytest
from pathlib import Path

# Import all teacher models
from models.teachers.resnet152_teacher import ResNet152Teacher
from models.teachers.convnext_l_teacher import ConvNeXtLTeacher
from models.teachers.convnext_s_teacher import ConvNeXtSTeacher
from models.teachers.efficientnet_l2_teacher import EfficientNetL2Teacher

# Import all student models
from models.students.resnet152_student import ResNet152Student
from models.students.resnet101_student import ResNetStudent as ResNet101Student
from models.students.resnet50_student import ResNet50Student
from models.students.shufflenet_v2_student import ShuffleNetV2Student
from models.students.mobilenet_v2_student import MobileNetV2Student
from models.students.efficientnet_b0_student import EfficientNetB0Student

# Import core builder
from core.builder import create_teacher_by_name, create_student_by_name


class TestTeacherModels:
    """Test all teacher models"""
    
    @pytest.mark.parametrize("teacher_name", [
        "resnet152_teacher",
        "convnext_l_teacher", 
        "convnext_s_teacher",
        "efficientnet_l2_teacher"
    ])
    def test_teacher_creation(self, teacher_name):
        """Test teacher model creation"""
        model = create_teacher_by_name(
            teacher_name=teacher_name,
            num_classes=100,
            pretrained=False,
            small_input=True
        )
        assert model is not None
        assert hasattr(model, 'forward')
        assert hasattr(model, 'extract_feats')
    
    @pytest.mark.parametrize("teacher_name", [
        "resnet152_teacher",
        "convnext_l_teacher", 
        "convnext_s_teacher",
        "efficientnet_l2_teacher"
    ])
    def test_teacher_forward(self, teacher_name):
        """Test teacher forward pass"""
        model = create_teacher_by_name(
            teacher_name=teacher_name,
            num_classes=100,
            pretrained=False,
            small_input=True
        )
        
        # Test forward pass
        x = torch.randn(2, 3, 32, 32)
        model.eval()
        with torch.no_grad():
            output = model(x)
            features_4d, features_2d = model.extract_feats(x)
        
        # Check output shapes
        assert isinstance(output, (torch.Tensor, tuple))
        assert features_4d is not None
        assert features_2d is not None
        assert features_4d.dim() == 4  # (B, C, H, W)
        assert features_2d.dim() == 2  # (B, C)


class TestStudentModels:
    """Test all student models"""
    
    @pytest.mark.parametrize("student_name", [
        "resnet152_pretrain",
        "resnet101_pretrain", 
        "resnet50_scratch",
        "shufflenet_v2_scratch",
        "mobilenet_v2_scratch",
        "efficientnet_b0_scratch"
    ])
    def test_student_creation(self, student_name):
        """Test student model creation"""
        model = create_student_by_name(
            student_name=student_name,
            num_classes=100,
            pretrained=False,
            small_input=True
        )
        assert model is not None
        assert hasattr(model, 'forward')
        assert hasattr(model, 'extract_feats')
    
    @pytest.mark.parametrize("student_name", [
        "resnet152_pretrain",
        "resnet101_pretrain", 
        "resnet50_scratch",
        "shufflenet_v2_scratch",
        "mobilenet_v2_scratch",
        "efficientnet_b0_scratch"
    ])
    def test_student_forward(self, student_name):
        """Test student forward pass"""
        model = create_student_by_name(
            student_name=student_name,
            num_classes=100,
            pretrained=False,
            small_input=True
        )
        
        # Test forward pass
        x = torch.randn(2, 3, 32, 32)
        model.eval()
        with torch.no_grad():
            output = model(x)
            features_4d, features_2d = model.extract_feats(x)
        
        # Check output shapes
        assert isinstance(output, (torch.Tensor, tuple))
        assert features_4d is not None
        assert features_2d is not None
        assert features_4d.dim() == 4  # (B, C, H, W)
        assert features_2d.dim() == 2  # (B, C)


class TestModelRegistry:
    """Test model registry functionality"""
    
    def test_teacher_registry(self):
        """Test teacher models are in registry"""
        from models.common import registry
        registry.ensure_scanned()
        
        teacher_keys = [
            "resnet152_teacher",
            "convnext_l_teacher", 
            "convnext_s_teacher",
            "efficientnet_l2_teacher"
        ]
        
        for key in teacher_keys:
            assert key in registry.MODEL_REGISTRY
    
    def test_student_registry(self):
        """Test student models are in registry"""
        from models.common import registry
        registry.ensure_scanned()
        
        student_keys = [
            "resnet152_pretrain",
            "resnet101_pretrain", 
            "resnet50_scratch",
            "shufflenet_v2_scratch",
            "mobilenet_v2_scratch",
            "efficientnet_b0_scratch"
        ]
        
        for key in student_keys:
            assert key in registry.MODEL_REGISTRY


class TestModelParameters:
    """Test model parameter counts"""
    
    def test_teacher_parameters(self):
        """Test teacher model parameter counts"""
        teachers = [
            ("resnet152_teacher", 60_000_000),  # ~60M
            ("convnext_s_teacher", 50_000_000),  # ~50M
            ("convnext_l_teacher", 200_000_000), # ~200M
            ("efficientnet_l2_teacher", 480_000_000), # ~480M
        ]
        
        for teacher_name, expected_min in teachers:
            model = create_teacher_by_name(
                teacher_name=teacher_name,
                num_classes=100,
                pretrained=False,
                small_input=True
            )
            param_count = sum(p.numel() for p in model.parameters())
            assert param_count > expected_min * 0.8  # Allow some variance
    
    def test_student_parameters(self):
        """Test student model parameter counts"""
        students = [
            ("resnet152_pretrain", 60_000_000),  # ~60M
            ("resnet101_pretrain", 45_000_000),  # ~45M
            ("resnet50_scratch", 25_000_000),    # ~25M
            ("shufflenet_v2_scratch", 4_000_000), # ~4M
            ("mobilenet_v2_scratch", 6_000_000),  # ~6M
            ("efficientnet_b0_scratch", 8_000_000), # ~8M
        ]
        
        for student_name, expected_min in students:
            model = create_student_by_name(
                student_name=student_name,
                num_classes=100,
                pretrained=False,
                small_input=True
            )
            param_count = sum(p.numel() for p in model.parameters())
            assert param_count > expected_min * 0.8  # Allow some variance 