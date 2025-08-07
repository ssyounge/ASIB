#!/usr/bin/env python3
"""Test modules.partial_freeze functions used in main.py"""

import torch
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from modules.partial_freeze import (
    apply_partial_freeze,
    partial_freeze_teacher_resnet,
    partial_freeze_teacher_efficientnet,
    partial_freeze_student_resnet,
)


class TestApplyPartialFreeze:
    """Test apply_partial_freeze function"""
    
    def test_apply_partial_freeze_basic(self):
        """Test basic apply_partial_freeze functionality"""
        # Create a simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 5)
        )
        
        # Apply partial freeze
        apply_partial_freeze(model, freeze_ratio=0.5)
        
        # Check that some parameters are frozen
        frozen_count = sum(1 for p in model.parameters() if not p.requires_grad)
        total_count = sum(1 for p in model.parameters())
        
        assert frozen_count > 0
        assert frozen_count <= total_count
    
    def test_apply_partial_freeze_zero_ratio(self):
        """Test with zero freeze ratio"""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 5)
        )
        
        apply_partial_freeze(model, freeze_ratio=0.0)
        
        # All parameters should be trainable
        frozen_count = sum(1 for p in model.parameters() if not p.requires_grad)
        assert frozen_count == 0
    
    def test_apply_partial_freeze_full_ratio(self):
        """Test with full freeze ratio"""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 5)
        )
        
        apply_partial_freeze(model, freeze_ratio=1.0)
        
        # All parameters should be frozen
        frozen_count = sum(1 for p in model.parameters() if not p.requires_grad)
        total_count = sum(1 for p in model.parameters())
        assert frozen_count == total_count
    
    def test_apply_partial_freeze_with_regex(self):
        """Test with regex pattern"""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20, name="layer1"),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 5, name="layer2")
        )
        
        # Freeze only layers with "layer1" in name
        apply_partial_freeze(model, freeze_ratio=0.5, freeze_regex=".*layer1.*")
        
        # Check that specific layers are frozen
        layer1_params = [p for name, p in model.named_parameters() if "layer1" in name]
        layer2_params = [p for name, p in model.named_parameters() if "layer2" in name]
        
        # All layer1 parameters should be frozen
        assert all(not p.requires_grad for p in layer1_params)
        # Some layer2 parameters might be frozen due to ratio
        frozen_layer2 = sum(1 for p in layer2_params if not p.requires_grad)
        assert frozen_layer2 >= 0


class TestPartialFreezeTeacherResnet:
    """Test partial_freeze_teacher_resnet function"""
    
    def test_partial_freeze_teacher_resnet_basic(self):
        """Test basic partial_freeze_teacher_resnet functionality"""
        # Create a mock ResNet-like model
        class MockResNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 64, 3, padding=1)
                self.layer1 = torch.nn.Sequential(
                    torch.nn.Conv2d(64, 64, 3, padding=1),
                    torch.nn.ReLU()
                )
                self.layer2 = torch.nn.Sequential(
                    torch.nn.Conv2d(64, 128, 3, padding=1),
                    torch.nn.ReLU()
                )
                self.fc = torch.nn.Linear(128, 100)
            
            def forward(self, x):
                x = self.conv1(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = x.mean([2, 3])  # Global average pooling
                x = self.fc(x)
                return x
        
        model = MockResNet()
        
        # Apply partial freeze
        partial_freeze_teacher_resnet(model, freeze_ratio=0.5)
        
        # Check that some parameters are frozen
        frozen_count = sum(1 for p in model.parameters() if not p.requires_grad)
        total_count = sum(1 for p in model.parameters())
        
        assert frozen_count > 0
        assert frozen_count <= total_count
    
    def test_partial_freeze_teacher_resnet_zero_ratio(self):
        """Test with zero freeze ratio"""
        class MockResNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 64, 3, padding=1)
                self.fc = torch.nn.Linear(64, 100)
            
            def forward(self, x):
                x = self.conv1(x)
                x = x.mean([2, 3])
                x = self.fc(x)
                return x
        
        model = MockResNet()
        
        partial_freeze_teacher_resnet(model, freeze_ratio=0.0)
        
        # All parameters should be trainable
        frozen_count = sum(1 for p in model.parameters() if not p.requires_grad)
        assert frozen_count == 0


class TestPartialFreezeTeacherEfficientnet:
    """Test partial_freeze_teacher_efficientnet function"""
    
    def test_partial_freeze_teacher_efficientnet_basic(self):
        """Test basic partial_freeze_teacher_efficientnet functionality"""
        # Create a mock EfficientNet-like model
        class MockEfficientNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_stem = torch.nn.Conv2d(3, 32, 3, padding=1)
                self.blocks = torch.nn.Sequential(
                    torch.nn.Conv2d(32, 64, 3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(64, 128, 3, padding=1),
                    torch.nn.ReLU()
                )
                self.conv_head = torch.nn.Conv2d(128, 256, 1)
                self.classifier = torch.nn.Linear(256, 100)
            
            def forward(self, x):
                x = self.conv_stem(x)
                x = self.blocks(x)
                x = self.conv_head(x)
                x = x.mean([2, 3])  # Global average pooling
                x = self.classifier(x)
                return x
        
        model = MockEfficientNet()
        
        # Apply partial freeze
        partial_freeze_teacher_efficientnet(model, freeze_ratio=0.5)
        
        # Check that some parameters are frozen
        frozen_count = sum(1 for p in model.parameters() if not p.requires_grad)
        total_count = sum(1 for p in model.parameters())
        
        assert frozen_count > 0
        assert frozen_count <= total_count
    
    def test_partial_freeze_teacher_efficientnet_full_ratio(self):
        """Test with full freeze ratio"""
        class MockEfficientNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_stem = torch.nn.Conv2d(3, 32, 3, padding=1)
                self.blocks = torch.nn.Sequential(
                    torch.nn.Conv2d(32, 64, 3, padding=1),
                    torch.nn.ReLU()
                )
                self.classifier = torch.nn.Linear(64, 100)
            
            def forward(self, x):
                x = self.conv_stem(x)
                x = self.blocks(x)
                x = x.mean([2, 3])
                x = self.classifier(x)
                return x
        
        model = MockEfficientNet()
        
        partial_freeze_teacher_efficientnet(model, freeze_ratio=1.0)
        
        # All parameters should be frozen
        frozen_count = sum(1 for p in model.parameters() if not p.requires_grad)
        total_count = sum(1 for p in model.parameters())
        assert frozen_count == total_count


class TestPartialFreezeStudentResnet:
    """Test partial_freeze_student_resnet function"""
    
    def test_partial_freeze_student_resnet_basic(self):
        """Test basic partial_freeze_student_resnet functionality"""
        # Create a mock ResNet-like student model
        class MockStudentResNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 64, 3, padding=1)
                self.layer1 = torch.nn.Sequential(
                    torch.nn.Conv2d(64, 64, 3, padding=1),
                    torch.nn.ReLU()
                )
                self.layer2 = torch.nn.Sequential(
                    torch.nn.Conv2d(64, 128, 3, padding=1),
                    torch.nn.ReLU()
                )
                self.fc = torch.nn.Linear(128, 100)
            
            def forward(self, x):
                x = self.conv1(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = x.mean([2, 3])  # Global average pooling
                x = self.fc(x)
                return x
        
        model = MockStudentResNet()
        
        # Apply partial freeze
        partial_freeze_student_resnet(model, freeze_ratio=0.5)
        
        # Check that some parameters are frozen
        frozen_count = sum(1 for p in model.parameters() if not p.requires_grad)
        total_count = sum(1 for p in model.parameters())
        
        assert frozen_count > 0
        assert frozen_count <= total_count
    
    def test_partial_freeze_student_resnet_with_adapter(self):
        """Test with adapter layers"""
        class MockStudentResNetWithAdapter(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 64, 3, padding=1)
                self.adapter1 = torch.nn.Conv2d(64, 64, 1)  # Adapter layer
                self.layer1 = torch.nn.Sequential(
                    torch.nn.Conv2d(64, 64, 3, padding=1),
                    torch.nn.ReLU()
                )
                self.adapter2 = torch.nn.Conv2d(64, 64, 1)  # Adapter layer
                self.fc = torch.nn.Linear(64, 100)
            
            def forward(self, x):
                x = self.conv1(x)
                x = self.adapter1(x)
                x = self.layer1(x)
                x = self.adapter2(x)
                x = x.mean([2, 3])
                x = self.fc(x)
                return x
        
        model = MockStudentResNetWithAdapter()
        
        # Apply partial freeze
        partial_freeze_student_resnet(model, freeze_ratio=0.5)
        
        # Check that some parameters are frozen
        frozen_count = sum(1 for p in model.parameters() if not p.requires_grad)
        total_count = sum(1 for p in model.parameters())
        
        assert frozen_count > 0
        assert frozen_count <= total_count


class TestIntegration:
    """Test integration scenarios"""
    
    def test_main_py_partial_freeze_flow(self):
        """Test the flow of partial freeze functions as used in main.py"""
        # Simulate main.py partial freeze setup
        class MockTeacher(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 64, 3, padding=1)
                self.layer1 = torch.nn.Sequential(
                    torch.nn.Conv2d(64, 64, 3, padding=1),
                    torch.nn.ReLU()
                )
                self.fc = torch.nn.Linear(64, 100)
            
            def forward(self, x):
                x = self.conv1(x)
                x = self.layer1(x)
                x = x.mean([2, 3])
                x = self.fc(x)
                return x
        
        class MockStudent(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 32, 3, padding=1)
                self.fc = torch.nn.Linear(32, 100)
            
            def forward(self, x):
                x = self.conv1(x)
                x = x.mean([2, 3])
                x = self.fc(x)
                return x
        
        # Create models
        teacher = MockTeacher()
        student = MockStudent()
        
        # Apply partial freeze to teacher
        partial_freeze_teacher_resnet(teacher, freeze_ratio=0.3)
        
        # Apply partial freeze to student
        partial_freeze_student_resnet(student, freeze_ratio=0.2)
        
        # Verify that parameters are frozen
        teacher_frozen = sum(1 for p in teacher.parameters() if not p.requires_grad)
        student_frozen = sum(1 for p in student.parameters() if not p.requires_grad)
        
        assert teacher_frozen > 0
        assert student_frozen > 0
        
        # Test forward pass still works
        x = torch.randn(2, 3, 32, 32)
        
        teacher.eval()
        student.eval()
        
        with torch.no_grad():
            teacher_output = teacher(x)
            student_output = student(x)
        
        assert teacher_output.shape == (2, 100)
        assert student_output.shape == (2, 100)
    
    def test_partial_freeze_edge_cases(self):
        """Test edge cases and error handling"""
        # Test with empty model
        class EmptyModel(torch.nn.Module):
            def forward(self, x):
                return x
        
        empty_model = EmptyModel()
        
        # Should handle gracefully
        apply_partial_freeze(empty_model, freeze_ratio=0.5)
        
        # Test with very small model
        small_model = torch.nn.Linear(1, 1)
        apply_partial_freeze(small_model, freeze_ratio=0.5)
        
        # Test with invalid freeze ratio
        model = torch.nn.Linear(10, 5)
        
        # Should handle negative ratio
        apply_partial_freeze(model, freeze_ratio=-0.1)
        
        # Should handle ratio > 1
        apply_partial_freeze(model, freeze_ratio=1.5)
