#!/usr/bin/env python3
"""Test all modules"""

import torch
import pytest
import torch.nn.functional as F

# Import all modules
from modules.losses import (
    kl_loss, mse_loss, ib_loss, ce_loss, 
    contrastive_loss, attention_loss, factor_transfer_loss
)
from modules.loss_safe import safe_kl_loss, safe_mse_loss
from modules.disagreement import compute_disagreement_rate
from modules.partial_freeze import (
    freeze_all, unfreeze_by_regex, apply_bn_ln_policy,
    get_freeze_schedule, apply_freeze_schedule
)
from modules.trainer_student import StudentTrainer
from modules.trainer_teacher import TeacherTrainer
from modules.cutmix_finetune_teacher import CutMixFinetuneTeacher


class TestLossFunctions:
    """Test all loss functions"""
    
    def test_kl_loss(self):
        """Test KL divergence loss"""
        student_logits = torch.randn(4, 100)
        teacher_logits = torch.randn(4, 100)
        temperature = 4.0
        
        loss = kl_loss(student_logits, teacher_logits, temperature)
        assert torch.isfinite(loss)
        assert loss > 0
    
    def test_mse_loss(self):
        """Test MSE loss"""
        student_feat = torch.randn(4, 128)
        teacher_feat = torch.randn(4, 128)
        
        loss = mse_loss(student_feat, teacher_feat)
        assert torch.isfinite(loss)
        assert loss > 0
    
    def test_ib_loss(self):
        """Test Information Bottleneck loss"""
        student_feat = torch.randn(4, 128)
        teacher_feat = torch.randn(4, 128)
        beta = 0.1
        
        loss = ib_loss(student_feat, teacher_feat, beta)
        assert torch.isfinite(loss)
        assert loss > 0
    
    def test_ce_loss(self):
        """Test Cross Entropy loss"""
        logits = torch.randn(4, 100)
        targets = torch.randint(0, 100, (4,))
        
        loss = ce_loss(logits, targets)
        assert torch.isfinite(loss)
        assert loss > 0
    
    def test_contrastive_loss(self):
        """Test Contrastive loss"""
        student_feat = torch.randn(4, 128)
        teacher_feat = torch.randn(4, 128)
        temperature = 0.1
        
        loss = contrastive_loss(student_feat, teacher_feat, temperature)
        assert torch.isfinite(loss)
        assert loss > 0
    
    def test_attention_loss(self):
        """Test Attention loss"""
        student_attn = torch.randn(4, 64, 64)
        teacher_attn = torch.randn(4, 64, 64)
        
        loss = attention_loss(student_attn, teacher_attn)
        assert torch.isfinite(loss)
        assert loss > 0
    
    def test_factor_transfer_loss(self):
        """Test Factor Transfer loss"""
        student_factor = torch.randn(4, 128)
        teacher_factor = torch.randn(4, 128)
        
        loss = factor_transfer_loss(student_factor, teacher_factor)
        assert torch.isfinite(loss)
        assert loss > 0


class TestSafeLossFunctions:
    """Test safe loss functions"""
    
    def test_safe_kl_loss(self):
        """Test safe KL loss"""
        student_logits = torch.randn(4, 100)
        teacher_logits = torch.randn(4, 100)
        temperature = 4.0
        
        loss = safe_kl_loss(student_logits, teacher_logits, temperature)
        assert torch.isfinite(loss)
        assert loss > 0
    
    def test_safe_mse_loss(self):
        """Test safe MSE loss"""
        student_feat = torch.randn(4, 128)
        teacher_feat = torch.randn(4, 128)
        
        loss = safe_mse_loss(student_feat, teacher_feat)
        assert torch.isfinite(loss)
        assert loss > 0


class TestDisagreement:
    """Test disagreement computation"""
    
    def test_compute_disagreement_rate(self):
        """Test disagreement rate computation"""
        # Create dummy models
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier = torch.nn.Linear(128, 100)
            
            def forward(self, x):
                return {"logit": self.classifier(x)}
        
        model1 = DummyModel()
        model2 = DummyModel()
        
        # Create dummy data loader
        data_loader = [
            (torch.randn(2, 128), torch.randint(0, 100, (2,)))
            for _ in range(5)
        ]
        
        # Test disagreement computation
        rate = compute_disagreement_rate(model1, model2, data_loader, device="cpu")
        assert isinstance(rate, float)
        assert 0 <= rate <= 100


class TestPartialFreeze:
    """Test partial freeze functionality"""
    
    def test_freeze_all(self):
        """Test freeze all parameters"""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        )
        
        # Check initial state
        initial_params = [p.requires_grad for p in model.parameters()]
        assert any(initial_params)
        
        # Freeze all
        freeze_all(model)
        
        # Check frozen state
        frozen_params = [p.requires_grad for p in model.parameters()]
        assert not any(frozen_params)
    
    def test_unfreeze_by_regex(self):
        """Test unfreeze by regex pattern"""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        )
        
        # Freeze all first
        freeze_all(model)
        
        # Unfreeze by regex - use the actual parameter name pattern
        unfreeze_by_regex(model, r"2\.weight")
        
        # Check specific layer is unfrozen
        params = list(model.parameters())
        # params[0]: 0.weight, params[1]: 0.bias, params[2]: 2.weight, params[3]: 2.bias
        assert not params[0].requires_grad  # 0.weight (frozen)
        assert params[2].requires_grad      # 2.weight (unfrozen, matches "2.weight" pattern)
    
    def test_apply_bn_ln_policy(self):
        """Test BN/LN policy application"""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.BatchNorm1d(5),
            torch.nn.Linear(5, 1)
        )
        
        # Apply policy
        apply_bn_ln_policy(model, train_bn=False, train_ln=True)
        
        # Check BN parameters are frozen
        bn_params = list(model[1].parameters())
        assert not bn_params[0].requires_grad  # weight
        assert not bn_params[1].requires_grad  # bias
    
    def test_get_freeze_schedule(self):
        """Test freeze schedule generation"""
        schedule = get_freeze_schedule("resnet", freeze_level=1)
        
        assert isinstance(schedule, dict)
        assert 'freeze_all' in schedule
        assert 'patterns' in schedule
        assert 'freeze_bn' in schedule
        assert 'freeze_ln' in schedule
    
    def test_apply_freeze_schedule(self):
        """Test freeze schedule application"""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        )
        
        schedule = get_freeze_schedule("resnet", freeze_level=1)
        
        # Apply schedule
        apply_freeze_schedule(model, schedule)
        
        # Check that some parameters are frozen
        params = list(model.parameters())
        # At least some parameters should be frozen
        assert any(not p.requires_grad for p in params)


class TestStudentTrainer:
    """Test student trainer"""
    
    def test_student_trainer_creation(self):
        """Test student trainer creation"""
        # Create dummy models
        class DummyStudent(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier = torch.nn.Linear(128, 100)
            
            def forward(self, x):
                return {"logit": self.classifier(x)}
        
        class DummyTeacher(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier = torch.nn.Linear(128, 100)
            
            def forward(self, x):
                return {"logit": self.classifier(x)}
        
        student = DummyStudent()
        teacher = DummyTeacher()
        
        # Create trainer
        trainer = StudentTrainer(
            student_model=student,
            teacher_models=[teacher],
            device="cpu"
        )
        
        assert trainer is not None
        assert hasattr(trainer, 'train_step')
    
    def test_student_trainer_step(self):
        """Test student trainer step"""
        # Create dummy models
        class DummyStudent(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier = torch.nn.Linear(128, 100)
            
            def forward(self, x):
                return {"logit": self.classifier(x)}
        
        class DummyTeacher(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier = torch.nn.Linear(128, 100)
            
            def forward(self, x):
                return {"logit": self.classifier(x)}
        
        student = DummyStudent()
        teacher = DummyTeacher()
        
        # Create trainer
        trainer = StudentTrainer(
            student_model=student,
            teacher_models=[teacher],
            device="cpu"
        )
        
        # Test training step
        x = torch.randn(4, 128)
        y = torch.randint(0, 100, (4,))
        batch = (x, y)
        optimizer = torch.optim.Adam(student.parameters())
        
        metrics = trainer.train_step(batch, optimizer)
        assert torch.isfinite(torch.tensor(metrics['total_loss']))
        assert metrics['total_loss'] > 0


class TestTeacherTrainer:
    """Test teacher trainer"""
    
    def test_teacher_trainer_creation(self):
        """Test teacher trainer creation"""
        # Create dummy model
        class DummyTeacher(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier = torch.nn.Linear(128, 100)
            
            def forward(self, x):
                return {"logit": self.classifier(x)}
        
        teacher = DummyTeacher()
        
        # Create trainer
        trainer = TeacherTrainer(
            teacher_models=[teacher],
            device="cpu"
        )
        
        assert trainer is not None
        assert hasattr(trainer, 'train_step')
    
    def test_teacher_trainer_step(self):
        """Test teacher trainer step"""
        # Create dummy model
        class DummyTeacher(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier = torch.nn.Linear(128, 100)
            
            def forward(self, x):
                return {"logit": self.classifier(x)}
        
        teacher = DummyTeacher()
        
        # Create trainer
        trainer = TeacherTrainer(
            teacher_models=[teacher],
            device="cpu"
        )
        
        # Test training step
        x = torch.randn(4, 128)
        y = torch.randint(0, 100, (4,))
        batch = (x, y)
        optimizer = torch.optim.Adam(teacher.parameters())
        
        metrics = trainer.train_step(batch, optimizer)
        assert torch.isfinite(torch.tensor(metrics['total_loss']))
        assert metrics['total_loss'] > 0


class TestCutMixFinetuneTeacher:
    """Test CutMix finetune teacher"""
    
    def test_cutmix_finetune_teacher_creation(self):
        """Test CutMix finetune teacher creation"""
        # Create dummy model
        class DummyTeacher(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier = torch.nn.Linear(128, 100)
            
            def forward(self, x):
                return {"logit": self.classifier(x)}
        
        teacher = DummyTeacher()
        
        # Create trainer
        trainer = CutMixFinetuneTeacher(
            teacher_model=teacher,
            device="cpu"
        )
        
        assert trainer is not None
        assert hasattr(trainer, 'train_step')
    
    def test_cutmix_finetune_teacher_step(self):
        """Test CutMix finetune teacher step"""
        # Create dummy model
        class DummyTeacher(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier = torch.nn.Linear(3072, 100)  # 3*32*32 = 3072
            
            def forward(self, x):
                # Flatten 4D input to 2D
                x_flat = x.view(x.size(0), -1)
                return {"logit": self.classifier(x_flat)}
        
        teacher = DummyTeacher()
        
        # Create trainer
        trainer = CutMixFinetuneTeacher(
            teacher_model=teacher,
            device="cpu"
        )
        
        # Test training step with 4D tensor (NCHW format)
        x = torch.randn(4, 3, 32, 32)  # 4D tensor for CutMix
        y = torch.randint(0, 100, (4,))
        batch = (x, y)
        optimizer = torch.optim.Adam(teacher.parameters())
        
        metrics = trainer.train_step(batch, optimizer)
        assert torch.isfinite(torch.tensor(metrics['loss']))
        assert metrics['loss'] > 0 