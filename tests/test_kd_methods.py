#!/usr/bin/env python3
"""Test all knowledge distillation methods"""

import torch
import pytest
import torch.nn.functional as F

# Import all KD methods
from methods.asib import ASIBDistiller
from methods.vanilla_kd import VanillaKDDistiller
from methods.dkd import DKDDistiller
from methods.crd import CRDDistiller
from methods.fitnet import FitNetDistiller
from methods.at import ATDistiller
from methods.simkd import SimKDDistiller
from methods.reviewkd import ReviewKDDistiller
from methods.sskd import SSKDDistiller
from methods.ab import ABDistiller
from methods.ft import FTDistiller

# Import MBM components
from models.mbm import IB_MBM, SynergyHead


class TestKDDistillers:
    """Test all KD distiller classes"""
    
    @pytest.fixture
    def dummy_models(self):
        """Create dummy teacher and student models for testing"""
        
        class DummyTeacher(torch.nn.Module):
            def __init__(self, feat_dim=128, num_classes=100):
                super().__init__()
                self.proj = torch.nn.Linear(3, feat_dim)
                self.classifier = torch.nn.Linear(feat_dim, num_classes)
                self._feat_dim = feat_dim
                
            def forward(self, x):
                feat = self.proj(x)
                logit = self.classifier(feat)
                return {
                    "logit": logit,
                    "feat_2d": feat,
                    "feat_4d_layer3": feat.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 8, 8)
                }
                
            def extract_feats(self, x):
                return self.proj(x)
                
            def get_feat_dim(self):
                return self._feat_dim

        class DummyStudent(torch.nn.Module):
            def __init__(self, feat_dim=128, num_classes=100):
                super().__init__()
                self.proj = torch.nn.Linear(3, feat_dim)
                self.classifier = torch.nn.Linear(feat_dim, num_classes)
                self._feat_dim = feat_dim
                
            def forward(self, x):
                feat = self.proj(x)
                logit = self.classifier(feat)
                return {
                    "feat_2d": feat,
                    "feat_4d_layer3": feat.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 8, 8)
                }, logit, None
                
            def extract_feats(self, x):
                return self.proj(x)
                
            def get_feat_dim(self):
                return self._feat_dim

        teacher1 = DummyTeacher()
        teacher2 = DummyTeacher()
        student = DummyStudent()
        
        return teacher1, teacher2, student
    
    def test_asib_distiller(self, dummy_models):
        """Test ASIB distiller"""
        teacher1, teacher2, student = dummy_models
    
        # kv_dim should be 2 * feat_dim since we stack two teacher features
        mbm = IB_MBM(q_dim=128, kv_dim=256, d_emb=128)
        synergy_head = SynergyHead(128, num_classes=100)
    
        distiller = ASIBDistiller(
            teacher1, teacher2, student, mbm, synergy_head, device="cpu"
        )
    
        x = torch.randn(4, 3)
        y = torch.randint(0, 100, (4,))
    
        loss, _ = distiller.forward(x, y)
        assert torch.isfinite(loss)
        assert loss > 0
    

    
    def test_vanilla_kd_distiller(self, dummy_models):
        """Test Vanilla KD distiller"""
        teacher1, teacher2, student = dummy_models
        
        distiller = VanillaKDDistiller(
            teacher1, student
        )
        
        x = torch.randn(4, 3)
        y = torch.randint(0, 100, (4,))
        
        loss, _ = distiller.forward(x, y)
        assert torch.isfinite(loss)
        assert loss > 0
    
    def test_dkd_distiller(self, dummy_models):
        """Test DKD distiller"""
        teacher1, teacher2, student = dummy_models
        
        distiller = DKDDistiller(
            teacher1, student, temperature=4.0
        )
        
        x = torch.randn(4, 3)
        y = torch.randint(0, 100, (4,))
        
        loss, _ = distiller.forward(x, y)
        assert torch.isfinite(loss)
        assert loss > 0
    
    def test_crd_distiller(self, dummy_models):
        """Test CRD distiller"""
        teacher1, teacher2, student = dummy_models
        
        distiller = CRDDistiller(
            teacher1, student
        )
        
        x = torch.randn(4, 3)
        y = torch.randint(0, 100, (4,))
        
        loss, _ = distiller.forward(x, y)
        assert torch.isfinite(loss)
        assert loss > 0
    
    def test_fitnet_distiller(self, dummy_models):
        """Test FitNet distiller"""
        teacher1, teacher2, student = dummy_models
        
        distiller = FitNetDistiller(
            teacher1, student, 
            hint_key="feat_4d_layer3",
            guided_key="feat_4d_layer3"
        )
        
        x = torch.randn(4, 3)
        y = torch.randint(0, 100, (4,))
        
        loss, _ = distiller.forward(x, y)
        assert torch.isfinite(loss)
        assert loss > 0
    
    def test_at_distiller(self, dummy_models):
        """Test AT distiller"""
        teacher1, teacher2, student = dummy_models
        
        distiller = ATDistiller(
            teacher1, student
        )
        
        x = torch.randn(4, 3)
        y = torch.randint(0, 100, (4,))
        
        loss, _ = distiller.forward(x, y)
        assert torch.isfinite(loss)
        assert loss > 0
    
    def test_simkd_distiller(self, dummy_models):
        """Test SimKD distiller"""
        teacher1, teacher2, student = dummy_models
        
        distiller = SimKDDistiller(
            teacher1, student
        )
        
        x = torch.randn(4, 3)
        y = torch.randint(0, 100, (4,))
        
        loss, _ = distiller.forward(x, y)
        assert torch.isfinite(loss)
        assert loss > 0
    
    def test_reviewkd_distiller(self, dummy_models):
        """Test ReviewKD distiller"""
        teacher1, teacher2, student = dummy_models
        
        distiller = ReviewKDDistiller(
            teacher1, student
        )
        
        x = torch.randn(4, 3)
        y = torch.randint(0, 100, (4,))
        
        loss, _ = distiller.forward(x, y)
        assert torch.isfinite(loss)
        assert loss > 0
    
    def test_sskd_distiller(self, dummy_models):
        """Test SSKD distiller"""
        teacher1, teacher2, student = dummy_models
        
        distiller = SSKDDistiller(
            teacher1, student
        )
        
        x = torch.randn(4, 3)
        y = torch.randint(0, 100, (4,))
        
        loss, _ = distiller.forward(x, y)
        assert torch.isfinite(loss)
        assert loss > 0
    
    def test_ab_distiller(self, dummy_models):
        """Test AB distiller"""
        teacher1, teacher2, student = dummy_models
        
        distiller = ABDistiller(
            teacher1, student
        )
        
        x = torch.randn(4, 3)
        y = torch.randint(0, 100, (4,))
        
        loss, _ = distiller.forward(x, y)
        assert torch.isfinite(loss)
        assert loss > 0
    
    def test_ft_distiller(self, dummy_models):
        """Test FT distiller"""
        teacher1, teacher2, student = dummy_models
        
        distiller = FTDistiller(
            teacher1, student
        )
        
        x = torch.randn(4, 3)
        y = torch.randint(0, 100, (4,))
        
        loss, _ = distiller.forward(x, y)
        assert torch.isfinite(loss)
        assert loss > 0


class TestMBMComponents:
    """Test MBM components"""
    
    def test_ib_mbm(self):
        """Test IB_MBM module"""
        mbm = IB_MBM(q_dim=128, kv_dim=128, d_emb=128)
    
        # Test forward pass
        q = torch.randn(4, 128)  # (batch_size, q_dim)
        kv = torch.randn(4, 128)  # (batch_size, kv_dim)
    
        output, mu, logvar = mbm(q, kv)
        assert output.shape == (4, 128)
        assert mu.shape == (4, 128)
        assert logvar.shape == (4, 128)
        assert torch.isfinite(output).all()
        assert torch.isfinite(mu).all()
        assert torch.isfinite(logvar).all()
    
    def test_synergy_head(self):
        """Test SynergyHead module"""
        head = SynergyHead(128, num_classes=100)
        
        # Test forward pass
        x = torch.randn(4, 128)
        
        output = head(x)
        assert output.shape == (4, 100)
        assert torch.isfinite(output).all()


class TestMethodRegistry:
    """Test method registry functionality"""
    
    def test_method_registry(self):
        """Test all methods are in registry"""
        from methods import __init__ as methods_init
    
        # This will trigger registry scanning
        methods_init
    
        # Check if methods are properly imported
        method_classes = [
            ASIBDistiller,
            VanillaKDDistiller,
            DKDDistiller,
            CRDDistiller,
            FitNetDistiller,
            ATDistiller,
            SimKDDistiller,
            ReviewKDDistiller,
            SSKDDistiller,
            ABDistiller,
            FTDistiller
        ]
        
        # Verify all classes exist
        for cls in method_classes:
            assert cls is not None
            assert hasattr(cls, '__init__')
            assert hasattr(cls, 'forward') 