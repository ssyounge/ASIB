import pytest
import torch
import torch.nn as nn
from models import IB_MBM, SynergyHead, build_ib_mbm_from_teachers as build_from_teachers

class TestMBMTensorShapes:
    """Test MBM tensor shape issues"""
    
    def test_mbm_forward_tensor_shapes(self):
        """Test that MBM forward handles tensor shapes correctly"""
        # Create MBM with typical dimensions
        mbm = IB_MBM(
            q_dim=512,
            kv_dim=512,
            d_emb=512,
            beta=1e-2,
            n_head=8,
        )
        
        # Test with correct shapes
        batch_size = 4
        q_feat = torch.randn(batch_size, 512)  # (batch_size, q_dim)
        kv_feats = torch.randn(batch_size, 512)  # (batch_size, kv_dim)
        
        # This should work
        z, mu, logvar = mbm(q_feat, kv_feats)
        
        # Check output shapes
        assert z.shape == (batch_size, 512)
        assert mu.shape == (batch_size, 512)
        assert logvar.shape == (batch_size, 512)
    
    def test_mbm_forward_with_3d_tensor_from_stack(self):
        """Test the specific issue with 3D tensors from torch.stack"""
        mbm = IB_MBM(
            q_dim=2048,  # student feature dimension
            kv_dim=2048,  # teacher feature dimension
            d_emb=512,
            beta=1e-2,
            n_head=8,
        )
        
        batch_size = 4
        q_feat = torch.randn(batch_size, 2048)  # (batch_size, q_dim) - student feature
        
        # This is the actual case from torch.stack([f1_2d, f2_2d], dim=1)
        # f1_2d and f2_2d are both (batch_size, 2048), so stack gives (batch_size, 2, 2048)
        kv_feats_3d = torch.randn(batch_size, 2, 2048)  # (batch_size, num_teachers, kv_dim)
        
        # This should now work with the fixed implementation
        z, mu, logvar = mbm(q_feat, kv_feats_3d)
        
        # Check output shapes
        assert z.shape == (batch_size, 512)
        assert mu.shape == (batch_size, 512)
        assert logvar.shape == (batch_size, 512)
    
    def test_mbm_forward_with_3d_tensor(self):
        """Test with 3D tensor (sequence of features)"""
        mbm = IB_MBM(
            q_dim=512,
            kv_dim=512,
            d_emb=512,
            beta=1e-2,
            n_head=8,
        )
        
        batch_size = 4
        seq_len = 3
        q_feat = torch.randn(batch_size, 512)  # (batch_size, q_dim)
        kv_feats_3d = torch.randn(batch_size, seq_len, 512)  # (batch_size, seq_len, kv_dim)
        
        # This should work
        z, mu, logvar = mbm(q_feat, kv_feats_3d)
        
        # Check output shapes
        assert z.shape == (batch_size, 512)
        assert mu.shape == (batch_size, 512)
        assert logvar.shape == (batch_size, 512)
    
    def test_synergy_head_forward(self):
        """Test SynergyHead forward pass"""
        head = SynergyHead(
            in_dim=512,
            num_classes=100,
            p=0.1
        )
        
        batch_size = 4
        x = torch.randn(batch_size, 512)
        
        # This should work
        logits = head(x)
        
        # Check output shape
        assert logits.shape == (batch_size, 100)
    
    def test_build_from_teachers(self):
        """Test build_from_teachers function"""
        # Mock teachers
        class MockTeacher(nn.Module):
            def __init__(self, feat_dim=512):
                super().__init__()
                self.feat_dim = feat_dim
            
            def get_feat_dim(self):
                return self.feat_dim
        
        teachers = [MockTeacher(512), MockTeacher(512)]
        cfg = {
            "ib_mbm_query_dim": 512,
            "ib_mbm_out_dim": 512,
            "ib_beta": 1e-2,
            "ib_mbm_n_head": 8,
            "num_classes": 100,
            "synergy_head_dropout": 0.1
        }
        
        # This should work
        mbm, head = build_from_teachers(teachers, cfg)
        
        # Test forward pass
        batch_size = 4
        q_feat = torch.randn(batch_size, 512)
        kv_feats = torch.randn(batch_size, 512)
        
        z, mu, logvar = mbm(q_feat, kv_feats)
        logits = head(z)
        
        # Check shapes
        assert z.shape == (batch_size, 512)
        assert logits.shape == (batch_size, 100) 