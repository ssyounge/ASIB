import pytest
import torch
from unittest.mock import MagicMock
from core.utils import auto_set_mbm_query_dim, auto_set_mbm_query_dim_with_model

class TestAutoSetMbmQueryDim:
    """Test auto_set_mbm_query_dim function signatures"""
    
    def test_auto_set_mbm_query_dim_signature(self):
        """Test that auto_set_mbm_query_dim takes only cfg"""
        cfg = {"mbm_query_dim": 0}
        
        # This should work
        result = auto_set_mbm_query_dim(cfg=cfg)
        assert result is not None
        
        # This should fail (wrong signature)
        with pytest.raises(TypeError):
            auto_set_mbm_query_dim(student_model=MagicMock(), cfg=cfg)
    
    def test_auto_set_mbm_query_dim_with_model_signature(self):
        """Test that auto_set_mbm_query_dim_with_model takes student_model and cfg"""
        cfg = {"mbm_query_dim": 0, "device": "cuda"}
        student_model = MagicMock()
        
        # Mock the model's forward pass
        mock_feat_dict = {"distill_feat": torch.randn(1, 512)}
        student_model.return_value = (mock_feat_dict, None, None)
        
        # This should work
        auto_set_mbm_query_dim_with_model(student_model=student_model, cfg=cfg)
        
        # This should fail (missing student_model)
        with pytest.raises(TypeError):
            auto_set_mbm_query_dim_with_model(cfg=cfg)
    
    def test_main_py_should_use_correct_function(self):
        """Test that main.py should use the correct function"""
        # Simulate what main.py is trying to do
        cfg = {"mbm_query_dim": 0, "device": "cuda"}
        student_model = MagicMock()
        
        # Mock the model's forward pass
        mock_feat_dict = {"distill_feat": torch.randn(1, 512)}
        student_model.return_value = (mock_feat_dict, None, None)
        
        # This is what main.py is doing (WRONG):
        # auto_set_mbm_query_dim(student_model, cfg)  # This fails
        
        # This is what main.py should do (CORRECT):
        auto_set_mbm_query_dim_with_model(student_model=student_model, cfg=cfg)  # This works
    
    def test_function_imports(self):
        """Test that both functions are properly imported"""
        from core.utils import auto_set_mbm_query_dim, auto_set_mbm_query_dim_with_model
        
        # Both should be callable
        assert callable(auto_set_mbm_query_dim)
        assert callable(auto_set_mbm_query_dim_with_model)
        
        # Test their signatures
        import inspect
        
        # auto_set_mbm_query_dim should take only cfg
        sig1 = inspect.signature(auto_set_mbm_query_dim)
        assert len(sig1.parameters) == 1
        assert 'cfg' in sig1.parameters
        
        # auto_set_mbm_query_dim_with_model should take student_model and cfg
        sig2 = inspect.signature(auto_set_mbm_query_dim_with_model)
        assert len(sig2.parameters) == 2
        assert 'student_model' in sig2.parameters
        assert 'cfg' in sig2.parameters 