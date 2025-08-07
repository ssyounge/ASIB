import pytest
from core.utils import _renorm_ce_kd, renorm_ce_kd

class TestRenormCeKd:
    """Test renorm_ce_kd function"""
    
    def test_renorm_ce_kd_config_function(self):
        """Test that renorm_ce_kd takes only cfg"""
        cfg = {"ce_alpha": 0.3, "kd_alpha": 0.7}
        
        # This should work
        renorm_ce_kd(cfg)
        
        # Check that values were normalized
        assert abs(cfg["ce_alpha"] + cfg["kd_alpha"] - 1.0) < 1e-5
    
    def test_renorm_ce_kd_with_losses(self):
        """Test that _renorm_ce_kd takes losses and alphas"""
        ce_loss = 1.0
        kd_loss = 2.0
        ce_alpha = 0.3
        kd_alpha = 0.7
        
        # This should work
        result = _renorm_ce_kd(ce_loss, kd_loss, ce_alpha, kd_alpha)
        assert result is not None
        
        # This should fail (wrong signature)
        with pytest.raises(TypeError):
            _renorm_ce_kd(cfg={})
    
    def test_main_py_should_use_correct_function(self):
        """Test that main.py should use the correct function"""
        # Simulate what main.py is trying to do
        cfg = {"ce_alpha": 0.3, "kd_alpha": 0.7}
        
        # This is what main.py is doing (WRONG):
        # _renorm_ce_kd(cfg)  # This fails
        
        # This is what main.py should do (CORRECT):
        renorm_ce_kd(cfg)  # This works
        
        # Check that values were normalized
        assert abs(cfg["ce_alpha"] + cfg["kd_alpha"] - 1.0) < 1e-5
    
    def test_function_imports(self):
        """Test that both functions are properly imported"""
        from core.utils import _renorm_ce_kd, renorm_ce_kd
        
        # Both should be callable
        assert callable(_renorm_ce_kd)
        assert callable(renorm_ce_kd)
        
        # Test their signatures
        import inspect
        
        # _renorm_ce_kd should take 4 arguments
        sig1 = inspect.signature(_renorm_ce_kd)
        assert len(sig1.parameters) == 4
        
        # renorm_ce_kd should take 1 argument
        sig2 = inspect.signature(renorm_ce_kd)
        assert len(sig2.parameters) == 1
        assert 'cfg' in sig2.parameters 