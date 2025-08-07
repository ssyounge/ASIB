import pytest
from core.utils import setup_partial_freeze_schedule, setup_partial_freeze_schedule_with_cfg

class TestSetupPartialFreezeSchedule:
    """Test setup_partial_freeze_schedule function"""
    
    def test_setup_partial_freeze_schedule_signature(self):
        """Test that setup_partial_freeze_schedule takes only num_stages"""
        # This should work
        schedule = setup_partial_freeze_schedule(num_stages=4)
        assert schedule is not None
        
        # This should fail (wrong signature)
        with pytest.raises(TypeError):
            setup_partial_freeze_schedule(cfg={}, num_stages=4)
    
    def test_setup_partial_freeze_schedule_with_cfg_signature(self):
        """Test that setup_partial_freeze_schedule_with_cfg takes cfg and num_stages"""
        cfg = {"some_config": "value"}
        # This should work
        setup_partial_freeze_schedule_with_cfg(cfg=cfg, num_stages=4)
        
        # This should fail (missing cfg)
        with pytest.raises(TypeError):
            setup_partial_freeze_schedule_with_cfg(num_stages=4)
    
    def test_main_py_should_use_correct_function(self):
        """Test that main.py should use the correct function"""
        # Simulate what main.py is trying to do
        cfg = {"some_config": "value"}
        num_stages = 4
        
        # This is what main.py is doing (WRONG):
        # setup_partial_freeze_schedule(cfg, num_stages)  # This fails
        
        # This is what main.py should do (CORRECT):
        setup_partial_freeze_schedule_with_cfg(cfg=cfg, num_stages=num_stages)  # This works
    
    def test_function_imports(self):
        """Test that both functions are properly imported"""
        from core.utils import setup_partial_freeze_schedule, setup_partial_freeze_schedule_with_cfg
        
        # Both should be callable
        assert callable(setup_partial_freeze_schedule)
        assert callable(setup_partial_freeze_schedule_with_cfg)
        
        # Test their signatures
        import inspect
        
        # setup_partial_freeze_schedule should take only num_stages
        sig1 = inspect.signature(setup_partial_freeze_schedule)
        assert len(sig1.parameters) == 1
        assert 'num_stages' in sig1.parameters
        
        # setup_partial_freeze_schedule_with_cfg should take cfg and num_stages
        sig2 = inspect.signature(setup_partial_freeze_schedule_with_cfg)
        assert len(sig2.parameters) == 2
        assert 'cfg' in sig2.parameters
        assert 'num_stages' in sig2.parameters 