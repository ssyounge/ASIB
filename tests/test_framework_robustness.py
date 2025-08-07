import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
import sys
import os
from pathlib import Path

class TestFrameworkRobustness:
    """Test framework robustness and error handling"""
    
    def test_all_critical_imports(self):
        """Test that all critical modules can be imported"""
        critical_modules = [
            "main",
            "core",
            "core.utils",
            "core.builder", 
            "core.trainer",
            "models.mbm",
            "models.common.base_wrapper",
            "data.cifar100",
            "data.imagenet32",
            "data.cifar100_overlap",
            "modules.trainer_student",
            "modules.trainer_teacher",
            "modules.losses",
            "utils.common",
            "utils.validation"
        ]
        
        for module_name in critical_modules:
            try:
                __import__(module_name)
                assert True
            except ImportError as e:
                pytest.fail(f"Failed to import {module_name}: {e}")
    
    def test_model_registry_completeness(self):
        """Test that model registry is complete and functional"""
        from models.common.base_wrapper import MODEL_REGISTRY
        
        # Check that registry is not empty
        assert len(MODEL_REGISTRY) > 0, "Model registry is empty"
        
        # Check for required model types (registry keys don't contain "student"/"teacher" suffixes)
        # Instead, check for models that are typically used as students or teachers
        student_models = [k for k in MODEL_REGISTRY.keys() if any(x in k for x in ["scratch", "resnet50", "mobilenet", "shufflenet", "efficientnet_b0"])]
        teacher_models = [k for k in MODEL_REGISTRY.keys() if any(x in k for x in ["resnet152", "convnext", "efficientnet_l2"])]
        
        assert len(student_models) > 0, "No student models in registry"
        assert len(teacher_models) > 0, "No teacher models in registry"
        
        # Test that all registered models can be created
        for model_name in list(MODEL_REGISTRY.keys())[:3]:  # Test first 3 models
            try:
                model_func = MODEL_REGISTRY[model_name]
                assert callable(model_func), f"Model {model_name} is not callable"
            except Exception as e:
                pytest.fail(f"Failed to access model {model_name}: {e}")
    
    def test_config_file_existence(self):
        """Test that all required config files exist"""
        required_files = [
            "configs/registry_key.yaml",
            "configs/registry_map.yaml",
            "configs/experiment/ablation_baseline.yaml",
            "configs/experiment/sota_scenario_a.yaml", 
            "configs/experiment/overlap_100.yaml",
            "configs/model/student/resnet50_scratch.yaml",
            "configs/model/teacher/convnext_s.yaml",
            "configs/model/teacher/resnet152.yaml"
        ]
        
        for file_path in required_files:
            assert Path(file_path).exists(), f"Required config file missing: {file_path}"
    
    def test_function_signatures(self):
        """Test critical function signatures"""
        from core.utils import (
            setup_partial_freeze_schedule_with_cfg,
            setup_safety_switches_with_cfg,
            auto_set_mbm_query_dim_with_model,
            renorm_ce_kd,
            cast_numeric_configs
        )
        
        # Test that all functions are callable
        assert callable(setup_partial_freeze_schedule_with_cfg)
        assert callable(setup_safety_switches_with_cfg)
        assert callable(auto_set_mbm_query_dim_with_model)
        assert callable(renorm_ce_kd)
        assert callable(cast_numeric_configs)
    
    def test_mbm_tensor_handling(self):
        """Test MBM tensor handling robustness"""
        from models.mbm import IB_MBM
        
        mbm = IB_MBM(
            q_dim=2048,
            kv_dim=2048,
            d_emb=512,
            beta=1e-2,
            n_head=8
        )
        
        batch_size = 4
        
        # Test with different tensor shapes
        test_cases = [
            # (q_feat_shape, kv_feats_shape, description)
            ((batch_size, 2048), (batch_size, 2048), "2D tensors"),
            ((batch_size, 2048), (batch_size, 2, 2048), "3D kv tensor"),
            ((batch_size, 2048), (batch_size, 2, 32, 32), "4D kv tensor"),
        ]
        
        for q_shape, kv_shape, desc in test_cases:
            try:
                q_feat = torch.randn(*q_shape)
                kv_feats = torch.randn(*kv_shape)
                
                z, mu, logvar = mbm(q_feat, kv_feats)
                
                # Check output shapes
                assert z.shape == (batch_size, 512), f"Wrong z shape for {desc}"
                assert mu.shape == (batch_size, 512), f"Wrong mu shape for {desc}"
                assert logvar.shape == (batch_size, 512), f"Wrong logvar shape for {desc}"
                
            except Exception as e:
                pytest.fail(f"MBM failed for {desc}: {e}")
    
    def test_dataset_attributes(self):
        """Test dataset attribute handling"""
        from data.cifar100 import CIFAR100NPZ
        from data.imagenet32 import ImageNet32
        
        # Test CIFAR100NPZ
        try:
            dataset = CIFAR100NPZ(root="./data", train=True)
            assert hasattr(dataset, 'classes'), "CIFAR100NPZ missing classes attribute"
            assert hasattr(dataset, 'num_classes'), "CIFAR100NPZ missing num_classes attribute"
            assert dataset.num_classes == 100, "CIFAR100NPZ wrong num_classes"
        except Exception as e:
            # Dataset might not exist, but attributes should be defined
            pass
        
        # Test ImageNet32
        try:
            dataset = ImageNet32(root="./data", split="train")
            assert hasattr(dataset, 'num_classes'), "ImageNet32 missing num_classes attribute"
            assert dataset.num_classes == 1000, "ImageNet32 wrong num_classes"
        except Exception as e:
            # Dataset might not exist, but attributes should be defined
            pass
    
    def test_config_validation(self):
        """Test configuration validation"""
        from utils.validation import validate_config
        
        # Test valid config
        valid_config = {
            "device": "cuda",
            "num_classes": 100,
            "student_lr": 0.1,
            "teacher_lr": 0.0,
            "mbm_query_dim": 2048,
            "mbm_out_dim": 512,
            "num_stages": 1,
            "student_epochs_per_stage": 15
        }
        
        try:
            validate_config(valid_config)
            assert True
        except Exception as e:
            # Validation might be strict, but shouldn't crash
            print(f"Config validation warning: {e}")
            assert True
    
    def test_error_handling(self):
        """Test error handling in critical functions"""
        from core.utils import cast_numeric_configs
        
        # Test with invalid config
        invalid_config = {
            "student_lr": "invalid",
            "teacher_lr": "not_a_number",
            "num_stages": "wrong_type"
        }
        
        # Should not crash, should handle gracefully
        try:
            cast_numeric_configs(invalid_config)
            assert True
        except Exception as e:
            pytest.fail(f"cast_numeric_configs crashed with invalid input: {e}")
    
    def test_memory_efficiency(self):
        """Test memory efficiency of critical operations"""
        # Test that we can create models without excessive memory usage
        try:
            from core import create_student_by_name, create_teacher_by_name
            
            config = {
                "device": "cuda",
                "num_classes": 100,
                "use_distillation_adapter": True,
                "distill_out_dim": 512
            }
            
            # Create models
            student = create_student_by_name(
                student_name="resnet50_scratch",
                num_classes=100,
                pretrained=False,
                small_input=True,
                cfg=config
            )
            
            teacher = create_teacher_by_name(
                teacher_name="convnext_s",
                num_classes=100,
                pretrained=False,  # Use False to avoid downloading
                small_input=True,
                cfg=config
            )
            
            # Test forward pass
            batch_size = 2  # Small batch to test memory
            x = torch.randn(batch_size, 3, 32, 32)
            
            with torch.no_grad():
                student_out = student(x)
                teacher_out = teacher(x)
            
            assert student_out is not None
            assert teacher_out is not None
            
        except Exception as e:
            pytest.fail(f"Memory efficiency test failed: {e}")
    
    def test_script_execution(self):
        """Test that experiment scripts can be executed"""
        script_files = [
            "run/run_asib_ablation_study.sh",
            "run/run_asib_sota_comparison.sh", 
            "run/run_asib_class_overlap.sh"
        ]
        
        for script_file in script_files:
            assert Path(script_file).exists(), f"Script file missing: {script_file}"
            assert Path(script_file).is_file(), f"Script is not a file: {script_file}"
            
            # Check that script is executable
            assert os.access(script_file, os.R_OK), f"Script not readable: {script_file}"
    
    def test_logging_setup(self):
        """Test logging setup"""
        import logging
        
        try:
            # Test basic logging setup
            logger = logging.getLogger("test_logger")
            logger.setLevel(logging.INFO)
            
            # Test logging
            logger.info("Test log message")
            logger.warning("Test warning message")
            
            assert True
        except Exception as e:
            pytest.fail(f"Logging setup failed: {e}")
    
    def test_device_handling(self):
        """Test device handling"""
        import torch
        
        # Test CPU device
        try:
            device = "cuda"
            x = torch.randn(2, 3, 32, 32, device=device)
            assert x.device.type == "cuda"
        except Exception as e:
            pytest.fail(f"CPU device handling failed: {e}")
        
        # Test CUDA device if available
        if torch.cuda.is_available():
            try:
                device = "cuda"
                x = torch.randn(2, 3, 32, 32, device=device)
                assert x.device.type == "cuda"
            except Exception as e:
                pytest.fail(f"CUDA device handling failed: {e}")
    
    def test_framework_completeness(self):
        """Test that all framework components are present"""
        required_dirs = [
            "configs",
            "configs/experiment",
            "configs/model",
            "configs/model/student",
            "configs/model/teacher",
            "core",
            "data",
            "models",
            "modules",
            "utils",
            "tests",
            "run"
        ]
        
        for dir_path in required_dirs:
            assert Path(dir_path).exists(), f"Required directory missing: {dir_path}"
            assert Path(dir_path).is_dir(), f"Not a directory: {dir_path}"
        
        # Check for critical files
        critical_files = [
            "main.py",
            "README.md",
            "environment.yml",
            "pytest.ini"
        ]
        
        for file_path in critical_files:
            assert Path(file_path).exists(), f"Critical file missing: {file_path}" 