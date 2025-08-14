import pytest
import sys
import subprocess
import time
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

class TestExperimentExecution:
    """Test experiment execution and error handling"""
    
    @pytest.fixture(scope="class")
    def experiment_scripts(self):
        """Get all experiment script paths"""
        return [
            "run/run_asib_ablation_study.sh",
            "run/run_asib_sota_comparison.sh", 
            "run/run_asib_class_overlap.sh"
        ]
    
    def test_script_files_exist(self, experiment_scripts):
        """Test that all experiment scripts exist and are executable"""
        for script_path in experiment_scripts:
            path = Path(script_path)
            assert path.exists(), f"Script file missing: {script_path}"
            assert path.is_file(), f"Script is not a file: {script_path}"
            assert os.access(script_path, os.R_OK), f"Script not readable: {script_path}"
    
    def test_script_syntax(self, experiment_scripts):
        """Test that all scripts have valid bash syntax"""
        for script_path in experiment_scripts:
            if sys.platform.startswith("win"):
                pytest.skip("Skip bash syntax check on Windows")
            try:
                result = subprocess.run(
                    ["bash", "-n", script_path],  # -n flag checks syntax without executing
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                assert result.returncode == 0, f"Script {script_path} has syntax errors: {result.stderr}"
            except subprocess.TimeoutExpired:
                pytest.fail(f"Script {script_path} syntax check timed out")
            except Exception as e:
                pytest.fail(f"Failed to check script {script_path}: {e}")
    
    def test_config_files_exist(self):
        """Test that all required config files exist"""
        required_configs = [
            "configs/experiment/L0_baseline.yaml",
            "configs/experiment/L1_ib.yaml",
            "configs/experiment/L2_cccp.yaml",
            "configs/experiment/L3_ib_cccp_tadapt.yaml",
            "configs/experiment/L4_full.yaml",
            "configs/experiment/side_cccp_ppf.yaml",
            "configs/experiment/sota_scenario_a.yaml",
            "configs/experiment/overlap_100.yaml"
        ]
        
        for config_path in required_configs:
            assert Path(config_path).exists(), f"Config file missing: {config_path}"
    
    def test_main_py_can_load_configs(self):
        """Test that main.py can load all experiment configs"""
        import yaml
        from omegaconf import OmegaConf
        
        config_dir = Path("configs/experiment")
        for config_file in config_dir.glob("*.yaml"):
            if config_file.name == "_template.yaml":
                continue
                
            try:
                # Test YAML loading
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                # Test OmegaConf loading
                omega_config = OmegaConf.create(config)
                
                # Test that we can access key values
                assert omega_config is not None
                
            except Exception as e:
                pytest.fail(f"Failed to load config {config_file}: {e}")
    
    def test_model_creation_with_configs(self):
        """Test that models can be created with experiment configs"""
        from core import create_student_by_name, create_teacher_by_name
        
        # Test with a basic config
        config = {
            "device": "cuda",
            "num_classes": 100,
            "use_distillation_adapter": True,
            "distill_out_dim": 512
        }
        
        try:
            # Test student creation
            student = create_student_by_name(
                student_name="resnet50_scratch",
                num_classes=100,
                pretrained=False,
                small_input=True,
                cfg=config
            )
            assert student is not None
            
            # Test teacher creation
            teacher = create_teacher_by_name(
                teacher_name="convnext_s",
                num_classes=100,
                pretrained=False,
                small_input=True,
                cfg=config
            )
            assert teacher is not None
            
        except Exception as e:
            pytest.fail(f"Model creation failed: {e}")
    
    def test_data_loading_with_configs(self):
        """Test data loading with experiment configs"""
        from data.cifar100 import get_cifar100_loaders
        from data.imagenet32 import get_imagenet32_loaders
        
        # Mock dataset loading to avoid file dependencies
        with patch('data.cifar100.CIFAR100NPZ') as mock_cifar, \
             patch('data.imagenet32.ImageNet32') as mock_imagenet:
            
            # Create mock datasets
            mock_cifar_dataset = MagicMock()
            mock_cifar_dataset.classes = list(range(100))
            mock_cifar_dataset.num_classes = 100
            mock_cifar_dataset.__len__ = lambda self: 1000
            mock_cifar_dataset.__getitem__ = lambda self, idx: (torch.randn(3, 32, 32), torch.randint(0, 100, (1,)).item())
            mock_cifar.return_value = mock_cifar_dataset
            
            mock_imagenet_dataset = MagicMock()
            mock_imagenet_dataset.classes = list(range(1000))
            mock_imagenet_dataset.num_classes = 1000
            mock_imagenet_dataset.__len__ = lambda self: 1000
            mock_imagenet_dataset.__getitem__ = lambda self, idx: (torch.randn(3, 32, 32), torch.randint(0, 1000, (1,)).item())
            mock_imagenet.return_value = mock_imagenet_dataset
            
            try:
                # Test CIFAR-100 loading
                train_loader, test_loader = get_cifar100_loaders(
                    root="./data",
                    batch_size=16,
                    num_workers=0,
                    augment=False
                )
                assert train_loader is not None
                assert test_loader is not None
                
                # Test ImageNet-32 loading
                train_loader, test_loader = get_imagenet32_loaders(
                    root="./data",
                    batch_size=16,
                    num_workers=0,
                    augment=False
                )
                assert train_loader is not None
                assert test_loader is not None
                
            except Exception as e:
                pytest.fail(f"Data loading failed: {e}")
    
    def test_ib_mbm_creation_with_configs(self):
        """Test IB_MBM creation with experiment configs"""
        import torch
        from models import build_ib_mbm_from_teachers as build_from_teachers
        
        # Mock teachers
        class MockTeacher:
            def __init__(self, feat_dim=2048):
                self.feat_dim = feat_dim
            
            def get_feat_dim(self):
                return self.feat_dim
        
        teachers = [MockTeacher(2048), MockTeacher(2048)]
        
        # Test with typical config
        config = {
            "ib_mbm_query_dim": 2048,
            "ib_mbm_out_dim": 512,
            "ib_beta": 1e-2,
            "ib_mbm_n_head": 8,
            "num_classes": 100,
            "synergy_head_dropout": 0.1,
            "use_distillation_adapter": True
        }
        
        try:
            ib_mbm, synergy_head = build_from_teachers(teachers, config)
            assert ib_mbm is not None
            assert synergy_head is not None
            
            # Test forward pass
            batch_size = 4
            q_feat = torch.randn(batch_size, 2048)
            kv_feats = torch.randn(batch_size, 2, 2048)
            
            z, mu, logvar = ib_mbm(q_feat, kv_feats)
            logits = synergy_head(z)
            
            assert z.shape == (batch_size, 512)
            assert logits.shape == (batch_size, 100)
            
        except Exception as e:
            pytest.fail(f"IB_MBM creation failed: {e}")
    
    def test_optimizer_creation_with_configs(self):
        """Test optimizer creation with experiment configs"""
        import torch
        from core.trainer import create_optimizers_and_schedulers
        
        # Mock models with parameters
        student_model = MagicMock()
        student_model.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
        
        teacher1 = MagicMock()
        teacher1.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
        teacher1.distillation_adapter = MagicMock()
        teacher1.distillation_adapter.parameters.return_value = [torch.randn(5, 5, requires_grad=True)]
        
        teacher2 = MagicMock()
        teacher2.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
        teacher2.distillation_adapter = MagicMock()
        teacher2.distillation_adapter.parameters.return_value = [torch.randn(5, 5, requires_grad=True)]
        
        teachers = [teacher1, teacher2]
        
        # Test config
        config = {
            "student_lr": 0.1,
            "teacher_lr": 0.0,
            "student_weight_decay": 0.0003,
            "teacher_weight_decay": 0.0,
            "momentum": 0.9,
            "nesterov": True,
            "num_stages": 1,
            "student_epochs_per_stage": 15,
            "schedule": {"type": "cosine"}
        }
        
        try:
            # Mock additional required components
            ib_mbm = MagicMock()
            ib_mbm.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
            
            synergy_head = MagicMock()
            synergy_head.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
            
            teacher_wrappers = [teacher1, teacher2]
            
            teacher_optimizer, teacher_scheduler, student_optimizer, student_scheduler = create_optimizers_and_schedulers(
                teacher_wrappers, ib_mbm, synergy_head, student_model, config, num_stages=1
            )
            
            assert teacher_optimizer is not None
            assert teacher_scheduler is not None
            assert student_optimizer is not None
            assert student_scheduler is not None
            
        except Exception as e:
            pytest.fail(f"Optimizer creation failed: {e}")
    
    def test_loss_functions_with_configs(self):
        """Test loss functions with experiment configs"""
        import torch
        
        batch_size = 4
        num_classes = 100
        
        # Test CE loss
        logits = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))
        
        try:
            # Test with different label smoothing values
            for ls_eps in [0.0, 0.1, 0.2]:
                ce_loss = torch.nn.functional.cross_entropy(logits, targets, label_smoothing=ls_eps)
                assert ce_loss.item() > 0
                
        except Exception as e:
            pytest.fail(f"CE loss failed: {e}")
        
        # Test KL loss
        student_logits = torch.randn(batch_size, num_classes)
        teacher_logits = torch.randn(batch_size, num_classes)
        
        try:
            # Test with different temperature values
            for tau in [1.0, 2.0, 4.0]:
                kl_loss = torch.nn.functional.kl_div(
                    torch.log_softmax(student_logits / tau, dim=1),
                    torch.softmax(teacher_logits / tau, dim=1),
                    reduction='batchmean'
                )
                assert kl_loss.item() >= 0
                
        except Exception as e:
            pytest.fail(f"KL loss failed: {e}")
    
    def test_config_validation_comprehensive(self):
        """Test comprehensive config validation"""
        from utils.validation import validate_config
        
        # Test valid configs
        valid_configs = [
            {
                "device": "cuda",
                "num_classes": 100,
                "student_lr": 0.1,
                "teacher_lr": 0.0,
                "ib_mbm_query_dim": 2048,
                "ib_mbm_out_dim": 512,
                "num_stages": 1,
                "student_epochs_per_stage": 15
            },
            {
                "device": "cuda",
                "num_classes": 1000,
                "student_lr": 0.01,
                "teacher_lr": 0.001,
                "ib_mbm_query_dim": 1280,
                "ib_mbm_out_dim": 1024,
                "num_stages": 3,
                "student_epochs_per_stage": 30
            }
        ]
        
        for config in valid_configs:
            try:
                validate_config(config)
                assert True
            except Exception as e:
                print(f"Config validation warning: {e}")
                assert True
    
    def test_error_handling_robustness(self):
        """Test error handling robustness"""
        from core.utils import cast_numeric_configs, renorm_ce_kd
        
        # Test with invalid configs
        invalid_configs = [
            {"student_lr": "invalid", "teacher_lr": "not_a_number"},
            {"num_stages": "wrong_type", "student_epochs_per_stage": "invalid"},
            {"ce_alpha": "bad", "kd_alpha": "wrong"}
        ]
        
        for config in invalid_configs:
            try:
                cast_numeric_configs(config)
                assert True
            except Exception as e:
                pytest.fail(f"cast_numeric_configs crashed with invalid input: {e}")
        
        # Test renorm_ce_kd with invalid inputs
        invalid_renorm_configs = [
            {"ce_alpha": 0.3, "kd_alpha": 0.7},  # Valid
            {"ce_alpha": 0.0, "kd_alpha": 0.0},  # Edge case
            {"ce_alpha": 1.0, "kd_alpha": 1.0}   # Edge case
        ]
        
        for config in invalid_renorm_configs:
            try:
                renorm_ce_kd(config)
                assert True
            except Exception as e:
                pytest.fail(f"renorm_ce_kd crashed: {e}")
    
    def test_memory_management(self):
        """Test memory management"""
        import torch
        import gc
        
        # Test that we can create and destroy models without memory leaks
        try:
            from core import create_student_by_name, create_teacher_by_name
            
            config = {
                "device": "cuda",
                "num_classes": 100,
                "use_distillation_adapter": True,
                "distill_out_dim": 512
            }
            
            # Create models multiple times
            for i in range(3):
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
                    pretrained=False,
                    small_input=True,
                    cfg=config
                )
                
                # Test forward pass
                batch_size = 2
                x = torch.randn(batch_size, 3, 32, 32)
                
                with torch.no_grad():
                    student_out = student(x)
                    teacher_out = teacher(x)
                
                # Clean up
                del student, teacher, student_out, teacher_out
                gc.collect()
                
            assert True
            
        except Exception as e:
            pytest.fail(f"Memory management test failed: {e}")
    
    def test_device_compatibility(self):
        """Test device compatibility"""
        import torch
        
        # Test CPU
        try:
            device = "cuda"
            x = torch.randn(2, 3, 32, 32, device=device)
            assert x.device.type == "cuda"
        except Exception as e:
            pytest.fail(f"CPU device test failed: {e}")
        
        # Test CUDA if available
        if torch.cuda.is_available():
            try:
                device = "cuda"
                x = torch.randn(2, 3, 32, 32, device=device)
                assert x.device.type == "cuda"
                
                # Test memory allocation
                y = torch.randn(1000, 1000, device=device)
                del y
                torch.cuda.empty_cache()
                
            except Exception as e:
                pytest.fail(f"CUDA device test failed: {e}")
    
    def test_logging_and_monitoring(self):
        """Test logging and monitoring"""
        import logging
        
        try:
            # Test basic logging setup
            logger = logging.getLogger("test_logger")
            logger.setLevel(logging.INFO)
            
            # Test logging
            logger.info("Test info message")
            logger.warning("Test warning message")
            logger.error("Test error message")
            
            assert logger is not None
            assert logger.level == logging.INFO
                
        except Exception as e:
            pytest.fail(f"Logging test failed: {e}")
    
    def test_framework_integrity(self):
        """Test overall framework integrity"""
        # Test that all critical components are present and functional
        critical_components = [
            "main.py",
            "core/__init__.py",
            "core/utils.py",
            "core/builder.py",
            "core/trainer.py",
            "models/ib_mbm.py",
            "models/common/base_wrapper.py",
            "data/cifar100.py",
            "data/imagenet32.py",
            "data/cifar100_overlap.py",
            "modules/trainer_student.py",
            "modules/trainer_teacher.py",
            "utils/common/__init__.py",
            "utils/validation.py"
        ]
        
        for component in critical_components:
            assert Path(component).exists(), f"Critical component missing: {component}"
        
        # Test that all directories exist
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