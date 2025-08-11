import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
import sys
import os
from pathlib import Path

class TestErrorPrevention:
    """Test error prevention and edge case handling"""
    
    def test_mbm_edge_cases(self):
        """Test IB_MBM with edge cases"""
        from models import IB_MBM
        
        # Test with minimal dimensions
        ib_mbm = IB_MBM(
            q_dim=1,
            kv_dim=1,
            d_emb=1,
            beta=1e-2,
            n_head=1
        )
        
        batch_size = 1
        q_feat = torch.randn(batch_size, 1)
        kv_feats = torch.randn(batch_size, 1)
        
        try:
            z, mu, logvar = ib_mbm(q_feat, kv_feats)
            assert z.shape == (batch_size, 1)
            assert mu.shape == (batch_size, 1)
            assert logvar.shape == (batch_size, 1)
        except Exception as e:
            pytest.fail(f"IB_MBM failed with minimal dimensions: {e}")
        
        # Test with very large dimensions
        ib_mbm_large = IB_MBM(
            q_dim=4096,
            kv_dim=4096,
            d_emb=1024,
            beta=1e-2,
            n_head=16
        )
        
        batch_size = 2
        q_feat = torch.randn(batch_size, 4096)
        kv_feats = torch.randn(batch_size, 2, 4096)
        
        try:
            z, mu, logvar = ib_mbm_large(q_feat, kv_feats)
            assert z.shape == (batch_size, 1024)
            assert mu.shape == (batch_size, 1024)
            assert logvar.shape == (batch_size, 1024)
        except Exception as e:
            pytest.fail(f"IB_MBM failed with large dimensions: {e}")
    
    def test_tensor_shape_validation(self):
        """Test tensor shape validation"""
        from models import IB_MBM
        
        ib_mbm = IB_MBM(
            q_dim=512,
            kv_dim=512,
            d_emb=256,
            beta=1e-2,
            n_head=8
        )
        
        batch_size = 4
        
        # Test various tensor shapes
        test_cases = [
            # (q_shape, kv_shape, should_work, description)
            ((batch_size, 512), (batch_size, 512), True, "2D tensors"),
            ((batch_size, 512), (batch_size, 2, 512), True, "3D kv tensor"),
            ((batch_size, 512), (batch_size, 1, 512), True, "Single teacher"),
            ((batch_size, 512), (batch_size, 10, 512), True, "Many teachers"),
        ]
        
        for q_shape, kv_shape, should_work, desc in test_cases:
            try:
                q_feat = torch.randn(*q_shape)
                kv_feats = torch.randn(*kv_shape)
                
                z, mu, logvar = ib_mbm(q_feat, kv_feats)
                
                if should_work:
                    assert z.shape == (batch_size, 256), f"Wrong z shape for {desc}"
                    assert mu.shape == (batch_size, 256), f"Wrong mu shape for {desc}"
                    assert logvar.shape == (batch_size, 256), f"Wrong logvar shape for {desc}"
                else:
                    pytest.fail(f"Should have failed for {desc}")
                    
            except Exception as e:
                if should_work:
                    pytest.fail(f"IB_MBM failed for {desc}: {e}")
                else:
                    assert True  # Expected to fail

    def test_teacher_feature_dim_mismatch_raises(self):
        """Teacher feature dims differ without adapter should raise in builder."""
        from models import build_ib_mbm_from_teachers as build_from_teachers
        import torch.nn as nn

        class T1(nn.Module):
            def get_feat_dim(self):
                return 1024

        class T2(nn.Module):
            def get_feat_dim(self):
                return 2048

        teachers = [T1(), T2()]
        cfg = {
            "use_distillation_adapter": False,
            "ib_mbm_query_dim": 1024,
            "ib_mbm_out_dim": 512,
            "ib_mbm_n_head": 8,
            "num_classes": 100,
        }

        with pytest.raises(ValueError):
            build_from_teachers(teachers, cfg)
    
    def test_config_edge_cases(self):
        """Test configuration edge cases"""
        from core.utils import cast_numeric_configs, renorm_ce_kd
        
        # Test with empty config
        empty_config = {}
        try:
            cast_numeric_configs(empty_config)
            assert True
        except Exception as e:
            pytest.fail(f"cast_numeric_configs failed with empty config: {e}")
        
        # Test with None values
        none_config = {
            "student_lr": None,
            "teacher_lr": None,
            "num_stages": None
        }
        try:
            cast_numeric_configs(none_config)
            assert True
        except Exception as e:
            pytest.fail(f"cast_numeric_configs failed with None values: {e}")
        
        # Test with extreme values
        extreme_config = {
            "student_lr": 1e-10,
            "teacher_lr": 1e10,
            "num_stages": 1000,
            "student_epochs_per_stage": 1,
            "momentum": 0.999999,
            "nesterov": True
        }
        try:
            cast_numeric_configs(extreme_config)
            assert True
        except Exception as e:
            pytest.fail(f"cast_numeric_configs failed with extreme values: {e}")
        
        # Test renorm_ce_kd edge cases
        renorm_cases = [
            {"ce_alpha": 0.0, "kd_alpha": 0.0},  # Both zero
            {"ce_alpha": 1.0, "kd_alpha": 0.0},  # Only CE
            {"ce_alpha": 0.0, "kd_alpha": 1.0},  # Only KD
            {"ce_alpha": 0.5, "kd_alpha": 0.5},  # Equal
            {"ce_alpha": 0.1, "kd_alpha": 0.9},  # Unequal
        ]
        
        for config in renorm_cases:
            try:
                renorm_ce_kd(config)
                assert True
            except Exception as e:
                pytest.fail(f"renorm_ce_kd failed for {config}: {e}")
    
    def test_model_creation_edge_cases(self):
        """Test model creation edge cases"""
        from core import create_student_by_name, create_teacher_by_name
        
        # Test with minimal config
        minimal_config = {
            "device": "cuda",
            "num_classes": 1,
            "use_distillation_adapter": False
        }
        
        try:
            student = create_student_by_name(
                student_name="resnet50_scratch",
                num_classes=1,
                pretrained=False,
                small_input=True,
                cfg=minimal_config
            )
            assert student is not None
        except Exception as e:
            pytest.fail(f"Student creation failed with minimal config: {e}")
        
        # Test with large config
        large_config = {
            "device": "cuda",
            "num_classes": 10000,
            "use_distillation_adapter": True,
            "distill_out_dim": 2048
        }
        
        try:
            teacher = create_teacher_by_name(
                teacher_name="convnext_s",
                num_classes=10000,
                pretrained=False,
                small_input=True,
                cfg=large_config
            )
            assert teacher is not None
        except Exception as e:
            pytest.fail(f"Teacher creation failed with large config: {e}")
    
    def test_data_loading_edge_cases(self):
        """Test data loading edge cases"""
        from data.cifar100 import get_cifar100_loaders
        from data.imagenet32 import get_imagenet32_loaders
        
        # Mock dataset loading
        with patch('data.cifar100.CIFAR100NPZ') as mock_cifar, \
             patch('data.imagenet32.ImageNet32') as mock_imagenet:
            
            # Create mock datasets
            mock_cifar_dataset = MagicMock()
            mock_cifar_dataset.classes = list(range(100))
            mock_cifar_dataset.num_classes = 100
            mock_cifar_dataset.__len__ = lambda self: 1  # Very small dataset
            mock_cifar_dataset.__getitem__ = lambda self, idx: (torch.randn(3, 32, 32), torch.randint(0, 100, (1,)).item())
            mock_cifar.return_value = mock_cifar_dataset
            
            mock_imagenet_dataset = MagicMock()
            mock_imagenet_dataset.classes = list(range(1000))
            mock_imagenet_dataset.num_classes = 1000
            mock_imagenet_dataset.__len__ = lambda self: 1  # Very small dataset
            mock_imagenet_dataset.__getitem__ = lambda self, idx: (torch.randn(3, 32, 32), torch.randint(0, 1000, (1,)).item())
            mock_imagenet.return_value = mock_imagenet_dataset
            
            # Test with very small batch size
            try:
                train_loader, test_loader = get_cifar100_loaders(
                    root="./data",
                    batch_size=1,
                    num_workers=0,
                    augment=False
                )
                assert train_loader is not None
                assert test_loader is not None
            except Exception as e:
                pytest.fail(f"CIFAR-100 loading failed with batch_size=1: {e}")
            
            # Test with large batch size
            try:
                train_loader, test_loader = get_imagenet32_loaders(
                    root="./data",
                    batch_size=128,
                    num_workers=0,
                    augment=False
                )
                assert train_loader is not None
                assert test_loader is not None
            except Exception as e:
                pytest.fail(f"ImageNet-32 loading failed with batch_size=128: {e}")
    
    def test_loss_function_edge_cases(self):
        """Test loss function edge cases"""
        import torch
        import torch.nn.functional as F
        
        # Test CE loss edge cases
        ce_cases = [
            # (logits_shape, targets_shape, label_smoothing, description)
            ((1, 1), (1,), 0.0, "Single sample, single class"),
            ((1, 2), (1,), 0.0, "Single sample, two classes"),
            ((2, 1), (2,), 0.0, "Two samples, single class"),
            ((1, 100), (1,), 0.1, "Single sample, many classes, label smoothing"),
            ((1, 100), (1,), 0.9, "Single sample, many classes, high label smoothing"),
        ]
        
        for logits_shape, targets_shape, ls_eps, desc in ce_cases:
            try:
                logits = torch.randn(*logits_shape)
                targets = torch.randint(0, logits_shape[1], targets_shape)
                
                ce_loss = F.cross_entropy(logits, targets, label_smoothing=ls_eps)
                assert ce_loss.item() >= 0, f"CE loss should be non-negative for {desc}"
                
            except Exception as e:
                pytest.fail(f"CE loss failed for {desc}: {e}")
        
        # Test KL loss edge cases
        kl_cases = [
            # (student_shape, teacher_shape, tau, description)
            ((1, 1), (1, 1), 1.0, "Single sample, single class"),
            ((1, 2), (1, 2), 0.1, "Single sample, two classes, low temperature"),
            ((1, 2), (1, 2), 10.0, "Single sample, two classes, high temperature"),
            ((2, 100), (2, 100), 4.0, "Two samples, many classes"),
        ]
        
        for student_shape, teacher_shape, tau, desc in kl_cases:
            try:
                student_logits = torch.randn(*student_shape)
                teacher_logits = torch.randn(*teacher_shape)
                
                kl_loss = F.kl_div(
                    F.log_softmax(student_logits / tau, dim=1),
                    F.softmax(teacher_logits / tau, dim=1),
                    reduction='batchmean'
                )
                assert kl_loss.item() >= 0, f"KL loss should be non-negative for {desc}"
                
            except Exception as e:
                pytest.fail(f"KL loss failed for {desc}: {e}")
    
    def test_memory_edge_cases(self):
        """Test memory edge cases"""
        import torch
        import gc
        
        # Test with very large tensors
        try:
            # Create large tensor
            large_tensor = torch.randn(1000, 1000, 1000)
            assert large_tensor.numel() == 1000000000
            
            # Delete and clean up
            del large_tensor
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            assert True
        except Exception as e:
            pytest.fail(f"Large tensor handling failed: {e}")
        
        # Test with many small tensors
        try:
            tensors = []
            for i in range(1000):
                tensor = torch.randn(100, 100)
                tensors.append(tensor)
            
            # Clean up
            del tensors
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            assert True
        except Exception as e:
            pytest.fail(f"Many small tensors handling failed: {e}")
    
    def test_device_edge_cases(self):
        """Test device edge cases"""
        import torch
        
        # Test CPU device
        try:
            device = "cuda"
            x = torch.randn(1, 1, 1, 1, device=device)
            assert x.device.type == "cuda"
            
            # Test with different dtypes (only float types for randn)
            for dtype in [torch.float32, torch.float64]:
                y = torch.randn(1, 1, device=device, dtype=dtype)
                assert x.device.type == "cuda"
                assert y.dtype == dtype
                
        except Exception as e:
            pytest.fail(f"CPU device edge cases failed: {e}")
        
        # Test CUDA device if available
        if torch.cuda.is_available():
            try:
                device = "cuda"
                x = torch.randn(1, 1, 1, 1, device=device)
                assert x.device.type == "cuda"
                
                # Test memory allocation and cleanup
                y = torch.randn(1000, 1000, device=device)
                del y
                torch.cuda.empty_cache()
                
                # Test with different dtypes
                for dtype in [torch.float32, torch.float16]:
                    z = torch.randn(1, 1, device=device, dtype=dtype)
                    assert z.device.type == "cuda"
                    assert z.dtype == dtype
                    del z
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                pytest.fail(f"CUDA device edge cases failed: {e}")
    
    def test_import_edge_cases(self):
        """Test import edge cases"""
        # Test importing with different Python versions
        try:
            import sys
            python_version = sys.version_info
            
            # Test critical imports
            critical_modules = [
                "torch",
                "torch.nn",
                "torch.nn.functional",
                "numpy",
                "yaml",
                "omegaconf"
            ]
            
            for module_name in critical_modules:
                try:
                    __import__(module_name)
                    assert True
                except ImportError as e:
                    pytest.fail(f"Failed to import {module_name}: {e}")
                    
        except Exception as e:
            pytest.fail(f"Import edge cases failed: {e}")
    
    def test_file_system_edge_cases(self):
        """Test file system edge cases"""
        from pathlib import Path
        
        # Test with non-existent paths
        non_existent_paths = [
            "/non/existent/path",
            "./non_existent_file.py",
            "configs/non_existent_config.yaml"
        ]
        
        for path_str in non_existent_paths:
            path = Path(path_str)
            assert not path.exists(), f"Path should not exist: {path_str}"
        
        # Test with existing paths
        existing_paths = [
            "main.py",
            "configs",
            "tests",
            "run"
        ]
        
        for path_str in existing_paths:
            path = Path(path_str)
            assert path.exists(), f"Path should exist: {path_str}"
        
        # Test file permissions
        try:
            # Test reading main.py
            with open("main.py", 'r', encoding='utf-8') as f:
                content = f.read()
                assert len(content) > 0
                
            # Test reading configs
            config_dir = Path("configs")
            config_files = list(config_dir.rglob("*.yaml"))
            assert len(config_files) > 0
            
        except Exception as e:
            pytest.fail(f"File system edge cases failed: {e}")
    
    def test_numerical_edge_cases(self):
        """Test numerical edge cases"""
        import torch
        import math
        
        # Test with very small numbers
        try:
            small_tensor = torch.tensor([1e-10, 1e-20, 1e-30])
            assert torch.all(torch.isfinite(small_tensor))
            
            # Test log operations
            log_small = torch.log(small_tensor)
            assert torch.all(torch.isfinite(log_small))
            
        except Exception as e:
            pytest.fail(f"Small numbers handling failed: {e}")
        
        # Test with very large numbers
        try:
            large_tensor = torch.tensor([1e10, 1e20, 1e30])
            assert torch.all(torch.isfinite(large_tensor))
            
            # Test exp operations (these will be inf, which is expected)
            exp_large = torch.exp(large_tensor)
            # Don't check isfinite since exp of large numbers produces inf
            assert torch.all(torch.isinf(exp_large))
            
        except Exception as e:
            pytest.fail(f"Large numbers handling failed: {e}")
        
        # Test with NaN and Inf
        try:
            nan_tensor = torch.tensor([float('nan'), float('inf'), float('-inf')])
            
            # Test isfinite
            finite_mask = torch.isfinite(nan_tensor)
            assert not torch.all(finite_mask)
            
            # Test isnan
            nan_mask = torch.isnan(nan_tensor)
            assert torch.any(nan_mask)
            
        except Exception as e:
            pytest.fail(f"NaN/Inf handling failed: {e}")
    
    def test_concurrent_access(self):
        """Test concurrent access scenarios"""
        import threading
        import time
        
        # Test multiple threads accessing the same model
        def create_model(thread_id):
            try:
                from core import create_student_by_name
                
                config = {
                    "device": "cuda",
                    "num_classes": 100,
                    "use_distillation_adapter": True,
                    "distill_out_dim": 512
                }
                
                model = create_student_by_name(
                    student_name="resnet50_scratch",
                    num_classes=100,
                    pretrained=False,
                    small_input=True,
                    cfg=config
                )
                
                # Test forward pass
                x = torch.randn(2, 3, 32, 32)
                with torch.no_grad():
                    output = model(x)
                
                return True
                
            except Exception as e:
                print(f"Thread {thread_id} failed: {e}")
                return False
        
        # Run multiple threads
        threads = []
        results = []
        
        for i in range(3):
            thread = threading.Thread(target=lambda i=i: results.append(create_model(i)))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(results) == 3, "Not all threads completed"
        assert all(results), "Some threads failed"
    
    def test_error_recovery(self):
        """Test error recovery mechanisms"""
        from core.utils import cast_numeric_configs, renorm_ce_kd
        
        # Test recovery from invalid config
        invalid_configs = [
            {"student_lr": "invalid", "teacher_lr": "not_a_number"},
            {"num_stages": "wrong_type", "student_epochs_per_stage": "invalid"},
            {"ce_alpha": "bad", "kd_alpha": "wrong"},
            {"student_lr": None, "teacher_lr": None},
            {"student_lr": float('nan'), "teacher_lr": float('inf')},
        ]
        
        for config in invalid_configs:
            try:
                # Should not crash
                cast_numeric_configs(config)
                assert True
            except Exception as e:
                pytest.fail(f"cast_numeric_configs crashed with invalid config: {e}")
        
        # Test recovery from invalid renorm config
        invalid_renorm_configs = [
            {"ce_alpha": float('nan'), "kd_alpha": float('inf')},
            {"ce_alpha": -1.0, "kd_alpha": -1.0},
            {"ce_alpha": 2.0, "kd_alpha": 2.0},
        ]
        
        for config in invalid_renorm_configs:
            try:
                # Should not crash
                renorm_ce_kd(config)
                assert True
            except Exception as e:
                pytest.fail(f"renorm_ce_kd crashed with invalid config: {e}") 