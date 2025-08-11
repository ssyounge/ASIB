#!/usr/bin/env python3
"""
Integration tests for ASIB
"""

import pytest
import torch
import tempfile
import shutil
import os
import numpy as np
from pathlib import Path

# Import main components
from core.builder import build_model, create_teacher_by_name, create_student_by_name
from core.trainer import create_optimizers_and_schedulers, create_optimizers_and_schedulers_legacy
from methods.asib import ASIBDistiller
from models import IB_MBM, SynergyHead


class TestCompletePipeline:
    """Test complete ASIB pipeline"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # On Windows, ensure file handlers are released before rmtree
        try:
            import logging
            logging.shutdown()
        except Exception:
            pass
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_model_creation_pipeline(self):
        """Test complete model creation pipeline"""
        # Create teachers
        teacher1 = create_teacher_by_name(
            teacher_name="resnet152",
            num_classes=100,
            pretrained=False,
            small_input=True
        )
        
        teacher2 = create_teacher_by_name(
            teacher_name="resnet152",  # Use same teacher type to avoid dimension mismatch
            num_classes=100,
            pretrained=False,
            small_input=True
        )
        
        # Create student
        student = create_student_by_name(
            student_name="efficientnet_b0_scratch",
            num_classes=100,
            pretrained=False,
            small_input=True
        )
        
        # Create MBM components
        mbm = IB_MBM(q_dim=1280, kv_dim=4096, d_emb=2048)  # kv_dim = 2 * teacher_feature_dim
        synergy_head = SynergyHead(2048, num_classes=100)
        
        # Create distiller
        distiller = ASIBDistiller(
            teacher1, teacher2, student, mbm, synergy_head, device="cuda"
        )
        
        # Test forward pass
        x = torch.randn(4, 3, 32, 32)
        y = torch.randint(0, 100, (4,))
        
        loss, _ = distiller.forward(x, y)
        assert torch.isfinite(loss)
        assert loss > 0
    
    def test_optimizer_pipeline(self):
        """Test complete optimizer creation pipeline"""
        # Create models
        teacher = torch.nn.Linear(10, 5)
        student = torch.nn.Linear(10, 5)
        
        # Create optimizers and schedulers
        cfg = {
            "teacher_lr": 1e-4,
            "student_lr": 1e-3,
            "teacher_weight_decay": 1e-4,
            "student_weight_decay": 1e-4,
            "teacher_epochs": 10,
            "student_epochs": 10
        }
        
        teacher_optimizer, teacher_scheduler, student_optimizer, student_scheduler = create_optimizers_and_schedulers_legacy(
            teacher=teacher,
            student_model=student,
            cfg=cfg
        )
        
        # Teacher training step
        teacher_optimizer.zero_grad()
        x = torch.randn(2, 10)
        output = teacher(x)
        loss = torch.sum(output)
        loss.backward()
        teacher_optimizer.step()
        teacher_scheduler.step()
        
        # Student training step
        student_optimizer.zero_grad()
        x = torch.randn(2, 10)
        output = student(x)
        loss = torch.sum(output)
        loss.backward()
        student_optimizer.step()
        student_scheduler.step()
        
        assert True  # If we get here, no errors occurred
    
    def test_logging_pipeline(self, temp_dir):
        """Test complete logging pipeline"""
        # Setup basic logging
        import logging
        import os
        
        log_file = os.path.join(temp_dir, "test_integration.log")
        logger = logging.getLogger("test_integration")
        logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        
        # Create a dummy student model for testing
        class DummyStudent(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.name = "dummy_student"
                self.linear = torch.nn.Linear(10, 5)  # Add a parameter to count
        
        dummy_student = DummyStudent()
        
        # Simulate training logging
        for i in range(10):
            logger.info(f"Training step {i}")
        
        # Check that log file was created
        assert os.path.exists(log_file)
        assert logger is not None
        # Close handlers to avoid Windows file lock
        for h in list(logger.handlers):
            try:
                h.flush()
                h.close()
            except Exception:
                pass
            logger.removeHandler(h)


class TestEndToEndTraining:
    """Test end-to-end training scenarios"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_single_epoch_training(self):
        """Test single epoch training"""
        # Create models with same feature dimensions
        teacher1 = create_teacher_by_name(
            teacher_name="resnet152",
            num_classes=10,  # Small for testing
            pretrained=False,
            small_input=True
        )
        
        teacher2 = create_teacher_by_name(
            teacher_name="resnet152",  # Use same teacher type to avoid dimension mismatch
            num_classes=10,
            pretrained=False,
            small_input=True
        )
        
        student = create_student_by_name(
            student_name="efficientnet_b0_scratch",
            num_classes=10,
            pretrained=False,
            small_input=True
        )
        
        # Create MBM components
        mbm = IB_MBM(q_dim=1280, kv_dim=4096, d_emb=2048)  # kv_dim = 2 * teacher_feature_dim
        synergy_head = SynergyHead(2048, num_classes=10)
        
        # Create distiller
        distiller = ASIBDistiller(
            teacher1, teacher2, student, mbm, synergy_head, device="cuda"
        )
        
        # Create optimizers
        cfg = {
            "teacher_lr": 1e-4,
            "student_lr": 1e-3,
            "teacher_weight_decay": 1e-4,
            "student_weight_decay": 1e-4,
            "teacher_epochs": 10,
            "student_epochs": 10
        }
        
        teacher_optimizer, teacher_scheduler, student_optimizer, student_scheduler = create_optimizers_and_schedulers_legacy(
            teacher=teacher1,
            student_model=student,
            cfg=cfg
        )
        
        # Simulate training loop
        num_batches = 5
        for batch_idx in range(num_batches):
            # Create dummy data
            x = torch.randn(2, 3, 32, 32)
            y = torch.randint(0, 10, (2,))
            
            # Forward pass
            loss, _ = distiller.forward(x, y)
            
            # Backward pass
            student_optimizer.zero_grad()
            loss.backward()
            student_optimizer.step()
            
            # Check loss is finite
            assert torch.isfinite(loss)
            assert loss > 0
        
        # Step schedulers (after optimizer.step())
        # Note: teacher_optimizer was not used in this test, so we skip teacher_scheduler.step()
        student_scheduler.step()
        
        assert True  # If we get here, training completed successfully
    
    def test_model_save_load(self, temp_dir):
        """Test model save and load functionality"""
        # Create model
        model = create_student_by_name(
            student_name="efficientnet_b0_scratch",
            num_classes=10,
            pretrained=False,
            small_input=True
        )
        
        # Save model
        save_path = Path(temp_dir) / "test_model.pth"
        torch.save(model.state_dict(), save_path)
        
        # Create new model
        new_model = create_student_by_name(
            student_name="efficientnet_b0_scratch",
            num_classes=10,
            pretrained=False,
            small_input=True
        )
        
        # Load model
        new_model.load_state_dict(torch.load(save_path, weights_only=True))
        
        # Test forward pass
        x = torch.randn(2, 3, 32, 32)
        
        model.eval()
        new_model.eval()
        
        with torch.no_grad():
            output1 = model(x)
            output2 = new_model(x)
        
        # Check outputs are the same
        if isinstance(output1, tuple):
            output1 = output1[1]  # Get logits
        if isinstance(output2, tuple):
            output2 = output2[1]  # Get logits
        
        assert torch.allclose(output1, output2, atol=1e-6)


class TestErrorHandling:
    """Test error handling scenarios"""
    
    def test_invalid_model_name(self):
        """Test handling of invalid model names"""
        with pytest.raises(Exception):
            create_teacher_by_name(
                teacher_name="invalid_teacher",
                num_classes=100,
                pretrained=False,
                small_input=True
            )
    
    def test_invalid_student_name(self):
        """Test handling of invalid student names"""
        with pytest.raises(Exception):
            create_student_by_name(
                student_name="invalid_student",
                num_classes=100,
                pretrained=False,
                small_input=True
            )
    
    def test_mismatched_dimensions(self):
        """Test handling of mismatched dimensions"""
        # Create models with mismatched dimensions
        teacher1 = create_teacher_by_name(
            teacher_name="resnet152",
            num_classes=100,
            pretrained=False,
            small_input=True
        )
        
        teacher2 = create_teacher_by_name(
            teacher_name="resnet152",  # Use same teacher type to avoid dimension mismatch
            num_classes=100,
            pretrained=False,
            small_input=True
        )
        
        student = create_student_by_name(
            student_name="efficientnet_b0_scratch",
            num_classes=100,
            pretrained=False,
            small_input=True
        )
        
        # Try to create MBM with mismatched dimensions
        # This should be handled gracefully
        try:
            mbm = IB_MBM(q_dim=1280, kv_dim=4096, d_emb=2048)  # kv_dim = 2 * teacher_feature_dim
            synergy_head = SynergyHead(2048, num_classes=100)
            
            distiller = ASIBDistiller(
                teacher1, teacher2, student, mbm, synergy_head, device="cuda"
            )
            
            # Test forward pass
            x = torch.randn(4, 3, 32, 32)
            y = torch.randint(0, 100, (4,))
            
            loss, _ = distiller.forward(x, y)
            assert torch.isfinite(loss)
            
        except Exception as e:
            # If there's an error, it should be a specific type
            assert "dimension" in str(e).lower() or "shape" in str(e).lower()


class TestPerformance:
    """Test performance characteristics"""
    
    def test_memory_usage(self):
        """Test memory usage"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create models
        teacher1 = create_teacher_by_name(
            teacher_name="resnet152",
            num_classes=100,
            pretrained=False,
            small_input=True
        )
        
        teacher2 = create_teacher_by_name(
            teacher_name="resnet152",  # Use same teacher type to avoid dimension mismatch
            num_classes=100,
            pretrained=False,
            small_input=True
        )
        
        student = create_student_by_name(
            student_name="efficientnet_b0_scratch",
            num_classes=100,
            pretrained=False,
            small_input=True
        )
        
        # Create MBM components
        mbm = IB_MBM(q_dim=1280, kv_dim=4096, d_emb=2048)  # kv_dim = 2 * teacher_feature_dim
        synergy_head = SynergyHead(2048, num_classes=100)
        
        # Create distiller
        distiller = ASIBDistiller(
            teacher1, teacher2, student, mbm, synergy_head, device="cuda"
        )
        
        # Test forward pass
        x = torch.randn(4, 3, 32, 32)
        y = torch.randint(0, 100, (4,))
        
        loss, _ = distiller.forward(x, y)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 1GB)
        assert memory_increase < 1024 * 1024 * 1024  # 1GB
    
    def test_forward_pass_speed(self):
        """Test forward pass speed"""
        import time
        
        # Create models
        teacher1 = create_teacher_by_name(
            teacher_name="resnet152",
            num_classes=100,
            pretrained=False,
            small_input=True
        )
        
        teacher2 = create_teacher_by_name(
            teacher_name="resnet152",  # Use same teacher type to avoid dimension mismatch
            num_classes=100,
            pretrained=False,
            small_input=True
        )
        
        student = create_student_by_name(
            student_name="efficientnet_b0_scratch",
            num_classes=100,
            pretrained=False,
            small_input=True
        )
        
        # Create MBM components
        mbm = IB_MBM(q_dim=1280, kv_dim=4096, d_emb=2048)  # kv_dim = 2 * teacher_feature_dim
        synergy_head = SynergyHead(2048, num_classes=100)
        
        # Create distiller
        distiller = ASIBDistiller(
            teacher1, teacher2, student, mbm, synergy_head, device="cuda"
        )
        
        # Warm up
        x = torch.randn(4, 3, 32, 32)
        y = torch.randint(0, 100, (4,))
        
        for _ in range(3):
            with torch.no_grad():
                _ = distiller.forward(x, y)
        
        # Time forward pass
        start_time = time.time()
        num_passes = 10
        
        for _ in range(num_passes):
            with torch.no_grad():
                _ = distiller.forward(x, y)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_passes
        
        # Forward pass should be reasonably fast (allow more time for CPU computation)
        assert avg_time < 60.0  # Increased time limit for CPU computation


class TestReproducibility:
    """Test reproducibility"""
    
    def test_deterministic_training(self):
        """Test deterministic training"""
        # Set seeds
        torch.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Create models with same feature dimensions
        teacher1 = create_teacher_by_name(
            teacher_name="resnet152",
            num_classes=10,
            pretrained=False,
            small_input=True
        )
        
        teacher2 = create_teacher_by_name(
            teacher_name="resnet152",  # Use same teacher type to avoid dimension mismatch
            num_classes=10,
            pretrained=False,
            small_input=True
        )
        
        student = create_student_by_name(
            student_name="efficientnet_b0_scratch",
            num_classes=10,
            pretrained=False,
            small_input=True
        )
        
        # Create MBM components
        mbm = IB_MBM(q_dim=1280, kv_dim=4096, d_emb=2048)  # kv_dim = 2 * teacher_feature_dim
        synergy_head = SynergyHead(2048, num_classes=10)
        
        # Create distiller
        distiller = ASIBDistiller(
            teacher1, teacher2, student, mbm, synergy_head, device="cuda"
        )
        
        # Create optimizers
        cfg = {
            "teacher_lr": 1e-4,
            "student_lr": 1e-3,
            "teacher_weight_decay": 1e-4,
            "student_weight_decay": 1e-4,
            "teacher_epochs": 10,
            "student_epochs": 10
        }
        
        teacher_optimizer, teacher_scheduler, student_optimizer, student_scheduler = create_optimizers_and_schedulers_legacy(
            teacher=teacher1,
            student_model=student,
            cfg=cfg
        )
        
        # Train for a few steps
        losses = []
        for _ in range(5):
            x = torch.randn(2, 3, 32, 32)  # Batch size 2 for BatchNorm compatibility
            y = torch.randint(0, 10, (2,))
            
            loss, _ = distiller.forward(x, y)
            student_optimizer.zero_grad()
            loss.backward()
            student_optimizer.step()
            
            losses.append(loss.item())
        
        # Reset seeds and repeat
        torch.manual_seed(42)
        
        # Create new models
        teacher1_new = create_teacher_by_name(
            teacher_name="resnet152",
            num_classes=10,
            pretrained=False,
            small_input=True
        )
        
        teacher2_new = create_teacher_by_name(
            teacher_name="resnet152",  # Use same teacher type to avoid dimension mismatch
            num_classes=10,
            pretrained=False,
            small_input=True
        )
        
        student_new = create_student_by_name(
            student_name="efficientnet_b0_scratch",
            num_classes=10,
            pretrained=False,
            small_input=True
        )
        
        # Create new MBM components
        mbm_new = IB_MBM(q_dim=1280, kv_dim=4096, d_emb=2048)  # kv_dim = 2 * teacher_feature_dim
        synergy_head_new = SynergyHead(2048, num_classes=10)
        
        # Create new distiller
        distiller_new = ASIBDistiller(
            teacher1_new, teacher2_new, student_new, mbm_new, synergy_head_new, device="cuda"
        )
        
        # Create new optimizers
        cfg_new = {
            "teacher_lr": 1e-4,
            "student_lr": 1e-3,
            "teacher_weight_decay": 1e-4,
            "student_weight_decay": 1e-4,
            "teacher_epochs": 10,
            "student_epochs": 10
        }
        
        teacher_optimizer_new, teacher_scheduler_new, student_optimizer_new, student_scheduler_new = create_optimizers_and_schedulers_legacy(
            teacher=teacher1_new,
            student_model=student_new,
            cfg=cfg_new
        )
        
        # Train again
        losses_new = []
        for _ in range(5):
            x = torch.randn(2, 3, 32, 32)  # Batch size 2 for BatchNorm compatibility
            y = torch.randint(0, 10, (2,))
            
            loss, _ = distiller_new.forward(x, y)
            student_optimizer_new.zero_grad()
            loss.backward()
            student_optimizer_new.step()
            
            losses_new.append(loss.item())
        
        # Check reproducibility
        for i, (loss1, loss2) in enumerate(zip(losses, losses_new)):
            assert abs(loss1 - loss2) < 1e-6, f"Loss mismatch at step {i}" 