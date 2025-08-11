# utils/optimization.py

"""Performance optimization utilities for ASIB."""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import logging


def enable_amp_autocast(device: str) -> bool:
    """Enable automatic mixed precision if supported."""
    if device == "cuda" and torch.cuda.is_available():
        # Check if AMP is supported
        if hasattr(torch, 'autocast'):
            return True
    return False


def optimize_memory_usage(model: nn.Module, device: str) -> None:
    """Optimize memory usage for the model."""
    if device == "cuda" and torch.cuda.is_available():
        # Enable gradient checkpointing for large models
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        # Clear cache
        torch.cuda.empty_cache()


def get_optimal_batch_size(
    model: nn.Module,
    input_shape: tuple,
    device: str,
    max_memory_gb: float = 24.0
) -> int:
    """Calculate optimal batch size based on available memory."""
    if device != "cuda" or not torch.cuda.is_available():
        return 32  # Default for CPU
    
    try:
        # Get available GPU memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        # Estimate memory per sample (rough approximation)
        with torch.no_grad():
            dummy_input = torch.randn(1, *input_shape[1:], device=device)
            model.eval()
            
            # Forward pass to estimate memory
            torch.cuda.reset_peak_memory_stats()
            _ = model(dummy_input)
            memory_per_sample = torch.cuda.max_memory_allocated() / 1024**3
            
            # Calculate optimal batch size
            available_memory = min(gpu_memory * 0.8, max_memory_gb)  # Use 80% of GPU memory
            optimal_batch_size = int(available_memory / memory_per_sample)
            
            # Clamp to reasonable range
            optimal_batch_size = max(1, min(optimal_batch_size, 128))
            
            logging.info(f"Optimal batch size: {optimal_batch_size}")
            return optimal_batch_size
            
    except Exception as e:
        logging.warning(f"Failed to calculate optimal batch size: {e}")
        return 32


def profile_model(
    model: nn.Module,
    input_shape: tuple,
    device: str,
    num_runs: int = 10
) -> Dict[str, float]:
    """Profile model performance."""
    model.eval()
    model.to(device)
    
    # Warm up
    with torch.no_grad():
        dummy_input = torch.randn(1, *input_shape[1:], device=device)
        for _ in range(3):
            _ = model(dummy_input)
    
    # Profile
    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    times = []
    for _ in range(num_runs):
        start_time.record()
        with torch.no_grad():
            _ = model(dummy_input)
        end_time.record()
        torch.cuda.synchronize()
        times.append(start_time.elapsed_time(end_time))
    
    avg_time = sum(times) / len(times)
    
    return {
        "avg_inference_time_ms": avg_time,
        "min_time_ms": min(times),
        "max_time_ms": max(times),
        "std_time_ms": torch.std(torch.tensor(times)).item()
    }


def optimize_data_loader(
    num_workers: int,
    pin_memory: bool,
    device: str
) -> Dict[str, Any]:
    """Optimize data loader settings."""
    if device == "cuda" and torch.cuda.is_available():
        # Optimize for GPU
        return {
            "num_workers": min(num_workers, 4),  # Limit workers
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 2
        }
    else:
        # Optimize for CPU
        return {
            "num_workers": 0,  # No multiprocessing for CPU
            "pin_memory": False,
            "persistent_workers": False,
            "prefetch_factor": 2
        }


def gradient_accumulation_steps(
    target_batch_size: int,
    actual_batch_size: int
) -> int:
    """Calculate gradient accumulation steps."""
    if actual_batch_size >= target_batch_size:
        return 1
    
    steps = target_batch_size // actual_batch_size
    if target_batch_size % actual_batch_size != 0:
        steps += 1
    
    return steps


def setup_optimization_flags(device: str) -> None:
    """Setup optimization flags for better performance."""
    if device == "cuda" and torch.cuda.is_available():
        # Enable cuDNN benchmarking for better performance
        torch.backends.cudnn.benchmark = True
        
        # Enable cuDNN deterministic mode for reproducibility
        torch.backends.cudnn.deterministic = False
        
        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(0.9)


def monitor_memory_usage(device: str) -> Dict[str, float]:
    """Monitor memory usage."""
    if device == "cuda" and torch.cuda.is_available():
        return {
            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
            "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
            "max_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2
        }
    else:
        return {}


def clear_memory_cache(device: str) -> None:
    """Clear memory cache."""
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def optimize_model_for_inference(model: nn.Module) -> nn.Module:
    """Optimize model for inference."""
    model.eval()
    
    # Enable JIT compilation if possible
    try:
        if hasattr(model, 'script'):
            model = torch.jit.script(model)
        elif hasattr(model, 'trace'):
            # Create dummy input for tracing
            dummy_input = torch.randn(1, 3, 224, 224)
            model = torch.jit.trace(model, dummy_input)
    except Exception as e:
        logging.warning(f"Failed to compile model with JIT: {e}")
    
    return model


def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in MB."""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def count_parameters(model: nn.Module, trainable_only: bool = True) -> Dict[str, int]:
    """Count model parameters."""
    total_params = 0
    trainable_params = 0
    
    for param in model.parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params if trainable_only else total_params,
        "non_trainable_parameters": total_params - trainable_params
    } 