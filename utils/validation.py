# utils/validation.py

"""Validation utilities for ASIB."""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
import logging
from utils.exceptions import ValidationError, ConfigurationError


def validate_config(cfg: Dict[str, Any]) -> None:
    """Validate configuration dictionary."""
    required_keys = [
        "num_stages",
        "teacher_lr",
        "student_lr",
        "device",
        "batch_size"
    ]
    
    for key in required_keys:
        if key not in cfg:
            raise ConfigurationError(f"Missing required config key: {key}")
    
    # Validate numeric values
    if cfg["num_stages"] <= 0:
        raise ConfigurationError("num_stages must be positive")
    
    if cfg["teacher_lr"] <= 0 or cfg["student_lr"] <= 0:
        raise ConfigurationError("Learning rates must be positive")
    
    if cfg["batch_size"] <= 0:
        raise ConfigurationError("batch_size must be positive")
    
    # Validate device
    if cfg["device"] not in ["cuda", "cpu"]:
        raise ConfigurationError("device must be 'cuda' or 'cpu'")
    
    # Validate ASIB specific parameters
    if cfg.get("use_ib", False):
        if "ib_beta" not in cfg:
            raise ConfigurationError("ib_beta is required when use_ib=True")
        if cfg["ib_beta"] < 0:
            raise ConfigurationError("ib_beta must be non-negative")
    
    # Validate IB_MBM parameters (legacy keys removed)
    qd = cfg.get("ib_mbm_query_dim")
    od = cfg.get("ib_mbm_out_dim")
    nh = cfg.get("ib_mbm_n_head")
    if qd is not None and qd <= 0:
        raise ConfigurationError("ib_mbm_query_dim must be positive")
    if od is not None and od <= 0:
        raise ConfigurationError("ib_mbm_out_dim must be positive")
    if nh is not None and nh <= 0:
        raise ConfigurationError("ib_mbm_n_head must be positive")


def validate_model(model: nn.Module, input_shape: Tuple[int, ...]) -> None:
    """Validate model architecture."""
    if not isinstance(model, nn.Module):
        raise ValidationError("Model must be a torch.nn.Module")
    
    # Check if model has required methods
    if not hasattr(model, 'forward'):
        raise ValidationError("Model must have a forward method")
    
    # Test forward pass
    try:
        model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, *input_shape[1:])
            output = model(dummy_input)
            
            if output is None:
                raise ValidationError("Model forward pass returned None")
            
            # Check if output is a tensor or tuple of tensors
            if isinstance(output, (list, tuple)):
                for i, out in enumerate(output):
                    if not isinstance(out, torch.Tensor):
                        raise ValidationError(f"Model output {i} is not a tensor")
            elif not isinstance(output, torch.Tensor):
                raise ValidationError("Model output is not a tensor")
                
    except Exception as e:
        raise ValidationError(f"Model forward pass failed: {e}")


def validate_data_loader(
    data_loader,
    expected_batch_size: int,
    expected_num_classes: int
) -> None:
    """Validate data loader."""
    if not hasattr(data_loader, '__iter__'):
        raise ValidationError("Data loader must be iterable")
    
    # Check first batch
    try:
        batch = next(iter(data_loader))
        if len(batch) != 2:
            raise ValidationError("Data loader must return (data, labels) pairs")
        
        data, labels = batch
        
        if not isinstance(data, torch.Tensor):
            raise ValidationError("Data must be a tensor")
        
        if not isinstance(labels, torch.Tensor):
            raise ValidationError("Labels must be a tensor")
        
        if data.size(0) != expected_batch_size:
            raise ValidationError(f"Expected batch size {expected_batch_size}, got {data.size(0)}")
        
        if labels.max() >= expected_num_classes:
            raise ValidationError(f"Label values exceed expected number of classes {expected_num_classes}")
        
        if labels.min() < 0:
            raise ValidationError("Label values must be non-negative")
            
    except StopIteration:
        raise ValidationError("Data loader is empty")
    except Exception as e:
        raise ValidationError(f"Data loader validation failed: {e}")


def validate_optimizer(optimizer, model: nn.Module) -> None:
    """Validate optimizer configuration."""
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise ValidationError("Optimizer must be a torch.optim.Optimizer")
    
    # Check if optimizer has model parameters
    optimizer_params = set()
    for param_group in optimizer.param_groups:
        optimizer_params.update(param_group['params'])
    
    model_params = set(model.parameters())
    
    if not optimizer_params.issubset(model_params):
        raise ValidationError("Optimizer contains parameters not in model")


def validate_loss_function(loss_fn, device: str) -> None:
    """Validate loss function."""
    if not callable(loss_fn):
        raise ValidationError("Loss function must be callable")
    
    # Test loss function
    try:
        dummy_output = torch.randn(2, 10, device=device)
        dummy_target = torch.randint(0, 10, (2,), device=device)
        
        loss = loss_fn(dummy_output, dummy_target)
        
        if not isinstance(loss, torch.Tensor):
            raise ValidationError("Loss function must return a tensor")
        
        if loss.dim() != 0:
            raise ValidationError("Loss function must return a scalar tensor")
        
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValidationError("Loss function returned NaN or Inf")
            
    except Exception as e:
        raise ValidationError(f"Loss function validation failed: {e}")


def validate_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Validate checkpoint file."""
    import os
    
    if not os.path.exists(checkpoint_path):
        raise ValidationError(f"Checkpoint file not found: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if not isinstance(checkpoint, dict):
            raise ValidationError("Checkpoint must be a dictionary")
        
        # Check for required keys
        if 'state_dict' not in checkpoint:
            raise ValidationError("Checkpoint must contain 'state_dict'")
        
        if not isinstance(checkpoint['state_dict'], dict):
            raise ValidationError("Checkpoint state_dict must be a dictionary")
        
        return checkpoint
        
    except Exception as e:
        raise ValidationError(f"Failed to load checkpoint: {e}")


def validate_device(device: str) -> str:
    """Validate and normalize device specification."""
    if device == "cuda":
        if not torch.cuda.is_available():
            logging.warning("CUDA requested but not available, falling back to CPU")
            return "cpu"
        return "cuda"
    elif device == "cpu":
        return "cpu"
    else:
        raise ValidationError(f"Invalid device: {device}. Must be 'cuda' or 'cpu'")


def validate_hyperparameters(cfg: Dict[str, Any]) -> None:
    """Validate hyperparameters."""
    # Learning rates
    if cfg.get("teacher_lr", 0) <= 0:
        raise ValidationError("teacher_lr must be positive")
    
    if cfg.get("student_lr", 0) <= 0:
        raise ValidationError("student_lr must be positive")
    
    # Weight decay
    if cfg.get("teacher_weight_decay", 0) < 0:
        raise ValidationError("teacher_weight_decay must be non-negative")
    
    if cfg.get("student_weight_decay", 0) < 0:
        raise ValidationError("student_weight_decay must be non-negative")
    
    # Loss weights
    if cfg.get("ce_alpha", 0) < 0:
        raise ValidationError("ce_alpha must be non-negative")
    
    if cfg.get("kd_alpha", 0) < 0:
        raise ValidationError("kd_alpha must be non-negative")
    
    if cfg.get("ib_beta", 0) < 0:
        raise ValidationError("ib_beta must be non-negative")
    
    # IB_MBM parameters (legacy keys removed)
    qd = cfg.get("ib_mbm_query_dim", 0)
    od = cfg.get("ib_mbm_out_dim", 0)
    nh = cfg.get("ib_mbm_n_head", 0)
    if qd <= 0:
        raise ValidationError("ib_mbm_query_dim must be positive")
    if od <= 0:
        raise ValidationError("ib_mbm_out_dim must be positive")
    if nh <= 0:
        raise ValidationError("ib_mbm_n_head must be positive")
    if (od % max(1, nh)) != 0:
        raise ValidationError("ib_mbm_out_dim must be divisible by ib_mbm_n_head")


def validate_training_setup(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader,
    device: str
) -> None:
    """Validate complete training setup."""
    logging.info("Validating training setup...")
    
    # Validate device
    device = validate_device(device)
    
    # Validate model
    validate_model(model, (1, 3, 224, 224))  # Assuming standard input shape
    
    # Validate optimizer
    validate_optimizer(optimizer, model)
    
    # Validate data loader
    validate_data_loader(data_loader, data_loader.batch_size, 100)  # Assuming CIFAR-100
    
    logging.info("Training setup validation passed")


def check_gradient_flow(model: nn.Module) -> Dict[str, float]:
    """Check gradient flow in the model."""
    gradients = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            gradients[name] = grad_norm
    
    return gradients


def validate_model_outputs(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    expected_output_shape: Optional[Tuple[int, ...]] = None
) -> None:
    """Validate model outputs."""
    model.eval()
    
    with torch.no_grad():
        dummy_input = torch.randn(1, *input_shape[1:])
        output = model(dummy_input)
        
        if isinstance(output, (list, tuple)):
            for i, out in enumerate(output):
                if not isinstance(out, torch.Tensor):
                    raise ValidationError(f"Output {i} is not a tensor")
                
                if expected_output_shape and out.shape[1:] != expected_output_shape[1:]:
                    raise ValidationError(f"Output {i} shape mismatch: expected {expected_output_shape[1:]}, got {out.shape[1:]}")
        else:
            if not isinstance(output, torch.Tensor):
                raise ValidationError("Model output is not a tensor")
            
            if expected_output_shape and output.shape[1:] != expected_output_shape[1:]:
                raise ValidationError(f"Output shape mismatch: expected {expected_output_shape[1:]}, got {output.shape[1:]}") 