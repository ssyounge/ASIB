# utils/misc.py

import random
import sys

import numpy as np
import torch

__all__ = [
    "set_random_seed",
    "get_amp_components",
    "mixup_data",
    "mixup_criterion",
]


def set_random_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_amp_components(cfg: dict):
    """Return autocast context and scaler based on config."""
    from contextlib import nullcontext

    from torch.amp import GradScaler, autocast

    use_amp = cfg.get("use_amp", False)
    init_scale = cfg.get("grad_scaler_init_scale", 1024)
    if use_amp and torch.cuda.is_available():
        dtype = cfg.get("amp_dtype", "float16")
        if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            dtype = "float16"
        autocast_ctx = autocast(
            device_type="cuda", dtype=getattr(torch, dtype, torch.float16)
        )
        scaler = GradScaler(init_scale=init_scale)
    else:
        autocast_ctx = nullcontext()
        scaler = None
    return autocast_ctx, scaler


def mixup_data(inputs: torch.Tensor, targets: torch.Tensor, alpha: float = 1.0):
    """Apply MixUp to a batch of inputs and targets."""
    if alpha <= 0.0:
        return inputs, targets, targets, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size = inputs.size(0)
    index = torch.randperm(batch_size, device=inputs.device)
    mixed = lam * inputs + (1.0 - lam) * inputs[index]
    targets_a = targets
    targets_b = targets[index]
    return mixed, targets_a, targets_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute loss for MixUp inputs."""
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)
