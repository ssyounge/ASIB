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
    "rand_bbox",
]


def set_random_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def get_amp_components(cfg: dict):
    """Return autocast context and scaler based on config."""
    from contextlib import nullcontext

    try:
        from torch.amp import GradScaler, autocast
    except Exception:  # pragma: no cover - fallback for old PyTorch
        from torch.cuda.amp import GradScaler, autocast

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


def rand_bbox(size, lam):
    """Generate a random square bounding box."""
    W = size[-1]
    H = size[-2]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2
