# utils/misc.py

import random
import sys

import numpy as np
import torch

__all__ = ["set_random_seed", "get_amp_components"]


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
    if use_amp and torch.cuda.is_available():
        dtype = cfg.get("amp_dtype", "float16")
        if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            dtype = "float16"
        autocast_ctx = autocast(
            device_type="cuda", dtype=getattr(torch, dtype, torch.float16)
        )
        scaler = GradScaler(init_scale=cfg.get("grad_scaler_init_scale", 1024))
    else:
        autocast_ctx = nullcontext()
        scaler = None
    return autocast_ctx, scaler
