# utils/misc.py

import os
import torch
import random
import numpy as np
import yaml

def set_random_seed(seed=42):
    """
    Fix random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_checkpoint(model, optimizer, epoch, save_path="checkpoint.pth"):
    """
    Save model & optimizer state_dict, plus metadata (epoch).
    """
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict()
    }
    torch.save(ckpt, save_path)

def load_checkpoint(model, optimizer, load_path):
    """
    Load model & optimizer from checkpoint file.
    """
    if not os.path.exists(load_path):
        print(f"[Warning] No checkpoint found at {load_path}")
        return 0  # or -1, indicating failure

    ckpt = torch.load(load_path)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optim_state"])
    start_epoch = ckpt["epoch"]
    return start_epoch

def cutmix_data(inputs, targets, alpha=1.0):
    """
    Simple example of CutMix, if you want it here.
    """
    if alpha <= 0.0:
        return inputs, targets, targets, 1.0
    # ... (이전에도 썼던 cutmix 코드)
    # ...
    return inputs_clone, target_a, target_b, lam


def parse_override_str(override: str):
    """Parse comma-separated KEY=VAL pairs into a dictionary."""
    overrides = {}
    if not override:
        return overrides
    for pair in override.split(','):
        if '=' not in pair:
            continue
        key, val = pair.split('=', 1)
        key = key.strip()
        val = val.strip()
        if not key:
            continue
        try:
            overrides[key] = yaml.safe_load(val)
        except Exception:
            overrides[key] = val
    return overrides
