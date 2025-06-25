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
    """Apply CutMix to a batch of inputs/targets.

    Args:
        inputs (Tensor): input images of shape ``[N, C, H, W]``.
        targets (Tensor): corresponding labels ``[N]``.
        alpha (float): beta distribution parameter. ``alpha<=0`` disables CutMix.

    Returns:
        Tuple of ``(inputs_clone, target_a, target_b, lam)`` where ``inputs_clone``
        contains mixed images, ``target_a`` and ``target_b`` are the original and
        permuted targets and ``lam`` is the mixing coefficient.
    """
    if alpha <= 0.0:
        # no cutmix
        return inputs, targets, targets, 1.0

    batch_size = inputs.size(0)
    indices = torch.randperm(batch_size, device=inputs.device)
    lam = random.betavariate(alpha, alpha)

    # crop box
    W, H = inputs.size(2), inputs.size(3)
    cut_w = int(W * (1 - lam))
    cut_h = int(H * (1 - lam))
    cx = random.randint(0, W)
    cy = random.randint(0, H)

    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)

    # clone and apply patch
    inputs_clone = inputs.clone()
    inputs_clone[:, :, x1:x2, y1:y2] = inputs[indices, :, x1:x2, y1:y2]

    lam = 1.0 - ((x2 - x1) * (y2 - y1) / (W * H))

    target_a = targets
    target_b = targets[indices]
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
