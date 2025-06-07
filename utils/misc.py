# utils/misc.py

import os
import torch
import random
import numpy as np

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
    """Apply CutMix augmentation.

    Parameters
    ----------
    inputs : Tensor
        Input tensor of shape ``(N, C, H, W)``.
    targets : Tensor
        Labels tensor of shape ``(N,)``.
    alpha : float, optional
        Beta distribution parameter. ``alpha <= 0`` disables CutMix.
    """
    if alpha <= 0.0:
        return inputs, targets, targets, 1.0

    batch_size = inputs.size(0)
    indices = torch.randperm(batch_size, device=inputs.device)
    lam = random.betavariate(alpha, alpha)

    W, H = inputs.size(2), inputs.size(3)
    cut_w = int(W * (1 - lam))
    cut_h = int(H * (1 - lam))
    cx = random.randint(0, W)
    cy = random.randint(0, H)

    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)

    inputs_clone = inputs.clone()
    inputs_clone[:, :, x1:x2, y1:y2] = inputs[indices, :, x1:x2, y1:y2]

    lam = 1.0 - ((x2 - x1) * (y2 - y1) / (W * H))
    target_a = targets
    target_b = targets[indices]
    return inputs_clone, target_a, target_b, lam


def mixup_data(inputs, targets, alpha=1.0):
    """Apply MixUp augmentation.

    Parameters
    ----------
    inputs : Tensor
        Input tensor of shape ``(N, C, H, W)``.
    targets : Tensor
        Ground-truth labels of shape ``(N,)``.
    alpha : float, optional
        MixUp beta distribution parameter. ``alpha <= 0`` disables MixUp.

    Returns
    -------
    Tuple[Tensor, Tensor, Tensor, float]
        Mixed inputs, first set of labels, second set of labels and mix
        coefficient ``lam``.
    """
    if alpha <= 0.0:
        return inputs, targets, targets, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size = inputs.size(0)
    index = torch.randperm(batch_size, device=inputs.device)

    mixed_inputs = lam * inputs + (1.0 - lam) * inputs[index]
    targets_a = targets
    targets_b = targets[index]
    return mixed_inputs, targets_a, targets_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute the criterion for mixed targets."""
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)
