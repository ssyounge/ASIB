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
    inputs: [N, C, H, W]
    targets: [N]
    alpha>0 => cutmix 적용
    alpha<=0 => cutmix 비활성 (return 그대로)
    ...
    indices = torch.randperm(batch_size, device=inputs.device)
    lam = random.betavariate(alpha, alpha)
    ...
    inputs_clone = inputs.clone()
    inputs_clone[:, :, x1:x2, y1:y2] = inputs[indices, :, x1:x2, y1:y2]
    ...
    return inputs_clone, target_a, target_b, lam
