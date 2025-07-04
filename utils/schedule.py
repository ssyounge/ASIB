# utils/schedule.py

import math
import torch


def get_tau(cfg: dict, epoch: int) -> float:
    """Return τ at a given *epoch* according to cfg."""
    sched = cfg.get("temperature_schedule", "fixed").lower()
    T0 = cfg.get("tau_start", 4.0)
    Tend = cfg.get("tau_end", 1.0)
    decay = max(int(cfg.get("tau_decay_epochs", 1)), 1)
    t = min(epoch, decay) / decay

    if sched in ("linear", "linear_decay"):
        tau = T0 + (Tend - T0) * t
    elif sched == "cosine":
        tau = Tend + 0.5 * (T0 - Tend) * (1 + math.cos(math.pi * t))
    else:  # fixed
        tau = T0
    return float(tau)


def cosine_lr_scheduler(
    optimizer,
    total_epochs,
    warmup_epochs: int = 0,
    min_lr_ratio: float = 0.05,
):
    def lr_lambda(cur_epoch):
        if cur_epoch < warmup_epochs:
            return (cur_epoch + 1) / warmup_epochs
        t = (cur_epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine = 0.5 * (1 + math.cos(math.pi * t))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
