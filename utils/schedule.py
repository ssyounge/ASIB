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


def cosine_lr_scheduler(optimizer, iters):
    """Return cosine scheduler over ``iters`` epochs."""
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters)
