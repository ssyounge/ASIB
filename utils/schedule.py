import math


def get_tau(cfg: dict, epoch: int) -> float:
    """Return τ at a given *epoch* according to cfg."""
    sched = cfg.get("temperature_schedule", "fixed").lower()
    T0 = cfg.get("tau_start", 4.0)
    Tend = cfg.get("tau_end", 1.0)
    decay = max(int(cfg.get("tau_decay_epochs", 1)), 1)
    t = min(epoch, decay) / decay

    if sched == "linear":
        tau = T0 + (Tend - T0) * t
    elif sched == "cosine":
        tau = Tend + 0.5 * (T0 - Tend) * (1 + math.cos(math.pi * t))
    else:  # fixed
        tau = T0
    return float(tau)
