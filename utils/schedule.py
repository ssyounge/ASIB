# utils/schedule.py

def get_tau(cfg: dict, epoch: int) -> float:
    """
    Polynomial temperature decay.

    The schedule follows:
      τ(e) = τ_end + (τ_start - τ_end) · (1 - e/E)^p

    where ``E`` is the total number of epochs (``T_max`` or ``total_epochs``)
    and ``p`` controls the curvature.
    """

    tau_start = float(cfg.get("tau_start", 4.0))
    tau_end = float(cfg.get("tau_end", tau_start))  # decay off if missing
    power = float(cfg.get("tau_decay_power", 1.0))
    total_ep = max(1, cfg.get("T_max", cfg.get("total_epochs", 1)))
    prog = min(epoch / total_ep, 1.0)
    return tau_end + (tau_start - tau_end) * (1.0 - prog) ** power


def get_beta(cfg: dict, epoch: int = 0) -> float:
    """Return β for the Information Bottleneck KL term at ``epoch``.

    Supports either a fixed value (``cfg.ib_beta``), a linear warmup via
    ``cfg.ib_beta_warmup_epochs`` or an explicit schedule specified by
    ``cfg.beta_schedule`` as ``[start, end, total_epoch]``.
    """

    if "beta_schedule" in cfg:
        start, end, total = cfg["beta_schedule"]
        t = min(epoch / max(1, total), 1.0)
        beta = start + t * (end - start)
    else:
        beta = float(cfg.get("ib_beta", 1e-3))
        warmup = int(cfg.get("ib_beta_warmup_epochs", 0))
        if warmup > 0:
            scale = min(float(epoch) / warmup, 1.0)
            beta *= scale
    return beta
