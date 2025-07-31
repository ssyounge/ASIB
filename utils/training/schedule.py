# utils/schedule.py

from typing import Optional


def get_tau(cfg: dict, epoch: int, total_epochs: Optional[int] = None) -> float:
    """Polynomial temperature decay.

    The schedule is defined as::

        τ(e) = τ_end + (τ_start - τ_end) · (1 - e/E)^p

    Parameters
    ----------
    cfg : dict
        Must provide ``tau_start``. If ``tau_end`` or ``tau_decay_power`` are
        missing the temperature remains fixed.
    epoch : int
        Current epoch index (0-based).
    total_epochs : int, optional
        Length ``E`` of the schedule. If omitted, ``cfg['T_max']`` then
        ``cfg['total_epochs']`` are consulted. If none are present ``E = 1`` and
        decay is disabled.
    """

    tau_start = float(cfg.get("tau_start", 4.0))
    tau_end = float(cfg.get("tau_end", tau_start))
    power = float(cfg.get("tau_decay_power", 1.0))

    if total_epochs is None:
        total_epochs = int(cfg.get("T_max", cfg.get("total_epochs", 1)))
    total_epochs = max(total_epochs, 1)

    progress = min(epoch / total_epochs, 1.0)
    return tau_end + (tau_start - tau_end) * (1.0 - progress) ** power


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
