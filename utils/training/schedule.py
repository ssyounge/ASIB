# utils/schedule.py

from typing import Optional


def get_tau(cfg: dict, epoch: int, total_epochs: Optional[int] = None) -> float:
    """Return temperature τ for KD.

    Priority:
      1) If ``cfg['tau']`` is set, return it as a fixed temperature.
      2) Else, if ``cfg['tau_schedule']`` is present, interpret it as
         ``[tau_start, tau_end]`` and use polynomial decay with
         ``cfg['tau_decay_power']`` (default 1.0).
      3) Else, fall back to legacy keys ``tau_start``/``tau_end`` with the same
         polynomial form. If only ``tau_start`` is provided, temperature stays fixed.

    The polynomial schedule is:
        τ(e) = τ_end + (τ_start - τ_end) · (1 - e/E)^p
    where ``E`` is ``total_epochs``.
    """

    # 1) Fixed tau if explicitly provided
    if "tau" in cfg:
        try:
            return float(cfg["tau"])
        except Exception:
            pass

    # 2) Schedule via two-value list [start, end]
    if "tau_schedule" in cfg:
        sched = cfg.get("tau_schedule")
        if isinstance(sched, (list, tuple)) and len(sched) >= 2:
            tau_start = float(sched[0])
            tau_end = float(sched[1])
            power = float(cfg.get("tau_decay_power", 1.0))
        else:
            # Fallback to fixed if malformed
            return float(sched[0]) if isinstance(sched, (list, tuple)) and len(sched) >= 1 else float(cfg.get("tau_start", 4.0))
    else:
        # 3) Legacy keys
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
