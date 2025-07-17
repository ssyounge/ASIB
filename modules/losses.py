# modules/losses.py
from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["kd_kl", "kd_loss_fn", "ce_loss_fn", "dkd_loss", "compute_vib_loss"]


def kd_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    T: float = 1.0,
    task_classes: list[int] | None = None,
) -> torch.Tensor:
    """Knowledge distillation KL loss with optional task masking."""
    if task_classes is not None:
        idx = torch.tensor(task_classes, device=student_logits.device)
        student_logits = student_logits.index_select(1, idx)
        teacher_logits = teacher_logits.index_select(1, idx)

    log_s = F.log_softmax(student_logits / T, dim=1)
    soft_t = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(log_s, soft_t, reduction="batchmean") * (T * T)


def kd_loss_fn(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    T: float = 1.0,
    task_classes: list[int] | None = None,
) -> torch.Tensor:
    """Backward compatibility wrapper around :func:`kd_kl`."""
    return kd_kl(student_logits, teacher_logits, T=T, task_classes=task_classes)


def ce_loss_fn(logits: torch.Tensor, labels: torch.Tensor,
               label_smoothing: float = 0.0) -> torch.Tensor:
    """
    Cross‑entropy that 자동판별:
      • labels int   → 기존 방식
      • labels float → one‑hot / mixup
    """
    # soft‑label(=MixUp/one‑hot) ↔ int label 모두 지원
    if labels.dtype.is_floating_point:
        log_p = F.log_softmax(logits, dim=1)
        return -(labels * log_p).sum(dim=1).mean()
    return F.cross_entropy(
        logits,
        labels,
        label_smoothing=label_smoothing,
    )


def dkd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor,
             alpha: float = 1.0, beta: float = 1.0, temperature: float = 4.0) -> torch.Tensor:
    """Simplified DKD loss combining KD and CE terms."""
    kd = kd_kl(student_logits, teacher_logits, T=temperature)
    ce = ce_loss_fn(student_logits, labels)
    return alpha * kd + beta * ce


def compute_vib_loss(latent: torch.Tensor) -> torch.Tensor:
    """Simple VIB regularization term for latent features."""
    return torch.mean(latent ** 2)
