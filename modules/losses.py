# modules/losses.py

import torch
import torch.nn.functional as F

__all__ = ["kd_loss_fn", "ce_loss_fn", "dkd_loss"]


def kd_loss_fn(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    """Basic knowledge distillation loss (KL divergence)."""
    log_s = F.log_softmax(student_logits / T, dim=1)
    soft_t = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(log_s, soft_t, reduction="batchmean") * (T * T)


def ce_loss_fn(logits: torch.Tensor, labels: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    """Cross-entropy loss with optional label smoothing."""
    return F.cross_entropy(logits, labels, label_smoothing=label_smoothing)


def dkd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor,
             alpha: float = 1.0, beta: float = 1.0, temperature: float = 4.0) -> torch.Tensor:
    """Simplified DKD loss combining KD and CE terms."""
    kd = kd_loss_fn(student_logits, teacher_logits, T=temperature)
    ce = ce_loss_fn(student_logits, labels)
    return alpha * kd + beta * ce
