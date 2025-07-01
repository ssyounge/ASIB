# modules/losses.py

import torch
import torch.nn.functional as F
from typing import Optional

def ce_loss_fn(student_logits, labels, label_smoothing: float = 0.0, reduction: str = "mean"):
    """Standard cross-entropy loss for classification.

    Parameters
    ----------
    student_logits : Tensor
        Model predictions of shape ``(N, C)`` or higher. If the logits have
        more than two dimensions, their spatial dimensions are averaged before
        computing the loss.
    labels : Tensor
        Ground-truth integer labels.
    label_smoothing : float, optional
        Amount of label smoothing to apply. ``0.0`` disables smoothing.
    """
    if student_logits.dim() > 2:
        student_logits = student_logits.mean(dim=tuple(range(2, student_logits.dim())))
    return F.cross_entropy(
        student_logits,
        labels,
        label_smoothing=label_smoothing,
        reduction=reduction,
    )

def kd_loss_fn(student_logits, teacher_logits, T=4.0, reduction="batchmean"):
    """
    KL-divergence based KD loss (Hinton et al., 2015).
    
    student_logits: (N, num_classes)
    teacher_logits: (N, num_classes)
    T: temperature
    reduction: typically "batchmean" (PyTorch default for KL)

    Returns scalar tensor (KD loss).
    """
    # average spatial dimensions if present
    if student_logits.dim() > 2:
        student_logits = student_logits.mean(dim=tuple(range(2, student_logits.dim())))
    if teacher_logits.dim() > 2:
        teacher_logits = teacher_logits.mean(dim=tuple(range(2, teacher_logits.dim())))

    # student prob (with log) under temperature
    s_log_probs = F.log_softmax(student_logits / T, dim=1)
    # teacher prob under temperature
    t_probs = F.softmax(teacher_logits / T, dim=1)

    # KL-div * T^2
    kl_div = F.kl_div(s_log_probs, t_probs, reduction=reduction) * (T * T)
    return kl_div

def hybrid_kd_loss_fn(student_logits, teacher_logits, labels, alpha=0.5, T=4.0):
    """
    Optionally provide a simple "CE + KD" combo in one function.
    total_loss = alpha*CE + (1-alpha)*KD
    """
    ce = ce_loss_fn(student_logits, labels)
    kd = kd_loss_fn(student_logits, teacher_logits, T=T)
    return alpha * ce + (1 - alpha) * kd


def dkd_loss(student_logits, teacher_logits, labels, alpha=1.0, beta=1.0, temperature=4.0):
    """Decoupled Knowledge Distillation loss."""
    if student_logits.dim() > 2:
        student_logits = student_logits.mean(dim=tuple(range(2, student_logits.dim())))
    if teacher_logits.dim() > 2:
        teacher_logits = teacher_logits.mean(dim=tuple(range(2, teacher_logits.dim())))

    s_logits = student_logits / temperature
    t_logits = teacher_logits / temperature

    s_probs = F.softmax(s_logits, dim=1)
    t_probs = F.softmax(t_logits, dim=1)

    num_classes = s_probs.size(1)
    one_hot = F.one_hot(labels, num_classes=num_classes).float()
    pos_mask = one_hot
    neg_mask = 1.0 - one_hot

    # positive (ground truth class)
    s_pos = (s_probs * pos_mask).sum(dim=1, keepdim=True)
    t_pos = (t_probs * pos_mask).sum(dim=1, keepdim=True)
    loss_pos = F.kl_div(torch.log(s_pos + 1e-12), t_pos, reduction="batchmean")

    # negative classes
    s_neg = s_probs * neg_mask
    t_neg = t_probs * neg_mask
    s_neg = s_neg / s_neg.sum(dim=1, keepdim=True)
    t_neg = t_neg / t_neg.sum(dim=1, keepdim=True)
    loss_neg = F.kl_div(torch.log(s_neg + 1e-12), t_neg, reduction="batchmean")

    loss = (alpha * loss_pos + beta * loss_neg) * (temperature ** 2)
    return loss



