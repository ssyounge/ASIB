# modules/losses.py

import torch
import torch.nn.functional as F

def ce_loss_fn(student_logits, labels, label_smoothing: float = 0.0):
    """Standard cross-entropy loss for classification.

    Parameters
    ----------
    student_logits : Tensor
        Model predictions of shape ``(N, C)``.
    labels : Tensor
        Ground-truth integer labels.
    label_smoothing : float, optional
        Amount of label smoothing to apply. ``0.0`` disables smoothing.
    """
    return F.cross_entropy(
        student_logits,
        labels,
        label_smoothing=label_smoothing,
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
