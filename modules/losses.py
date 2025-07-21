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

    # half-precision overflow 방지
    student_logits = student_logits.float()
    teacher_logits = teacher_logits.float()

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


# ---------- Information Bottleneck ----------
def ib_loss(mu, logvar, beta: float = 1e-3):
    r"""Return β · KL\big(N(μ,σ²) \| N(0, 1)\big).

    fp16 under-/overflow 방지를 위해 float32 캐스팅 후 clipping을 적용한다."""  # <- r-string 로 SyntaxWarning 제거

    mu = mu.float()
    logvar = torch.clamp(logvar.float(), -10.0, 10.0)

    kl_elem = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return beta * kl_elem.mean()


def certainty_weights(logvar: torch.Tensor) -> torch.Tensor:
    """Return per-element certainty weights ``1 / exp(logvar)``.

    Parameters
    ----------
    logvar : Tensor
        Log variance tensor from the Information Bottleneck module.

    Returns
    -------
    Tensor
        ``1 / exp(logvar)`` with the same shape as ``logvar``.
    """

    return 1.0 / logvar.exp()


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


def rkd_distance_loss(
    student_feat,
    teacher_feat,
    eps: float = 1e-12,
    reduction: str = "mean",
    max_clip: Optional[float] = None,
):
    """Relational KD distance loss.

    Parameters
    ----------
    reduction : str, optional
        "mean" to return a scalar or "none" to return per-sample losses.
    """
    if student_feat.dim() > 2:
        student_feat = student_feat.view(student_feat.size(0), -1)
    if teacher_feat.dim() > 2:
        teacher_feat = teacher_feat.view(teacher_feat.size(0), -1)

    if student_feat.size(0) < 2:
        return torch.tensor(0.0, device=student_feat.device)

    diff_s = student_feat.unsqueeze(0) - student_feat.unsqueeze(1)
    diff_t = teacher_feat.unsqueeze(0) - teacher_feat.unsqueeze(1)

    dist_s = diff_s.pow(2).sum(dim=2).sqrt()
    dist_t = diff_t.pow(2).sum(dim=2).sqrt()

    pos_s = dist_s > 0
    pos_t = dist_t > 0

    if pos_s.any():
        mean_s = dist_s[pos_s].mean()
    else:
        mean_s = dist_s.new_tensor(1.0)

    if pos_t.any():
        mean_t = dist_t[pos_t].mean()
    else:
        mean_t = dist_t.new_tensor(1.0)

    dist_s = dist_s / (mean_s + eps)
    dist_t = dist_t / (mean_t + eps)

    if reduction == "none":
        losses = F.smooth_l1_loss(dist_s, dist_t, reduction="none").mean(dim=1)
    else:
        losses = F.smooth_l1_loss(dist_s, dist_t, reduction=reduction)
    if max_clip is not None:
        losses = torch.clamp(losses, max=max_clip)
    return losses


def rkd_angle_loss(
    student_feat,
    teacher_feat,
    eps: float = 1e-12,
    reduction: str = "mean",
):
    """Relational KD angle loss.

    Parameters
    ----------
    reduction : str, optional
        "mean" to return a scalar or "none" to return per-sample losses.
    """
    if student_feat.dim() > 2:
        student_feat = student_feat.view(student_feat.size(0), -1)
    if teacher_feat.dim() > 2:
        teacher_feat = teacher_feat.view(teacher_feat.size(0), -1)

    if student_feat.size(0) < 3:
        return torch.tensor(0.0, device=student_feat.device)

    diff_s = student_feat.unsqueeze(0) - student_feat.unsqueeze(1)
    diff_t = teacher_feat.unsqueeze(0) - teacher_feat.unsqueeze(1)

    n = student_feat.size(0)
    mask = ~torch.eye(n, dtype=torch.bool, device=student_feat.device)
    diff_s = diff_s[mask].view(n, n - 1, -1)
    diff_t = diff_t[mask].view(n, n - 1, -1)

    norm_s = F.normalize(diff_s, p=2, dim=2)
    norm_t = F.normalize(diff_t, p=2, dim=2)

    angle_s = torch.bmm(norm_s, norm_s.transpose(1, 2))
    angle_t = torch.bmm(norm_t, norm_t.transpose(1, 2))

    diag_mask = ~torch.eye(n - 1, dtype=torch.bool, device=student_feat.device)
    angle_s = angle_s[:, diag_mask].view(n, -1)
    angle_t = angle_t[:, diag_mask].view(n, -1)

    if reduction == "none":
        return F.smooth_l1_loss(angle_s, angle_t, reduction="none").mean(dim=1)
    return F.smooth_l1_loss(angle_s, angle_t, reduction=reduction)
