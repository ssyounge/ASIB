# modules/losses.py

import torch
import torch.nn.functional as F
from typing import Optional

# ---------- Safe loss helpers (AMP-friendly) ----------
_EPS = 1e-6  # for log/softmax flooring

def _smooth_one_hot(target: torch.Tensor, num_classes: int, eps: float) -> torch.Tensor:
    """Return label-smoothed one-hot distribution."""
    smooth = torch.full((target.size(0), num_classes), eps / (num_classes - 1), device=target.device)
    smooth.scatter_(1, target.unsqueeze(1), 1.0 - eps)
    return smooth

def ce_safe(logits: torch.Tensor, target: torch.Tensor, ls_eps: float = 0.0) -> torch.Tensor:
    """Cross-entropy in float32 under autocast-off region with optional label smoothing.

    - Prevents fp16 underflow by computing in float32
    - Floors probabilities to avoid log(0)
    """
    with torch.autocast('cuda', enabled=False):
        logits = logits.float()
        if ls_eps and ls_eps > 0.0:
            tgt_prob = _smooth_one_hot(target, logits.size(1), float(ls_eps))
            log_prob = torch.log_softmax(logits, dim=1)
            return -(tgt_prob * log_prob).sum(dim=1).mean()
        prob = torch.softmax(logits, dim=1).clamp(min=_EPS)
        return F.nll_loss(prob.log(), target, reduction="mean")

def kl_safe(p_logits: torch.Tensor, q_logits: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Numerically stable KL(p||q) with temperature and float32 math."""
    with torch.autocast('cuda', enabled=False):
        p = torch.softmax(p_logits.float() / tau, dim=1).clamp(_EPS, 1.0)
        q = torch.softmax(q_logits.float() / tau, dim=1).clamp(_EPS, 1.0)
        return F.kl_div(p.log(), q, reduction="batchmean") * (tau * tau)

def safe_kl_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 4.0) -> torch.Tensor:
    """Safe KL divergence loss wrapper (kept for backward compatibility)."""
    # Note: preserves original argument order used in tests
    return kl_safe(teacher_logits, student_logits, tau=temperature)

def safe_mse_loss(student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
    """Safe MSE loss between features in float32 under autocast-off region."""
    with torch.autocast('cuda', enabled=False):
        return F.mse_loss(student_feat.float(), teacher_feat.float(), reduction="mean")

def soft_clip_loss(loss: torch.Tensor, max_val: float) -> torch.Tensor:
    """Softly scale the loss to not exceed max_val while preserving gradients.

    loss' = loss * min(1, max_val / loss_detached)
    """
    with torch.no_grad():
        scale = torch.clamp(max_val / (loss.detach() + 1e-8), max=1.0)
    return loss * scale

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

def kl_safe_vec_probs(p_logits: torch.Tensor, q_probs: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Return per-sample KL(student||teacher) where teacher is given as probabilities.

    Computes KL over classes for each sample and scales by tau^2, using float32 math
    inside an autocast-disabled region for numerical stability.
    """
    with torch.autocast('cuda', enabled=False):
        p_log = F.log_softmax(p_logits.float() / tau, dim=1)
        q = q_probs.float()
        kl = F.kl_div(p_log, q, reduction="none")  # [B, C]
        return kl.sum(dim=1) * (tau * tau)

def hybrid_kd_loss_fn(student_logits, teacher_logits, labels, alpha=0.5, T=4.0):
    """
    Optionally provide a simple "CE + KD" combo in one function.
    total_loss = alpha*CE + (1-alpha)*KD
    """
    ce = ce_loss_fn(student_logits, labels)
    kd = kd_loss_fn(student_logits, teacher_logits, T=T)
    return alpha * ce + (1 - alpha) * kd


def feat_mse_loss(s_feat, t_feat, norm: str = "none", reduction="mean"):
    """Return the MSE between two features after optional normalization."""
    if s_feat.dim() > 2:
        s_feat = s_feat.flatten(1)
    if t_feat.dim() > 2:
        t_feat = t_feat.flatten(1)

    if norm == "l2":
        s_feat = F.normalize(s_feat, dim=1)
        t_feat = F.normalize(t_feat, dim=1)

    loss = F.mse_loss(s_feat, t_feat, reduction=reduction)
    return loss


# ---------- Information Bottleneck ----------
def ib_loss(mu, logvar, beta: float = 1e-3):
    r"""Return β · KL\big(N(μ,σ²) \| N(0, 1)\big).

    fp16 under-/overflow 방지를 위해 float32 캐스팅 후 clipping을 적용한다."""  # <- r-string 로 SyntaxWarning 제거

    mu = mu.float()
    logvar = torch.clamp(logvar.float(), -10.0, 10.0)

    kl_elem = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    
    # 추가 안전장치: loss가 너무 크면 clipping
    kl_elem = torch.clamp(kl_elem, -100.0, 100.0)
    
    loss = beta * kl_elem.mean()
    
    return loss


# ---------- Test compatibility functions ----------
def kl_loss(student_logits, teacher_logits, temperature=4.0):
    """Wrapper for kd_loss_fn for test compatibility."""
    return kd_loss_fn(student_logits, teacher_logits, T=temperature)


def mse_loss(student_feat, teacher_feat):
    """Wrapper for feat_mse_loss for test compatibility."""
    return feat_mse_loss(student_feat, teacher_feat)


def ce_loss(logits, targets, label_smoothing=0.0):
    """Wrapper for ce_loss_fn for test compatibility."""
    return ce_loss_fn(logits, targets, label_smoothing=label_smoothing)


def contrastive_loss(student_feat, teacher_feat, temperature=0.1):
    """Contrastive loss between student and teacher features."""
    # Normalize features
    student_feat = F.normalize(student_feat, dim=1)
    teacher_feat = F.normalize(teacher_feat, dim=1)
    
    # Compute similarity matrix
    sim_matrix = torch.mm(student_feat, teacher_feat.T) / temperature
    
    # Contrastive loss (InfoNCE style)
    labels = torch.arange(student_feat.size(0), device=student_feat.device)
    loss = F.cross_entropy(sim_matrix, labels)
    
    return loss


def attention_loss(student_attn, teacher_attn):
    """Attention transfer loss between student and teacher attention maps."""
    # Ensure same shape
    if student_attn.shape != teacher_attn.shape:
        # Resize if needed
        student_attn = F.interpolate(
            student_attn.unsqueeze(1), 
            size=teacher_attn.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        ).squeeze(1)
    
    # MSE loss between attention maps
    loss = F.mse_loss(student_attn, teacher_attn)
    return loss


def factor_transfer_loss(student_factor, teacher_factor):
    """Factor transfer loss between student and teacher factors."""
    # Ensure same shape
    if student_factor.shape != teacher_factor.shape:
        # Linear projection if needed
        if student_factor.dim() == 2 and teacher_factor.dim() == 2:
            if student_factor.size(1) != teacher_factor.size(1):
                projection = torch.nn.Linear(
                    student_factor.size(1), 
                    teacher_factor.size(1)
                ).to(student_factor.device)
                student_factor = projection(student_factor)
    
    # MSE loss between factors
    loss = F.mse_loss(student_factor, teacher_factor)
    return loss


def certainty_weights(logvar: torch.Tensor) -> torch.Tensor:
    """Return certainty weights based on log variance."""
    # Higher variance = lower certainty = lower weight
    weights = torch.exp(-logvar)
    return weights


def dkd_loss(student_logits, teacher_logits, labels, alpha=1.0, beta=1.0, temperature=4.0):
    """
    Decoupled Knowledge Distillation (DKD) loss.
    
    Parameters:
    -----------
    student_logits : torch.Tensor
        Student model logits
    teacher_logits : torch.Tensor
        Teacher model logits  
    labels : torch.Tensor
        Ground truth labels
    alpha : float
        Weight for target class knowledge distillation
    beta : float
        Weight for non-target class knowledge distillation
    temperature : float
        Temperature for softmax
    """
    # Ensure temperature is positive
    temperature = max(temperature, 1e-6)
    
    # Target class knowledge distillation
    target_kd = kd_loss_fn(student_logits, teacher_logits, T=temperature)
    
    # Non-target class knowledge distillation
    # Create mask for non-target classes
    batch_size, num_classes = student_logits.shape
    target_mask = torch.zeros_like(student_logits, dtype=torch.bool)
    target_mask.scatter_(1, labels.unsqueeze(1), True)
    
    # Mask out target classes with a large negative value instead of -inf
    large_negative = -1e6
    student_logits_nt = student_logits.masked_fill(target_mask, large_negative)
    teacher_logits_nt = teacher_logits.masked_fill(target_mask, large_negative)
    
    # Apply softmax to get probabilities
    student_probs_nt = F.softmax(student_logits_nt / temperature, dim=1)
    teacher_probs_nt = F.softmax(teacher_logits_nt / temperature, dim=1)
    
    # Add small epsilon to prevent log(0)
    eps = 1e-8
    student_probs_nt = student_probs_nt + eps
    teacher_probs_nt = teacher_probs_nt + eps
    
    # Normalize to sum to 1
    student_probs_nt = student_probs_nt / student_probs_nt.sum(dim=1, keepdim=True)
    teacher_probs_nt = teacher_probs_nt / teacher_probs_nt.sum(dim=1, keepdim=True)
    
    # KL divergence for non-target classes
    non_target_kd = F.kl_div(
        torch.log(student_probs_nt + eps),
        teacher_probs_nt,
        reduction='batchmean'
    ) * (temperature ** 2)
    
    total_loss = alpha * target_kd + beta * non_target_kd
    return total_loss


def rkd_distance_loss(
    student_feat,
    teacher_feat,
    eps: float = 1e-12,
    reduction: str = "mean",
    max_clip: Optional[float] = None,
):
    """Relational Knowledge Distillation distance loss."""
    # Flatten spatial dimensions
    if student_feat.dim() > 2:
        student_feat = student_feat.flatten(1)
    if teacher_feat.dim() > 2:
        teacher_feat = teacher_feat.flatten(1)
    
    # Compute pairwise distances
    def pairwise_distance(feat):
        feat_square = torch.sum(feat ** 2, dim=1, keepdim=True)
        feat_prod = torch.mm(feat, feat.t())
        dist = feat_square + feat_square.t() - 2 * feat_prod
        dist = torch.clamp(dist, min=eps)
        return torch.sqrt(dist)
    
    student_dist = pairwise_distance(student_feat)
    teacher_dist = pairwise_distance(teacher_feat)
    
    # Normalize distances
    student_dist = student_dist / (student_dist.max() + eps)
    teacher_dist = teacher_dist / (teacher_dist.max() + eps)
    
    # Compute loss
    loss = F.mse_loss(student_dist, teacher_dist, reduction=reduction)
    
    if max_clip is not None:
        loss = torch.clamp(loss, max=max_clip)
    
    return loss


def rkd_angle_loss(
    student_feat,
    teacher_feat,
    eps: float = 1e-12,
    reduction: str = "mean",
):
    """Relational Knowledge Distillation angle loss."""
    # Flatten spatial dimensions
    if student_feat.dim() > 2:
        student_feat = student_feat.flatten(1)
    if teacher_feat.dim() > 2:
        teacher_feat = teacher_feat.flatten(1)
    
    # Normalize features
    student_feat = F.normalize(student_feat, dim=1)
    teacher_feat = F.normalize(teacher_feat, dim=1)
    
    # Compute pairwise angles
    def pairwise_angle(feat):
        # Compute cosine similarity
        cos_sim = torch.mm(feat, feat.t())
        cos_sim = torch.clamp(cos_sim, min=-1 + eps, max=1 - eps)
        # Convert to angle
        angle = torch.acos(cos_sim)
        return angle
    
    student_angle = pairwise_angle(student_feat)
    teacher_angle = pairwise_angle(teacher_feat)
    
    # Compute loss
    loss = F.mse_loss(student_angle, teacher_angle, reduction=reduction)
    
    return loss


def masked_kd_loss(student_logits, teacher_logits, seen_classes, T=4.0, reduction="batchmean"):
    """
    Class-IL용 KD: seen_classes 열만 골라 KL(student||teacher).
    seen_classes: 1D LongTensor (e.g., tensor([0,1,2,...]))
    """
    if not torch.is_tensor(seen_classes):
        seen_classes = torch.tensor(seen_classes, device=student_logits.device, dtype=torch.long)
    if seen_classes.numel() == 0:
        # 첫 태스크 등: KD 항 없음
        return student_logits.new_zeros(())
    s = student_logits.index_select(dim=1, index=seen_classes)
    t = teacher_logits.index_select(dim=1, index=seen_classes)
    s_log = F.log_softmax(s / T, dim=1)
    t_prob = F.softmax(t / T, dim=1)
    return F.kl_div(s_log, t_prob, reduction=reduction) * (T * T)


def class_il_total_loss(student_logits, teacher_logits, targets, seen_classes,
                        ce_weight=1.0, kd_weight=1.0, T=4.0):
    ce = ce_loss_fn(student_logits, targets)
    kd = masked_kd_loss(student_logits, teacher_logits, seen_classes, T=T) if kd_weight > 0 else 0.0
    total = ce_weight * ce + kd_weight * kd
    return total
