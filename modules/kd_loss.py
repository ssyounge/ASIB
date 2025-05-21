
"""
kd_loss.py

- KD(Knowledge Distillation) 관련 Loss 함수 모음
- KL-div, CE, 혹은 CRD, etc.
"""

import torch
import torch.nn.functional as F

def kd_loss_fn(student_logits, teacher_logits, T=4.0, reduction="batchmean"):
    """
    KL divergence-based KD
    """
    # softmax w/ temperature
    s_log_probs = F.log_softmax(student_logits / T, dim=1)
    t_probs = F.softmax(teacher_logits / T, dim=1)
    kl = F.kl_div(s_log_probs, t_probs, reduction=reduction) * (T * T)
    return kl

def ce_loss_fn(student_logits, labels):
    return F.cross_entropy(student_logits, labels)

def hybrid_kd_loss(student_logits, teacher_logits, labels, alpha=0.5, T=4.0):
    """
    alpha * CE + (1-alpha) * KD
    """
    ce = ce_loss_fn(student_logits, labels)
    kd = kd_loss_fn(student_logits, teacher_logits, T)
    return alpha * ce + (1 - alpha) * kd

# 필요 시 CRD, PKT, etc. 다른 KD 방식도 추가 가능
def fitnet_loss_fn(teacher_feat, student_feat):
    """
    FitNet style: MSE between teacher feature & student feature
    teacher_feat, student_feat: shape [N, C, H, W] or [N, D]
    """
    return F.mse_loss(student_feat, teacher_feat)

def at_loss_fn(teacher_feat, student_feat, p=2):
    """
    Attention Transfer:
      - Compute norm over channel dimension => attention map (H×W)
      - L2 distance between teacher's attention map & student's
    teacher_feat, student_feat: shape [N, C, H, W]
    """
    # 1) teacher attention
    #   A^T = sum(|F^T|^p, dim=1)^(1/p)
    t_atten = teacher_feat.abs().pow(p).sum(dim=1)  # [N, H, W]
    t_atten = t_atten.pow(1.0 / p)

    # 2) student attention
    s_atten = student_feat.abs().pow(p).sum(dim=1)
    s_atten = s_atten.pow(1.0 / p)

    # 3) normalize (Z-K 2017)
    t_norm = F.normalize(t_atten.view(t_atten.size(0), -1), dim=1)
    s_norm = F.normalize(s_atten.view(s_atten.size(0), -1), dim=1)

    # 4) MSE
    return F.mse_loss(s_norm, t_norm)

def crd_loss_fn(student_feat, teacher_feat, index, 
                nce_t=0.07, nce_k=16384, 
                feat_dim=128):
    """
    Pseudocode: 
      - student_feat, teacher_feat: shape [N, D]
      - index: label or instance index [N]
      - build positives & negatives
      - compute InfoNCE or similar contrastive loss
    """
    # typically we need a memory bank or queue with size nce_k
    # This code is non-trivial => see official CRD repo
    # ...
    loss = torch.zeros(1, device=student_feat.device)
    return loss

def pkt_loss_fn(teacher_feat, student_feat):
    """
    PKT(Probabilistic Knowledge Transfer):
     - 1) Normalize teacher_feat => p^T
     - 2) Normalize student_feat => p^S
     - 3) compute distance (KL, JS, etc.)
    """
    # teacher_feat, student_feat: [N, D]
    t_norm = F.normalize(teacher_feat, dim=1)
    s_norm = F.normalize(student_feat, dim=1)
    # Probability distribution => outer product?
    # e.g. p^T(i, j) = t_norm[i]*t_norm[j], same for p^S
    # Then measure KL( p^T || p^S )
    # ...
    loss = torch.zeros(1, device=teacher_feat.device)
    return loss

def dkd_loss_fn(s_logit, t_logit, alpha=1.0, beta=0.5, T=4.0):
    """
    Pseudocode for DKD
    - alpha, beta: positive/negative part 가중치
    - T: temperature
    - ...
    """
    # 실제 계산 로직은 꽤 길지만, 요약하면
    # p_s, p_t = softmax(s_logit/T), softmax(t_logit/T)
    # DKD separates "target-class" vs "non-target" terms
    # see official code: https://github.com/Jossome/Decoupled-KD
    
    loss = torch.zeros(1, device=s_logit.device)
    return loss
