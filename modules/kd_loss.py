
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
