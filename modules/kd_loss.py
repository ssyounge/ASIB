# modules/partial_freeze.py

"""
kd_loss.py
import torch
import torch.nn as nn

- KD(Knowledge Distillation) 관련 Loss 함수 모음
- KL-div, CE, 혹은 CRD, etc.
"""
def freeze_all_params(model: nn.Module):
    """모델 내 모든 파라미터를 학습 불가능(requires_grad=False)하게 만든다."""
    for param in model.parameters():
        param.requires_grad = False

import torch
import torch.nn.functional as F
def freeze_bn_params(module: nn.Module):
    """
    BatchNorm 계열 모듈(gamma/beta)도 학습하지 않도록 동결한다.
    model.apply(freeze_bn_params) 형태로 호출.
    """
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        for p in module.parameters():
            p.requires_grad = False

def kd_loss_fn(student_logits, teacher_logits, T=4.0, reduction="batchmean"):
def freeze_ln_params(module: nn.Module):
    """
    KL divergence-based KD
    LayerNorm 계열 모듈에서 gamma/beta도 동결한다.
    (Swin/ViT 등에서 사용)
    """
    # softmax w/ temperature
    s_log_probs = F.log_softmax(student_logits / T, dim=1)
    t_probs = F.softmax(teacher_logits / T, dim=1)
    kl = F.kl_div(s_log_probs, t_probs, reduction=reduction) * (T * T)
    return kl
    if isinstance(module, nn.LayerNorm):
        for p in module.parameters():
            p.requires_grad = False

def ce_loss_fn(student_logits, labels):
    return F.cross_entropy(student_logits, labels)

def hybrid_kd_loss(student_logits, teacher_logits, labels, alpha=0.5, T=4.0):
###########################################################
# (A) Teacher partial-freeze: 백본 동결 + BN/Head/MBM 업데이트
###########################################################

def partial_freeze_teacher_resnet(model: nn.Module, freeze_bn=True):
    """
    alpha * CE + (1-alpha) * KD
    Teacher (ResNet101):
      - 백본(하위 레이어) 동결
      - BN/Head/MBM만 업데이트하는 논문 설정을 가정.
      - 여기서는 'fc' (헤드)와 'mbm.'(있다면)만 unfreeze,
        BN을 업데이트하려면 freeze_bn=False로 호출.
    """
    ce = ce_loss_fn(student_logits, labels)
    kd = kd_loss_fn(student_logits, teacher_logits, T)
    return alpha * ce + (1 - alpha) * kd
    freeze_all_params(model)

    # 1) fc(헤드) & mbm(있다면)만 열기
    for name, param in model.named_parameters():
        # 예: backbone.fc -> "fc."
        #     MBM 파라미터 -> "mbm." (가령 Teacher 래퍼 안에 있다면)
        if "fc." in name or "mbm." in name:
            param.requires_grad = True

    # 2) BN 업데이트 여부
    if not freeze_bn:
        # BN 레이어를 다시 unfreeze
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                for p in m.parameters():
                    p.requires_grad = True


def fitnet_loss_fn(teacher_feat, student_feat):
def partial_freeze_teacher_efficientnet(model: nn.Module, freeze_bn=True):
    """
    FitNet style: MSE between teacher feature & student feature
    teacher_feat, student_feat: shape [N, C, H, W] or [N, D]
    Teacher (EfficientNet-B2):
      - 백본(features) 동결
      - BN/Head/MBM만 업데이트.
      - 여기서는 'classifier.'(헤드) + 'mbm.'(있다면)만 unfreeze
    """
    return F.mse_loss(student_feat, teacher_feat)
    freeze_all_params(model)
    for name, param in model.named_parameters():
        if "classifier." in name or "mbm." in name:
            param.requires_grad = True

def at_loss_fn(teacher_feat, student_feat, p=2):
    if not freeze_bn:
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                for p in m.parameters():
                    p.requires_grad = True


def partial_freeze_teacher_swin(model: nn.Module, freeze_bn=True, freeze_ln=True):
    """
    Attention Transfer:
      - Compute norm over channel dimension => attention map (H×W)
      - L2 distance between teacher's attention map & student's
    teacher_feat, student_feat: shape [N, C, H, W]
    Teacher (Swin Tiny):
      - 백본 동결
      - BN/Head/MBM 업데이트 (논문 설정)
      - 여기서는 'head.'(헤드), 'mbm.'(있다면)만 열고,
        BN/LN 업데이트는 옵션
    """
    # 1) teacher attention
    #   A^T = sum(|F^T|^p, dim=1)^(1/p)
    t_atten = teacher_feat.abs().pow(p).sum(dim=1)  # [N, H, W]
    t_atten = t_atten.pow(1.0 / p)
    freeze_all_params(model)
    for name, param in model.named_parameters():
        if "head." in name or "mbm." in name:
            param.requires_grad = True

    # 2) student attention
    s_atten = student_feat.abs().pow(p).sum(dim=1)
    s_atten = s_atten.pow(1.0 / p)
    # Swin에서 BN/LN 동결 옵션
    if not freeze_bn:
        model.apply(freeze_bn_params)
        # Swin은 보통 LayerNorm 쓰므로, BN이 있을지 여부는 모델 구조에 따라.
    if not freeze_ln:
        model.apply(freeze_ln_params)

    # 3) normalize (Z-K 2017)
    t_norm = F.normalize(t_atten.view(t_atten.size(0), -1), dim=1)
    s_norm = F.normalize(s_atten.view(s_atten.size(0), -1), dim=1)

    # 4) MSE
    return F.mse_loss(s_norm, t_norm)
###########################################################
# (B) Student partial-freeze: 상부 레이어(later stage + fc)만 학습
###########################################################

def crd_loss_fn(student_feat, teacher_feat, index, 
                nce_t=0.07, nce_k=16384, 
                feat_dim=128):
def partial_freeze_student_resnet(model: nn.Module, freeze_bn=True):
    """
    Pseudocode: 
      - student_feat, teacher_feat: shape [N, D]
      - index: label or instance index [N]
      - build positives & negatives
      - compute InfoNCE or similar contrastive loss
    Student (ResNet101):
      - 논문 예시: layer4 + fc 등 상위 레이어만 학습,
        하위 레이어 동결.
    """
    # typically we need a memory bank or queue with size nce_k
    # This code is non-trivial => see official CRD repo
    # ...
    loss = torch.zeros(1, device=student_feat.device)
    return loss
    freeze_all_params(model)
    for name, param in model.named_parameters():
        # layer4, fc 파라미터만 unfreeze
        if "layer4." in name or "fc." in name:
            param.requires_grad = True

    if not freeze_bn:
        # BN도 업데이트하려면 다시 열기
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                for p in m.parameters():
                    p.requires_grad = True


def pkt_loss_fn(teacher_feat, student_feat):
def partial_freeze_student_efficientnet(model: nn.Module, freeze_bn=True):
    """
    PKT(Probabilistic Knowledge Transfer):
     - 1) Normalize teacher_feat => p^T
     - 2) Normalize student_feat => p^S
     - 3) compute distance (KL, JS, etc.)
    Student (EfficientNet-B2):
      - 여기서는 백본(features) 동결,
        classifier 등 상부만 학습한다고 가정
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
    freeze_all_params(model)
    for name, param in model.named_parameters():
        if "classifier." in name:
            param.requires_grad = True

def dkd_loss_fn(s_logit, t_logit, alpha=1.0, beta=0.5, T=4.0):
    if not freeze_bn:
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                for p in m.parameters():
                    p.requires_grad = True


def partial_freeze_student_swin(model: nn.Module, freeze_bn=True, freeze_ln=True):
    """
    Pseudocode for DKD
    - alpha, beta: positive/negative part 가중치
    - T: temperature
    - ...
    Student (Swin Tiny):
      - 논문 예시에 맞춰 '마지막 stage + head'만 학습 예시 => "layers.3." & "head."
    """
    # 실제 계산 로직은 꽤 길지만, 요약하면
    # p_s, p_t = softmax(s_logit/T), softmax(t_logit/T)
    # DKD separates "target-class" vs "non-target" terms
    # see official code: https://github.com/Jossome/Decoupled-KD
    
    loss = torch.zeros(1, device=s_logit.device)
    return loss
    freeze_all_params(model)
    for name, param in model.named_parameters():
        if "layers.3." in name or "head." in name:
            param.requires_grad = True

    if not freeze_bn:
        model.apply(freeze_bn_params)
    if not freeze_ln:
        model.apply(freeze_ln_params)
