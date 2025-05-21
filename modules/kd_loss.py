# modules/partial_freeze.py

import torch
import torch.nn as nn

def freeze_all_params(model: nn.Module):
    """모델 내 모든 파라미터를 학습 불가능(requires_grad=False)하게 만든다."""
    for param in model.parameters():
        param.requires_grad = False

def freeze_bn_params(module: nn.Module):
    """
    BatchNorm 계열 모듈(gamma/beta)도 학습하지 않도록 동결한다.
    model.apply(freeze_bn_params) 형태로 호출.
    """
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        for p in module.parameters():
            p.requires_grad = False

def freeze_ln_params(module: nn.Module):
    """
    LayerNorm 계열 모듈에서 gamma/beta도 동결한다.
    (Swin/ViT 등에서 사용)
    """
    if isinstance(module, nn.LayerNorm):
        for p in module.parameters():
            p.requires_grad = False


###########################################################
# (A) Teacher partial-freeze: 백본 동결 + BN/Head/MBM 업데이트
###########################################################

def partial_freeze_teacher_resnet(model: nn.Module, freeze_bn=True):
    """
    Teacher (ResNet101):
      - 백본(하위 레이어) 동결
      - BN/Head/MBM만 업데이트하는 논문 설정을 가정.
      - 여기서는 'fc' (헤드)와 'mbm.'(있다면)만 unfreeze,
        BN을 업데이트하려면 freeze_bn=False로 호출.
    """
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


def partial_freeze_teacher_efficientnet(model: nn.Module, freeze_bn=True):
    """
    Teacher (EfficientNet-B2):
      - 백본(features) 동결
      - BN/Head/MBM만 업데이트.
      - 여기서는 'classifier.'(헤드) + 'mbm.'(있다면)만 unfreeze
    """
    freeze_all_params(model)
    for name, param in model.named_parameters():
        if "classifier." in name or "mbm." in name:
            param.requires_grad = True

    if not freeze_bn:
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                for p in m.parameters():
                    p.requires_grad = True


def partial_freeze_teacher_swin(model: nn.Module, freeze_bn=True, freeze_ln=True):
    """
    Teacher (Swin Tiny):
      - 백본 동결
      - BN/Head/MBM 업데이트 (논문 설정)
      - 여기서는 'head.'(헤드), 'mbm.'(있다면)만 열고,
        BN/LN 업데이트는 옵션
    """
    freeze_all_params(model)
    for name, param in model.named_parameters():
        if "head." in name or "mbm." in name:
            param.requires_grad = True

    # Swin에서 BN/LN 동결 옵션
    if not freeze_bn:
        model.apply(freeze_bn_params)
        # Swin은 보통 LayerNorm 쓰므로, BN이 있을지 여부는 모델 구조에 따라.
    if not freeze_ln:
        model.apply(freeze_ln_params)


###########################################################
# (B) Student partial-freeze: 상부 레이어(later stage + fc)만 학습
###########################################################

def partial_freeze_student_resnet(model: nn.Module, freeze_bn=True):
    """
    Student (ResNet101):
      - 논문 예시: layer4 + fc 등 상위 레이어만 학습,
        하위 레이어 동결.
    """
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


def partial_freeze_student_efficientnet(model: nn.Module, freeze_bn=True):
    """
    Student (EfficientNet-B2):
      - 여기서는 백본(features) 동결,
        classifier 등 상부만 학습한다고 가정
    """
    freeze_all_params(model)
    for name, param in model.named_parameters():
        if "classifier." in name:
            param.requires_grad = True

    if not freeze_bn:
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                for p in m.parameters():
                    p.requires_grad = True


def partial_freeze_student_swin(model: nn.Module, freeze_bn=True, freeze_ln=True):
    """
    Student (Swin Tiny):
      - 논문 예시에 맞춰 '마지막 stage + head'만 학습 예시 => "layers.3." & "head."
    """
    freeze_all_params(model)
    for name, param in model.named_parameters():
        if "layers.3." in name or "head." in name:
            param.requires_grad = True

    if not freeze_bn:
        model.apply(freeze_bn_params)
    if not freeze_ln:
        model.apply(freeze_ln_params)
