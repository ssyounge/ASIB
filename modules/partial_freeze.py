# modules/partial_freeze.py

import torch
import torch.nn as nn

def freeze_all_params(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False

def freeze_bn_params(module: nn.Module):
    """
    BatchNorm 계열 모듈에 대해 gamma/beta조차 학습 안 하도록 동결
    """
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        for p in module.parameters():
            p.requires_grad = False

def freeze_ln_params(module: nn.Module):
    """
    LayerNorm 계열 모듈 (예: Swin, ViT 등)에 대해 gamma/beta 동결
    """
    if isinstance(module, nn.LayerNorm):
        for p in module.parameters():
            p.requires_grad = False


###########################################################
# (A) Teacher partial-freeze
###########################################################

def partial_freeze_teacher_resnet(model: nn.Module, freeze_bn=True):
    """
    ResNet101 Teacher:
    - layer4, fc만 학습 => 나머지 requires_grad=False
    """
    freeze_all_params(model)
    for name, param in model.named_parameters():
        # TeacherResNetWrapper안에 self.backbone이 있을 수 있으므로
        # backbone. 접두사를 뺀 다음 검사하거나, 그대로 문자열 매칭
        if ("layer4." in name) or ("fc." in name):
            param.requires_grad = True

    if freeze_bn:
        model.apply(freeze_bn_params)

def partial_freeze_teacher_efficientnet(model: nn.Module, freeze_bn=True):
    """
    EfficientNet-B2 Teacher:
    - classifier.* 만 학습 => 나머지 requires_grad=False
    """
    freeze_all_params(model)
    for name, param in model.named_parameters():
        # TeacherEfficientNetWrapper의 self.backbone.classifier?
        # 효율을 위해 'classifier.' substring 매칭
        if "classifier." in name:
            param.requires_grad = True

    if freeze_bn:
        model.apply(freeze_bn_params)

def partial_freeze_teacher_swin(model: nn.Module, freeze_bn=True, freeze_ln=True):
    """
    Swin Tiny Teacher:
    - 마지막 stage (layers.3.) + head 만 학습, 예시
    """
    freeze_all_params(model)
    for name, param in model.named_parameters():
        # layers.3 -> 마지막 stage, head. -> 최종 헤드
        if ("layers.3." in name) or ("head." in name):
            param.requires_grad = True

    if freeze_bn:
        model.apply(freeze_bn_params)
    # Swin은 주로 LayerNorm -> freeze_ln도 옵션화
    if freeze_ln:
        model.apply(freeze_ln_params)


###########################################################
# (B) Student partial-freeze
###########################################################

def partial_freeze_student_resnet_adapter(model: nn.Module, freeze_bn=True):
    """
    ResNet101 Student w/ Adapter:
    - adapter_* + layer3, layer4, fc 등을 학습 
    """
    freeze_all_params(model)
    for name, param in model.named_parameters():
        if any(tag in name for tag in ["adapter_", "layer3.", "layer4.", "fc."]):
            param.requires_grad = True

    if freeze_bn:
        model.apply(freeze_bn_params)

def partial_freeze_student_effnet_adapter(model: nn.Module, freeze_bn=True):
    """
    EfficientNet-B2 Student w/ Adapter
    - adapter_* + classifier
    """
    freeze_all_params(model)
    for name, param in model.named_parameters():
        # adapter_*, classifier => unfreeze
        if any(tag in name for tag in ["adapter_", "classifier"]):
            param.requires_grad = True

    if freeze_bn:
        model.apply(freeze_bn_params)

def partial_freeze_student_swin_adapter(
    model: nn.Module,
    freeze_ln=True
):
    """
    Swin Student w/ Adapter:
    - 예) adapter + head + 일부 block만 학습
    (실제 구현 예시는 프로젝트 따라 맞추면 됨)
    """
    freeze_all_params(model)
    for name, param in model.named_parameters():
        # 여기서는 단순하게 adapter_, head. 만 unfreeze
        if ("adapter" in name) or ("head." in name):
            param.requires_grad = True

    # LayerNorm 동결 여부
    if freeze_ln:
        model.apply(freeze_ln_params)
