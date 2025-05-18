"""
partial_freeze.py

- Teacher/Student에서 특정 레이어만 학습 가능하도록 requires_grad 설정
- BN 동결 여부도 이 파일에 정의
"""

import torch
import torch.nn as nn

def freeze_all_params(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_all_params(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True

def freeze_bn_params(module: nn.Module):
    """
    BatchNorm 계열 모듈에 대해 gamma/beta조차 학습 안 하도록 동결
    """
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        for p in module.parameters():
            p.requires_grad = False

###########################################################
# 예시1) ResNet101 Teacher: layer4 + fc 만 학습
###########################################################
def partial_freeze_teacher_resnet(model: nn.Module, freeze_bn=True):
    """
    ResNet101 Teacher.
    layer4, fc만 학습 => 나머지 requires_grad=False
    """
    freeze_all_params(model)
    # layer4, fc만 unfreeze
    for name, param in model.named_parameters():
        if "layer4." in name or "fc." in name:
            param.requires_grad = True

    if freeze_bn:
        model.apply(freeze_bn_params)

###########################################################
# 예시2) EfficientNet Teacher: classifier만 학습
###########################################################
def partial_freeze_teacher_efficientnet(model: nn.Module, freeze_bn=True):
    freeze_all_params(model)
    # EfficientNet-B2 => classifier.* 을 학습
    for name, param in model.classifier.named_parameters():
        param.requires_grad = True

    if freeze_bn:
        model.apply(freeze_bn_params)

###########################################################
# 예시3) Student(ResNet w/ Adapter): 
#        adapter + layer3,4 + fc 학습
###########################################################
def partial_freeze_student_resnet_adapter(model: nn.Module, freeze_bn=True):
    freeze_all_params(model)
    # ex) adapter_conv, adapter_gn, layer3, layer4, fc
    for name, param in model.named_parameters():
        if any(tag in name for tag in ["adapter_conv", "adapter_gn", "layer3.", "layer4.", "fc."]):
            param.requires_grad = True

    if freeze_bn:
        model.apply(freeze_bn_params)

###########################################################
# 예시4) Student(EfficientNet w/ Adapter):
###########################################################
def partial_freeze_student_effnet_adapter(model: nn.Module, freeze_bn=True):
    freeze_all_params(model)
    # adapter_conv, adapter_gn, classifier?
    for name, param in model.named_parameters():
        if any(tag in name for tag in ["adapter_conv", "adapter_gn", "classifier"]):
            param.requires_grad = True

    if freeze_bn:
        model.apply(freeze_bn_params)

###########################################################
# 필요 시 MobileNet/DenseNet도 유사하게 정의
###########################################################
