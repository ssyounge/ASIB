import pytest

torch = pytest.importorskip("torch")

from modules.partial_freeze import (
    partial_freeze_teacher_efficientnet,
    partial_freeze_teacher_resnet,
)


class DummyResNetTeacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Module()
        self.backbone.layer4 = torch.nn.Linear(1, 1)
        self.backbone.layer3 = torch.nn.Linear(1, 1)
        self.backbone.fc = torch.nn.Linear(1, 1)
        self.backbone.layer4.adapter_conv = torch.nn.Linear(1, 1)
        self.mbm = torch.nn.Linear(1, 1)


class DummyEfficientNetTeacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Module()
        self.backbone.features = torch.nn.Linear(1, 1)
        self.backbone.classifier = torch.nn.Linear(1, 1)
        self.extra = torch.nn.Linear(1, 1)



def _req_dict(model):
    return {n: p.requires_grad for n, p in model.named_parameters()}


def test_resnet_layer4_fc():
    m = DummyResNetTeacher()
    partial_freeze_teacher_resnet(m, freeze_level=2)
    req = _req_dict(m)
    assert req["backbone.layer4.weight"]
    assert req["backbone.fc.weight"]
    assert req["mbm.weight"]
    assert not req["backbone.layer3.weight"]


def test_efficientnet_features_classifier():
    m = DummyEfficientNetTeacher()
    partial_freeze_teacher_efficientnet(m, freeze_level=2)
    req = _req_dict(m)
    assert req["backbone.features.weight"]
    assert req["backbone.classifier.weight"]
    assert not req["extra.weight"]



def test_adapter_pattern():
    m = DummyResNetTeacher()
    partial_freeze_teacher_resnet(m, freeze_level=0, use_adapter=True)
    req = _req_dict(m)
    assert req["backbone.fc.weight"]
    assert req["backbone.layer4.adapter_conv.weight"]
