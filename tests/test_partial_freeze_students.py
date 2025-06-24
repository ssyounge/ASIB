import pytest

torch = pytest.importorskip("torch")

from modules.partial_freeze import partial_freeze_student_swin


class DummyStudentSwin(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.swin = torch.nn.Module()
        self.swin.layers = torch.nn.Module()
        self.swin.layers.add_module("3", torch.nn.Linear(1, 1))
        self.fc = torch.nn.Linear(1, 1)
        self.adapter_block = torch.nn.Linear(1, 1)
        self.add_module("adapter_conv", self.adapter_block)


def _req_dict(model):
    return {n: p.requires_grad for n, p in model.named_parameters()}


def test_swin_student_unfreeze_default():
    m = DummyStudentSwin()
    partial_freeze_student_swin(m)
    req = _req_dict(m)
    assert req["swin.layers.3.weight"]
    assert req["fc.weight"]

