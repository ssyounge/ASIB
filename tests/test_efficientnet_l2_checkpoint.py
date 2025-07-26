import types
import sys
import pytest

from models.teachers.efficientnet_l2_teacher import create_efficientnet_l2

class DummyBackbone(types.SimpleNamespace):
    def __init__(self):
        super().__init__()
        self.num_features = 4
        self.blocks = "orig"


def setup_timm(monkeypatch, modules):
    timm_mod = types.ModuleType("timm")
    timm_mod.create_model = lambda *a, **k: DummyBackbone()
    monkeypatch.setitem(sys.modules, "timm", timm_mod)
    for name, mod in modules.items():
        monkeypatch.setitem(sys.modules, name, mod)


def test_import_from_layers(monkeypatch):
    layers = types.ModuleType("timm.layers")
    layers.checkpoint_seq = lambda x: "layers"
    modules = {"timm.layers": layers, "timm.utils": types.ModuleType("timm.utils")}
    setup_timm(monkeypatch, modules)
    teacher = create_efficientnet_l2(use_checkpointing=True)
    assert teacher.backbone.blocks == "layers"


def test_import_from_utils(monkeypatch):
    layers = types.ModuleType("timm.layers")
    utils = types.ModuleType("timm.utils")
    utils.checkpoint_seq = lambda x: "utils"
    modules = {"timm.layers": layers, "timm.utils": utils}
    setup_timm(monkeypatch, modules)
    teacher = create_efficientnet_l2(use_checkpointing=True)
    assert teacher.backbone.blocks == "utils"


def test_import_from_utils_checkpoint(monkeypatch):
    layers = types.ModuleType("timm.layers")
    utils = types.ModuleType("timm.utils")
    utils_checkpoint = types.ModuleType("timm.utils.checkpoint")
    utils_checkpoint.checkpoint_seq = lambda x: "checkpoint"
    modules = {
        "timm.layers": layers,
        "timm.utils": utils,
        "timm.utils.checkpoint": utils_checkpoint,
    }
    setup_timm(monkeypatch, modules)
    teacher = create_efficientnet_l2(use_checkpointing=True)
    assert teacher.backbone.blocks == "checkpoint"


def test_import_error(monkeypatch):
    layers = types.ModuleType("timm.layers")
    utils = types.ModuleType("timm.utils")
    utils_checkpoint = types.ModuleType("timm.utils.checkpoint")
    modules = {
        "timm.layers": layers,
        "timm.utils": utils,
        "timm.utils.checkpoint": utils_checkpoint,
    }
    setup_timm(monkeypatch, modules)
    with pytest.raises(ImportError):
        create_efficientnet_l2(use_checkpointing=True)
