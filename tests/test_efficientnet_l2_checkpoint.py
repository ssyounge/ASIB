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
    setup_timm(monkeypatch, {"timm.layers": layers})
    teacher = create_efficientnet_l2(use_checkpointing=True)
    assert teacher.backbone.blocks == "layers"


def test_import_from_factory(monkeypatch):
    # ``timm.layers`` exists but lacks ``checkpoint_seq`` forcing fallback
    layers = types.ModuleType("timm.layers")
    factory = types.ModuleType("timm.models._factory")
    factory.checkpoint_seq = lambda x: "factory"
    modules = {"timm.layers": layers, "timm.models._factory": factory}
    setup_timm(monkeypatch, modules)
    teacher = create_efficientnet_l2(use_checkpointing=True)
    assert teacher.backbone.blocks == "factory"




def test_import_error(monkeypatch):
    layers = types.ModuleType("timm.layers")
    factory = types.ModuleType("timm.models._factory")
    modules = {"timm.layers": layers, "timm.models._factory": factory}
    setup_timm(monkeypatch, modules)
    with pytest.raises(ImportError):
        create_efficientnet_l2(use_checkpointing=True)
