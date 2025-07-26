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


def test_import_from_layers_helpers(monkeypatch):
    layers_helpers = types.ModuleType("timm.layers.helpers")
    layers_helpers.checkpoint_seq = lambda x: "layers_helpers"
    setup_timm(monkeypatch, {"timm.layers.helpers": layers_helpers})
    teacher = create_efficientnet_l2(use_checkpointing=True)
    assert teacher.backbone.blocks == "layers_helpers"


def test_import_from_layers(monkeypatch):
    layers = types.ModuleType("timm.layers")
    layers.checkpoint_seq = lambda x: "layers"
    setup_timm(monkeypatch, {"timm.layers": layers})
    teacher = create_efficientnet_l2(use_checkpointing=True)
    assert teacher.backbone.blocks == "layers"


def test_import_from_models_helpers(monkeypatch):
    models_helpers = types.ModuleType("timm.models.helpers")
    models_helpers.checkpoint_seq = lambda x: "models_helpers"
    setup_timm(monkeypatch, {"timm.models.helpers": models_helpers})
    teacher = create_efficientnet_l2(use_checkpointing=True)
    assert teacher.backbone.blocks == "models_helpers"


def test_fallback_on_signature_mismatch(monkeypatch):
    """Fallback to next module when checkpoint_seq signature is unexpected."""
    bad_helpers = types.ModuleType("timm.models.helpers")
    # invalid signature: requires two params not matching (x, y)
    bad_helpers.checkpoint_seq = lambda x, y: "bad"
    layers = types.ModuleType("timm.layers")
    layers.checkpoint_seq = lambda x: "layers"
    setup_timm(
        monkeypatch,
        {
            "timm.models.helpers": bad_helpers,
            "timm.layers": layers,
        },
    )
    teacher = create_efficientnet_l2(use_checkpointing=True)
    assert teacher.backbone.blocks == "layers"


def test_import_error(monkeypatch):
    setup_timm(monkeypatch, {})
    with pytest.warns(RuntimeWarning):
        teacher = create_efficientnet_l2(use_checkpointing=True)
    assert teacher.backbone.blocks == "orig"
