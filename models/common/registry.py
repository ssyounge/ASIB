"""Simple model registry without automatic scanning.

Models must be registered using ``@register("key")`` or listed in
``configs/registry_key.yaml`` for lazy import.
"""

from pathlib import Path
from importlib import import_module
import yaml

MODEL_REGISTRY: dict[str, type] = {}


def register(key: str):
    """Decorator to register a model class under ``key``."""

    def _wrap(cls):
        if key in MODEL_REGISTRY:
            raise KeyError(f"[registry] duplicate key: {key}")
        MODEL_REGISTRY[key] = cls
        return cls

    return _wrap


_CFG = Path(__file__).resolve().parent.parent.parent / "configs" / "registry_key.yaml"

if _CFG.is_file():
    with _CFG.open() as f:
        _cfg = yaml.safe_load(f) or {}
    _KEYS = _cfg.get("student_keys", []) + _cfg.get("teacher_keys", [])

    for k in _KEYS:
        if k.endswith("_teacher"):
            mod = f"models.teachers.{k}"
        elif k.endswith("_student"):
            mod = f"models.students.{k}"
        else:
            mod = k
        try:
            import_module(mod)
        except ModuleNotFoundError:
            pass
