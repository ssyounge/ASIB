# models/common/registry.py

"""Minimal registry with a static YAML map.

Models are registered either via ``@register("key")`` decorators or
``configs/registry_map.yaml`` which lists callables for each key.
No automatic directory scanning is performed.
"""

from importlib import import_module
from pathlib import Path
from functools import partial
import yaml

MODEL_REGISTRY: dict[str, callable] = {}


def register(key: str):
    """Decorator for manual registration."""

    def _wrap(cls):
        if key in MODEL_REGISTRY:
            # 동일 클래스면 조용히 패스, 아니면 에러
            if MODEL_REGISTRY[key] is cls:
                return cls
            raise KeyError(f"[registry] duplicate key: {key}")
        MODEL_REGISTRY[key] = cls
        return cls

    return _wrap


# ---------------------------------------------------------
# ❶  Load registry entries from configs/registry_map.yaml
# ---------------------------------------------------------

_MAP_PATH = Path(__file__).parent.parent.parent / "configs" / "registry_map.yaml"
if not _MAP_PATH.is_file():
    raise FileNotFoundError(f"[registry] manual map file missing → {_MAP_PATH}")

with open(_MAP_PATH, "r") as f:
    _MAP = yaml.safe_load(f)

def _make_lazy_builder(mod_path: str, attr: str):
    """Return a callable(**kw) that lazy-imports *mod_path.attr* and instantiates."""

    def _builder(*args, **kwargs):
        mod = import_module(mod_path)
        cls_or_fn = getattr(mod, attr)
        return cls_or_fn(*args, **kwargs)

    return _builder

for section in ("teachers", "students"):
    for key, target in (_MAP.get(section, {}) or {}).items():
        mod_path, attr = target.rsplit(".", 1)
        MODEL_REGISTRY[key] = _make_lazy_builder(mod_path, attr)


# ---------------------------------------------------------
# ❷  decorator 등록 방식 유지
# ---------------------------------------------------------


# ---------------------------------------------------------
# ❸  Compatibility dummy – scanning no longer needed
# ---------------------------------------------------------
def ensure_scanned(*, slim: bool = False):  # noqa: D401
    """Registry is ready at import-time (no auto-scan needed)."""
    return

