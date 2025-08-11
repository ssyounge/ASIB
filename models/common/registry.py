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


def register(_: str):
    """NO‑OP decorator.

    코드에 남은 ``@register("…")`` 줄을 지우지 않아도 되고,
    레지스트리에 아무 것도 넣지 않고 클래스를 그대로 돌려준다.
    """

    def _wrap(cls):             # pylint: disable=unused-argument
        return cls              # 아무 일도 하지 않는다

    return _wrap


# ---------------------------------------------------------
# ❶  Load registry entries from configs/registry_map.yaml
# ---------------------------------------------------------

_MAP_PATH = Path(__file__).parent.parent.parent / "configs" / "registry_map.yaml"
if not _MAP_PATH.is_file():
    raise FileNotFoundError(f"[registry] manual map file missing → {_MAP_PATH}")

with open(_MAP_PATH, "r", encoding="utf-8") as f:
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
def ensure_scanned(*_, **__):  # noqa: D401
    """Legacy stub — does nothing anymore."""
    return

