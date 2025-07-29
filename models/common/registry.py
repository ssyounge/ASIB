# models/common/registry.py

"""Minimal registry – **자동 스캔은 사용하지 않습니다**.
   - ``@register("key")`` 로만 수동 등록
   - 또는 ``configs/registry_key.yaml`` 에 명시된 모듈만 import 합니다.
"""

from pathlib import Path
from importlib import import_module
import yaml

MODEL_REGISTRY: dict[str, type] = {}


def register(key: str):
    """Decorator for manual registration."""

    def _wrap(cls):
        if key in MODEL_REGISTRY:
            raise KeyError(f"[registry] duplicate key: {key}")
        MODEL_REGISTRY[key] = cls
        return cls

    return _wrap


# ---------------------------------------------------------------------------
# configs/registry_key.yaml 에 명시된 모듈들만 import 하여 레지스트리 등록
# ---------------------------------------------------------------------------
_CFG = Path(__file__).resolve().parent.parent.parent / "configs" / "registry_key.yaml"

if _CFG.is_file():
    with _CFG.open() as f:
        _cfg = yaml.safe_load(f) or {}
    _KEYS = _cfg.get("student_keys", []) + _cfg.get("teacher_keys", [])

    for k in _KEYS:
        # 간단한 휴리스틱으로 모듈 import 시도
        if k.endswith("_teacher"):
            mod = f"models.teachers.{k}"
        elif k.endswith("_student"):
            mod = f"models.students.{k}"
        else:
            mod = k
        try:
            import_module(mod)
        except ModuleNotFoundError:
            # 사용자가 외부에서 import 할 수도 있으므로 조용히 무시
            pass

