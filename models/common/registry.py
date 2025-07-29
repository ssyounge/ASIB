# models/common/registry.py

"""Minimal registry – **자동 스캔은 사용하지 않습니다**.
   - ``@register("key")`` 로만 수동 등록
   - 또는 ``configs/registry_key.yaml`` 에 명시된 모듈만 import 합니다.
"""

from pathlib import Path
from importlib import import_module
import re
import yaml

MODEL_REGISTRY: dict[str, type] = {}


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
# 내부 유틸: 서브모듈 import + BaseKDModel 자동 등록
#   ‐ 직접 호출 금지, 아래 ensure_scanned() 에서만 사용
# ---------------------------------------------------------
def _import_submodules(pkg_root: str):
    pkg = import_module(pkg_root)
    root = Path(pkg.__file__).parent
    for p in root.glob("**/*.py"):
        if p.name.startswith("_"):
            continue
        rel = p.with_suffix("").relative_to(root)
        mod = f"{pkg_root}.{rel.as_posix().replace('/', '.')}"
        import_module(mod)


# ---------------------------------------------------------
# ❶  BaseKDModel → registry 자동 등록
# ---------------------------------------------------------
def _snake(s: str) -> str:
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", s)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def _auto_register(slim: bool = False):
    """Register subclasses of :class:`BaseKDModel` using multiple keys."""
    from models.common.base_wrapper import BaseKDModel  # noqa: E402

    for cls in BaseKDModel.__subclasses__():
        camel = cls.__name__
        snake = _snake(camel)
        keys = (camel, snake) if slim else (
            camel,
            snake,
            snake.replace("_student", "").replace("_teacher", ""),
        )
        for k in keys:
            MODEL_REGISTRY.setdefault(k, cls)


# ---------------------------------------------------------
# ❷  registry_key.yaml  (허용 키 필터)
# ---------------------------------------------------------
import yaml, pkg_resources, json
_CFG_PATH = pkg_resources.resource_filename(__name__, "../../configs/registry_key.yaml")
if Path(_CFG_PATH).is_file():
    with open(_CFG_PATH, "r") as f:
        _key_cfg = yaml.safe_load(f)
    _ALLOW_KEYS = set(_key_cfg.get("student_keys", []) + _key_cfg.get("teacher_keys", []))
else:
    _ALLOW_KEYS = set()

# ---------------------------------------------------------------------------
# 호출 여부 플래그와 헬퍼
_SCANNED = False


def ensure_scanned(*, slim: bool = False):
    """첫 호출 시 students / teachers 서브모듈 import → 자동 등록"""
    global _SCANNED
    if _SCANNED:
        return

    # ------------------------------------------------------------------
    #  A) 1차:  registry_key.yaml 의 key → 대응 모듈만 import
    # ------------------------------------------------------------------
    def _deduce_module(key: str) -> str | None:
        """
        heuristic:
          · '..._student' → models.students.<key>
          · '..._teacher' → models.teachers.<key>
        """
        if key.endswith("_student"):
            return f"models.students.{key}"
        if key.endswith("_teacher"):
            return f"models.teachers.{key}"
        return None

    mods = {_deduce_module(k) for k in _ALLOW_KEYS}
    mods.discard(None)
    for m in mods:
        try:
            import_module(m)
        except ModuleNotFoundError:
            # key 는 있는데 모듈이 실제로 없으면 경고만 출력
            import logging
            logging.warning("[registry] module '%s' not found for key-based import", m)

    _auto_register(slim=slim)

    # ------------------------------------------------------------------
    #  B) 2차: allow-list 필터  (scan 후에도 key 누락시 제거)
    # ------------------------------------------------------------------
    if _ALLOW_KEYS:
        for k in list(MODEL_REGISTRY.keys()):
            if k not in _ALLOW_KEYS:
                MODEL_REGISTRY.pop(k)

    _SCANNED = True

