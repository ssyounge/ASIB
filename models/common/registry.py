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
            raise KeyError(f"[registry] duplicate key: {key}")
        MODEL_REGISTRY[key] = cls
        return cls

    return _wrap


# ---------------------------------------------------------------------------
# 1)  하위 패키지 재귀 import → 클래스 정의 로드
# ---------------------------------------------------------------------------
def _import_submodules(pkg_root: str):
    pkg = import_module(pkg_root)
    root = Path(pkg.__file__).parent
    for p in root.glob("**/*.py"):
        if p.name.startswith("_"):
            continue
        rel = p.with_suffix("").relative_to(root)
        mod = f"{pkg_root}.{rel.as_posix().replace('/', '.')}"
        import_module(mod)


# ---------------------------------------------------------------------------
# scan_submodules() : 나중에 필요할 때 호출하도록 제공
# ---------------------------------------------------------------------------
def scan_submodules():
    _import_submodules("models.students")
    _import_submodules("models.teachers")


# ---------------------------------------------------------------------------
# 2)  BaseKDModel 파생 클래스 자동 등록
# ---------------------------------------------------------------------------
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


def auto_register(*, slim: bool = False):
    _auto_register(slim=slim)


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

# ---------------------------------------------------------------------------
# 호출 여부 플래그와 헬퍼
_SCANNED = False


def ensure_scanned():
    """첫 호출 시에만 scan+auto_register 수행"""
    global _SCANNED
    if not _SCANNED:
        scan_submodules()
        auto_register()
        _SCANNED = True

