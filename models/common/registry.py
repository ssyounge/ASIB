# models/common/registry.py

from importlib import import_module
from pathlib import Path
import re

# ------------------------------------------------------------------
# (A) 선택적 수동 등록 – @register 데코레이터
# ------------------------------------------------------------------
MODEL_REGISTRY: dict[str, type] = {}

def register(key: str):
    def _wrap(cls):
        MODEL_REGISTRY[key] = cls
        return cls
    return _wrap

# ---------------------------------------------------------
# 1)  하위 패키지 재귀 import  →  클래스 정의 로드
#     (register 가 이미 존재하므로 안전)
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

# 외부에서 늦게 호출하도록 노출
def scan_submodules():
    _import_submodules("models.students")
    _import_submodules("models.teachers")

# ---------------------------------------------------------
# 2)  선택형 데코레이터  (기존 코드 호환)
# ---------------------------------------------------------

# ---------------------------------------------------------
# 3)  **BaseKDModel 파생 클래스 자동 등록**
#     ‑ @register() 안 붙여도 작동
# ---------------------------------------------------------
def _snake(s: str) -> str:
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", s)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

def _auto_register(slim: bool = False):
    """Register subclasses of :class:`BaseKDModel` using multiple keys."""
    # BaseKDModel import 는 scan 이후 base_wrapper 쪽에서 호출
    from models.common.base_wrapper import BaseKDModel  # noqa

    for cls in BaseKDModel.__subclasses__():
        camel = cls.__name__           # e.g. ResNet152Student
        snake = _snake(camel)          # -> res_net152_student
        keys = (camel, snake) if slim else (
            camel,
            snake,
            snake.replace("_student", "").replace("_teacher", ""),
        )
        for k in keys:
            MODEL_REGISTRY.setdefault(k, cls)

# 외부에서 늦게 호출하도록 노출
import yaml
import pkg_resources

_CFG_PATH = pkg_resources.resource_filename(
    __name__, "../../configs/registry_key.yaml"
)
if Path(_CFG_PATH).is_file():
    with open(_CFG_PATH, "r") as f:
        _key_cfg = yaml.safe_load(f)
    ALLOW_KEYS = set(
        _key_cfg.get("student_keys", []) + _key_cfg.get("teacher_keys", [])
    )
else:
    ALLOW_KEYS = set()


def auto_register(*, slim: bool = False):
    """Auto register subclasses, then optionally filter by registry_key.yaml."""
    _auto_register(slim=slim)

    if ALLOW_KEYS:
        for k in list(MODEL_REGISTRY.keys()):
            if k not in ALLOW_KEYS:
                MODEL_REGISTRY.pop(k)
