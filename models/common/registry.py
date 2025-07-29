from importlib import import_module
from pathlib import Path
import re

# ---------------------------------------------------------
# 0)  공용 레지스트리  &  데코레이터  **먼저 선언**
# ---------------------------------------------------------
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

_import_submodules("models.students")
_import_submodules("models.teachers")

# ---------------------------------------------------------
# 2)  선택형 데코레이터  (기존 코드 호환)
# ---------------------------------------------------------

# ---------------------------------------------------------
# 3)  **BaseKDModel 파생 클래스 자동 등록**
#     ‑ @register() 안 붙여도 작동
# ---------------------------------------------------------
from models.common.base_wrapper import BaseKDModel   # pylint: disable=cyclic-import

def _snake(s: str) -> str:
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", s)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

def _auto_register():
    for cls in BaseKDModel.__subclasses__():
        camel = cls.__name__
        snake = _snake(camel)
        short = snake.replace("_student", "").replace("_teacher", "")
        for k in (camel, snake, short):
            MODEL_REGISTRY.setdefault(k, cls)

_auto_register()
