from .mbm import (
    SynergyHead,
    IB_MBM,
    build_from_teachers,
)

__all__ = [
    "SynergyHead",
    "IB_MBM",
    "build_from_teachers",
]

from models.common.base_wrapper import MODEL_REGISTRY

# --------------------------------------------------------------------
#  모든 *teachers/*.py · *students/*.py 파일을 자동 import 해서
#  파일 안에 @register("…") 데코레이터가 실행되도록 한다.
# --------------------------------------------------------------------
from importlib import import_module
import pkgutil as _pkgutil

# 현재 패키지(models)의 모든 하위 모듈 순회
for _finder, _mod_name, _is_pkg in _pkgutil.walk_packages(__path__, prefix=__name__ + "."):
    if ".teachers." in _mod_name or ".students." in _mod_name:
        import_module(_mod_name)
