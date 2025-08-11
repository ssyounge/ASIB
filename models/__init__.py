from .ib_mbm import (
    SynergyHead,
    IB_MBM,
    build_ib_mbm_from_teachers,
)

build_from_teachers = build_ib_mbm_from_teachers

__all__ = [
    "SynergyHead",
    "IB_MBM",
    "build_from_teachers",
]

# 패키지 초기화 단계에 서브모듈을 **자동 import 하지 않는다**.
# 필요할 때 registry.ensure_scanned() 가 수행됨.
from models.common.base_wrapper import MODEL_REGISTRY
