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
MODEL_REGISTRY.pop("resnet152_student", None)
