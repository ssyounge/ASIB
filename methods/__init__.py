# methods/__init__.py
from .crd import CRDDistiller
from .dkd import DKDDistiller
from .vanilla_kd import VanillaKDDistiller
from .feature_kd import FeatureKD

__all__ = [
    "CRDDistiller",
    "DKDDistiller",
    "VanillaKDDistiller",
    "FeatureKD",
]
