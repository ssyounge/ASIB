# methods/__init__.py
from .crd import CRDDistiller
from .dkd import DKDDistiller
from .vanilla_kd import VanillaKDDistiller
from .at import ATDistiller
from .fitnet import FitNetDistiller

__all__ = [
    "CRDDistiller",
    "DKDDistiller",
    "VanillaKDDistiller",
    "ATDistiller",
    "FitNetDistiller",
]
