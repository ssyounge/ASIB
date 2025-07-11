# methods/__init__.py
from .crd import CRDDistiller
from .dkd import DKDDistiller
from .vanilla_kd import VanillaKDDistiller
from .at import ATDistiller
from .fitnet import FitNetDistiller
from .ewc import EWC
import torch

class EWCMethod:
    """Simple cross-entropy method for EWC-only experiments."""

    def __init__(self):
        self.criterion = torch.nn.CrossEntropyLoss()

    def loss(self, logits, target):
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        return self.criterion(logits, target)

class VIBMethod:
    pass


class KDMethod:
    pass


class CEMethod:
    pass


__all__ = [
    "CRDDistiller",
    "DKDDistiller",
    "VanillaKDDistiller",
    "ATDistiller",
    "FitNetDistiller",
]

# ---------------------------------------------------------------------------
# Experimental registry for simple method wrappers
# ---------------------------------------------------------------------------
METHODS_REGISTRY = {
    "vib": VIBMethod,
    "kd": KDMethod,
    "ce": CEMethod,
    "ewc": EWCMethod,   # EWC 단독 실험용(선택)
}
