# utils/misc.py

import random
import numpy as np
import torch
from tqdm import tqdm
import sys

__all__ = [
    "set_random_seed",
    "progress",
    "check_label_range",
    "get_model_num_classes",
    "get_amp_components",
]

def set_random_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

def progress(iterable, desc=None, **kwargs):
    """Simple tqdm wrapper with sane defaults."""
    kwargs.setdefault("file", sys.stdout)
    kwargs.setdefault("leave", False)
    if "disable" not in kwargs:
        kwargs["disable"] = not sys.stdout.isatty()
    return tqdm(iterable, desc=desc, **kwargs)


def check_label_range(dataset, num_classes: int) -> None:
    """Ensure all dataset labels fall within ``[0, num_classes)``."""
    labels = getattr(dataset, "targets", None)
    if labels is None:
        labels = getattr(dataset, "labels", None)
    if labels is None:
        # Fallback: iterate over dataset
        labels = [y for _, y in dataset]
    for lbl in labels:
        l = int(lbl)
        if l < 0 or l >= num_classes:
            raise ValueError(
                f"Label {l} outside expected range [0, {num_classes})"
            )


def get_model_num_classes(model) -> int:
    """Best-effort check for the classifier output dimension."""
    # Common attribute
    if hasattr(model, "num_classes"):
        return int(model.num_classes)

    backbone = getattr(model, "backbone", model)
    # torchvision ResNet / ConvNeXt style
    fc = getattr(backbone, "fc", None)
    if isinstance(fc, torch.nn.Linear):
        return int(fc.out_features)
    classifier = getattr(backbone, "classifier", None)
    if isinstance(classifier, torch.nn.Linear):
        return int(classifier.out_features)
    if isinstance(classifier, (torch.nn.Sequential, list)):
        # assume last element is Linear
        last = classifier[-1]
        if isinstance(last, torch.nn.Linear):
            return int(last.out_features)

    raise ValueError("Unable to determine number of classes for the model")


def get_amp_components(cfg: dict):
    """Return autocast context and scaler based on config."""
    from contextlib import nullcontext
    from torch.cuda.amp import autocast, GradScaler

    use_amp = cfg.get("use_amp", False)
    if use_amp and torch.cuda.is_available():
        dtype = cfg.get("amp_dtype", "float16")
        autocast_ctx = autocast(dtype=getattr(torch, dtype, torch.float16))
        scaler = GradScaler(init_scale=cfg.get("grad_scaler_init_scale", 1024))
    else:
        autocast_ctx = nullcontext()
        scaler = None
    return autocast_ctx, scaler

