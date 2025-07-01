# utils/misc.py

import random
import numpy as np
import torch
from tqdm import tqdm
import sys

__all__ = ["set_random_seed", "progress"]

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
