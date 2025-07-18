utils/ema.py

import torch
from copy import deepcopy

class ModelEMA:
    """Simple EMA wrapper for model parameters."""

    def __init__(self, model, decay=0.999, device=None, skip_keys=()):
        self.ema = deepcopy(model).eval()
        self.decay = decay
        self.skip_keys = tuple(skip_keys)
        if device is not None:
            self.ema.to(device)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            for (name, ema_v), (_, model_v) in zip(self.ema.state_dict().items(), model.state_dict().items()):
                if any(name.startswith(k) for k in self.skip_keys):
                    ema_v.copy_(model_v)
                else:
                    if ema_v.dtype.is_floating_point:
                        ema_v.mul_(self.decay).add_(model_v, alpha=1 - self.decay)
                    else:
                        ema_v.copy_(model_v)
