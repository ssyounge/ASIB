import copy
import torch

class LwF:
    """Placeholder implementation of Learning without Forgetting regularizer."""

    def __init__(self, model, cfg):
        self.prev_model = copy.deepcopy(model).eval()
        self.temperature = float(cfg.get("lwf_T", 2.0))
        self.alpha = float(cfg.get("lwf_alpha", 1.0))

    def penalty(self, model):
        # Actual knowledge distillation loss requires current batch.
        # This minimal stub returns zero and serves as an interface example.
        return torch.tensor(0.0, device=next(model.parameters()).device)
