# utils/eval.py

import torch

@torch.no_grad()
def evaluate_acc(model, loader, device="cuda", cfg=None, mixup_active: bool = False):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if mixup_active:
            y = y.argmax(dim=1)
        out = model(x)
        if isinstance(out, tuple):
            logits = out[1]
        elif isinstance(out, dict):
            logits = out.get("logit", out)
        else:
            logits = out
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total
