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


@torch.no_grad()
def evaluate_mbm_acc(teacher1, teacher2, vib_mbm, loader, device="cuda"):
    """Evaluate classification accuracy of ``vib_mbm`` using frozen teachers."""
    vib_mbm.eval()
    teacher1.eval()
    teacher2.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out1 = teacher1(x)
        out2 = teacher2(x)
        t1_dict = out1[0] if isinstance(out1, tuple) else out1
        t2_dict = out2[0] if isinstance(out2, tuple) else out2
        logits = vib_mbm(t1_dict["feat_2d"], t2_dict["feat_2d"])[1]
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total
