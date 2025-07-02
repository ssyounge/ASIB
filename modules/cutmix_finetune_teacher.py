"""Minimal CutMix fine-tuning utilities used by scripts.fine_tuning."""
import os
import numpy as np
import torch
import torch.nn.functional as F
from utils.eval import evaluate_acc

__all__ = ["finetune_teacher_cutmix", "eval_teacher"]


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def finetune_teacher_cutmix(
    model,
    train_loader,
    test_loader,
    alpha=1.0,
    lr=1e-3,
    weight_decay=1e-4,
    epochs=10,
    device="cuda",
    ckpt_path="teacher_finetuned.pth",
    label_smoothing: float = 0.0,
    cfg=None,
):
    """Simple CutMix fine-tune loop."""
    model = model.to(device)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(cfg.get("adam_beta1", 0.9) if cfg else 0.9,
               cfg.get("adam_beta2", 0.999) if cfg else 0.999),
    )
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True), strict=False)
        acc = evaluate_acc(model, test_loader, device)
        return model, acc

    best_acc = 0.0
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
            rand_index = torch.randperm(x.size(0)).to(device)
            target_a = y
            target_b = y[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
            x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))

            out = model(x)
            logit = out["logit"] if isinstance(out, dict) else out
            loss = criterion(logit, target_a) * lam + criterion(logit, target_b) * (1 - lam)
            optim.zero_grad()
            loss.backward()
            optim.step()
        acc = evaluate_acc(model, test_loader, device)
        if acc > best_acc:
            best_acc = acc
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
    return model, best_acc


def eval_teacher(model, loader, device="cuda"):
    """Evaluate teacher accuracy (wrapper around utils.eval)."""
    return evaluate_acc(model.to(device), loader, device)
