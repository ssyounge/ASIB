# modules/cutmix_finetune_teacher.py

import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional
from utils.progress import smart_tqdm

from utils.misc import cutmix_data, get_amp_components, check_label_range


def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    """CutMix-adjusted cross-entropy.

    Computed as ``lam * CE(pred, y_a) + (1 - lam) * CE(pred, y_b)``.

    ``pred`` can have spatial dimensions. In that case, the predictions are
    averaged across all non-batch, non-class dimensions before calculating the
    loss.
    """
    if pred.dim() > 2:
        pred = pred.mean(dim=tuple(range(2, pred.dim())))

    # Clamp labels to avoid invalid class indices from CutMix augmentation
    num_classes = pred.size(1)
    y_a = torch.clamp(y_a, 0, num_classes - 1)
    y_b = torch.clamp(y_b, 0, num_classes - 1)

    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch_cutmix(
    teacher_model,
    loader,
    optimizer,
    alpha=1.0,
    device="cuda",
    label_smoothing: float = 0.0,
    num_classes: Optional[int] = None,
    cfg=None,
):
    """
    ``teacher_model`` should implement ``forward(x, y=None)`` and return a
    dictionary containing ``"logit"``. Only ``"logit"`` is used for the
    cross-entropy calculation.

    ``label_smoothing`` controls the amount of smoothing applied to the loss.
    """
    teacher_model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    autocast_ctx, scaler = get_amp_components(cfg or {})
    total_loss = 0.0
    correct, total = 0, 0

    for batch_idx, (x, y) in enumerate(smart_tqdm(loader, desc="[CutMix Train]")):
        x, y = x.to(device), y.to(device)

        if num_classes is not None:
            _batch_ds = type("BatchDataset", (), {})()
            _batch_ds.targets = y
            check_label_range(_batch_ds, num_classes)

        # 1) cutmix
        x_cm, y_a, y_b, lam = cutmix_data(x, y, alpha=alpha)

        # 2) forward + loss
        with autocast_ctx:
            out = teacher_model(x_cm)  # we only need `logits` for classification
            logits = out["logit"]
            if num_classes is None:
                num_classes = logits.size(1)
            min_label = int(torch.cat((y_a, y_b)).min())
            max_label = int(torch.cat((y_a, y_b)).max())
            if min_label < 0 or max_label >= num_classes:
                raise ValueError(
                    f"CutMix labels must be within [0, {num_classes - 1}], "
                    f"got min={min_label}, max={max_label}"
                )
            loss = cutmix_criterion(criterion, logits, y_a, y_b, lam)

        # 4) backward
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # 5) log
        bs = x.size(0)
        total_loss += loss.item() * bs

        # Accuracy is computed against ``y_a`` for simplicity.
        # The mixed pixels make this slightly optimistic but it is commonly used.
        preds = logits.argmax(dim=1)
        correct += (preds == y_a).sum().item()
        total   += bs

    epoch_loss = total_loss / total
    epoch_acc  = 100.0 * correct / total
    return epoch_loss, epoch_acc

@torch.no_grad()
def eval_teacher(teacher_model, loader, device="cuda", cfg=None):
    """Evaluate classification accuracy of a model.

    This mirrors :func:`evaluate_acc` in ``eval.py`` but lives here so the
    training utilities can depend on it without a circular import.
    ``teacher_model`` may return either a dict containing ``"logit"``, a tuple
    of ``(..., logits, ...)`` or logits directly.
    """

    autocast_ctx, _ = get_amp_components(cfg or {})
    teacher_model.eval()
    correct, total = 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with autocast_ctx:
            out = teacher_model(x)

            if isinstance(out, tuple):
                logits = out[1]
            elif isinstance(out, dict):
                logits = out["logit"]
            else:
                logits = out

            if logits.dim() > 2:
                logits = logits.mean(dim=tuple(range(2, logits.dim())))

            preds = logits.argmax(dim=1)

        correct += (preds == y).sum().item()
        total += y.size(0)

    return 100.0 * correct / total

def finetune_teacher_cutmix(
    teacher_model,
    train_loader,
    test_loader,
    alpha=1.0,           # CutMix alpha
    lr=1e-3,
    weight_decay=1e-4,
    epochs=10,
    device="cuda",
    ckpt_path="teacher_finetuned_cutmix.pth",
    label_smoothing: float = 0.0,
    cfg=None,
):
    """
    teacher_model: must produce a dict containing "logit". Only the
    logits are used to do classification during fine-tuning.
    train_loader, test_loader: standard classification dataset
    alpha: cutmix alpha
    lr, weight_decay, epochs, etc. for AdamW optimizer
    label_smoothing: passed to ``CrossEntropyLoss`` during training
    """
    teacher_model = teacher_model.to(device)

    if os.path.exists(ckpt_path):
        print(f"[CutMix] Found checkpoint => load {ckpt_path}")
        teacher_model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True),
            strict=False,
        )
        test_acc = eval_teacher(teacher_model, test_loader, device=device, cfg=None)
        print(f"[CutMix] loaded => testAcc={test_acc:.2f}")
        return teacher_model, test_acc

    num_classes = len(getattr(train_loader.dataset, "classes", []))
    if num_classes == 0:
        from utils.misc import get_model_num_classes
        num_classes = get_model_num_classes(teacher_model)

    optimizer = optim.AdamW(
        teacher_model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    best_state = copy.deepcopy(teacher_model.state_dict())

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch_cutmix(
            teacher_model,
            train_loader,
            optimizer,
            alpha=alpha,
            device=device,
            label_smoothing=label_smoothing,
            num_classes=num_classes,
            cfg=None,
        )
        te_acc = eval_teacher(teacher_model, test_loader, device=device, cfg=None)

        if te_acc > best_acc:
            best_acc = te_acc
            best_state = copy.deepcopy(teacher_model.state_dict())
            # --- save best checkpoint whenever best accuracy is updated ---
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(best_state, ckpt_path)

        scheduler.step()

        print(
            f"[CutMix|ep={ep}/{epochs}] lr={scheduler.get_last_lr()[0]:.6f}, "
            f"trainAcc={tr_acc:.2f}, testAcc={te_acc:.2f}, best={best_acc:.2f}"
        )

    print(f"[CutMix] Fine-tune done => bestAcc={best_acc:.2f}, saved={ckpt_path}")
    # reload the best checkpoint (already saved during training)
    teacher_model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True)
    )
    return teacher_model, best_acc

def standard_ce_finetune(
    teacher_model,
    train_loader,
    test_loader,
    lr=1e-3,
    weight_decay=1e-4,
    epochs=10,
    device="cuda",
    ckpt_path="teacher_finetuned_ce.pth",
    label_smoothing: float = 0.0,
    cfg=None,
):
    """Simple cross-entropy fine-tuning loop.

    Parameters
    ----------
    label_smoothing : float, optional
        Amount of label smoothing for ``CrossEntropyLoss``.
    """
    teacher_model = teacher_model.to(device)
    autocast_ctx, scaler = get_amp_components(cfg or {})

    if os.path.exists(ckpt_path):
        print(f"[CEFineTune] Found checkpoint => load {ckpt_path}")
        teacher_model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True),
            strict=False,
        )
        test_acc = eval_teacher(teacher_model, test_loader, device=device, cfg=cfg)
        print(f"[CEFineTune] loaded => testAcc={test_acc:.2f}")
        return teacher_model, test_acc

    optimizer = optim.AdamW(
        teacher_model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(
            cfg.get("adam_beta1", 0.9) if cfg is not None else 0.9,
            cfg.get("adam_beta2", 0.999) if cfg is not None else 0.999,
        ),
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    best_acc = 0.0
    best_state = copy.deepcopy(teacher_model.state_dict())

    for ep in range(1, epochs + 1):
        teacher_model.train()
        for x, y in smart_tqdm(train_loader, desc=f"[CE FineTune ep={ep}]"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast_ctx:
                out = teacher_model(x)
                loss = criterion(out["logit"], y)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        te_acc = eval_teacher(teacher_model, test_loader, device=device, cfg=cfg)
        if te_acc > best_acc:
            best_acc = te_acc
            best_state = copy.deepcopy(teacher_model.state_dict())
            # save checkpoint whenever best accuracy improves
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(best_state, ckpt_path)
        print(f"[CE FineTune|ep={ep}/{epochs}] testAcc={te_acc:.2f}, best={best_acc:.2f}")

    print(f"[CEFineTune] done => bestAcc={best_acc:.2f}, saved={ckpt_path}")
    teacher_model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True)
    )
    return teacher_model, best_acc
