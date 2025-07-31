# modules/cutmix_finetune_teacher.py

import os
import copy
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional
from utils.common import smart_tqdm, cutmix_data, get_amp_components, check_label_range, get_model_num_classes


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
            if isinstance(out, tuple):
                logits = out[1]
            elif isinstance(out, dict):
                logits = out["logit"]
            else:
                logits = out
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
        logging.info("[CutMix] Found checkpoint => load %s", ckpt_path)
        teacher_model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True),
            strict=False,
        )
        test_acc = eval_teacher(teacher_model, test_loader, device=device, cfg=None)
        logging.info("[CutMix] loaded => testAcc=%.2f", test_acc)
        return teacher_model, test_acc

    num_classes = len(getattr(train_loader.dataset, "classes", []))
    if num_classes == 0:
        num_classes = get_model_num_classes(teacher_model)

    # ------------------------------------------------------------------
    # 0) Optimizer / Scheduler
    # ------------------------------------------------------------------
    optimizer = optim.AdamW(
        teacher_model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    warm_epochs = int(cfg.get("warmup_epochs", 0))
    if warm_epochs >= epochs:            # Guard
        logging.warning(
            "[CutMix] warmup_epochs(%d) >= epochs(%d) \u27A1 warmup_epochs = %d",
            warm_epochs, epochs, max(0, epochs - 1)
        )
        warm_epochs = max(0, epochs - 1)

    # (1) eta_min 방어구문
    eta_min = cfg.get("min_lr")
    if eta_min is None:
        eta_min = 1e-6

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs - warm_epochs),
        eta_min=eta_min,
    )
    base_lr = lr

    best_acc = 0.0
    best_state = copy.deepcopy(teacher_model.state_dict())

    for ep in range(1, epochs + 1):
        # ------------------------------------------------------------------
        # 1) 1-epoch CutMix train
        # ------------------------------------------------------------------
        if warm_epochs and ep <= warm_epochs:             # linear warm-up
            warm_lr = base_lr * ep / warm_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = warm_lr

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
            ckpt_dir = os.path.dirname(ckpt_path)
            if ckpt_dir:                      # ← 폴더가 있을 때만 생성
                os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(best_state, ckpt_path)

        if ep > warm_epochs:                              # cosine step
            scheduler.step()

        logging.info(
            "[CutMix|ep=%d/%d] lr=%.6f, trainAcc=%.2f, testAcc=%.2f, best=%.2f",
            ep,
            epochs,
            scheduler.get_last_lr()[0],
            tr_acc,
            te_acc,
            best_acc,
        )

    logging.info(
        "[CutMix] Fine-tune done => bestAcc=%.2f, saved=%s", best_acc, ckpt_path
    )
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
        logging.info("[CEFineTune] Found checkpoint => load %s", ckpt_path)
        teacher_model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True),
            strict=False,
        )
        test_acc = eval_teacher(teacher_model, test_loader, device=device, cfg=cfg)
        logging.info("[CEFineTune] loaded => testAcc=%.2f", test_acc)
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
    # --- Scheduler : linear warm-up + cosine ------------------------------------
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs - cfg.get("warmup_epochs", 0),
        eta_min=cfg.get("min_lr", 1e-6),
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    best_acc = 0.0
    best_state = copy.deepcopy(teacher_model.state_dict())

    for ep in range(1, epochs + 1):
        teacher_model.train()
        # optional warm-up
        if ep <= cfg.get("warmup_epochs", 0):
            warm_lr = (ep / cfg["warmup_epochs"]) * cfg["finetune_lr"]
            for pg in optimizer.param_groups:
                pg["lr"] = warm_lr
        for x, y in smart_tqdm(train_loader, desc=f"[CE FineTune ep={ep}]"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast_ctx:
                out = teacher_model(x)
                if isinstance(out, tuple):
                    logits = out[1]
                elif isinstance(out, dict):
                    logits = out["logit"]
                else:
                    logits = out
                loss = criterion(logits, y)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        te_acc = eval_teacher(teacher_model, test_loader, device=device, cfg=cfg)
        if ep > cfg.get("warmup_epochs", 0):
            scheduler.step()
        if te_acc > best_acc:
            best_acc = te_acc
            best_state = copy.deepcopy(teacher_model.state_dict())
            # save checkpoint whenever best accuracy improves
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(best_state, ckpt_path)
        logging.info(
            "[CE FineTune|ep=%d/%d] testAcc=%.2f, best=%.2f",
            ep,
            epochs,
            te_acc,
            best_acc,
        )

    logging.info(
        "[CEFineTune] done => bestAcc=%.2f, saved=%s", best_acc, ckpt_path
    )
    teacher_model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True)
    )
    return teacher_model, best_acc
