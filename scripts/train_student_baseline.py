#!/usr/bin/env python3
# scripts/train_student_baseline.py
"""Train a student model with cross-entropy only."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import copy
import torch
import torch.nn as nn
import torch.optim as optim
import hydra
from omegaconf import DictConfig, OmegaConf

from utils.misc import set_random_seed, check_label_range
from data.cifar100 import get_cifar100_loaders
from data.imagenet100 import get_imagenet100_loaders
from main import create_student_by_name, apply_partial_freeze
from modules.cutmix_finetune_teacher import eval_teacher
from utils.progress import smart_tqdm
from utils.misc import get_amp_components





def train_student_ce(
    student_model,
    train_loader,
    test_loader,
    lr=1e-3,
    weight_decay=1e-4,
    epochs=10,
    device="cuda",
    ckpt_path="student_baseline.pth",
    label_smoothing: float = 0.0,
    cfg=None,
):
    student_model = student_model.to(device)
    autocast_ctx, scaler = get_amp_components(cfg or {})

    if os.path.exists(ckpt_path):
        print(f"[StudentCE] Found checkpoint => load {ckpt_path}")
        student_model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True),
            strict=False,
        )
        test_acc = eval_teacher(student_model, test_loader, device=device, cfg=cfg)
        print(f"[StudentCE] loaded => testAcc={test_acc:.2f}")
        return test_acc

    optimizer = optim.AdamW(
        student_model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(
            cfg.get("adam_beta1", 0.9) if cfg is not None else 0.9,
            cfg.get("adam_beta2", 0.999) if cfg is not None else 0.999,
        ),
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    best_acc = 0.0
    best_state = copy.deepcopy(student_model.state_dict())

    for ep in range(1, epochs + 1):
        student_model.train()
        for x, y in smart_tqdm(train_loader, desc=f"[StudentCE ep={ep}]"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast_ctx:
                out = student_model(x)
                logit = out["logit"] if isinstance(out, dict) else out
                loss = criterion(logit, y)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        te_acc = eval_teacher(student_model, test_loader, device=device, cfg=cfg)
        if te_acc > best_acc:
            best_acc = te_acc
            best_state = copy.deepcopy(student_model.state_dict())
        print(f"[StudentCE|ep={ep}/{epochs}] testAcc={te_acc:.2f}, best={best_acc:.2f}")

    student_model.load_state_dict(best_state)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(student_model.state_dict(), ckpt_path)
    print(f"[StudentCE] done => bestAcc={best_acc:.2f}, saved={ckpt_path}")
    return best_acc


@hydra.main(config_path="../configs", config_name="base", version_base="1.3")
def main(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    from utils.config_utils import flatten_hydra_config
    cfg = flatten_hydra_config(cfg)

    device = cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    set_random_seed(cfg.get("seed", 42))

    dataset = cfg.get("dataset_name", "cifar100")
    batch_size = cfg.get("batch_size", 128)
    data_root = cfg.get("data_root", "./data")
    if dataset == "cifar100":
        train_loader, test_loader = get_cifar100_loaders(
            root=data_root,
            batch_size=batch_size,
            num_workers=cfg.get("num_workers", 2),
            augment=cfg.get("data_aug", True),
        )
    else:
        train_loader, test_loader = get_imagenet100_loaders(
            root=data_root,
            batch_size=batch_size,
            num_workers=cfg.get("num_workers", 2),
            augment=cfg.get("data_aug", True),
        )

    num_classes = len(train_loader.dataset.classes)
    check_label_range(train_loader.dataset, num_classes)
    check_label_range(test_loader.dataset, num_classes)

    small_input = cfg.get("small_input")
    if small_input is None:
        small_input = dataset == "cifar100"

    student = create_student_by_name(
        cfg.get("student_type", "resnet_adapter"),
        pretrained=cfg.get("student_pretrained", True),
        small_input=small_input,
        num_classes=num_classes,
        cfg=cfg,
    ).to(device)
    if cfg.get("student_ckpt"):
        student.load_state_dict(
            torch.load(cfg["student_ckpt"], map_location=device, weights_only=True),
            strict=False,
        )

    apply_partial_freeze(
        student,
        cfg.get("student_freeze_level", -1 if not cfg.get("use_partial_freeze") else 0),
        cfg.get("student_freeze_bn", False),
    )

    os.makedirs(cfg.get("results_dir", "results"), exist_ok=True)
    ckpt = os.path.join(cfg["results_dir"], "student_baseline.pth")
    acc = train_student_ce(
        student,
        train_loader,
        test_loader,
        lr=cfg.get("student_lr", 1e-3),
        weight_decay=cfg.get("weight_decay", cfg.get("student_weight_decay", 1e-4)),
        epochs=cfg.get("epochs", 10),
        device=device,
        ckpt_path=ckpt,
        label_smoothing=cfg.get("label_smoothing", 0.0),
        cfg=cfg,
    )

    print(f"[train_student_baseline] final_acc={acc:.2f}% -> {ckpt}")


if __name__ == "__main__":
    main()

