#!/usr/bin/env python3
# scripts/train_student_baseline.py
"""Train a student model with cross-entropy only."""
import argparse
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim

from utils.misc import set_random_seed, check_label_range
from data.cifar100 import get_cifar100_loaders
from data.imagenet100 import get_imagenet100_loaders
from main import create_student_by_name, partial_freeze_student_auto
from modules.cutmix_finetune_teacher import eval_teacher
from utils.progress import smart_tqdm
from utils.misc import get_amp_components


def parse_args():
    p = argparse.ArgumentParser(description="Student baseline training")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--student_type", type=str)
    p.add_argument("--student_ckpt", type=str)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--student_lr", type=float)
    p.add_argument("--weight_decay", type=float)
    p.add_argument("--sgd_momentum", type=float)
    p.add_argument("--epochs", type=int)
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str)
    p.add_argument("--dataset", "--dataset_name", dest="dataset_name", type=str)
    p.add_argument("--data_aug", type=int)
    p.add_argument("--label_smoothing", type=float)
    p.add_argument("--small_input", type=int)
    p.add_argument("--student_freeze_level", type=int)
    return p.parse_args()


def load_config(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            import yaml
            return yaml.safe_load(f) or {}
    return {}


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

    optimizer = optim.SGD(
        student_model.parameters(),
        lr=lr,
        momentum=cfg.get("sgd_momentum", 0.9) if cfg else 0.9,
        weight_decay=weight_decay,
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


def main():
    args = parse_args()
    base_cfg = load_config(args.config)
    cli_cfg = {k: v for k, v in vars(args).items() if v is not None}
    cfg = {**base_cfg, **cli_cfg}

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

    partial_freeze_student_auto(
        student,
        student_name=cfg.get("student_type", "resnet_adapter"),
        freeze_bn=cfg.get("student_freeze_bn", True),
        freeze_ln=cfg.get("student_freeze_ln", True),
        use_adapter=cfg.get("student_use_adapter", False),
        freeze_level=cfg.get("student_freeze_level", 1),
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

