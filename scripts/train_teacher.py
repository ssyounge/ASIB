#!/usr/bin/env python3
"""Simple teacher CE training script."""
import argparse
import os
import sys
from typing import Dict

import torch
import yaml

sys.path.append(os.path.dirname(__file__) + "/..")

from data.cifar100 import get_cifar100_loaders
from models.teachers.teacher_efficientnet import create_efficientnet_b2
from models.teachers.teacher_resnet import create_resnet152
from utils.freeze import freeze_all
from utils.misc import set_random_seed

TEACHER_CHOICES = ["resnet152", "efficientnet_b2"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Teacher fine-tuning")
    p.add_argument("--config", default="configs/minimal.yaml")
    p.add_argument("--teacher", choices=TEACHER_CHOICES, required=True)
    p.add_argument("--epochs", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--wd", type=float)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_cfg(path: str) -> Dict:
    if os.path.exists(path):
        with open(path) as f:
            return yaml.safe_load(f)
    return {}


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config)
    set_random_seed(cfg.get("seed", 42))

    batch_size = args.batch_size or cfg.get("batch_size", 128)
    train_loader, test_loader = get_cifar100_loaders(
        batch_size=batch_size, num_workers=cfg.get("num_workers", 2)
    )
    num_cls = len(train_loader.dataset.classes)

    if args.teacher == "resnet152":
        model = create_resnet152(num_classes=num_cls, small_input=True)
    else:
        model = create_efficientnet_b2(num_classes=num_cls, small_input=True)

    if cfg.get("finetune_partial_freeze", False):
        freeze_all(model)

    device = args.device
    model = model.to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr or cfg.get("finetune_lr", cfg.get("teacher_lr", 1e-4)),
        weight_decay=args.wd
        or cfg.get("finetune_weight_decay", cfg.get("teacher_weight_decay", 5e-4)),
    )
    crit = torch.nn.CrossEntropyLoss()

    best = 0.0
    best_state = model.state_dict()
    epochs = args.epochs or cfg.get("finetune_epochs", 3)
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)[1]
            loss = crit(logits, y)
            loss.backward()
            opt.step()

        model.eval()
        corr = tot = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)[1].argmax(1)
                corr += (pred == y).sum().item()
                tot += y.size(0)
        acc = 100.0 * corr / tot
        print(f"[TeacherCE] ep={ep}/{epochs} acc={acc:.2f}")
        if acc > best:
            best = acc
            best_state = model.state_dict()

    torch.save(best_state, args.ckpt)
    print(f"[TeacherCE] best={best:.2f} saved={args.ckpt}")


if __name__ == "__main__":
    main()
