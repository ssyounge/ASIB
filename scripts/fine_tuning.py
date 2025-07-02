#!/usr/bin/env python3
"""Lightweight teacher fine-tuning script used by run_ibkd.sh."""
import argparse
import os
import sys
import yaml
import torch

sys.path.append(os.path.dirname(__file__) + "/..")

from data.cifar100 import get_cifar100_loaders
from utils.model_factory import create_teacher_by_name
from utils.misc import set_random_seed
from trainer import simple_finetune


TEACHER_CHOICES = ["resnet152", "efficientnet_b2"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Teacher fine-tuning")
    p.add_argument("--config", default="configs/minimal.yaml")
    p.add_argument("--teacher_type", choices=TEACHER_CHOICES, required=True)
    p.add_argument("--finetune_ckpt_path", required=True)
    p.add_argument("--finetune_epochs", type=int)
    p.add_argument("--finetune_lr", type=float)
    p.add_argument("--finetune_weight_decay", type=float)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_cfg(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return yaml.safe_load(f)
    return {}


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config)
    set_random_seed(cfg.get("seed", 42))

    loader, _ = get_cifar100_loaders(
        batch_size=cfg.get("batch_size", 128),
        num_workers=cfg.get("num_workers", 0),
    )

    model = create_teacher_by_name(
        args.teacher_type,
        num_classes=100,
        pretrained=True,
        small_input=True,
    ).to(args.device)

    epochs = (
        args.finetune_epochs
        if args.finetune_epochs is not None
        else cfg.get("finetune_epochs", 3)
    )
    lr = (
        args.finetune_lr
        if args.finetune_lr is not None
        else cfg.get("finetune_lr", 1e-4)
    )
    wd = (
        args.finetune_weight_decay
        if args.finetune_weight_decay is not None
        else cfg.get("finetune_weight_decay", 0.0)
    )

    os.makedirs(os.path.dirname(args.finetune_ckpt_path) or ".", exist_ok=True)

    simple_finetune(
        model,
        loader,
        lr=lr,
        epochs=epochs,
        device=args.device,
        weight_decay=wd,
        cfg=cfg,
        ckpt_path=args.finetune_ckpt_path,
    )

    # best and final checkpoints are saved within ``simple_finetune``


if __name__ == "__main__":
    main()
