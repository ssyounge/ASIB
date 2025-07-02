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
    p.add_argument("--finetune_epochs", type=int, default=3)
    p.add_argument("--finetune_lr", type=float, default=1e-4)
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

    simple_finetune(
        model,
        loader,
        lr=args.finetune_lr,
        epochs=args.finetune_epochs,
        device=args.device,
        cfg=cfg,
    )

    torch.save(model.state_dict(), args.finetune_ckpt_path)
    print(f"[FINETUNE] saved => {args.finetune_ckpt_path}")


if __name__ == "__main__":
    main()
