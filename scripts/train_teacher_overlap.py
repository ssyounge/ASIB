#!/usr/bin/env python3
"""Train two teachers on overlapping CIFAR-100 splits."""

from __future__ import annotations

import argparse
import os
import sys
import yaml

sys.path.append(os.path.dirname(__file__) + "/..")

from utils.model_factory import create_teacher_by_name
from trainer import simple_finetune
from data.overlap import get_overlap_loaders

TEACHER_CHOICES = ["resnet152", "efficientnet_b2"]


def _load_cfg(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Train two teachers on overlapping CIFAR-100 splits"
    )
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--teacher1", choices=TEACHER_CHOICES, default="resnet152")
    ap.add_argument("--teacher2", choices=TEACHER_CHOICES, default="efficientnet_b2")
    ap.add_argument("--rho", type=float, required=True, help="Overlap ratio [0,1]")
    ap.add_argument("--out_dir", required=True, help="Directory to save checkpoints")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--wd", type=float)
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_cfg(args.config)

    batch_size = args.batch_size or cfg.get("batch_size", 128)
    num_workers = cfg.get("num_workers", 2)
    train1, train2, test_loader = get_overlap_loaders(
        args.rho,
        root=cfg.get("dataset_root", "./data"),
        batch_size=batch_size,
        num_workers=num_workers,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt1 = os.path.join(args.out_dir, f"T1_rho{args.rho}.pth")
    ckpt2 = os.path.join(args.out_dir, f"T2_rho{args.rho}.pth")

    n_cls = len(test_loader.dataset.classes)
    t1 = create_teacher_by_name(
        args.teacher1,
        num_classes=n_cls,
        pretrained=True,
        small_input=True,
        cfg=cfg,
    ).to(args.device)
    t2 = create_teacher_by_name(
        args.teacher2,
        num_classes=n_cls,
        pretrained=True,
        small_input=True,
        cfg=cfg,
    ).to(args.device)

    epochs = args.epochs or cfg.get("finetune_epochs", 3)
    lr = args.lr or cfg.get("finetune_lr", 1e-4)
    wd = args.wd or cfg.get("finetune_weight_decay", 0.0)

    ft_cfg = dict(cfg)
    ft_cfg["finetune_eval_loader"] = test_loader

    simple_finetune(
        t1,
        train1,
        lr=lr,
        epochs=epochs,
        device=args.device,
        weight_decay=wd,
        cfg=ft_cfg,
        ckpt_path=ckpt1,
    )
    simple_finetune(
        t2,
        train2,
        lr=lr,
        epochs=epochs,
        device=args.device,
        weight_decay=wd,
        cfg=ft_cfg,
        ckpt_path=ckpt2,
    )


if __name__ == "__main__":
    main()
