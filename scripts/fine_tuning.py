# scripts/fine_tuning.py
"""
Example: Fine-tuning Teacher (ResNet/EfficientNet/Swin) on either CIFAR-100 or ImageNet100
using optional CutMix or standard CE training.

Usage:
  python fine_tuning.py --config configs/hparams.yaml

All fine-tuning options live in `configs/hparams.yaml`.
"""

import argparse
import os
import copy
import torch
import yaml

from utils.misc import set_random_seed, check_label_range

# data loaders
from data.cifar100 import get_cifar100_loaders
from data.imagenet100 import get_imagenet100_loaders

# teacher factories
from models.teachers.teacher_resnet import create_resnet101, create_resnet152
from models.teachers.teacher_efficientnet import create_efficientnet_b2
from models.teachers.teacher_swin import create_swin_t

# partial freeze
from modules.partial_freeze import (
    partial_freeze_teacher_resnet,
    partial_freeze_teacher_efficientnet,
    partial_freeze_teacher_swin,
)
from utils.freeze import freeze_all

# cutmix finetune
from modules.cutmix_finetune_teacher import finetune_teacher_cutmix, eval_teacher

def parse_args():
    parser = argparse.ArgumentParser(description="Teacher Fine-tuning Script")

    # ① YAML 기본값
    parser.add_argument(
        "--config",
        type=str,
        default="configs/hparams.yaml",
        help="Path to YAML config for fine-tuning",
    )

    # ② run_experiments.sh 가 전달하는 옵션들(없으면 None)
    parser.add_argument("--teacher_type", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--finetune_ckpt_path", type=str)

    # (필요 시 추가) lr·epoch 등 sweep 파라미터
    parser.add_argument("--finetune_lr", type=float)
    parser.add_argument("--finetune_epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--finetune_weight_decay", type=float)
    # ↓ run_experiments.sh 에서 CutMix 알파도 변경할 수 있도록
    parser.add_argument("--finetune_cutmix_alpha", type=float)
    parser.add_argument("--data_aug", type=int)
    parser.add_argument("--small_input", type=int)
    parser.add_argument("--dropout_p", type=float)
    parser.add_argument("--use_amp", type=int)
    parser.add_argument("--amp_dtype", type=str)
    parser.add_argument("--grad_scaler_init_scale", type=int)

    return parser.parse_args()

def load_config(cfg_path):
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

def get_data_loaders(dataset_name, batch_size=128, num_workers=2, augment=True):
    """
    Returns train_loader, test_loader based on dataset_name.
    """
    if dataset_name == "cifar100":
        return get_cifar100_loaders(
            batch_size=batch_size,
            num_workers=num_workers,
            augment=augment,
        )
    elif dataset_name == "imagenet100":
        return get_imagenet100_loaders(
            batch_size=batch_size,
            num_workers=num_workers,
            augment=augment,
        )
    else:
        raise ValueError(f"Unknown dataset_name={dataset_name}")

def create_teacher_by_name(
    teacher_type,
    num_classes=100,
    pretrained=True,
    small_input=False,
    dropout_p=0.3,
    cfg: dict | None = None,
):
    """
    Extends to handle resnet152, resnet101, efficientnet_b2, swin_tiny, etc.
    """
    if teacher_type == "resnet101":
        return create_resnet101(
            num_classes=num_classes,
            pretrained=pretrained,
            small_input=small_input,
            cfg=cfg,
        )
    elif teacher_type == "resnet152":
        return create_resnet152(
            num_classes=num_classes,
            pretrained=pretrained,
            small_input=small_input,
            cfg=cfg,
        )
    elif teacher_type == "efficientnet_b2":
        return create_efficientnet_b2(
            num_classes=num_classes,
            pretrained=pretrained,
            small_input=small_input,
            dropout_p=dropout_p,
            cfg=cfg,
        )
    elif teacher_type == "swin_tiny":
        return create_swin_t(
            num_classes=num_classes,
            pretrained=pretrained,
            cfg=cfg,
        )
    else:
        raise ValueError(f"[fine_tuning.py] Unknown teacher_type={teacher_type}")

def partial_freeze_teacher_auto(
    model,
    teacher_type,
    freeze_bn=True,
    freeze_ln=True,
    use_adapter=False,
    bn_head_only=False,
    freeze_level=1,
):
    """
    If needed, partial freeze for fine-tune. Or you can freeze nothing if you want full fine-tune.
    """
    if teacher_type in ("resnet101", "resnet152"):
        partial_freeze_teacher_resnet(
            model,
            freeze_bn=freeze_bn,
            use_adapter=use_adapter,
            bn_head_only=bn_head_only,
            freeze_level=freeze_level,
        )
    elif teacher_type == "efficientnet_b2":
        partial_freeze_teacher_efficientnet(
            model,
            freeze_bn=freeze_bn,
            use_adapter=use_adapter,
            bn_head_only=bn_head_only,
            freeze_level=freeze_level,
        )
    elif teacher_type == "swin_tiny":
        partial_freeze_teacher_swin(
            model,
            freeze_ln=freeze_ln,
            use_adapter=use_adapter,
            freeze_level=freeze_level,
        )
    else:
        raise ValueError(f"Unknown teacher_type={teacher_type}")

def standard_ce_finetune(
    model,
    train_loader,
    test_loader,
    lr,
    weight_decay,
    epochs,
    device,
    ckpt_path,
    label_smoothing: float = 0.0,
):
    """Simple fine-tune loop using cross-entropy loss.

    Parameters
    ----------
    label_smoothing : float, optional
        Passed to ``CrossEntropyLoss``.
    """
    model = model.to(device)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    crit  = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    best_acc = 0.0
    for ep in range(1, epochs+1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            out = model(x)
            loss = crit(out["logit"], y)
            loss.backward()
            optim.step()
        acc = eval_teacher(model, test_loader, device)
        if acc > best_acc:
            best_acc = acc
            ckpt_dir = os.path.dirname(ckpt_path)
            if ckpt_dir:
                os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
    return model, best_acc

def main():
    args = parse_args()
    base_cfg = load_config(args.config)

    # argparse 값이 None 이면 YAML 값을 유지, 아니면 덮어쓰기
    cli_cfg  = {k: v for k, v in vars(args).items() if v is not None}
    cfg      = {**base_cfg, **cli_cfg}
    device = cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("[Warning] No CUDA => Using CPU")
        device = "cpu"

    seed = cfg.get("seed", 42)
    deterministic = cfg.get("deterministic", True)
    set_random_seed(seed, deterministic=deterministic)

    # 1) dataset
    dataset_name = cfg.get("dataset_name", "cifar100")
    batch_size   = cfg.get("batch_size", 128)
    train_loader, test_loader = get_data_loaders(
        dataset_name,
        batch_size=batch_size,
        num_workers=cfg.get("num_workers", 2),
        augment=cfg.get("data_aug", True),
    )

    num_classes = len(train_loader.dataset.classes)
    check_label_range(train_loader.dataset, num_classes)
    check_label_range(test_loader.dataset, num_classes)

    small_input = cfg.get("small_input")
    if small_input is None:
        small_input = dataset_name == "cifar100"

    # 2) teacher
    teacher_type = cfg.get("teacher_type", cfg.get("default_teacher_type"))  # e.g. "resnet152", "efficientnet_b2", "swin_tiny"
    print(f"[FineTune] ===== Now fine-tuning teacher: {teacher_type} =====")
    teacher_model = create_teacher_by_name(
        teacher_type,
        num_classes=num_classes,
        pretrained=cfg.get("teacher_pretrained", True),
        small_input=small_input,
        dropout_p=cfg.get("efficientnet_dropout", 0.3),
        cfg=cfg,
    ).to(device)

    # optional load ckpt
    if cfg.get("finetune_ckpt_path") and os.path.isfile(cfg["finetune_ckpt_path"]):
        teacher_model.load_state_dict(
            torch.load(
                cfg["finetune_ckpt_path"], map_location=device, weights_only=True
            ),
            strict=False,
        )
        print(f"[FineTune] ckpt exists → fine-tune 스킵 ({cfg['finetune_ckpt_path']})")
        # 평가만 한 번 찍고 바로 반환
        best_acc = eval_teacher(teacher_model, test_loader, device)
        print(f"[FineTune] testAcc={best_acc:.2f}")
        return

    # 3) partial freeze or full fine-tune?
    if cfg.get("finetune_partial_freeze", False):
        # e.g. freeze backbone, unfreeze head
        freeze_bn = cfg.get("teacher_freeze_bn", True)
        freeze_ln = cfg.get("teacher_freeze_ln", True)
        partial_freeze_teacher_auto(
            teacher_model,
            teacher_type,
            freeze_bn=freeze_bn,
            freeze_ln=freeze_ln,
            use_adapter=cfg.get("teacher_use_adapter", False),
            bn_head_only=cfg.get("teacher_bn_head_only", False),
            freeze_level=cfg.get("teacher_freeze_level", 1),
        )
        print("[FineTune] partial freeze mode => only head is trainable (example).")
    else:
        # full fine-tune => do nothing or freeze_all if you want the opposite
        print("[FineTune] full fine-tune => no partial freeze applied.")

    # 4) use cutmix or standard CE?
    use_cutmix = cfg.get("finetune_use_cutmix", True)
    cutmix_alpha = cfg.get("finetune_cutmix_alpha", 1.0)

    finetune_epochs = cfg.get("finetune_epochs", 10)
    lr = cfg.get("finetune_lr", 1e-3)
    weight_decay = cfg.get("finetune_weight_decay", 1e-4)
    ckpt_path = cfg.get("finetune_ckpt_path", "teacher_finetuned_cutmix.pth")
    ckpt_dir = os.path.dirname(ckpt_path)  # ''(빈 문자열) 이면 폴더 없는 케이스
    if ckpt_dir:                            # 폴더가 실제로 있을 때만
        os.makedirs(ckpt_dir, exist_ok=True)

    if use_cutmix:
        # => call finetune_teacher_cutmix
        print(f"[FineTune] Using CutMix alpha={cutmix_alpha}, epochs={finetune_epochs}, lr={lr}")
        teacher_model, best_acc = finetune_teacher_cutmix(
            teacher_model,
            train_loader,
            test_loader,
            alpha=cutmix_alpha,
            lr=lr,
            weight_decay=weight_decay,
            epochs=finetune_epochs,
            device=device,
            ckpt_path=ckpt_path,
            label_smoothing=cfg.get("label_smoothing", 0.0),
        )
    else:
        # => implement your own standard CE fine-tune loop or reuse a function
        teacher_model, best_acc = standard_ce_finetune(
            teacher_model,
            train_loader,
            test_loader,
            lr=lr,
            weight_decay=weight_decay,
            epochs=finetune_epochs,
            device=device,
            ckpt_path=ckpt_path,
            label_smoothing=cfg.get("label_smoothing", 0.0),
        )

    print(f"[FineTune] done => bestAcc={best_acc:.2f}, final ckpt={ckpt_path}")

if __name__ == "__main__":
    main()
