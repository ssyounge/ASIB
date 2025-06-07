# scripts/fine_tuning.py
"""
Example: Fine-tuning Teacher (ResNet/EfficientNet/Swin) on either CIFAR-100 or ImageNet100
using optional CutMix or standard CE training.

Usage:
  python fine_tuning.py --config configs/fine_tune.yaml
"""

import argparse
import os
import copy
import torch
import yaml

# data loaders
from data.cifar100 import get_cifar100_loaders
from data.imagenet100 import get_imagenet100_loaders

# teacher factories
from models.teachers.teacher_resnet import create_resnet101
from models.teachers.teacher_efficientnet import create_efficientnet_b2
from models.teachers.teacher_swin import create_swin_t

# partial freeze
from modules.partial_freeze import (
    partial_freeze_teacher_resnet,
    partial_freeze_teacher_efficientnet,
    partial_freeze_teacher_swin,
    freeze_all_params
)

# cutmix finetune
from modules.cutmix_finetune_teacher import finetune_teacher_cutmix, eval_teacher

def parse_args():
    parser = argparse.ArgumentParser(description="Teacher Fine-tuning Script")

    # ① YAML 기본값
    parser.add_argument("--config", type=str, default="configs/fine_tune.yaml",
                        help="Path to YAML config for fine-tuning")

    # ② run_many.sh 가 던지는 옵션들(없으면 None)
    parser.add_argument("--teacher_name", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--finetune_ckpt_path", type=str)

    # (필요 시 추가) lr·epoch 등 sweep 파라미터
    parser.add_argument("--finetune_lr", type=float)
    parser.add_argument("--finetune_epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--finetune_weight_decay", type=float)
    # ↓ run_many.sh 에서 CutMix 알파도 변경할 수 있도록
    parser.add_argument("--cutmix_alpha", type=float)

    # (공통) config 값을 key=value 형태로 덮어쓰기
    parser.add_argument("--override", type=str,
                        help="comma-separated KEY=VAL pairs to override config")
    
    return parser.parse_args()

def load_config(cfg_path):
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

def get_data_loaders(dataset_name, batch_size=128):
    """
    Returns train_loader, test_loader based on dataset_name.
    """
    if dataset_name == "cifar100":
        return get_cifar100_loaders(batch_size=batch_size)
    elif dataset_name == "imagenet100":
        return get_imagenet100_loaders(batch_size=batch_size)
    else:
        raise ValueError(f"Unknown dataset_name={dataset_name}")

def create_teacher_by_name(teacher_name, num_classes=100, pretrained=True):
    """
    Extends to handle resnet101, efficientnet_b2, swin_tiny, etc.
    """
    if teacher_name == "resnet101":
        return create_resnet101(num_classes=num_classes, pretrained=pretrained)
    elif teacher_name == "efficientnet_b2":
        return create_efficientnet_b2(num_classes=num_classes, pretrained=pretrained)
    elif teacher_name == "swin_tiny":
        return create_swin_t(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"[fine_tuning.py] Unknown teacher_name={teacher_name}")

def partial_freeze_teacher_auto(model, teacher_name, freeze_bn=True, freeze_ln=True):
    """
    If needed, partial freeze for fine-tune. Or you can freeze nothing if you want full fine-tune.
    """
    if teacher_name == "resnet101":
        partial_freeze_teacher_resnet(model, freeze_bn=freeze_bn, freeze_scope=None)
    elif teacher_name == "efficientnet_b2":
        partial_freeze_teacher_efficientnet(model, freeze_bn=freeze_bn, freeze_scope=None)
    elif teacher_name == "swin_tiny":
        partial_freeze_teacher_swin(model, freeze_ln=freeze_ln, freeze_scope=None)
    else:
        raise ValueError(f"Unknown teacher_name={teacher_name}")

def standard_ce_finetune(model, train_loader, test_loader,
                         lr, weight_decay, epochs, device, ckpt_path):
    model = model.to(device)
    optim = torch.optim.SGD(model.parameters(), lr=lr,
                            momentum=0.9, weight_decay=weight_decay)
    crit  = torch.nn.CrossEntropyLoss()
    best_acc = 0.0
    for ep in range(1, epochs+1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            _, logits, _ = model(x)          # teacher wrapper: (dict, logit, ce)
            loss = crit(logits, y)
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
    override_str = cli_cfg.pop("override", None)
    cfg = {**base_cfg, **cli_cfg}
    if override_str:
        from utils.misc import parse_override_str
        cfg.update(parse_override_str(override_str))
    device = cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("[Warning] No CUDA => Using CPU")
        device = "cpu"

    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)

    # 1) dataset
    dataset_name = cfg.get("dataset_name", "cifar100")
    batch_size   = cfg.get("batch_size", 128)
    train_loader, test_loader = get_data_loaders(dataset_name, batch_size=batch_size)

    # 2) teacher
    teacher_name = cfg.get("teacher_name", "resnet101")  # e.g. "resnet101", "efficientnet_b2", "swin_tiny"
    print(f"[FineTune] ===== Now fine-tuning teacher: {teacher_name} =====")
    teacher_model = create_teacher_by_name(
        teacher_name,
        num_classes=cfg.get("num_classes", 100),
        pretrained=cfg.get("teacher_pretrained", True)
    ).to(device)

    # optional load ckpt
    if cfg.get("finetune_ckpt_path") and os.path.isfile(cfg["finetune_ckpt_path"]):
        teacher_model.load_state_dict(
            torch.load(cfg["finetune_ckpt_path"], map_location=device)
        )
        print(f"[FineTune] ckpt exists → fine-tune 스킵 ({cfg['finetune_ckpt_path']})")
        # 평가만 한 번 찍고 바로 반환
        best_acc = eval_teacher(teacher_model, test_loader, device)
        print(f"[FineTune] testAcc={best_acc:.2f}")
        return

    # 3) partial freeze or full fine-tune?
    if cfg.get("fine_tune_partial_freeze", False):
        # e.g. freeze backbone, unfreeze head
        freeze_bn = cfg.get("teacher_freeze_bn", True)
        freeze_ln = cfg.get("teacher_freeze_ln", True)
        partial_freeze_teacher_auto(teacher_model, teacher_name, freeze_bn=freeze_bn, freeze_ln=freeze_ln)
        print("[FineTune] partial freeze mode => only head is trainable (example).")
    else:
        # full fine-tune => do nothing or freeze_all_params if you want the opposite
        print("[FineTune] full fine-tune => no partial freeze applied.")

    # 4) use cutmix or standard CE?
    use_cutmix = cfg.get("use_cutmix", True)
    cutmix_alpha = cfg.get("cutmix_alpha", 1.0)

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
            ckpt_path=ckpt_path
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
            ckpt_path=ckpt_path
        )

    print(f"[FineTune] done => bestAcc={best_acc:.2f}, final ckpt={ckpt_path}")

if __name__ == "__main__":
    main()
