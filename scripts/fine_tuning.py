# scripts/fine_tuning.py
"""
Example: Fine-tuning Teacher (ResNet/EfficientNet/Swin) on either CIFAR-100 or ImageNet100
using optional CutMix or standard CE training.

Usage:
  python scripts/fine_tuning.py --config-name base \
      +teacher_type=resnet152 +finetune_epochs=100 +finetune_lr=0.0005

Change datasets with a ``dataset.name`` override or a dataset YAML file.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import logging
from typing import Optional
import hydra
from omegaconf import DictConfig, OmegaConf
from utils.logging_utils import init_logger

from utils.misc import (
    set_random_seed,
    check_label_range,
    get_model_num_classes,
    get_amp_components,
)

# data loaders
from data.cifar100 import get_cifar100_loaders
from data.imagenet32 import get_imagenet32_loaders

# teacher factories
from models.teachers.resnet152_teacher import create_resnet152

# partial freeze
from modules.partial_freeze import (
    partial_freeze_teacher_resnet,
    partial_freeze_teacher_efficientnet,
)

# cutmix finetune
from modules.cutmix_finetune_teacher import finetune_teacher_cutmix, eval_teacher

def get_data_loaders(
    dataset_name, batch_size=128, num_workers=2, augment=True, root=None
):
    """
    Returns train_loader, test_loader based on dataset_name.
    """
    if dataset_name == "cifar100":
        return get_cifar100_loaders(
            root=root or "./data",
            batch_size=batch_size,
            num_workers=num_workers,
            augment=augment,
        )
    elif dataset_name == "imagenet32":
        return get_imagenet32_loaders(
            root or "./data/imagenet32",
            batch_size=batch_size,
            num_workers=num_workers,
        )
    else:
        raise ValueError(f"Unknown dataset_name={dataset_name}")

def create_teacher_by_name(
    teacher_type,
    num_classes=100,
    pretrained=True,
    small_input=False,
    dropout_p: float | None = None,
    cfg: Optional[dict] = None,
):
    """
    Extends to handle resnet152 and efficientnet_l2 models.
    """
    if teacher_type == "resnet152":
        return create_resnet152(
            num_classes=num_classes,
            pretrained=pretrained,
            small_input=small_input,
            cfg=cfg,
        )
    elif teacher_type in ("efficientnet_l2", "effnet_l2"):
        from models.teachers.efficientnet_l2_teacher import create_efficientnet_l2
        if dropout_p is None and cfg is not None:
            dropout_p = cfg.get("efficientnet_dropout", 0.3)
        return create_efficientnet_l2(
            num_classes=num_classes,
            pretrained=pretrained,
            small_input=small_input,
            dropout_p=dropout_p if dropout_p is not None else 0.3,
            use_checkpointing=cfg.get("teacher_use_checkpointing", False),
            cfg=cfg,
        )
    elif teacher_type in ("convnext_l_teacher", "convnext_l"):
        # registry 에 이미 올라와 있으므로 factory 호출
        from models.common.base_wrapper import MODEL_REGISTRY
        return MODEL_REGISTRY[teacher_type](
            num_classes=num_classes,
            pretrained=pretrained,
            small_input=small_input,
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
    if teacher_type == "resnet152":
        partial_freeze_teacher_resnet(
            model,
            freeze_bn=freeze_bn,
            use_adapter=use_adapter,
            bn_head_only=bn_head_only,
            freeze_level=freeze_level,
        )
    elif teacher_type in ("efficientnet_l2", "effnet_l2"):
        partial_freeze_teacher_efficientnet(
            model,
            freeze_bn=freeze_bn,
            use_adapter=use_adapter,
            bn_head_only=bn_head_only,
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
    cfg=None,
): 
    """Simple fine-tune loop using cross-entropy loss.

    Parameters
    ----------
    label_smoothing : float, optional
        Passed to ``CrossEntropyLoss``.
    """
    model = model.to(device)
    autocast_ctx, scaler = get_amp_components(cfg or {})
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(
            cfg.get("adam_beta1", 0.9) if cfg is not None else 0.9,
            cfg.get("adam_beta2", 0.999) if cfg is not None else 0.999,
        ),
    )
    crit  = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    best_acc = 0.0
    for ep in range(1, epochs+1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            with autocast_ctx:
                out = model(x)
                if isinstance(out, tuple):
                    logits = out[1]
                elif isinstance(out, dict):
                    logits = out["logit"]
                else:
                    logits = out
                loss = crit(logits, y)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                optim.step()
        acc = eval_teacher(model, test_loader, device, cfg=cfg)
        if acc > best_acc:
            best_acc = acc
            ckpt_dir = os.path.dirname(ckpt_path)
            if ckpt_dir:
                os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
    return model, best_acc

@hydra.main(config_path="../configs", config_name="base", version_base="1.3")
def main(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    from utils.config_utils import flatten_hydra_config
    cfg = flatten_hydra_config(cfg)
    init_logger(cfg.get("log_level", "INFO"))
    device = cfg.get("device", "cuda")
    if device == "cuda":
        if torch.cuda.is_available():
            os.environ.setdefault(
                "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
            )
        else:
            logging.warning("[FineTune] No CUDA => Using CPU")
            device = "cpu"

    seed = cfg.get("seed", 42)
    deterministic = cfg.get("deterministic", True)
    set_random_seed(seed, deterministic=deterministic)

    # 1) dataset
    dataset_name = cfg.get("dataset_name", "cifar100")
    batch_size   = cfg.get("batch_size", 128)
    if cfg.get("class_subset"):
        from data.cifar100_overlap import _make_loader
        subset = [int(x) for x in str(cfg.get("class_subset")).split(",")]
        train_loader = _make_loader(
            subset,
            True,
            batch_size,
            cfg.get("num_workers", 2),
            cfg.get("data_aug", True),
        )
        test_loader = _make_loader(
            subset,
            False,
            batch_size,
            cfg.get("num_workers", 2),
            False,
        )
    else:
        train_loader, test_loader = get_data_loaders(
            dataset_name,
            batch_size=batch_size,
            num_workers=cfg.get("num_workers", 2),
            augment=cfg.get("data_aug", True),
            root=cfg.get("data_root"),
        )

    if isinstance(train_loader.dataset, torch.utils.data.ConcatDataset):
        # 첫 번째 Subset → 원본 CIFAR‑100 Dataset
        origin_ds = train_loader.dataset.datasets[0].dataset
    elif isinstance(train_loader.dataset, torch.utils.data.Subset):
        origin_ds = train_loader.dataset.dataset
    else:
        origin_ds = train_loader.dataset
    num_classes = len(origin_ds.classes)
    check_label_range(train_loader.dataset, num_classes)
    check_label_range(test_loader.dataset, num_classes)

    small_input = cfg.get("small_input")
    if small_input is None:
        small_input = dataset_name in ("cifar100", "imagenet32")

    # 2) teacher
    teacher_type = cfg.get("teacher_type", cfg.get("default_teacher_type"))
    logging.info(
        "[FineTune] ===== Now fine-tuning teacher: %s =====", teacher_type
    )
    teacher_model = create_teacher_by_name(
        teacher_type,
        num_classes=num_classes,
        pretrained=cfg.get("teacher_pretrained", True),
        small_input=small_input,
        dropout_p=cfg.get("efficientnet_dropout"),
        cfg=cfg,
    ).to(device)

    model_classes = get_model_num_classes(teacher_model)
    if model_classes != num_classes:
        raise ValueError(
            f"Teacher head expects {model_classes} classes but dataset provides {num_classes}"
        )

    # optional load ckpt (unless force_refinetune=True)
    ckpt_path_cfg = cfg.get("finetune_ckpt_path")
    if ckpt_path_cfg and os.path.isfile(ckpt_path_cfg) and not cfg.get("force_refinetune", False):
        teacher_model.load_state_dict(
            torch.load(
                ckpt_path_cfg, map_location=device, weights_only=True
            ),
            strict=False,
        )
        logging.info("[FineTune] ckpt exists → fine-tune 스킵 (%s)", ckpt_path_cfg)
        # 평가만 한 번 찍고 바로 반환
        best_acc = eval_teacher(teacher_model, test_loader, device)
        logging.info("[FineTune] testAcc=%.2f", best_acc)
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
            use_adapter=cfg.get("use_distillation_adapter", False),
            bn_head_only=cfg.get("teacher_bn_head_only", False),
            freeze_level=cfg.get("teacher_freeze_level", 1),
        )
        logging.info("[FineTune] partial freeze mode => only head is trainable (example).")
    else:
        # full fine-tune => do nothing or freeze_all if you want the opposite
        logging.info("[FineTune] full fine-tune => no partial freeze applied.")

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
        logging.info(
            "[FineTune] Using CutMix alpha=%s, epochs=%s, lr=%s",
            cutmix_alpha,
            finetune_epochs,
            lr,
        )
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
            cfg=cfg,
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
            cfg=cfg,
        )

    logging.info(
        "[FineTune] done => bestAcc=%.2f, final ckpt=%s", best_acc, ckpt_path
    )

if __name__ == "__main__":
    main()
