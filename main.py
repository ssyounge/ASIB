#!/usr/bin/env python3
"""
main.py

Implements a multi-stage distillation flow using:
 (A) Teacher adaptive update (Teacher/MBM partial freeze)
 (B) Student distillation
Repeated for 'num_stages' times, as in ASMB multi-stage self-training.
"""

import logging
import os
import json
import torch
from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from utils.logger import ExperimentLogger
from utils.logging_setup import get_logger
# --- 중복 방지 플래그
_HP_LOGGED = False
# — W&B --------------------------------------------------------
try:
    import wandb
except ModuleNotFoundError:
    wandb = None
from utils.misc import set_random_seed, check_label_range, get_model_num_classes
from modules.disagreement import compute_disagreement_rate
from modules.trainer_teacher import teacher_adaptive_update
from modules.trainer_student import student_distillation_update
from data.cifar100 import get_cifar100_loaders
from data.imagenet32 import get_imagenet32_loaders

# partial freeze
from modules.partial_freeze import (
    apply_partial_freeze,
    partial_freeze_teacher_resnet,
    partial_freeze_teacher_efficientnet,
    partial_freeze_teacher_swin,
    partial_freeze_student_resnet,
    partial_freeze_student_efficientnet,
    partial_freeze_student_swin
)

# Teacher creation (factory):
from models.teachers.teacher_resnet import create_resnet101
from models.teachers.teacher_efficientnet import create_efficientnet_b2
from models.teachers.teacher_swin import create_swin_t

def create_student_by_name(
    student_name: str,
    pretrained: bool = True,
    small_input: bool = False,
    num_classes: int = 100,
    cfg: Optional[dict] = None,
):
    """
    Returns a student model that follows the common interface
    (feature_dict, logits, ce_loss).
    """
    if student_name == "resnet_adapter":
        from models.students.student_resnet_adapter import (
            create_resnet101_with_extended_adapter,
        )
        return create_resnet101_with_extended_adapter(
            pretrained=pretrained,
            num_classes=num_classes,
            small_input=small_input,
        )

    elif student_name == "resnet152_adapter":
        from models.students.student_resnet152_adapter import (
            create_resnet152_with_extended_adapter,
        )
        return create_resnet152_with_extended_adapter(
            pretrained=pretrained,
            num_classes=num_classes,
            small_input=small_input,
        )

    elif student_name == "efficientnet_adapter":
        from models.students.student_efficientnet_adapter import (
            create_efficientnet_b2_with_adapter,
        )
        return create_efficientnet_b2_with_adapter(
            pretrained=pretrained,
            num_classes=num_classes,
            small_input=small_input,
        )

    elif student_name == "swin_adapter":
        from models.students.student_swin_adapter import (
            create_swin_adapter_student,
        )
        adapter_dim = 64
        if cfg is not None:
            adapter_dim = cfg.get("swin_adapter_dim", adapter_dim)
        return create_swin_adapter_student(
            pretrained=pretrained,
            small_input=small_input,
            num_classes=num_classes,
            adapter_dim=adapter_dim,
            cfg=cfg,
        )

    else:
        raise ValueError(
            f"[create_student_by_name] unknown student_name={student_name}"
        )

from models.mbm import build_from_teachers



def create_teacher_by_name(
    teacher_name,
    num_classes=100,
    pretrained=True,
    small_input=False,
    cfg: Optional[dict] = None,
):
    if teacher_name == "resnet101":
        return create_resnet101(
            num_classes=num_classes,
            pretrained=pretrained,
            small_input=small_input,
            cfg=cfg,
        )
    elif teacher_name == "resnet152":
        from models.teachers.teacher_resnet import create_resnet152
        return create_resnet152(
            num_classes=num_classes,
            pretrained=pretrained,
            small_input=small_input,
            cfg=cfg,
        )
    elif teacher_name == "efficientnet_b2":
        return create_efficientnet_b2(
            num_classes=num_classes,
            pretrained=pretrained,
            small_input=small_input,
            cfg=cfg,
        )
    elif teacher_name == "swin_tiny":
        return create_swin_t(
            num_classes=num_classes,
            pretrained=pretrained,
            cfg=cfg,
        )
    else:
        raise ValueError(f"[create_teacher_by_name] Unknown teacher_name={teacher_name}")

def partial_freeze_teacher_auto(
    model,
    teacher_name,
    freeze_bn=True,
    freeze_ln=True,
    use_adapter=False,
    bn_head_only=False,
    freeze_level=1,
    train_distill_adapter_only=False,
):
    if teacher_name == "resnet101" or teacher_name == "resnet152":
        partial_freeze_teacher_resnet(
            model,
            freeze_bn=freeze_bn,
            use_adapter=use_adapter,
            bn_head_only=bn_head_only,
            freeze_level=freeze_level,
            train_distill_adapter_only=train_distill_adapter_only,
        )
    elif teacher_name == "efficientnet_b2":
        partial_freeze_teacher_efficientnet(
            model,
            freeze_bn=freeze_bn,
            use_adapter=use_adapter,
            bn_head_only=bn_head_only,
            freeze_level=freeze_level,
            train_distill_adapter_only=train_distill_adapter_only,
        )
    elif teacher_name == "swin_tiny":
        partial_freeze_teacher_swin(
            model,
            freeze_ln=freeze_ln,
            use_adapter=use_adapter,
            freeze_level=freeze_level,
            train_distill_adapter_only=train_distill_adapter_only,
        )
    else:
        raise ValueError(f"[partial_freeze_teacher_auto] Unknown teacher_name={teacher_name}")

def partial_freeze_student_auto(
    model,
    student_name="resnet_adapter",
    freeze_bn=True,
    freeze_ln=True,
    use_adapter=False,
    freeze_level=1,
):
    if freeze_level < 0:
        for p in model.parameters():
            p.requires_grad = True
        return
    if student_name == "resnet_adapter":
        partial_freeze_student_resnet(
            model,
            freeze_bn=freeze_bn,
            use_adapter=use_adapter,
            freeze_level=freeze_level,
        )
    elif student_name == "efficientnet_adapter":
        partial_freeze_student_efficientnet(
            model,
            freeze_bn=freeze_bn,
            use_adapter=use_adapter,
            freeze_level=freeze_level,
        )
    elif student_name == "swin_adapter":
        partial_freeze_student_swin(
            model,
            freeze_ln=freeze_ln,
            use_adapter=use_adapter,
            freeze_level=freeze_level,
        )
    else:
        partial_freeze_student_resnet(
            model,
            freeze_bn=freeze_bn,
            freeze_level=freeze_level,
        )

@hydra.main(config_path="configs", config_name="base", version_base="1.3")
def main(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    from utils.config_utils import flatten_hydra_config
    cfg = flatten_hydra_config(cfg)

    fl = cfg.get("student_freeze_level", -1)
    if fl is None:
        fl = -1
    cfg["student_freeze_level"] = fl

    # ------------------------------------------------------------------
    # (NEW)  student_pretrained 기본값 자동 결정
    #   · freeze_level ≥0  →  pretrained 켜기
    #   · freeze_level  <0 →  scratch (꺼두기)
    # ------------------------------------------------------------------
    if "student_pretrained" not in cfg:
        cfg["student_pretrained"] = bool(fl >= 0)
        print(
            f"[Auto-cfg] student_pretrained\u2190{cfg['student_pretrained']} "
            f"(freeze_level={fl})"
        )

    if fl >= 0 and not cfg.get("student_pretrained", False):
        print(
            "[Warn] freeze_level ≥0 인데 student_pretrained=False ‑‑ "
            "동결된 층이 랜덤 초기화 상태가 됩니다."
        )

    # --------------- α/β 재정규화 (한 번만) -------------
    def _renorm_ce_kd(d):
        if "ce_alpha" in d and "kd_alpha" in d:
            ce, kd = float(d["ce_alpha"]), float(d["kd_alpha"])
            if abs(ce + kd - 1) > 1e-5:
                tot = ce + kd
                d["ce_alpha"], d["kd_alpha"] = ce / tot, kd / tot
                print(
                    f"[Auto-cfg] ce_alpha+kd_alpha \u22601 \u2192 \uc7ac\uc815\uaddc\ud654 "
                    f"(ce={d['ce_alpha']:.3f}, kd={d['kd_alpha']:.3f})"
                )

    _renorm_ce_kd(cfg)

    # ------------------------------------------------------------------
    # [Safety switch]  partial freeze OFF → freeze_level = -1 (no‑freeze)
    #  — sweep 중 조건부 파라미터 처리가 어려우므로 코드에서 보정
    # ------------------------------------------------------------------
    if not cfg.get("use_partial_freeze", False):
        cfg["student_freeze_level"]  = -1
        cfg["teacher1_freeze_level"] = -1
        cfg["teacher2_freeze_level"] = -1
    # ------------------------------------------- #

    # ── tqdm 전체 OFF ─────────────────────────────
    if cfg.get("disable_tqdm", False):
        os.environ["PROGRESS"] = "0"    # utils.progress 에서 사용

    # ── W&B API-key (config 우선) ────────────────
    if cfg.get("use_wandb", False) and cfg.get("wandb_api_key", ""):
        os.environ.setdefault("WANDB_API_KEY", str(cfg["wandb_api_key"]))

    # ────────────────── LOGGING & W&B ──────────────────
    exp_dir = cfg.get("results_dir", ".")
    logger = get_logger(
        exp_dir,
        level=cfg.get("log_level", "INFO"),
        stream_level="INFO" if cfg.get("log_level", "INFO").upper() == "DEBUG" else cfg.get("log_level", "INFO")
    )

    global _HP_LOGGED
    if not _HP_LOGGED:
        safe_cfg = {k: v for k, v in cfg.items() if not isinstance(v, logging.Logger)}
        logger.info("HParams:\n%s", json.dumps(safe_cfg, indent=2))
        if wandb is not None and wandb.run:
            wandb.config.update(cfg, allow_val_change=False)
        _HP_LOGGED = True

    if cfg.get("use_wandb", False):
        if wandb is None:
            logging.warning("[W&B] wandb not installed ‑‑ skipping")
        else:
            wandb.init(
                project=cfg.get("wandb_project", "kd_monitor"),
                entity=cfg.get("wandb_entity") or None,
                name=cfg.get("wandb_run_name") or cfg.get("exp_id", "run"),
                config=cfg,
            )
            logging.info("[W&B] dashboard → %s", wandb.run.url)

    # convert learning rate and weight decay fields to float when merged
    for key in (
        "teacher_weight_decay",
        "student_weight_decay",
        "finetune_weight_decay",
        "teacher_lr",
        "student_lr",
        "finetune_lr",
    ):
        if key in cfg:
            cfg[key] = float(cfg[key])

    exp_logger = ExperimentLogger(cfg, exp_name="asmb_experiment")
    exp_logger.update_metric("use_amp", cfg.get("use_amp", False))
    exp_logger.update_metric("amp_dtype", cfg.get("amp_dtype", "float16"))
    exp_logger.update_metric("mbm_type", cfg.get("mbm_type", "MLP"))
    exp_logger.update_metric("mbm_r", cfg.get("mbm_r"))
    exp_logger.update_metric("mbm_n_head", cfg.get("mbm_n_head"))
    exp_logger.update_metric("mbm_learnable_q", cfg.get("mbm_learnable_q"))
    # overlap_pct 기록 (없으면 -1)
    cfg_overlap = cfg.get("overlap_pct", -1)
    exp_logger.update_metric("overlap_pct", cfg_overlap)
    if wandb and wandb.run:
        wandb.run.summary["overlap_pct"] = cfg_overlap

    device = cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("[Warning] No CUDA => Using CPU")
        device = "cpu"

    # fix seed
    seed = cfg.get("seed", 42)
    deterministic = cfg.get("deterministic", True)
    set_random_seed(seed, deterministic=deterministic)

    # 3) Data
    dataset = cfg.get("dataset_name", "cifar100")
    batch_size = cfg.get("batch_size", 128)
    data_root = cfg.get("data_root", "./data")
    if cfg.get("overlap_pct", -1) >= 0:
        from data.cifar100_overlap import get_overlap_loaders
        (A_tr, A_te), (B_tr, B_te), _ = get_overlap_loaders(
            pct_overlap=cfg["overlap_pct"],
            batch_size=batch_size,
            num_workers=cfg.get("num_workers", 2),
            augment=cfg.get("data_aug", True),
            seed=seed,
        )
        # 학생은 **두 교사 클래스의 합집합**을 모두 보게 해야 함
        train_loader = torch.utils.data.ConcatDataset([A_tr.dataset, B_tr.dataset])
        test_loader  = torch.utils.data.ConcatDataset([A_te.dataset, B_te.dataset])
        train_loader = torch.utils.data.DataLoader(
            train_loader, batch_size=batch_size, shuffle=True,
            num_workers=cfg.get("num_workers", 2), pin_memory=True
        )
        test_loader  = torch.utils.data.DataLoader(
            test_loader,  batch_size=batch_size, shuffle=False,
            num_workers=cfg.get("num_workers", 2), pin_memory=True
        )
    elif dataset == "cifar100":
        train_loader, test_loader = get_cifar100_loaders(
            root=data_root,
            batch_size=batch_size,
            num_workers=cfg.get("num_workers", 2),
            augment=cfg.get("data_aug", True),
        )
    elif dataset == "imagenet32":
        train_loader, test_loader = get_imagenet32_loaders(
            root=data_root,
            batch_size=batch_size,
            num_workers=cfg.get("num_workers", 2),
        )
    else:
        raise ValueError(f"Unknown dataset_name={dataset}")

    if isinstance(train_loader.dataset, torch.utils.data.ConcatDataset):
        num_classes = 100   # CIFAR-100 fixed
    else:
        num_classes = len(train_loader.dataset.classes)
    cfg["num_classes"] = num_classes
    exp_logger.update_metric("num_classes", num_classes)
    check_label_range(train_loader.dataset, num_classes)
    check_label_range(test_loader.dataset, num_classes)

    small_input = cfg.get("small_input")
    if small_input is None:
        small_input = dataset == "cifar100"

    # 4) Create teacher1, teacher2
    teacher1_type = cfg.get("teacher1_type", "resnet152")
    teacher2_type = cfg.get("teacher2_type", "resnet152")

    teacher1_ckpt_path = cfg.get("teacher1_ckpt", f"./checkpoints/{teacher1_type}_ft.pth")
    teacher1 = create_teacher_by_name(
        teacher_name=teacher1_type,
        num_classes=num_classes,
        pretrained=cfg.get("teacher1_pretrained", True),
        small_input=small_input,
        cfg=cfg,
    ).to(device)
    model_classes = get_model_num_classes(teacher1)
    if model_classes != num_classes:
        raise ValueError(
            f"Teacher1 head expects {model_classes} classes but dataset provides {num_classes}"
        )

    if os.path.exists(teacher1_ckpt_path):
        teacher1.load_state_dict(
            torch.load(teacher1_ckpt_path, map_location=device, weights_only=True),
            strict=False,
        )
        print(f"[Main] Loaded teacher1 from {teacher1_ckpt_path}")

    if cfg.get("use_partial_freeze", True):
        partial_freeze_teacher_auto(
            teacher1, teacher1_type,
            freeze_bn=cfg.get("teacher1_freeze_bn", True),
            freeze_ln=cfg.get("teacher1_freeze_ln", True),
            use_adapter=cfg.get("teacher1_use_adapter", False),
            bn_head_only=cfg.get("teacher1_bn_head_only", False),
            freeze_level=cfg.get("teacher1_freeze_level", 1),
            train_distill_adapter_only=cfg.get("use_distillation_adapter", False),
        )

    teacher2_ckpt_path = cfg.get("teacher2_ckpt", f"./checkpoints/{teacher2_type}_ft.pth")
    teacher2 = create_teacher_by_name(
        teacher_name=teacher2_type,
        num_classes=num_classes,
        pretrained=cfg.get("teacher2_pretrained", True),
        small_input=small_input,
        cfg=cfg,
    ).to(device)
    model_classes = get_model_num_classes(teacher2)
    if model_classes != num_classes:
        raise ValueError(
            f"Teacher2 head expects {model_classes} classes but dataset provides {num_classes}"
        )

    if os.path.exists(teacher2_ckpt_path):
        teacher2.load_state_dict(
            torch.load(teacher2_ckpt_path, map_location=device, weights_only=True),
            strict=False,
        )
        print(f"[Main] Loaded teacher2 from {teacher2_ckpt_path}")

    if cfg.get("use_partial_freeze", True):
        partial_freeze_teacher_auto(
            teacher2, teacher2_type,
            freeze_bn=cfg.get("teacher2_freeze_bn", True),
            freeze_ln=cfg.get("teacher2_freeze_ln", True),
            use_adapter=cfg.get("teacher2_use_adapter", False),
            bn_head_only=cfg.get("teacher2_bn_head_only", False),
            freeze_level=cfg.get("teacher2_freeze_level", 1),
            train_distill_adapter_only=cfg.get("use_distillation_adapter", False),
        )

    # optional fine-tuning of teachers before ASMB stages
    finetune_epochs = int(cfg.get("finetune_epochs", 0))
    if finetune_epochs > 0:
        from modules.cutmix_finetune_teacher import (
            finetune_teacher_cutmix,
            standard_ce_finetune,
        )

        print(f"[Main] Fine-tuning teachers for {finetune_epochs} epochs")
        lr = cfg.get("finetune_lr", 1e-3)
        wd = cfg.get("finetune_weight_decay", 1e-4)
        use_cutmix = bool(cfg.get("finetune_use_cutmix", True))
        alpha = cfg.get("finetune_alpha", 1.0)
        ckpt1 = cfg.get("finetune_ckpt1", "teacher1_finetuned.pth")
        ckpt2 = cfg.get("finetune_ckpt2", "teacher2_finetuned.pth")

        if use_cutmix:
            teacher1, _ = finetune_teacher_cutmix(
                teacher1,
                train_loader,
                test_loader,
                alpha=alpha,
                lr=lr,
                weight_decay=wd,
                epochs=finetune_epochs,
                device=device,
                ckpt_path=ckpt1,
            )
            teacher2, _ = finetune_teacher_cutmix(
                teacher2,
                train_loader,
                test_loader,
                alpha=alpha,
                lr=lr,
                weight_decay=wd,
                epochs=finetune_epochs,
                device=device,
                ckpt_path=ckpt2,
            )
        else:
            teacher1, _ = standard_ce_finetune(
                teacher1,
                train_loader,
                test_loader,
                lr=lr,
                weight_decay=wd,
                epochs=finetune_epochs,
                device=device,
                ckpt_path=ckpt1,
            )
            teacher2, _ = standard_ce_finetune(
                teacher2,
                train_loader,
                test_loader,
                lr=lr,
                weight_decay=wd,
                epochs=finetune_epochs,
                device=device,
                ckpt_path=ckpt2,
            )

    # Evaluate teacher performance before distillation begins
    from modules.cutmix_finetune_teacher import eval_teacher

    te1_acc = eval_teacher(teacher1, test_loader, device=device, cfg=cfg)
    te2_acc = eval_teacher(teacher2, test_loader, device=device, cfg=cfg)
    print(f"[Main] Teacher1 ({teacher1_type}) testAcc={te1_acc:.2f}%")
    print(f"[Main] Teacher2 ({teacher2_type}) testAcc={te2_acc:.2f}%")
    exp_logger.update_metric("teacher1_test_acc", te1_acc)
    exp_logger.update_metric("teacher2_test_acc", te2_acc)

    # 5) Student
    student_name  = cfg.get("student_type", "resnet_adapter")   # e.g. resnet_adapter / efficientnet_adapter / swin_adapter
    student_model = create_student_by_name(
        student_name,
        pretrained=cfg.get("student_pretrained", True),
        small_input=small_input,
        num_classes=num_classes,
        cfg=cfg,
    ).to(device)

    # ── NEW: ① 학생 feature dim 자동 추론 → mbm_query_dim -----------------
    if cfg.get("mbm_query_dim", 0) in (0, None):
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
            dummy = torch.randn(1, 3, 32, 32, device=device)
            feat_dict, _, _ = student_model(dummy)
            qdim = feat_dict.get("distill_feat", feat_dict.get("feat_2d")).shape[-1]
            cfg["mbm_query_dim"] = int(qdim)
            print(f"[Auto-cfg] mbm_query_dim ← {qdim}")

    # ── NEW: ② 문자열 숫자 → float/int 캐스팅 --------------------------------
    _num_keys = [
        "teacher_lr",
        "student_lr",
        "teacher_weight_decay",
        "student_weight_decay",
        "reg_lambda",
        "mbm_reg_lambda",
        "kd_alpha",
        "ce_alpha",
        "ib_beta",
    ]
    for k in _num_keys:
        if k in cfg and isinstance(cfg[k], str):
            try:
                cfg[k] = float(cfg[k])
            except ValueError:
                pass  # ignore


    if cfg.get("student_ckpt"):
        student_model.load_state_dict(
            torch.load(
                cfg["student_ckpt"], map_location=device, weights_only=True
            ),
            strict=False,
        )
        print(f"[Main] Loaded student from {cfg['student_ckpt']}")

    apply_partial_freeze(
        student_model,
        cfg.get("student_freeze_level", -1),
        cfg.get("student_freeze_bn", False),
    )

    # ───────────────────────── debug: trainable 파라미터 개수 로그 ──────────────
    def _count_trainable(m):
        """Return the number of trainable elements in ``m``."""
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    n_trainable = _count_trainable(student_model)
    print(f"[Debug] Student trainable **elements** → {n_trainable:,}")
    if cfg.get("debug_verbose", True):
        # freeze‑level 검증
        frz_lvl = cfg.get("student_freeze_level", -1)
        n_requires_grad = sum(1 for p in student_model.parameters() if p.requires_grad)
        print(f"[DBG] freeze_level={frz_lvl}, tensors_grad={n_requires_grad}")
    exp_logger.update_metric("n_trainable_student", n_trainable)

    # Obtain student feature dimension for MBM defaults
    feat_dim = None
    if hasattr(student_model, "get_feat_dim"):
        feat_dim = student_model.get_feat_dim()
        if cfg.get("mbm_out_dim") in (None, 0):
            cfg["mbm_out_dim"] = feat_dim
            exp_logger.update_metric("mbm_out_dim", feat_dim)
            print(
                f"[Info] mbm_out_dim set to student feature dimension {feat_dim}"
            )
        elif cfg["mbm_out_dim"] != feat_dim:
            print(
                f"[Warning] mbm_out_dim ({cfg['mbm_out_dim']}) does not match the student feature dimension ({feat_dim})."
            )

        # 이미 student_feat_dim 을 이용해 mbm_query_dim 은 채워놓은 상태
        if "mbm_query_dim" not in cfg or cfg["mbm_query_dim"] <= 0:
            cfg["mbm_query_dim"] = feat_dim
            print(f"[Auto-cfg] mbm_query_dim ← {cfg['mbm_query_dim']}")

        # mbm_out_dim 이 query_dim 과 다르면 경고만 뜨는데,
        # 특별한 이유가 없으면 동일하게 맞춰 주는 편이 안전하다.
        if cfg.get("mbm_out_dim") != cfg["mbm_query_dim"]:
            print(
                f"[Auto-cfg] mbm_out_dim 조정: {cfg['mbm_out_dim']} → {cfg['mbm_query_dim']}"
            )
            cfg["mbm_out_dim"] = cfg["mbm_query_dim"]

    # Validate or infer MBM query dimension
    mbm_query_dim = cfg.get("mbm_query_dim", 0)
    if cfg.get("mbm_type", "MLP") == "LA":
        if mbm_query_dim <= 0:
            if feat_dim is not None:
                mbm_query_dim = feat_dim
                cfg["mbm_query_dim"] = mbm_query_dim
                print(
                    f"[Info] mbm_query_dim not specified; using student feature dimension {mbm_query_dim}"
                )
            else:
                print(
                    "[Warning] Student model does not expose get_feat_dim(); please set mbm_query_dim manually"
                )
        elif feat_dim is not None and mbm_query_dim != feat_dim:
            raise ValueError(
                f"mbm_query_dim ({mbm_query_dim}) does not match the student feature dimension ({feat_dim})."
            )

    # 6) MBM and synergy head
    mbm_query_dim = cfg.get("mbm_query_dim")
    mbm, synergy_head = build_from_teachers(
        [teacher1, teacher2], cfg, query_dim=mbm_query_dim
    )
    mbm = mbm.to(device)
    synergy_head = synergy_head.to(device)

    # 7) teacher wrappers
    teacher_wrappers = [teacher1, teacher2]

    # 7b) create optimizers and schedulers
    num_stages = cfg.get("num_stages", 2)

    teacher_params = []
    use_da = cfg.get("use_distillation_adapter", False)
    for tw in teacher_wrappers:
        src = (
            tw.distillation_adapter.parameters()
            if use_da and hasattr(tw, "distillation_adapter")
            else tw.parameters()
        )
        for p in src:
            if p.requires_grad:
                teacher_params.append(p)
    mbm_params = [p for p in mbm.parameters() if p.requires_grad]
    syn_params = [p for p in synergy_head.parameters() if p.requires_grad]

    teacher_optimizer = optim.Adam(
        [
            {"params": teacher_params, "lr": cfg["teacher_lr"]},
            {
                "params": mbm_params,
                "lr": cfg["teacher_lr"] * cfg.get("mbm_lr_factor", 1.0),
            },
            {
                "params": syn_params,
                "lr": cfg["teacher_lr"] * cfg.get("mbm_lr_factor", 1.0),
            },
        ],
        weight_decay=cfg["teacher_weight_decay"],
        betas=(
            cfg.get("adam_beta1", 0.9),
            cfg.get("adam_beta2", 0.999),
        ),
    )

    teacher_total_epochs = num_stages * cfg.get("teacher_iters", cfg.get("teacher_adapt_epochs", 5))
    if cfg.get("lr_schedule", "step") == "cosine":
        teacher_scheduler = CosineAnnealingLR(teacher_optimizer, T_max=teacher_total_epochs)
    else:
        teacher_scheduler = StepLR(
            teacher_optimizer,
            step_size=cfg.get("teacher_step_size", 10),
            gamma=cfg.get("teacher_gamma", 0.1),
        )

    # ↓ requires_grad=True 파라미터만 Optimizer에 등록
    student_optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, student_model.parameters()),
        lr=cfg["student_lr"],
        weight_decay=cfg["student_weight_decay"],
        betas=(
            cfg.get("adam_beta1", 0.9),
            cfg.get("adam_beta2", 0.999),
        ),
        eps=1e-8,
    )

    student_total_epochs = num_stages * cfg.get("student_iters", cfg.get("student_epochs_per_stage", 15))
    if cfg.get("lr_schedule", "step") == "cosine":
        student_scheduler = CosineAnnealingLR(student_optimizer, T_max=student_total_epochs)
    else:
        student_scheduler = StepLR(
            student_optimizer,
            step_size=cfg.get("student_step_size", 10),
            gamma=cfg.get("student_gamma", 0.1),
        )

    # ------------------------------------------------------------
    # 8) IID vs. Continual-Learning loop
    # ------------------------------------------------------------

    if cfg.get("cl_mode", False):
        print("[main] Continual-Learning mode ON (β=%.3f)" % cfg.get("ib_beta", 0.01))

        from utils.data import get_split_cifar100_loaders
        from utils.cl_utils import ReplayBuffer, EWC
        from modules.trainer_student import eval_student

        num_tasks = cfg.get("num_tasks", 5)
        replay_ratio = cfg.get("replay_ratio", 0.5)
        replay_cap = cfg.get("replay_capacity", 2000)

        task_loaders = get_split_cifar100_loaders(
            num_tasks=num_tasks,
            batch_size=cfg.get("batch_size", 128),
            augment=cfg.get("data_aug", True),
            root=cfg.get("data_root", "./data"),
        )

        buffer = ReplayBuffer(replay_cap)
        ewc = EWC(cfg.get("lambda_ewc", 0.4))
        device = cfg["device"]

        global_ep = 0
        for task_id, (tl, vl) in enumerate(task_loaders):
            print(f"\n=== Task {task_id + 1}/{num_tasks} ===")
            student_model.train()
            epochs = cfg.get("epochs", 1)
            for ep in range(epochs):
                for x, y in tl:
                    x, y = x.to(device), y.to(device)
                    r_bs = int(replay_ratio * x.size(0))
                    if len(buffer.buffer) == 0:
                        x_batch, y_batch = x, y
                    else:
                        xs, ys, _ = buffer.sample(r_bs, device=device)
                        x_batch = torch.cat([x, xs], dim=0)
                        y_batch = torch.cat([y, ys], dim=0)

                    out = student_model(x_batch)
                    if isinstance(out, tuple):
                        out = out[1]
                    loss = torch.nn.functional.cross_entropy(out, y_batch)
                    loss += ewc.penalty(student_model)

                    student_optimizer.zero_grad()
                    loss.backward()
                    student_optimizer.step()

                global_ep += 1
                if student_scheduler is not None:
                    student_scheduler.step()

            # add to replay buffer and update fisher
            buffer.add(tl.dataset, task_id)
            ewc.update_fisher(student_model, tl, device=device)

            acc = eval_student(student_model, vl, device, cfg)
            exp_logger.update_metric(f"task{task_id + 1}_acc", acc)
            print(f"[Task {task_id + 1}] acc={acc:.2f}%")
    else:
        global_ep = 0
        for stage_id in range(1, num_stages + 1):
            print(f"\n=== Stage {stage_id}/{num_stages} ===")

            # ---------- DEBUG: disagreement weight 파라미터 확인 ----------
            dbg_mode = cfg.get("disagree_mode", "both_wrong")
            dbg_lh   = cfg.get("disagree_lambda_high", 1.2)
            dbg_ll   = cfg.get("disagree_lambda_low", 0.8)
            print(
                f"[DBG] disagree_mode={dbg_mode}, "
                f"λ_high={dbg_lh}, λ_low={dbg_ll}"
            )
            # --------------------------------------------------------------

            teacher_epochs = cfg.get("teacher_iters", cfg.get("teacher_adapt_epochs", 5))
            student_epochs = cfg.get("student_iters", cfg.get("student_epochs_per_stage", 15))

            # (A) Teacher adaptive update
            teacher_adaptive_update(
                teacher_wrappers=teacher_wrappers,
                mbm=mbm,
                synergy_head=synergy_head,
                student_model=student_model,
                trainloader=train_loader,
                testloader=test_loader,
                cfg=cfg,
                logger=exp_logger,
                optimizer=teacher_optimizer,
                scheduler=teacher_scheduler,
                global_ep=global_ep,
            )

            global_ep += teacher_epochs

            dis_rate = compute_disagreement_rate(
                teacher1,
                teacher2,
                test_loader,
                device=device,
                cfg=cfg,
                mode=cfg.get("disagree_mode", "both_wrong"),
            )
            print(f"[Stage {stage_id}] Teacher disagreement= {dis_rate:.2f}%")
            exp_logger.update_metric(f"stage{stage_id}_disagreement_rate", dis_rate)

            # (B) Student distillation
            final_acc = student_distillation_update(
                teacher_wrappers=teacher_wrappers,
                mbm=mbm,
                synergy_head=synergy_head,
                student_model=student_model,
                trainloader=train_loader,
                testloader=test_loader,
                cfg=cfg,
                logger=exp_logger,
                optimizer=student_optimizer,
                scheduler=student_scheduler,
                global_ep=global_ep,
            )
            global_ep += student_epochs
            print(f"[Stage {stage_id}] Student final acc= {final_acc:.2f}%")
            exp_logger.update_metric(f"stage{stage_id}_student_acc", final_acc)

    # 8) save final
    ckpt_dir = cfg.get("ckpt_dir", cfg["results_dir"])
    os.makedirs(ckpt_dir, exist_ok=True)
    student_ckpt_path = os.path.join(ckpt_dir, "student_final.pth")
    torch.save(student_model.state_dict(), student_ckpt_path)
    print(f"[main] Distillation done => {student_ckpt_path}")
    exp_logger.update_metric("final_student_ckpt", student_ckpt_path)

    # ---- Key 동기화 ----
    final_key = f"stage{cfg['num_stages']}_student_acc"
    exp_logger.update_metric(final_key, final_acc)   # CSV / JSON 기록

    if wandb and wandb.run:
        # sweep metric 이 'stage4_student_acc' 같은 형식일 때 그대로 복사
        wandb.run.summary[final_key] = final_acc
        # 선택: 별도 generic 키도 남기고 싶다면
        wandb.run.summary["best_student_acc"] = final_acc
    exp_logger.finalize()

if __name__ == "__main__":
    main()
    logging.shutdown()
