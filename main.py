#!/usr/bin/env python3
# main.py
import logging
import os
import json
from typing import Optional, List

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from utils.logging import ExperimentLogger, get_logger

from core import (
    create_student_by_name,
    create_teacher_by_name,
    run_training_stages,
    run_continual_learning,
    renorm_ce_kd,
    setup_partial_freeze_schedule_with_cfg,
    setup_safety_switches_with_cfg,
    auto_set_ib_mbm_query_dim_with_model,
    cast_numeric_configs,
)

from utils.common import (
    set_random_seed,
    check_label_range,
    get_model_num_classes,
)

from data.cifar100 import get_cifar100_loaders
from data.imagenet32 import get_imagenet32_loaders
from modules.partial_freeze import apply_partial_freeze

try:
    import wandb
except ModuleNotFoundError:
    wandb = None


def _to_dict(cfg: DictConfig):
    return OmegaConf.to_container(cfg, resolve=True)

def normalize_exp(exp: dict):
    """Flatten any remaining nesting in experiment config and promote method values to top level"""
    if isinstance(exp, dict):
        # dataset/schedule/method가 이중 중첩이면 평탄화
        for k in ("dataset", "schedule", "method"):
            v = exp.get(k)
            if isinstance(v, dict) and k in v and isinstance(v[k], dict):
                exp[k] = v[k]
        
        # teacher1, teacher2, model.student의 중첩도 평탄화
        for k in ("teacher1", "teacher2"):
            v = exp.get(k)
            if isinstance(v, dict) and k in v and isinstance(v[k], dict):
                exp[k] = v[k]
        
        # model.student 중첩 평탄화
        model = exp.get("model", {})
        if isinstance(model, dict):
            student = model.get("student", {})
            if isinstance(student, dict) and "student" in student and isinstance(student["student"], dict):
                model["student"] = student["student"]
        
        # method 값을 최상위로 반영(이름 제외 선택)
        if isinstance(exp.get("method"), dict):
            for mk, mv in exp["method"].items():
                if mk != "name" and mk not in exp:
                    exp[mk] = mv
    
    return exp


def _log_and_save_meta(logger, exp_logger: ExperimentLogger, exp: dict,
                       t1_name: str, t2_name: str, s_name: str,
                       t1_acc: float = 0.0, t2_acc: float = 0.0):
    """Print a concise experiment metadata banner and write meta.json."""
    method_cfg = exp.get("method", {}) or {}
    kd_method = method_cfg.get("name", "asib")
    dataset = exp.get("dataset", {}) or {}
    schedule = exp.get("schedule", {}) or {}
    student_cfg = (exp.get("model", {}) or {}).get("student", {}) or {}

    meta = {
        "exp_id": exp.get("exp_id"),
        "seed": exp.get("seed", 42),
        "dataset": {
            "name": dataset.get("name"),
            "batch_size": dataset.get("batch_size"),
            "num_workers": dataset.get("num_workers"),
            "data_aug": dataset.get("data_aug", 1),
        },
        "teachers": [
            {"name": t1_name, "ckpt": exp.get("teacher1_ckpt"), "test_acc": round(float(t1_acc or 0.0), 2)},
            {"name": t2_name, "ckpt": exp.get("teacher2_ckpt"), "test_acc": round(float(t2_acc or 0.0), 2)},
        ],
        "student": {
            "name": s_name,
            "pretrained": bool(student_cfg.get("pretrained", False)),
            "use_adapter": bool(student_cfg.get("use_adapter", False)),
        },
        "kd": {
            "method": kd_method,
            "ce_alpha": exp.get("ce_alpha"),
            "kd_alpha": exp.get("kd_alpha", 0.0),
            "kd_ens_alpha": exp.get("kd_ens_alpha", 0.0),
            "teacher_adapt_kd_warmup": exp.get("teacher_adapt_kd_warmup", 0),
        },
        "ib": {
            "use_ib": exp.get("use_ib", False),
            "ib_beta": exp.get("ib_beta"),
            "ib_epochs_per_stage": exp.get("ib_epochs_per_stage"),
            "ib_mbm_out_dim": exp.get("ib_mbm_out_dim"),
            "ib_mbm_n_head": exp.get("ib_mbm_n_head"),
        },
        "cccp": {
            "use_cccp": exp.get("use_cccp", False),
        },
        "optim": {
            "optimizer": exp.get("optimizer", "adamw"),
            "student_lr": exp.get("student_lr"),
            "student_weight_decay": exp.get("student_weight_decay"),
            "grad_clip_norm": exp.get("grad_clip_norm", 0),
        },
        "schedule": {
            "type": schedule.get("type"),
            "lr_warmup_epochs": schedule.get("lr_warmup_epochs"),
            "min_lr": schedule.get("min_lr"),
            "step_size": schedule.get("step_size"),
            "gamma": schedule.get("gamma"),
        },
        "cl": {
            "cl_mode": exp.get("cl_mode", False),
            "cl_method": (exp.get("cl_method", "asib_cl") if exp.get("cl_mode", False) else None),
        },
        "amp": {
            "use_amp": exp.get("use_amp", False),
            "amp_dtype": exp.get("amp_dtype"),
        },
        "ppf": {
            "use_partial_freeze": exp.get("use_partial_freeze", False),
            "use_teacher_finetuning": exp.get("use_teacher_finetuning", False),
            "student_freeze_level": exp.get("student_freeze_level", -1),
            "teacher1_freeze_level": exp.get("teacher1_freeze_level", -1),
            "teacher2_freeze_level": exp.get("teacher2_freeze_level", -1),
            "student_freeze_bn": exp.get("student_freeze_bn", False),
            "teacher1_freeze_bn": exp.get("teacher1_freeze_bn", True),
            "teacher2_freeze_bn": exp.get("teacher2_freeze_bn", True),
        },
    }

    banner = [
        "================= EXPERIMENT META =================",
        f"ExpID          : {meta['exp_id']}",
        f"Dataset        : {meta['dataset']['name']} | BS={meta['dataset']['batch_size']} | Aug={meta['dataset']['data_aug']}",
        f"Teacher1       : {t1_name} | ckpt={exp.get('teacher1_ckpt')} | acc={meta['teachers'][0]['test_acc']:.2f}%",
        f"Teacher2       : {t2_name} | ckpt={exp.get('teacher2_ckpt')} | acc={meta['teachers'][1]['test_acc']:.2f}%",
        f"Student        : {s_name} | pretrained={meta['student']['pretrained']} | adapter={meta['student']['use_adapter']}",
        f"KD             : {meta['kd']['method']} | ce={meta['kd']['ce_alpha']} kd={meta['kd']['kd_alpha']} ens={meta['kd']['kd_ens_alpha']} warmup={meta['kd']['teacher_adapt_kd_warmup']}",
        f"IB/CCCP        : use_ib={meta['ib']['use_ib']} β={meta['ib']['ib_beta']} ib_ep={meta['ib']['ib_epochs_per_stage']} (out={meta['ib']['ib_mbm_out_dim']} n_head={meta['ib']['ib_mbm_n_head']}) | use_cccp={meta['cccp']['use_cccp']}",
        f"PPF            : partial_freeze={meta['ppf']['use_partial_freeze']} t_finetune={meta['ppf']['use_teacher_finetuning']} | s_freeze={meta['ppf']['student_freeze_level']} t1_freeze={meta['ppf']['teacher1_freeze_level']} t2_freeze={meta['ppf']['teacher2_freeze_level']} | bn(s/t1/t2)={meta['ppf']['student_freeze_bn']}/{meta['ppf']['teacher1_freeze_bn']}/{meta['ppf']['teacher2_freeze_bn']}",
        f"Optim/Sched    : {meta['optim']['optimizer']} lr={meta['optim']['student_lr']} wd={meta['optim']['student_weight_decay']} | sch={meta['schedule']['type']}",
        f"AMP            : use_amp={meta['amp']['use_amp']} dtype={meta['amp']['amp_dtype']}",
        f"CL             : cl_mode={meta['cl']['cl_mode']} method={meta['cl']['cl_method']}",
        "===================================================",
    ]
    for line in banner:
        logger.info(line)
    exp_logger.save_meta(meta)
    # PPF 설정 불일치 경고
    if not meta["ppf"]["use_partial_freeze"] and any(lv != -1 for lv in [meta["ppf"]["student_freeze_level"], meta["ppf"]["teacher1_freeze_level"], meta["ppf"]["teacher2_freeze_level"]]):
        logger.warning("[PPF] use_partial_freeze=False이지만 freeze_level!= -1 설정이 감지되었습니다. 설정 확인 필요.")
    
    # Also persist a few meta fields into CSV summary
    exp_logger.update_metric("optimizer", meta["optim"]["optimizer"])
    exp_logger.update_metric("student_lr", meta["optim"]["student_lr"])
    exp_logger.update_metric("student_weight_decay", meta["optim"]["student_weight_decay"])
    exp_logger.update_metric("kd_alpha", meta["kd"]["kd_alpha"])
    exp_logger.update_metric("ce_alpha", meta["kd"]["ce_alpha"])
    exp_logger.update_metric("use_ib", int(bool(meta["ib"]["use_ib"])) )
    exp_logger.update_metric("ib_beta", meta["ib"]["ib_beta"]) 
    exp_logger.update_metric("ib_epochs_per_stage", meta["ib"]["ib_epochs_per_stage"]) 
    exp_logger.update_metric("use_cccp", int(bool(meta["cccp"]["use_cccp"])) )
    
    # PPF 정보도 CSV에 추가
    exp_logger.update_metric("use_partial_freeze", int(bool(meta["ppf"]["use_partial_freeze"])))
    exp_logger.update_metric("use_teacher_finetuning", int(bool(meta["ppf"]["use_teacher_finetuning"])))
    exp_logger.update_metric("student_freeze_level", meta["ppf"]["student_freeze_level"])
    exp_logger.update_metric("teacher1_freeze_level", meta["ppf"]["teacher1_freeze_level"])
    exp_logger.update_metric("teacher2_freeze_level", meta["ppf"]["teacher2_freeze_level"])


@torch.no_grad()
def _quick_eval(model: torch.nn.Module, loader, device: str) -> float:
    """Lightweight top-1 accuracy eval on a dataloader."""
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        # unwrap logits from common wrappers
        if isinstance(out, tuple):
            logits = out[1]
        elif isinstance(out, dict) and "logit" in out:
            logits = out["logit"]
        else:
            logits = out
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / max(1, total)

@hydra.main(config_path="configs", version_base="1.3")
def main(cfg: DictConfig):
    # 1) experiment 서브트리만 사용 (nested experiment 방어)
    exp = cfg.experiment if "experiment" in cfg else cfg
    # If composed config nests another 'experiment', unwrap it
    try:
        if isinstance(exp, DictConfig) and "experiment" in exp and isinstance(exp.experiment, (DictConfig, dict)):
            exp = exp.experiment
    except Exception:
        pass
    exp_dict = _to_dict(exp)

    # 2-a) 루트 키 오버라이드 병합 지원: -cn으로 experiment 선택 시에도
    # root 수준의 오버라이드(예: seed=42, kd_alpha=0.3)를 exp로 병합한다.
    try:
        root_dict = _to_dict(cfg)
        for rk, rv in list(root_dict.items()):
            if rk in ("experiment", "hydra", "defaults"):
                continue
            # 루트 오버라이드는 실험 기본값을 덮어쓴다
            exp_dict[rk] = rv
    except Exception:
        pass
    
    # 2) 중첩 평탄화
    exp_dict = normalize_exp(exp_dict)

    # 2-a) method.* 잔존 키로 인한 충돌 방지:
    # normalize_exp는 method의 값을 최상위에 채우되(없을 때만)
    # 잔존 method 서브트리가 남아있으면 일부 모듈이 cfg["method"]를 참조해
    # 의도치 않게 다른 하이퍼파라미터를 사용할 수 있다.
    # 혼선을 방지하기 위해 method 서브트리를 제거한다.
    try:
        if isinstance(exp_dict.get("method"), dict):
            exp_dict.pop("method", None)
    except Exception:
        pass

    # 2-b) 학습 스케줄 키 정규화: trainer 모듈이 기대하는 키로 매핑
    # student_epochs_per_stage -> student_epochs_schedule
    if "student_epochs_per_stage" in exp_dict and "student_epochs_schedule" not in exp_dict:
        try:
            sel = exp_dict.get("student_epochs_per_stage")
            if isinstance(sel, (list, tuple)) and len(sel) > 0:
                exp_dict["student_epochs_schedule"] = list(sel)
                # student_iters fallback도 동일 값으로 설정하여 1 epoch로 고정되는 것 방지
                exp_dict["student_iters"] = int(sel[0])
        except Exception:
            pass

    # 2-b-1) 간단 오버라이드 지원: student_epochs, stages
    # 사용 예) +experiment.student_epochs=5 +experiment.stages=1
    if "stages" in exp_dict:
        try:
            exp_dict["num_stages"] = int(exp_dict["stages"])
        except Exception:
            pass
    if "student_epochs" in exp_dict:
        try:
            se = int(exp_dict["student_epochs"])
            # num_stages가 아직 정해지지 않았다면 1로 가정
            ns = int(exp_dict.get("num_stages", 1))
            exp_dict["student_epochs_schedule"] = [se for _ in range(ns)]
            exp_dict["student_iters"] = se
        except Exception:
            pass

    # 2-b-2) 간단 오버라이드 지원: dataset_batch_size (중첩 struct 회피)
    # 사용 예) +experiment.dataset_batch_size=16
    if "dataset_batch_size" in exp_dict:
        try:
            bs = int(exp_dict["dataset_batch_size"])
            if not isinstance(exp_dict.get("dataset"), dict):
                exp_dict["dataset"] = {}
            exp_dict["dataset"]["batch_size"] = bs
        except Exception:
            pass

    # 2-c) 안전장치: num_stages 보정
    if "num_stages" not in exp_dict and "student_epochs_schedule" in exp_dict:
        try:
            exp_dict["num_stages"] = int(len(exp_dict["student_epochs_schedule"]))
        except Exception:
            exp_dict["num_stages"] = 1

    # 3) 로거
    exp_dir = exp_dict.get("results_dir", ".")
    logger = get_logger(exp_dir, level=exp_dict.get("log_level", "INFO"))
    # Effective config (post-normalize) quick check
    try:
        logger.info(
            "[CFG] kd_target=%s ce=%s kd=%s ib_beta=%s",
            str(exp_dict.get("kd_target")),
            str(exp_dict.get("ce_alpha")),
            str(exp_dict.get("kd_alpha")),
            str(exp_dict.get("ib_beta")),
        )
    except Exception:
        pass
    # Clean internal/derived keys from HParams for clarity
    _hp = {k: v for k, v in exp_dict.items()}
    for _k in ("student_epochs_schedule", "student_iters", "stages", "student_epochs", "dataset_batch_size"):
        if _k in _hp:
            _hp.pop(_k, None)
    logger.info("HParams:\n%s", json.dumps(_hp, indent=2))

    # 3) W&B (옵션)
    if exp_dict.get("use_wandb", False):
        if wandb is None:
            logger.warning("[W&B] wandb not installed – skipping")
        else:
            wandb.init(
                project=exp_dict.get("wandb_project", "kd_monitor"),
                entity=exp_dict.get("wandb_entity"),
                name=exp_dict.get("wandb_run_name", exp_dict.get("exp_id", "run")),
                config=exp_dict,
            )
            logger.info("[W&B] %s", wandb.run.url)

    # 4) 숫자 캐스팅/안전 스위치
    cast_numeric_configs(exp_dict)

    # 5) 로그 저장기
    exp_logger = ExperimentLogger(exp_dict, exp_name="asib")

    # 6) 디바이스/시드
    device = exp_dict.get("device", "cuda")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available. Falling back to CPU.")
        device = "cpu"

    set_random_seed(exp_dict.get("seed", 42), deterministic=exp_dict.get("deterministic", True))
    
    # 6.5) CUDA 최적화 설정 (채널-라스트, TF32, cuDNN)
    if device == "cuda" and torch.cuda.is_available():
        # cuDNN 벤치마크 모드 활성화 (Conv 연산 자동 최적화)
        torch.backends.cudnn.benchmark = True
        # TF32 활성화 (A100/3090에서 Conv/Matmul 가속)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Float32 matmul precision 설정
        torch.set_float32_matmul_precision("high")
        logger.info("CUDA optimizations enabled: cuDNN benchmark, TF32, high precision matmul")

    # 7) 데이터 로더
    ds_cfg = exp_dict.get("dataset", {})
    dataset_name = ds_cfg.get("name", "cifar100")
    data_root = ds_cfg.get("root", "./data")
    batch_size = int(ds_cfg.get("batch_size", 128))
    num_workers = int(ds_cfg.get("num_workers", 2))
    data_aug = ds_cfg.get("data_aug", True)
    small_input = bool(exp_dict.get("small_input", ds_cfg.get("small_input", dataset_name == "cifar100")))

    if exp_dict.get("overlap_pct", -1) >= 0:
        from data.cifar100_overlap import get_overlap_loaders, CIFAR100OverlapDataset
        (A_tr, A_te), (B_tr, B_te), _ = get_overlap_loaders(
            pct_overlap=exp_dict["overlap_pct"],
            batch_size=batch_size,
            num_workers=num_workers,
            augment=data_aug,
            seed=exp_dict.get("seed", 42),
        )
        base_train_loader, base_test_loader = get_cifar100_loaders(
            root=data_root, batch_size=batch_size, num_workers=num_workers, augment=data_aug
        )
        all_classes = sorted(list(set(A_tr.dataset.class_indices + B_tr.dataset.class_indices)))
        combined_train_dataset = CIFAR100OverlapDataset(base_train_loader.dataset, all_classes)
        combined_test_dataset = CIFAR100OverlapDataset(base_test_loader.dataset, all_classes)
        train_loader = torch.utils.data.DataLoader(
            combined_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
        )
        test_loader = torch.utils.data.DataLoader(
            combined_test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
    elif dataset_name == "cifar100":
        train_loader, test_loader = get_cifar100_loaders(
            root=data_root, batch_size=batch_size, num_workers=num_workers, augment=data_aug
        )
    elif dataset_name == "imagenet32":
        train_loader, test_loader = get_imagenet32_loaders(
            root=data_root, batch_size=batch_size, num_workers=num_workers, augment=data_aug
        )
    else:
        raise ValueError(f"Unknown dataset.name={dataset_name}")

    # 8) 클래스 수
    if isinstance(train_loader.dataset, torch.utils.data.ConcatDataset):
        num_classes = 100
    else:
        n_classes = getattr(train_loader.dataset, "classes", None)
        if n_classes is None:
            n_classes = getattr(train_loader.dataset, "num_classes", None)
        if n_classes is None:
            raise AttributeError("Dataset must expose `classes` or `num_classes`")
        num_classes = len(n_classes) if not isinstance(n_classes, int) else n_classes

    exp_logger.update_metric("num_classes", num_classes)
    check_label_range(train_loader.dataset, num_classes)
    check_label_range(test_loader.dataset, num_classes)

    # 9) 교사/학생 생성
    # teacher1
    t1 = exp_dict.get("teacher1", {})
    t1_name = t1.get("name")
    if not t1_name:
        raise ValueError("Missing 'experiment.teacher1.name'")
    t1_ckpt = exp_dict.get("teacher1_ckpt", f"./checkpoints/teachers/{t1_name}_ft.pth")

    logging.getLogger().setLevel(logging.WARNING)
    teacher1 = create_teacher_by_name(
        teacher_name=t1_name,
        num_classes=num_classes,
        pretrained=t1.get("pretrained", True),
        small_input=small_input,
        cfg=exp_dict,
    ).to(device)
    logging.getLogger().setLevel(logging.INFO)

    if os.path.exists(t1_ckpt):
        try:
            sd = torch.load(t1_ckpt, map_location=device, weights_only=True)
        except TypeError:
            sd = torch.load(t1_ckpt, map_location=device)
        teacher1.load_state_dict(sd, strict=False)
        logging.info("Loaded teacher1 from %s", t1_ckpt)

    # teacher2
    t2 = exp_dict.get("teacher2", {})
    t2_name = t2.get("name")
    if not t2_name:
        raise ValueError("Missing 'experiment.teacher2.name'")
    t2_ckpt = exp_dict.get("teacher2_ckpt", f"./checkpoints/teachers/{t2_name}_ft.pth")

    logging.getLogger().setLevel(logging.WARNING)
    teacher2 = create_teacher_by_name(
        teacher_name=t2_name,
        num_classes=num_classes,
        pretrained=t2.get("pretrained", True),
        small_input=small_input,
        cfg=exp_dict,
    ).to(device)
    logging.getLogger().setLevel(logging.INFO)

    if os.path.exists(t2_ckpt):
        try:
            sd = torch.load(t2_ckpt, map_location=device, weights_only=True)
        except TypeError:
            sd = torch.load(t2_ckpt, map_location=device)
        teacher2.load_state_dict(sd, strict=False)
        logging.info("Loaded teacher2 from %s", t2_ckpt)

    # Optional quick eval of teachers
    teacher1_acc = 0.0
    teacher2_acc = 0.0
    if exp_dict.get("compute_teacher_eval", False):
        try:
            teacher1_acc = _quick_eval(teacher1, test_loader, device)
            logging.info("Teacher1 (%s) testAcc=%.2f%%", t1_name, teacher1_acc)
            teacher2_acc = _quick_eval(teacher2, test_loader, device)
            logging.info("Teacher2 (%s) testAcc=%.2f%%", t2_name, teacher2_acc)
        except Exception as e:
            logging.warning("Teacher quick eval failed: %s", e)

    # 학생
    s = exp_dict.get("model", {}).get("student", {})
    s_name = s.get("name")
    if not s_name:
        raise ValueError("Missing 'experiment.model.student.name'")

    logging.getLogger().setLevel(logging.WARNING)
    student = create_student_by_name(
        s_name,
        pretrained=s.get("pretrained", False),
        small_input=small_input,
        num_classes=num_classes,
        cfg=exp_dict,
    ).to(device)
    logging.getLogger().setLevel(logging.INFO)

    # 10) 학습 전 설정 (IB_MBM 생성 전에 q_dim 설정)
    num_stages = int(exp_dict.get("num_stages", 1))
    setup_partial_freeze_schedule_with_cfg(exp_dict, num_stages)
    setup_safety_switches_with_cfg(exp_dict, num_stages)
    auto_set_ib_mbm_query_dim_with_model(student, exp_dict)  # ← IB_MBM 생성 전에 호출
    renorm_ce_kd(exp_dict)

    # 11) IB_MBM & Synergy Head (q_dim 설정 후 생성)
    from models import build_ib_mbm_from_teachers as build_from_teachers
    ib_mbm, synergy_head = build_from_teachers([teacher1, teacher2], exp_dict)
    ib_mbm = ib_mbm.to(device)
    synergy_head = synergy_head.to(device)
    
    # 11-a) 채널-라스트 메모리 포맷 적용 (Conv 연산 가속)
    if device == "cuda" and torch.cuda.is_available():
        # ConvNet 계열 모델들을 채널-라스트 포맷으로 변환
        for model, name in [(teacher1, "teacher1"), (teacher2, "teacher2"), (student, "student")]:
            try:
                model = model.to(memory_format=torch.channels_last)
                logger.info(f"{name} converted to channels_last memory format")
            except Exception as e:
                logger.debug(f"Could not convert {name} to channels_last: {e}")
        # IB_MBM과 synergy_head는 주로 Linear 레이어이므로 제외

    # 11-b) 메타데이터 배너 및 저장
    try:
        # teacher1/2 acc가 별도 측정되지 않았다면 0.0으로
        _log_and_save_meta(
            logger,
            exp_logger,
            exp_dict,
            t1_name=t1_name,
            t2_name=t2_name,
            s_name=s_name,
            t1_acc=locals().get("teacher1_acc", 0.0),
            t2_acc=locals().get("teacher2_acc", 0.0),
        )
    except Exception as _e:
        logger.warning("[META] banner/meta save skipped: %s", _e)

    # 12) 로그
    logging.info("🚀 Starting training process...")
    logging.info(f"CL mode: {exp_dict.get('cl_mode', False)}")
    logging.info(f"Number of stages: {num_stages}")
    logging.info(f"Student epochs per stage: {exp_dict.get('student_epochs_per_stage', [])}")
    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Device: {device}")
    logging.info(f"Train loader size: {len(train_loader)} batches")
    logging.info(f"Test loader size: {len(test_loader)} batches")
    logging.info(f"Dataset classes: {num_classes}")

    # 13) 트레이닝
    if exp_dict.get("cl_mode", False):
        logging.info("📚 Running in Continual Learning mode...")
        final_acc = run_continual_learning(
            [teacher1, teacher2], ib_mbm, synergy_head, student, exp_dict, exp_logger
        )
    else:
        logging.info("🎯 Running in Standard training mode...")
        try:
            logging.info("📊 Training will start now - you should see epoch progress logs...")
            final_acc = run_training_stages(
                [teacher1, teacher2],
                ib_mbm,
                synergy_head,
                student,
                train_loader,
                test_loader,
                exp_dict,
                exp_logger,
                num_stages,
            )
            logging.info(f"✅ run_training_stages completed with accuracy: {final_acc:.2f}%")
        except Exception as e:
            logging.error(f"❌ run_training_stages failed: {e}", exc_info=True)
            final_acc = 0.0

    logging.info(f"✅ Training completed. Final student accuracy: {final_acc:.2f}%")
    exp_logger.update_metric("final_student_acc", final_acc)
    exp_logger.save_results()
    return final_acc


if __name__ == "__main__":
    main() 