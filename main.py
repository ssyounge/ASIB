#!/usr/bin/env python3
# main.py
import logging
import os
import json
import copy
import math
import hashlib
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

# Runtime hardening: enable faulthandler, stabilize allocator
import faulthandler, signal
faulthandler.enable(all_threads=True)
os.environ.setdefault("PYTHONMALLOC", "malloc")

try:
    import wandb
except ModuleNotFoundError:
    wandb = None


def _ckpt_from_fixed_map(teacher_name: str, dataset_name: str, override: Optional[str]) -> Optional[str]:
    if override:
        return override
    ds = (dataset_name or "").lower()
    table = {
        "cifar100": {
            "resnet152":       "checkpoints/teachers/resnet152_cifar100.pth",
            "convnext_s":      "checkpoints/teachers/convnext_s_cifar100.pth",
            "convnext_l":      "checkpoints/teachers/convnext_l_cifar100.pth",
            "efficientnet_l2": "checkpoints/teachers/efficientnet_l2_cifar100.pth",
        },
        "imagenet32": {
            "resnet152":       "checkpoints/teachers/resnet152_imagenet32.pth",
            "convnext_s":      "checkpoints/teachers/convnext_s_imagenet32.pth",
            "convnext_l":      "checkpoints/teachers/convnext_l_imagenet32_improved.pth",
            "efficientnet_l2": "checkpoints/teachers/efficientnet_l2_imagenet32_improved.pth",
        },
    }
    return table.get(ds, {}).get(teacher_name)

def _to_dict(cfg: DictConfig):
    return OmegaConf.to_container(cfg, resolve=True)


def _apply_env_shortcuts(exp_dict: dict):
    import os as _os
    if not bool(exp_dict.get("allow_env_overrides", False)):
        return
    tp = _os.environ.get("TEACHER_PAIR")
    if tp:
        t = [s.strip() for s in tp.split(",")]
        if len(t) == 2:
            exp_dict.setdefault("teacher1", {}); exp_dict["teacher1"]["name"] = t[0]
            exp_dict.setdefault("teacher2", {}); exp_dict["teacher2"]["name"] = t[1]
            exp_dict["teacher1"]["pretrained"] = exp_dict["teacher1"].get("pretrained", True)
            exp_dict["teacher2"]["pretrained"] = exp_dict["teacher2"].get("pretrained", True)
            logging.info("[ENV] TEACHER_PAIR override applied: %s", tp)
    stu = _os.environ.get("STUDENT")
    if stu:
        exp_dict.setdefault("model", {}).setdefault("student", {})
        name = stu if stu.endswith("_scratch") else f"{stu}_scratch"
        exp_dict["model"]["student"]["name"] = name
        exp_dict["model"]["student"]["pretrained"] = False
        # Do not force use_adapter; preserve config value if present
        logging.info("[ENV] STUDENT override applied: %s", stu)
    m = _os.environ.get("METHOD")
    if m:
        exp_dict["method_name"] = m
        logging.info("[ENV] METHOD override applied: %s", m)
    sd = _os.environ.get("SEED")
    if sd:
        try:
            exp_dict["seed"] = int(sd)
        except Exception:
            pass


def _deep_merge(dst: dict, src: dict) -> None:
    """Recursively merge src into dst (in-place)."""
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)


def _auto_results(exp_dict: dict):
    base = exp_dict.get("results_dir_base", "experiments/auto")
    t1 = ((exp_dict.get("teacher1") or {}).get("name") or "t1")
    t2 = ((exp_dict.get("teacher2") or {}).get("name") or "t2")
    stu = ((exp_dict.get("model") or {}).get("student") or {}).get("name") or "student"
    method = exp_dict.get("method_name", "method")
    seed = int(exp_dict.get("seed", 42))
    stu_short = str(stu).replace("_scratch", "")
    rd = f"{base}/{t1}-{t2}/{stu_short}/{method}/seed{seed}"
    exp_dict.setdefault("results_dir", rd)
    exp_dict.setdefault("exp_id", f"{method}__{t1}-{t2}__{stu_short}__s{seed}")
    return exp_dict


def _is_small_student(exp: dict) -> bool:
    s_name = (((exp.get("model", {}) or {}).get("student", {}) or {}).get("name") or "").lower()
    return any(k in s_name for k in ("mobilenet_v2", "efficientnet_b0", "shufflenet_v2"))


def finalize_config(exp: dict) -> None:
    """Lock 전에 파생/자동 키를 확정해 config 불변성을 보장한다.
    사용자/anchor 지정값은 존중하고, 없을 때만 채운다.
    """
    try:
        cast_numeric_configs(exp)
    except Exception:
        pass

    # Schedules → ensure student_epochs_schedule/num_stages/student_iters
    if "student_epochs_per_stage" in exp and "student_epochs_schedule" not in exp:
        sel = exp.get("student_epochs_per_stage")
        if isinstance(sel, (list, tuple)) and len(sel) > 0:
            exp["student_epochs_schedule"] = list(sel)
            exp.setdefault("student_iters", int(sel[0]))
    if "stages" in exp and "num_stages" not in exp:
        try:
            exp["num_stages"] = int(exp["stages"])
        except Exception:
            pass
    if "student_epochs" in exp:
        try:
            se = int(exp["student_epochs"])
            ns = int(exp.get("num_stages", 1))
            exp["student_epochs_schedule"] = [se for _ in range(ns)]
            exp.setdefault("student_iters", se)
        except Exception:
            pass
    if "num_stages" not in exp and "student_epochs_schedule" in exp:
        try:
            exp["num_stages"] = int(len(exp["student_epochs_schedule"]))
        except Exception:
            exp["num_stages"] = 1

    ns = int(exp.get("num_stages", 1))

    # 0) Method profile → set sane defaults (exposed knobs reduction)
    try:
        # Allow both top-level and nested method.profile
        profile = (
            str((exp.get("profile") or
                 ((exp.get("method") or {}).get("profile")) or "").strip().lower())
        )
        if profile:
            # Defaults per profile (only fill if absent)
            if profile == "stable":
                exp.setdefault("kd_two_view_start_epoch", 20)
                exp.setdefault("kd_uncertainty_weight", 0.3)
                exp.setdefault("kd_ens_alpha", 0.0)
                # Keep tau_syn conservative (same as tau) unless explicitly set
                exp.setdefault("tau_syn", float(exp.get("tau", 4.0)))
                exp.setdefault("auto_tune_target_ratio", 0.35)
            elif profile == "balanced":
                exp.setdefault("kd_two_view_start_epoch", 10)
                exp.setdefault("kd_uncertainty_weight", 0.4)
                exp.setdefault("kd_ens_alpha", 0.0)
                exp.setdefault("tau_syn", float(exp.get("tau", 4.0)))
                exp.setdefault("auto_tune_target_ratio", 0.40)
            elif profile == "aggressive":
                exp.setdefault("kd_two_view_start_epoch", 1)
                exp.setdefault("kd_uncertainty_weight", 0.5)
                # Allow non-zero only if user intentionally overrides later
                exp.setdefault("kd_ens_alpha", 0.0)
                exp.setdefault("tau_syn", float(exp.get("tau", 4.0)))
                exp.setdefault("auto_tune_target_ratio", 0.45)
            # Optional LR log interval reasonable default
            exp.setdefault("lr_log_every", 10)
            # Record applied profile (for meta/debug)
            exp["_profile_applied"] = profile
    except Exception:
        pass

    # Partial freeze / safety switches (should respect existing values)
    try:
        setup_partial_freeze_schedule_with_cfg(exp, ns)
        setup_safety_switches_with_cfg(exp, ns)
    except Exception:
        pass

    # Distillation dims / IB alignment (small student defaults)
    if bool(exp.get("use_distillation_adapter", False)):
        d_out = int(exp.get("distill_out_dim", 0) or 0)
        if d_out <= 0 and _is_small_student(exp):
            exp["distill_out_dim"] = 256
            d_out = 256
        ib_out = int(exp.get("ib_mbm_out_dim", 0) or 0)
        if d_out > 0 and ib_out != d_out:
            exp["ib_mbm_out_dim"] = d_out
        # Enforce query dim alignment for small students
        try:
            if _is_small_student(exp):
                exp.setdefault("ib_mbm_query_dim", d_out or 256)
        except Exception:
            pass

    # Teacher optimizer alias (pre-lock)
    try:
        a_lr = float(exp.get("a_step_lr", 0.0) or 0.0)
        if float(exp.get("teacher_lr", 0.0) or 0.0) == 0.0 and a_lr > 0.0:
            exp["teacher_lr"] = a_lr
        a_wd = float(exp.get("a_step_weight_decay", 0.0) or 0.0)
        if float(exp.get("teacher_weight_decay", 0.0) or 0.0) == 0.0 and a_wd > 0.0:
            exp["teacher_weight_decay"] = a_wd
    except Exception:
        pass

    # Pre-resolve teacher ckpt paths to reduce post-lock mutations (optional)
    try:
        ds_name = (exp.get("dataset", {}) or {}).get("name", "cifar100")
        t1_name = (exp.get("teacher1", {}) or {}).get("name")
        t2_name = (exp.get("teacher2", {}) or {}).get("name")
        if t1_name and not exp.get("teacher1_ckpt"):
            exp["teacher1_ckpt"] = _ckpt_from_fixed_map(t1_name, ds_name, exp.get("teacher1_ckpt"))
        if t2_name and not exp.get("teacher2_ckpt"):
            exp["teacher2_ckpt"] = _ckpt_from_fixed_map(t2_name, ds_name, exp.get("teacher2_ckpt"))
    except Exception:
        pass


def _resolve_method_name(cfg: DictConfig, exp_dict: dict) -> str:
    """
    Hydra choices와 experiment.method.*를 훑어 최종 method_name을 결정.
    """
    def sel(path: str):
        try:
            return OmegaConf.select(cfg, path)
        except Exception:
            return None

    # 1) Hydra choices 우선
    for key in (
        "hydra.runtime.choices.experiment/method",
        "hydra.runtime.choices.method",
    ):
        v = sel(key)
        if v:
            return str(v)

    # 2) composed config 내 name 필드
    for key in (
        "experiment.experiment.method.name",
        "experiment.method.name",
        "method.name",
        "method_name",
    ):
        v = sel(key)
        if v:
            return str(v)

    # 3) defaults 역순 스캔
    try:
        dfl = OmegaConf.to_container(cfg.get("defaults"), resolve=True) or []
        for item in reversed(dfl):
            if isinstance(item, dict):
                for k in ("experiment/method", "method", "experiment.method"):
                    if k in item:
                        return str(item[k])
    except Exception:
        pass
    return str(exp_dict.get("method_name", "asib"))


def _apply_method_policy(exp_dict: dict, strict_baseline: bool = True) -> None:
    """
    최종 method_name에 따라 안전 가드와 공정성 규칙을 적용.
    """
    mn = str(exp_dict.get("method_name", "asib")).lower()

    # Treat all ASIB-family methods equivalently; rely on YAML/CLI (no hardcoding).
    if mn in ("asib", "asib_stage", "asib_ablation_stage", "asib_ablation_ce"):
        return
    elif mn in ("avg_kd", "avg", "ensemble_kd"):
        # Fairness defaults live in anchor; do not force here
        exp_dict["use_ib"] = False
        exp_dict["teacher_adapt_epochs"] = 0
        exp_dict["use_cccp"] = False
        if strict_baseline:
            exp_dict["feat_kd_alpha"] = 0.0
    else:
        # Default non-ASIB methods: rely on anchor for fairness keys
        exp_dict["use_ib"] = False
        exp_dict["teacher_adapt_epochs"] = 0
        exp_dict["use_cccp"] = False
        if strict_baseline:
            exp_dict["feat_kd_alpha"] = 0.0

    # Anchor policy handled by validator; do not mutate protected keys here


def _resolve_method_from_cli_overrides(cfg: DictConfig) -> Optional[str]:
    """Parse Hydra CLI overrides to extract method name even from incorrect group keys."""
    try:
        ov = OmegaConf.to_container(OmegaConf.select(cfg, "hydra.overrides.task"), resolve=True)
        if not ov:
            return None
        for it in ov:
            s = str(it)
            if "=" not in s:
                continue
            k, v = s.split("=", 1)
            k = k.strip(); v = v.strip()
            if any(sub in k for sub in (
                "method@experiment.experiment.method",
                "method@experiment.method",
                "experiment.method",
                "method@method",
                "method",
            )):
                return v
    except Exception:
        return None
    return None

def _sync_method_name_from_hydra(cfg: DictConfig) -> None:
    """
    최종 선택된 method 이름을 여러 경로에서 탐색해 cfg.method_name에 주입.
    우선순위: experiment.method.name → hydra.runtime.choices.* → defaults 역순 스캔.
    experiment 노드에도 복사해둔다.
    """
    def _sel(path: str):
        try:
            return OmegaConf.select(cfg, path)
        except Exception:
            return None

    try:
        selected: Optional[str] = None
        # 1) 가장 확실: experiment.method.name (defaults 반영 결과)
        for key in (
            "experiment.experiment.method.name",
            "experiment.method.name",
            "method.name",
            "hydra.runtime.choices.method",
            "hydra.runtime.choices.experiment/method",
            "hydra.runtime.choices.experiment.method",
        ):
            val = _sel(key)
            if val:
                selected = str(val)
                break

        # 2) fallback: defaults 리스트에서 마지막 method 항목 찾기
        if selected is None:
            try:
                dfl = OmegaConf.to_container(cfg.get("defaults"), resolve=True) or []
                for item in reversed(dfl):
                    if isinstance(item, dict):
                        if "method" in item:
                            selected = str(item["method"]); break
                        if "experiment/method" in item:
                            selected = str(item["experiment/method"]); break
                        if "experiment.method" in item:
                            selected = str(item["experiment.method"]); break
            except Exception:
                pass

        # 3) 주입
        if selected:
            prev = cfg.get("method_name", None)
            cfg["method_name"] = selected
            try:
                if "experiment" in cfg:
                    cfg.experiment["method_name"] = selected
            except Exception:
                pass
            print(f"[hydra-sync] method_name <- {selected} (was: {prev})")
    except Exception:
        # 내부 구조 변화로 인한 실패는 학습을 막지 않음
        pass

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
        
        # method 값을 최상위로 반영(이름 제외)
        # 정책: 메소드 값은 기본값만 채운다(fill-only). 실험(YAML)에서 명시된 값이 우선.
        if isinstance(exp.get("method"), dict):
            method_dict = exp["method"]
            for mk, mv in method_dict.items():
                if mk != "name" and mk not in exp:
                    exp[mk] = mv
            # Certain trainer-critical knobs should override base defaults.
            # Experiment/rung YAML will still override after this step.
            for mk in (
                "ce_alpha", "kd_alpha",
                "ib_beta", "ib_beta_warmup_epochs", "ib_epochs_per_stage",
                "ib_mbm_query_dim", "ib_mbm_out_dim", "ib_mbm_n_head",
                "ib_mbm_feature_norm", "ib_mbm_logvar_clip", "ib_mbm_min_std",
                "ib_mbm_lr_factor",
                # optimization/data knobs commonly defined in method
                "student_lr", "student_weight_decay",
                "mixup_alpha", "cutmix_alpha_distill", "ce_label_smoothing",
            ):
                if mk in method_dict:
                    exp[mk] = method_dict[mk]
        # 중첩 experiment.experiment.method 역시 기본값만 채우도록 처리
        try:
            nested_exp = exp.get("experiment")
            if isinstance(nested_exp, dict):
                nested_method = nested_exp.get("method")
                if isinstance(nested_method, dict):
                    for mk, mv in nested_method.items():
                        if mk != "name" and mk not in exp:
                            exp[mk] = mv
                    for mk in (
                        "ce_alpha", "kd_alpha",
                        "student_lr", "student_weight_decay",
                        "mixup_alpha", "cutmix_alpha_distill", "ce_label_smoothing",
                        "ib_beta", "ib_beta_warmup_epochs", "ib_epochs_per_stage",
                        "ib_mbm_query_dim", "ib_mbm_out_dim", "ib_mbm_n_head",
                        "ib_mbm_feature_norm", "ib_mbm_logvar_clip", "ib_mbm_min_std",
                        "ib_mbm_lr_factor",
                    ):
                        if mk in nested_method:
                            exp[mk] = nested_method[mk]
                # Promote other experiment.* leaves to top-level with override semantics
                # Explicit rung/experiment YAML must take precedence over method/base
                for nk, nv in nested_exp.items():
                    if nk == "method":
                        continue
                    exp[nk] = nv
        except Exception:
            pass
    
    return exp


def _sanitize_config_for_hash(cfg: dict) -> dict:
    """Return a sanitized copy of cfg for hashing and comparisons.
    Strips runtime/derived keys and normalizes number/string types.
    """
    _cfg = copy.deepcopy(cfg)
    try:
        cast_numeric_configs(_cfg)
    except Exception:
        pass
    IGNORE_TOP_LEVEL_KEYS = {
        "config_sha256",
        "locked",
        "use_amp",
        "teacher1_ckpt",
        "teacher2_ckpt",
        "ib_mbm_out_dim",
        "ib_mbm_query_dim",
        "auto_align_ib_out_dim",
        "_locked_config",
        "csv_filename",
        "total_time_sec",
        "final_student_acc",
        "last_synergy_acc",
        "last_synergy_acc_pct",
        "kd_gate_on",
        "optimizer",
        "hydra_method",
        "cur_stage",
        "effective_teacher_lr",
        "effective_teacher_wd",
        # Runtime/auto-tuned or meta keys (ignored to prevent lock violations)
        "kd_sample_thr",
        "kd_two_view_start_epoch",
        "_profile_applied",
        "auto_tune_target_ratio",
        "lr_log_every",
    }
    for k in IGNORE_TOP_LEVEL_KEYS:
        _cfg.pop(k, None)
    for k in list(_cfg.keys()):
        if k.startswith(("student_ep", "teacher_ep", "epoch", "csv_", "ep", "stage")):
            _cfg.pop(k, None)
    for k in ("num_classes",):
        _cfg.pop(k, None)
    return _cfg

def _log_and_save_meta(logger, exp_logger: ExperimentLogger, exp: dict,
                       t1_name: str, t2_name: str, s_name: str,
                       t1_acc: float = 0.0, t2_acc: float = 0.0):
    """Print a concise experiment metadata banner and write meta.json."""
    # method name 우선순위(실제 선택값 우선): experiment.method.name → 중첩된 method.name → 기존 method_name → 기본값
    kd_method = (
        ((exp.get("experiment", {}) or {}).get("method", {}) or {}).get("name")
        or (exp.get("method", {}) or {}).get("name")
        or exp.get("method_name")
        or "asib"
    )
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
            "kd_warmup_epochs": exp.get("kd_warmup_epochs", 0),
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
        "ema": {
            "use_ema_teacher": exp.get("use_ema_teacher", False),
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

    # Derived flags for readability
    try:
        tadapt = (int(exp.get("teacher_adapt_epochs", 0)) > 0) and bool(exp.get("train_distill_adapter_only", False))
    except Exception:
        tadapt = False
    meta["tadapt"] = bool(tadapt)
    # KD policy flags
    try:
        kd_mode = str(exp.get("kd_target_mode", "")).strip().lower()
    except Exception:
        kd_mode = ""
    two_view = (kd_mode == "two_view")
    center_teacher = bool(exp.get("kd_center_teacher", False))
    kd_adapter = bool(exp.get("use_distillation_adapter", False))
    kd_target = str(exp.get("kd_target", "avg")).lower()
    need_syn = (kd_target in ("synergy", "auto", "auto_min")) or bool(exp.get("use_ib", False))
    # display용 need_teachers: 실제 교사 로드 여부 기준
    disp_need_teachers = (
        float(exp.get("kd_alpha", 0.0) or 0.0) > 0.0
        or bool(exp.get("use_ib", False))
        or bool(exp.get("compute_teacher_eval", False))
        or (kd_target in ("synergy", "auto", "auto_min"))
    )

    # Teacher acc/ckpt display (handle compute_teacher_eval=false)
    _eval_on = bool(exp.get("compute_teacher_eval", True))
    t1_acc_str = ("n/a" if not _eval_on else f"{meta['teachers'][0]['test_acc']:.2f}%")
    t2_acc_str = ("n/a" if not _eval_on else f"{meta['teachers'][1]['test_acc']:.2f}%")
    t1_ckpt_str = (exp.get("teacher1_ckpt") if disp_need_teachers else "n/a")
    t2_ckpt_str = (exp.get("teacher2_ckpt") if disp_need_teachers else "n/a")

    banner = [
        "================= EXPERIMENT META =================",
        f"ExpID          : {meta['exp_id']}",
        f"Dataset        : {meta['dataset']['name']} | BS={meta['dataset']['batch_size']} | Aug={meta['dataset']['data_aug']}",
        f"Teacher1       : {t1_name} | ckpt={t1_ckpt_str} | acc={t1_acc_str}",
        f"Teacher2       : {t2_name} | ckpt={t2_ckpt_str} | acc={t2_acc_str}",
        f"Student        : {s_name} | pretrained={meta['student']['pretrained']} | student_adapter={meta['student']['use_adapter']}",
        f"KD-Policy      : target={kd_target} two_view={two_view} center_teacher={center_teacher} kd_adapter={kd_adapter} need_syn={need_syn}",
        f"T-Adapt/EMA    : tadapt={meta['tadapt']} ema={meta['ema']['use_ema_teacher']}",
        f"KD             : {meta['kd']['method']} | ce={meta['kd']['ce_alpha']} kd={meta['kd']['kd_alpha']} ens={meta['kd']['kd_ens_alpha']} warmup={meta['kd']['kd_warmup_epochs']} | smoothing={exp.get('ce_label_smoothing', 'n/a')}",
        f"Aug            : mixup={exp.get('mixup_alpha', 'n/a')} cutmix={exp.get('cutmix_alpha_distill', 'n/a')}",
        f"IB/CCCP        : use_ib={meta['ib']['use_ib']} β={meta['ib']['ib_beta']} ib_ep={meta['ib']['ib_epochs_per_stage']} (eff={min(int(exp.get('teacher_adapt_epochs', 0)), int(exp.get('ib_epochs_per_stage', 0) or 0))}) (out={meta['ib']['ib_mbm_out_dim']} n_head={meta['ib']['ib_mbm_n_head']}) | use_cccp={meta['cccp']['use_cccp']}",
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
    exp_logger.update_metric("csv_optimizer", meta["optim"]["optimizer"])
    exp_logger.update_metric("csv_student_lr", meta["optim"]["student_lr"])
    exp_logger.update_metric("csv_student_weight_decay", meta["optim"]["student_weight_decay"])
    exp_logger.update_metric("csv_kd_alpha", meta["kd"]["kd_alpha"])
    exp_logger.update_metric("csv_ce_alpha", meta["kd"]["ce_alpha"])
    try:
        exp_logger.update_metric("csv_kd_target", kd_target)
        exp_logger.update_metric("csv_need_synergy", int(bool(need_syn)))
        exp_logger.update_metric("csv_need_teachers", int(bool(disp_need_teachers)))
    except Exception:
        pass
    exp_logger.update_metric("csv_use_ib", int(bool(meta["ib"]["use_ib"])) )
    exp_logger.update_metric("csv_ib_beta", meta["ib"]["ib_beta"]) 
    exp_logger.update_metric("csv_ib_epochs_per_stage", meta["ib"]["ib_epochs_per_stage"]) 
    exp_logger.update_metric("csv_use_cccp", int(bool(meta["cccp"]["use_cccp"])) )
    # Persist selected hydra method for clarity
    try:
        exp_logger.update_metric("csv_hydra_method", exp.get("method_name", "unknown"))
    except Exception:
        pass
    
    # PPF 정보도 CSV에 추가
    exp_logger.update_metric("csv_use_partial_freeze", int(bool(meta["ppf"]["use_partial_freeze"])))
    exp_logger.update_metric("csv_use_teacher_finetuning", int(bool(meta["ppf"]["use_teacher_finetuning"])))
    exp_logger.update_metric("csv_student_freeze_level", meta["ppf"]["student_freeze_level"])
    exp_logger.update_metric("csv_teacher1_freeze_level", meta["ppf"]["teacher1_freeze_level"])
    exp_logger.update_metric("csv_teacher2_freeze_level", meta["ppf"]["teacher2_freeze_level"])
    # Adapter flags to CSV
    try:
        exp_logger.update_metric("csv_kd_adapter", int(bool(exp.get("use_distillation_adapter", False))))
        exp_logger.update_metric("csv_student_adapter", int(bool(meta["student"]["use_adapter"])))
    except Exception:
        pass


@torch.inference_mode()
def _quick_eval(
    model: torch.nn.Module,
    loader,
    device: str,
    max_batches: int | None = None,
    use_amp: bool = True,
) -> float:
    """Lightweight top-1 accuracy eval (optional AMP, partial batches)."""
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    # AMP for GPU
    if str(device).startswith("cuda") and use_amp:
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        from contextlib import nullcontext
        autocast_ctx = nullcontext()
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast_ctx:
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
    if was_training:
        model.train(True)
    return 100.0 * correct / max(1, total)


def _make_eval_loader_from(base_loader, batch_size: int | None = None, num_workers: int | None = None):
    import torch as _torch
    bs = int(batch_size) if (batch_size is not None and int(batch_size) > 0) else getattr(base_loader, "batch_size", 128)
    nw = int(num_workers) if (num_workers is not None) else int(getattr(base_loader, "num_workers", 0))
    return _torch.utils.data.DataLoader(
        base_loader.dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=max(0, nw),
        pin_memory=True if nw and nw > 0 else False,
        persistent_workers=False,
    )

@hydra.main(config_path="configs", version_base="1.3")
def main(cfg: DictConfig):
    # Sync selected method name from Hydra choices early
    _sync_method_name_from_hydra(cfg)
    # Optional: show hydra choices once for debugging
    try:
        print("hydra choices:", OmegaConf.to_container(cfg.hydra.runtime.choices, resolve=True))
    except Exception:
        pass
    # optional: show composed experiment.method.name
    try:
        print("experiment.method.name:", OmegaConf.select(cfg, "experiment.method.name"))
    except Exception:
        pass
    # 1) experiment 서브트리만 사용 (nested experiment 방어)
    exp = cfg.experiment if "experiment" in cfg else cfg
    # If composed config nests another 'experiment', unwrap it
    try:
        if isinstance(exp, DictConfig) and "experiment" in exp and isinstance(exp.experiment, (DictConfig, dict)):
            exp = exp.experiment
    except Exception:
        pass
    exp_dict = _to_dict(exp)

    # 2-a) 루트 키 오버라이드 병합(화이트리스트):
    # 실험 핵심 설정은 experiment.*를 신뢰하고, 루트에서는 런타임/로그용 최소 키만 병합한다.
    try:
        root_dict = _to_dict(cfg)
        ALLOWED_ROOT = {"results_dir", "exp_id", "seed", "log_level", "device"}
        for rk, rv in list(root_dict.items()):
            if rk in ("experiment", "hydra", "defaults"):
                continue
            if rk not in ALLOWED_ROOT:
                continue
            exp_dict[rk] = rv
        # 주의: root.method.name을 강제로 method_name에 주입하면 실제 선택된 experiment.method가 가려질 수 있음
        # (의도적으로 주입하지 않음)
    except Exception:
        pass
    
    # 2) 중첩 평탄화
    exp_dict = normalize_exp(exp_dict)

    # ENV 반영 (teacher/student/seed 등 – method는 Hydra 그룹을 신뢰)
    _apply_env_shortcuts(exp_dict)

    # 메소드 이름은 experiment.method.name만 신뢰 (그룹 단일화 가정)
    method_node = exp_dict.get("method") or {}
    mname = str((method_node.get("name") or "").strip().lower()) if isinstance(method_node, dict) else ""
    if not mname:
        raise ValueError(
            "experiment.method.name이 비어 있습니다. defaults에서 `- experiment/method@experiment.method: asib`를 사용하고, "
            "실행 시 `experiment.method=<name>`로 오버라이드하세요."
        )
    exp_dict["method_name"] = mname

    # 2.1) 앵커 적용 및 스냅샷 저장 (보호 키 비교용)
    try:
        anchor_node = exp_dict.get("anchor", None)
        if isinstance(anchor_node, dict) and anchor_node:
            # (0) 보호키 사전 진단을 위한 pre-merge 값 저장 (dotted keys 추적)
            def _capture(dct: dict, keys: list[str]):
                def _get(d, dotted):
                    cur = d
                    for t in dotted.split('.'):
                        if not isinstance(cur, dict):
                            return None
                        cur = cur.get(t)
                    return cur
                snap = {}
                for kk in keys:
                    snap[kk] = _get(dct, kk)
                return snap
            exp_dict["_pre_anchor_values"] = _capture(exp_dict, [
                "ce_alpha","kd_alpha","tau","tau_schedule","ce_label_smoothing",
                "optimizer","student_lr","student_weight_decay","dataset.batch_size",
                "mixup_alpha","cutmix_alpha_distill","use_partial_freeze","student_freeze_bn",
                "student_freeze_level","student_freeze_level_schedule",
                "teacher1_freeze_level","teacher2_freeze_level",
                "teacher1_freeze_level_schedule","teacher2_freeze_level_schedule",
                "teacher1_freeze_bn","teacher2_freeze_bn",
            ])
            # (1) 앵커를 effective config에 병합하여 실제 학습이 앵커 하이퍼를 사용하도록 함
            _deep_merge(exp_dict, anchor_node)
            # (2) 검증용 스냅샷 저장
            exp_dict["_anchor_snapshot"] = copy.deepcopy(anchor_node)
            # (3) 런타임 혼선을 줄이기 위해 제거
            exp_dict.pop("anchor", None)
    except Exception:
        pass

    # 정책 적용 → 결과경로 자동 생성 (보호 키는 여기서 변경하지 않음)
    _apply_method_policy(exp_dict, strict_baseline=True)
    # Remove deprecated override key to avoid confusion in logs
    if "force_ppf_off" in exp_dict:
        try:
            exp_dict.pop("force_ppf_off", None)
        except Exception:
            pass
    _auto_results(exp_dict)

    # 공정 가중치 정규화 및 결과 디렉토리 보장 (로거 생성 이전)
    # strict_mode=True면 자동 renorm 금지 → 검증 단계에서 에러 발생
    if not bool(exp_dict.get("strict_mode", True)) and bool(exp_dict.get("auto_renorm", False)):
        renorm_ce_kd(exp_dict)
    try:
        os.makedirs(exp_dict["results_dir"], exist_ok=True)
    except Exception:
        pass

    # 2-a) method.* 잔존 키로 인한 충돌 방지:
    # normalize_exp는 method의 값을 최상위에 채우되(없을 때만)
    # 잔존 method 서브트리가 남아있으면 일부 모듈이 cfg["method"]를 참조해
    # 의도치 않게 다른 하이퍼파라미터를 사용할 수 있다.
    # 혼선을 방지하기 위해 method 서브트리를 제거한다.
    try:
        if isinstance(exp_dict.get("method"), dict):
            # method.name을 보존하여 로깅/메타에 사용
            mn = exp_dict["method"].get("name")
            if mn and "method_name" not in exp_dict:
                exp_dict["method_name"] = mn
            exp_dict.pop("method", None)
    except Exception:
        pass

    # 2-a-1) (제거) 메소드별 가드는 초기에 일괄 적용됨

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

    # >>> finalize derived/runtime keys BEFORE lock to ensure immutability
    try:
        finalize_config(exp_dict)
    except Exception as _e:
        logging.debug("finalize_config skipped: %s", _e)

    # 3) 로거
    exp_dir = exp_dict.get("results_dir", ".")
    logger = get_logger(exp_dir, level=exp_dict.get("log_level", "INFO"))
    # Profile/auto defaults banner for quick diagnostics
    try:
        prof = exp_dict.get("_profile_applied") or exp_dict.get("profile")
        if prof:
            tvs = int(exp_dict.get("kd_two_view_start_epoch", -1))
            taus = float(exp_dict.get("tau_syn", exp_dict.get("tau", 4.0)))
            kuw = float(exp_dict.get("kd_uncertainty_weight", -1))
            logger.info("[AUTO] profile=%s tv_start=%s tau_syn=%.2f kd_unc_w=%.2f", str(prof), str(tvs), float(taus), float(kuw))
    except Exception:
        pass

    # 3-a) 최종 구성 검증 및 락(해시 저장)
    try:
        def _get(dct: dict, dotted: str):
            cur = dct
            for tok in dotted.split('.'):
                if not isinstance(cur, dict):
                    return None
                cur = cur.get(tok)
            return cur

        # 1) ce/kd 합 검증 (strict_mode 시 필수)
        ce = float(exp_dict.get("ce_alpha", 0.0))
        kd = float(exp_dict.get("kd_alpha", 0.0))
        if bool(exp_dict.get("strict_mode", True)) and not math.isclose(ce + kd, 1.0, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(f"ce_alpha+kd_alpha != 1 (got {ce}+{kd})")
        if (not bool(exp_dict.get("strict_mode", True))) and not math.isclose(ce + kd, 1.0, rel_tol=0.0, abs_tol=1e-8):
            if bool(exp_dict.get("auto_renorm", False)):
                tot = max(1e-12, ce + kd)
                exp_dict["ce_alpha"], exp_dict["kd_alpha"] = ce / tot, kd / tot
                logger.warning("[RENORM] ce/kd → %.6f/%.6f", exp_dict["ce_alpha"], exp_dict["kd_alpha"])

        # 2) 보호 키 위반 검사 (앵커 기준)
        # Protected fairness keys — kd_target excluded here to allow method-specific modes
        PROTECTED = {
            "ce_alpha","kd_alpha","tau","tau_schedule",
            "ce_label_smoothing","optimizer","student_lr","student_weight_decay",
            "dataset.batch_size","mixup_alpha","cutmix_alpha_distill",
            "use_partial_freeze","student_freeze_bn",
            "student_freeze_level","student_freeze_level_schedule",
            "teacher1_freeze_level","teacher2_freeze_level",
            "teacher1_freeze_level_schedule","teacher2_freeze_level_schedule",
            "teacher1_freeze_bn","teacher2_freeze_bn",
        }
        anchor = exp_dict.get("_anchor_snapshot", None)
        if isinstance(anchor, dict) and anchor:
            violations = []
            premerge_values = exp_dict.get("_pre_anchor_values", {})
            for k in PROTECTED:
                av = _get(anchor, k)
                ev = _get(exp_dict, k)
                if av is not None and ev is not None and av != ev:
                    pv = premerge_values.get(k)
                    violations.append((k, av, ev, pv))
            if violations:
                details = "\n".join([
                    f" - {k}: anchor={a} effective={e} pre_merge={p}"
                    for (k,a,e,p) in violations
                ])
                raise ValueError("Protected keys changed by method/overrides:\n" + details)

        # 3) 락 + 해시 기록
        locked_cfg = _sanitize_config_for_hash(exp_dict)
        eff = json.dumps(locked_cfg, sort_keys=True, default=str).encode("utf-8")
        exp_dict["config_sha256"] = hashlib.sha256(eff).hexdigest()
        # store for debugging; ignored by sanitizer
        try:
            exp_dict["_locked_config"] = locked_cfg
        except Exception:
            pass
        exp_dict["locked"] = True
        logger.info("[LOCK] config_sha256=%s", exp_dict["config_sha256"])
    except Exception as _ve:
        logger.error("[CONFIG-VALIDATION] %s", _ve)
        raise
    # Effective config (post-policy, post-normalize) quick check
    try:
        # Runtime stack quick banner
        logger.info(
            "[RUNTIME] Torch %s | CUDA build %s | cuDNN enabled=%s",
            str(torch.__version__),
            str(getattr(torch.version, "cuda", "none")),
            str(torch.backends.cudnn.enabled),
        )
        logger.info(
            "[CFG] method=%s kd_target=%s ce=%s kd=%s ib_beta=%s",
            str(exp_dict.get("method_name")),
            str(exp_dict.get("kd_target")),
            str(exp_dict.get("ce_alpha")),
            str(exp_dict.get("kd_alpha")),
            str(exp_dict.get("ib_beta")),
        )
    except Exception:
        pass
    # (정리) HParams 직전 추가 보정/가드는 제거됨 – 앞단에서 일괄 확정
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

    # 4) 숫자 캐스팅은 finalize_config에서 수행됨 (락 전 확정)
    # 5) 로그 저장기
    exp_logger = ExperimentLogger(exp_dict, exp_name=str(exp_dict.get("method_name", "exp")))

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
        # Safe mode can disable aggressive CUDA optimizations to avoid rare segfaults
        use_safe = bool(exp_dict.get("use_safe_mode", False))
        if use_safe:
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.enabled = False
            torch.set_float32_matmul_precision("high")
            logger.info("CUDA safe-mode: cuDNN benchmark OFF, TF32 OFF, cuDNN disabled")
        else:
            # cuDNN 벤치마크 모드 활성화 (Conv 연산 자동 최적화)
            torch.backends.cudnn.benchmark = True
            # TF32 활성화 (A100/3090에서 Conv/Matmul 가속)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Float32 matmul precision 설정
            torch.set_float32_matmul_precision("high")
            logger.info("CUDA optimizations enabled: cuDNN benchmark, TF32, high precision matmul")
            # Optional: allow disabling cuDNN selectively in perf mode for troubleshooting
            if bool(exp_dict.get("disable_cudnn", False)):
                torch.backends.cudnn.enabled = False
                torch.backends.cudnn.benchmark = False
                logger.info("cuDNN disabled by flag (perf mode)")

    # Dry-run: print config only, then exit (after merge/validation/lock)
    if bool(exp_dict.get("print_config_only", False)):
        try:
            with open(os.path.join(exp_dir, "effective_config.json"), "w", encoding="utf-8") as f:
                json.dump(exp_dict, f, indent=2)
            logging.info("[DRY-RUN] effective_config.json written. sha256=%s", exp_dict.get("config_sha256"))
        except Exception:
            pass
        return 0.0

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
            use_spawn_dl=bool(exp_dict.get("use_spawn_dl", False)),
            seed=exp_dict.get("seed", 42),
        )
        base_train_loader, base_test_loader = get_cifar100_loaders(
            root=data_root, batch_size=batch_size, num_workers=num_workers, augment=data_aug,
            use_spawn_dl=bool(exp_dict.get("use_spawn_dl", False)),
            backend=str(ds_cfg.get("backend", "npz")).lower(),
            log_first_batch_stats=bool(ds_cfg.get("log_first_batch_stats", False)),
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
            root=data_root, batch_size=batch_size, num_workers=num_workers, augment=data_aug,
            use_spawn_dl=bool(exp_dict.get("use_spawn_dl", False)),
            backend=str(ds_cfg.get("backend", "npz")).lower(),
            log_first_batch_stats=bool(ds_cfg.get("log_first_batch_stats", False)),
        )
    elif dataset_name == "imagenet32":
        train_loader, test_loader = get_imagenet32_loaders(
            root=data_root, batch_size=batch_size, num_workers=num_workers, augment=data_aug,
            use_spawn_dl=bool(exp_dict.get("use_spawn_dl", False))
        )
    else:
        raise ValueError(f"Unknown dataset.name={dataset_name}")

    # Rebuild loaders with persistent_workers=False for stability
    try:
        import torch as _torch
        def _rebuild_loader(_ldr, _bs, _nw, _pin=False, *, shuffle=False):
            return _torch.utils.data.DataLoader(
                _ldr.dataset,
                batch_size=_bs,
                shuffle=bool(shuffle),
                num_workers=max(0, int(_nw)),
                pin_memory=_pin,
                persistent_workers=False,
            )
        if isinstance(train_loader, _torch.utils.data.DataLoader) and getattr(train_loader, "persistent_workers", False):
            train_loader = _rebuild_loader(train_loader, batch_size, num_workers, False, shuffle=True)
        if isinstance(test_loader, _torch.utils.data.DataLoader) and getattr(test_loader, "persistent_workers", False):
            test_loader = _rebuild_loader(test_loader, batch_size, num_workers, False, shuffle=False)
    except Exception:
        pass

    # 8) 클래스 수 (래퍼 안전 추출: Subset/ConcatDataset/커스텀 래퍼 대응)
    def _infer_num_classes_from(ds) -> int | None:
        nc = getattr(ds, "num_classes", None)
        if isinstance(nc, int):
            return nc
        cls = getattr(ds, "classes", None)
        if cls is not None:
            return (len(cls) if not isinstance(cls, int) else cls)
        return None

    base_ds = train_loader.dataset
    num_classes = None
    try:
        import torch.utils.data as _tud
        visited = 0
        while visited < 10 and num_classes is None and base_ds is not None:
            visited += 1
            n = _infer_num_classes_from(base_ds)
            if n is not None:
                num_classes = int(n)
                break
            if isinstance(base_ds, _tud.Subset):
                base_ds = base_ds.dataset
            elif isinstance(base_ds, _tud.ConcatDataset):
                # any child that exposes classes/num_classes
                for ch in getattr(base_ds, "datasets", []) or []:
                    n = _infer_num_classes_from(ch)
                    if n is not None:
                        num_classes = int(n)
                        break
                break
            elif hasattr(base_ds, "dataset"):
                base_ds = getattr(base_ds, "dataset")
            else:
                break
        if num_classes is None:
            # 마지막 안전 가드: CIFAR-100 기본값 시도
            num_classes = 100
    except Exception:
        # 실패 시 이전 방식으로 폴백 (예외 방지)
        n_classes = getattr(train_loader.dataset, "classes", None) or getattr(train_loader.dataset, "num_classes", None)
        if n_classes is None:
            num_classes = 100
        else:
            num_classes = len(n_classes) if not isinstance(n_classes, int) else int(n_classes)

    exp_logger.update_metric("num_classes", num_classes)
    check_label_range(train_loader.dataset, num_classes)
    check_label_range(test_loader.dataset, num_classes)

    # 9) 교사/학생 생성
    # Determine if teachers are needed at all
    need_teachers = (
        bool(exp_dict.get("kd_alpha", 0.0) > 0.0)
        or bool(exp_dict.get("use_ib", False))
        or bool(exp_dict.get("compute_teacher_eval", False))
    )
    # Edge case: kd_target requires synergy routing even when kd_alpha/use_ib/eval are off
    try:
        kd_tgt_for_build = str(exp_dict.get("kd_target", "avg")).lower()
        if kd_tgt_for_build in ("synergy", "auto", "auto_min"):
            need_teachers = True
    except Exception:
        pass
    # teacher1
    t1 = exp_dict.get("teacher1", {})
    t1_name = t1.get("name")
    ds_name = (exp_dict.get("dataset", {}) or {}).get("name", "cifar100")
    t1_ckpt = _ckpt_from_fixed_map(t1_name, ds_name, exp_dict.get("teacher1_ckpt")) if need_teachers else None
    if need_teachers and not t1_name:
        raise ValueError("Missing 'experiment.teacher1.name'")

    teacher1 = None
    teacher2 = None
    if need_teachers:
        logging.getLogger().setLevel(logging.WARNING)
        teacher1 = create_teacher_by_name(
            teacher_name=t1_name,
            num_classes=num_classes,
            pretrained=t1.get("pretrained", True),
            small_input=small_input,
            cfg=exp_dict,
        )
        logging.getLogger().setLevel(logging.INFO)

    if t1_ckpt and os.path.exists(t1_ckpt):
        # Always load checkpoint to CPU first to avoid early CUDA init segfaults
        try:
            sd = torch.load(t1_ckpt, map_location="cpu", weights_only=True)
        except TypeError:
            sd = torch.load(t1_ckpt, map_location="cpu")
        teacher1.load_state_dict(sd, strict=False)
        logging.info("Loaded teacher1 state_dict from %s to CPU (will move to %s)", t1_ckpt, device)
    if need_teachers:
        teacher1 = teacher1.to(device)
    try:
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    # teacher2
    t2 = exp_dict.get("teacher2", {})
    t2_name = t2.get("name")
    t2_ckpt = _ckpt_from_fixed_map(t2_name, ds_name, exp_dict.get("teacher2_ckpt")) if need_teachers else None
    if need_teachers and not t2_name:
        raise ValueError("Missing 'experiment.teacher2.name'")

    # 존재 검사 (없으면 즉시 에러로 중단, 1% 방지)
    def _must_exist(p: Optional[str], who: str):
        if not p or not os.path.exists(p):
            raise FileNotFoundError(f"[CKPT] {who} ckpt not found for dataset={ds_name}: {p}")
        return p

    if need_teachers:
        t1_ckpt = _must_exist(t1_ckpt, f"teacher1:{t1_name}")
        t2_ckpt = _must_exist(t2_ckpt, f"teacher2:{t2_name}")

    # 메타/로그에 실제 경로 반영
    if need_teachers:
        exp_dict["teacher1_ckpt"] = t1_ckpt
        exp_dict["teacher2_ckpt"] = t2_ckpt

    if need_teachers:
        logging.getLogger().setLevel(logging.WARNING)
        teacher2 = create_teacher_by_name(
            teacher_name=t2_name,
            num_classes=num_classes,
            pretrained=t2.get("pretrained", True),
            small_input=small_input,
            cfg=exp_dict,
        )
        logging.getLogger().setLevel(logging.INFO)

    if t2_ckpt and os.path.exists(t2_ckpt):
        try:
            sd = torch.load(t2_ckpt, map_location="cpu", weights_only=True)
        except TypeError:
            sd = torch.load(t2_ckpt, map_location="cpu")
        teacher2.load_state_dict(sd, strict=False)
        logging.info("Loaded teacher2 state_dict from %s to CPU (will move to %s)", t2_ckpt, device)
    if need_teachers:
        teacher2 = teacher2.to(device)
    try:
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    # Optional quick eval of teachers (respect flag)
    teacher1_acc = 0.0
    teacher2_acc = 0.0
    _do_eval = bool(exp_dict.get("compute_teacher_eval", False))
    eval_on_gpu = bool(exp_dict.get("teacher_eval_on_gpu", True))
    eval_max_batches = int(exp_dict.get("teacher_eval_max_batches", 0)) or None
    eval_bs = int(exp_dict.get("teacher_eval_batch_size", 0)) or None
    eval_amp = bool(exp_dict.get("teacher_eval_amp", True))
    eval_force_cudnn = bool(exp_dict.get("teacher_eval_force_cudnn", False))
    if _do_eval and need_teachers:
        try:
            if eval_on_gpu and device == "cuda" and torch.cuda.is_available():
                # Optional: temporarily enable cuDNN/TF32 for faster eval even in safe-mode
                prev_cudnn_en = torch.backends.cudnn.enabled
                prev_cudnn_bn = torch.backends.cudnn.benchmark
                prev_tf32     = torch.backends.cuda.matmul.allow_tf32
                if eval_force_cudnn:
                    torch.backends.cudnn.enabled = True
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cuda.matmul.allow_tf32 = True
                try:
                    # Optional smaller eval loader for VRAM safety
                    eval_num_workers = exp_dict.get("teacher_eval_num_workers", None)
                    eval_loader = _make_eval_loader_from(
                        test_loader,
                        batch_size=eval_bs,
                        num_workers=(int(eval_num_workers) if eval_num_workers is not None else (exp_dict.get("dataset", {}) or {}).get("num_workers", 0)),
                    )
                    teacher1_acc = _quick_eval(teacher1, eval_loader, device, max_batches=eval_max_batches, use_amp=eval_amp)
                    logging.info("Teacher1 (%s) testAcc=%.2f%% (GPU eval)", t1_name, teacher1_acc)
                    teacher2_acc = _quick_eval(teacher2, eval_loader, device, max_batches=eval_max_batches, use_amp=eval_amp)
                    logging.info("Teacher2 (%s) testAcc=%.2f%% (GPU eval)", t2_name, teacher2_acc)
                finally:
                    if eval_force_cudnn:
                        torch.backends.cudnn.enabled = prev_cudnn_en
                        torch.backends.cudnn.benchmark = prev_cudnn_bn
                        torch.backends.cuda.matmul.allow_tf32 = prev_tf32
            else:
                # CPU path (safer, slower)
                teacher1_acc = _quick_eval(teacher1.to("cpu"), test_loader, device="cpu", max_batches=eval_max_batches, use_amp=False)
                logging.info("Teacher1 (%s) testAcc=%.2f%% (CPU eval)", t1_name, teacher1_acc)
                teacher2_acc = _quick_eval(teacher2.to("cpu"), test_loader, device="cpu", max_batches=eval_max_batches, use_amp=False)
                logging.info("Teacher2 (%s) testAcc=%.2f%% (CPU eval)", t2_name, teacher2_acc)
                teacher1 = teacher1.to(device); teacher2 = teacher2.to(device)
        except Exception as e:
            logging.warning("GPU teacher eval failed, falling back to CPU: %s", e)
            teacher1_acc = _quick_eval(teacher1.to("cpu"), test_loader, device="cpu", max_batches=eval_max_batches, use_amp=False)
            teacher2_acc = _quick_eval(teacher2.to("cpu"), test_loader, device="cpu", max_batches=eval_max_batches, use_amp=False)
            teacher1 = teacher1.to(device); teacher2 = teacher2.to(device)

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

    # Quick CUDA sanity log
    try:
        if device == "cuda" and torch.cuda.is_available():
            logger.info(
                "[CUDA] available=%s | device=%s | student_param_device=%s | mem=%.2fMB",
                str(torch.cuda.is_available()),
                str(device),
                str(next(student.parameters()).device),
                float(torch.cuda.memory_allocated())/1024.0/1024.0,
            )
    except Exception:
        pass

    # 10) 학습 전 설정 (IB_MBM 생성 전에 q_dim 설정)
    num_stages = int(exp_dict.get("num_stages", 1))
    # partial_freeze/safety switches were finalized pre-lock
    auto_set_ib_mbm_query_dim_with_model(student, exp_dict)  # ← IB_MBM 생성 전에 호출

    # 11) IB_MBM & Synergy Head (q_dim 설정 후 생성)
    from models import build_ib_mbm_from_teachers as build_from_teachers
    # IB out dim alignment was finalized pre-lock
    # -----------------------------------------------------------------
    # Build IB_MBM/Synergy only when needed
    # - IB enabled (A-step/IB weighting)
    # - or KD target requires synergy routing (synergy/auto)
    kd_tgt = str(exp_dict.get("kd_target", "avg")).lower()
    use_ib_flag = bool(exp_dict.get("use_ib", False))
    # Build path whenever kd_target requires synergy routing, or IB is enabled for KL
    # Even when use_ib is False, we still need synergy logits for kd_target in {synergy, auto, auto_min}
    need_synergy = (kd_tgt in ("synergy", "auto", "auto_min")) or use_ib_flag
    ib_mbm = None
    synergy_head = None
    if need_synergy:
        ib_mbm, synergy_head = build_from_teachers([teacher1, teacher2], exp_dict)
        ib_mbm = ib_mbm.to(device)
        synergy_head = synergy_head.to(device)
    
    # 11-a) 채널-라스트 메모리 포맷 적용 (Conv 연산 가속)
    if device == "cuda" and torch.cuda.is_available():
        # ConvNet 계열 모델들을 채널-라스트 포맷으로 변환 (변수에 재대입)
        if bool(exp_dict.get("use_channels_last", True)):
            try:
                if need_teachers and (teacher1 is not None):
                    teacher1 = teacher1.to(memory_format=torch.channels_last)
                if need_teachers and (teacher2 is not None):
                    teacher2 = teacher2.to(memory_format=torch.channels_last)
                student  = student.to(memory_format=torch.channels_last)
                logger.info("teachers/students converted to channels_last")
            except Exception as e:
                logger.debug(f"channels_last conversion skipped: {e}")

    # 11-a-5) (제거) 메소드별 가드/보정은 앞단에서 일괄 적용됨

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

    # Assemble teachers list for trainer
    teachers = []
    if need_teachers:
        teachers = [t for t in (teacher1, teacher2) if t is not None]

    # 3-b) 해시 재검증 헬퍼 (변이 감지)
    def _assert_config_hash(tag: str):
        try:
            eff_now = json.dumps(_sanitize_config_for_hash(exp_dict), sort_keys=True, default=str).encode("utf-8")
            sha_now = hashlib.sha256(eff_now).hexdigest()
            if exp_dict.get("config_sha256") and sha_now != exp_dict["config_sha256"]:
                # Optional diff to speed up debug of future regressions
                try:
                    locked_cfg = exp_dict.get("_locked_config", {})
                    now_cfg = _sanitize_config_for_hash(exp_dict)
                    def _flat(d, p=""):
                        out = {}
                        for k, v in d.items():
                            kp = f"{p}.{k}" if p else k
                            if isinstance(v, dict):
                                out.update(_flat(v, kp))
                            else:
                                out[kp] = v
                        return out
                    L, N = _flat(locked_cfg), _flat(now_cfg)
                    added = sorted(set(N) - set(L))
                    removed = sorted(set(L) - set(N))
                    changed = sorted(k for k in set(L) & set(N) if L[k] != N[k])
                    logging.error("[LOCK-DIFF] added=%s", added)
                    logging.error("[LOCK-DIFF] removed=%s", removed)
                    logging.error("[LOCK-DIFF] changed(sample)=%s", [(k, L[k], N[k]) for k in changed[:20]])
                finally:
                    raise RuntimeError(f"[LOCK] Config mutated after lock ({tag}). expected={exp_dict['config_sha256']} got={sha_now}")
        except Exception as _e:
            raise

    # 13) 트레이닝
    if exp_dict.get("cl_mode", False):
        logging.info("📚 Running in Continual Learning mode...")
        _assert_config_hash("before_cl")
        final_acc = run_continual_learning(
            teachers, ib_mbm, synergy_head, student, exp_dict, exp_logger
        )
    else:
        logging.info("🎯 Running in Standard training mode...")
        try:
            logging.info("📊 Training will start now - you should see epoch progress logs...")
            _assert_config_hash("before_run")
            final_acc = run_training_stages(
                teachers,
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
            # SAFE RETRY: num_workers=0, AMP off, no channels_last
            try:
                import torch as _torch
                exp_dict["use_amp"] = False
                # Do not mutate channels_last after lock; leave it as-is to avoid hash mismatch
                _assert_config_hash("before_safe_retry")
                train_loader = _torch.utils.data.DataLoader(train_loader.dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False, persistent_workers=False)
                test_loader  = _torch.utils.data.DataLoader(test_loader.dataset,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, persistent_workers=False)
                logging.warning("[SAFE-RETRY] Re-running with num_workers=0 and AMP off.")
                final_acc = run_training_stages(
                    teachers,
                    ib_mbm,
                    synergy_head,
                    student,
                    train_loader,
                    test_loader,
                    exp_dict,
                    exp_logger,
                    num_stages,
                )
            except Exception as e2:
                logging.error(f"❌ SAFE-RETRY failed: {e2}", exc_info=True)
                final_acc = 0.0

    logging.info(f"✅ Training completed. Final student accuracy: {final_acc:.2f}%")
    _assert_config_hash("after_run")
    exp_logger.update_metric("final_student_acc", final_acc)
    exp_logger.save_results()
    return final_acc


if __name__ == "__main__":
    main() 