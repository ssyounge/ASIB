# core/utils.py

import torch
import logging
from typing import Dict, Any


def _renorm_ce_kd(ce_loss, kd_loss, ce_alpha, kd_alpha):
    """Renormalize ce_alpha and kd_alpha to sum to 1."""
    total_alpha = ce_alpha + kd_alpha
    if total_alpha > 0:
        ce_alpha = ce_alpha / total_alpha
        kd_alpha = kd_alpha / total_alpha
    return ce_alpha * ce_loss + kd_alpha * kd_loss

def renorm_ce_kd(cfg: Dict[str, Any]):
    """Renormalize ce_alpha and kd_alpha to sum to 1."""
    if "ce_alpha" in cfg and "kd_alpha" in cfg:
        ce, kd = float(cfg["ce_alpha"]), float(cfg["kd_alpha"])
        if abs(ce + kd - 1) > 1e-5:
            tot = ce + kd
            # Handle zero division case
            if tot == 0:
                cfg["ce_alpha"], cfg["kd_alpha"] = 0.5, 0.5
            else:
                cfg["ce_alpha"], cfg["kd_alpha"] = ce / tot, kd / tot
            if cfg.get("debug_verbose"):
                logging.debug(
                    "[Auto-cfg] ce_alpha+kd_alpha !=1 → 재정규화 (ce=%.3f, kd=%.3f)",
                    cfg["ce_alpha"],
                    cfg["kd_alpha"],
                )


def setup_partial_freeze_schedule(num_stages: int):
    """Setup partial freeze schedule."""
    return [-1] * num_stages

def setup_partial_freeze_schedule_with_cfg(cfg: Dict[str, Any], num_stages: int):
    """Setup partial freeze schedule."""
    fl = int(cfg.get("student_freeze_level", -1) or -1)

    plan = cfg.get("student_freeze_schedule")
    if plan is None:
        # fallback :  init_lvl,  max(init_lvl-1,-1), …
        plan = [max(-1, fl - s) for s in range(num_stages)]
    if len(plan) < num_stages:
        raise ValueError(
            f"student_freeze_schedule 길이({len(plan)}) < num_stages({num_stages})"
        )
    cfg["student_freeze_schedule"] = plan

    # Auto-set student_pretrained based on freeze schedule
    if "student_pretrained" not in cfg:
        # 스케줄에 0 이상 값이 하나라도 있으면 pretrained 권장
        need_pt = any(lvl >= 0 for lvl in cfg["student_freeze_schedule"])
        cfg["student_pretrained"] = need_pt
        if cfg.get("debug_verbose"):
            logging.debug(
                "[Auto-cfg] student_pretrained←%s (freeze_sched=%s)",
                cfg["student_pretrained"],
                cfg["student_freeze_schedule"],
            )

    if fl >= 0 and not cfg.get("student_pretrained", False):
        logging.warning(
            "freeze_level ≥0 인데 student_pretrained=False ‑‑ 동결된 층이 랜덤 초기화 상태가 됩니다."
        )


def setup_safety_switches(num_stages: int):
    """Setup safety switches for partial freeze."""
    return {"student_freeze_level": -1, "teacher1_freeze_level": -1, "teacher2_freeze_level": -1, "student_freeze_schedule": [-1] * num_stages}

def setup_safety_switches_with_cfg(cfg: Dict[str, Any], num_stages: int):
    """Setup safety switches for partial freeze."""
    if not cfg.get("use_partial_freeze", False):
        cfg["student_freeze_level"] = -1
        cfg["teacher1_freeze_level"] = -1
        cfg["teacher2_freeze_level"] = -1
        cfg["student_freeze_schedule"] = [-1] * num_stages


def auto_set_mbm_query_dim(cfg: Dict[str, Any]):
    """Auto-set mbm_query_dim based on student model."""
    if cfg.get("mbm_query_dim", 0) in (0, None):
        cfg["mbm_query_dim"] = 512  # Default value
    return cfg

def auto_set_mbm_query_dim_with_model(student_model, cfg: Dict[str, Any]):
    """Auto-set mbm_query_dim based on student model."""
    if cfg.get("mbm_query_dim", 0) in (0, None):
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
            dummy = torch.randn(1, 3, 32, 32, device=cfg["device"])
            feat_dict, _, _ = student_model(dummy)
            qdim = feat_dict.get("distill_feat", feat_dict.get("feat_2d")).shape[-1]
            cfg["mbm_query_dim"] = int(qdim)
            if cfg.get("debug_verbose"):
                logging.debug("[Auto-cfg] mbm_query_dim ← %d", qdim)


def cast_numeric_configs(cfg: Dict[str, Any]):
    """Cast string numeric values to float/int/bool."""
    _float_keys = [
        "teacher_lr",
        "student_lr",
        "teacher_weight_decay",
        "student_weight_decay",
        "reg_lambda",
        "mbm_reg_lambda",
        "kd_alpha",
        "ce_alpha",
        "ib_beta",
        "momentum",
    ]
    _int_keys = [
        "num_stages",
        "student_epochs_per_stage",
        "teacher_adapt_epochs",
    ]
    _bool_keys = [
        "nesterov",
    ]
    
    for k in _float_keys:
        if k in cfg and isinstance(cfg[k], str):
            try:
                cfg[k] = float(cfg[k])
            except ValueError:
                pass  # ignore
    
    for k in _int_keys:
        if k in cfg and isinstance(cfg[k], str):
            try:
                cfg[k] = int(cfg[k])
            except ValueError:
                pass  # ignore
    
    for k in _bool_keys:
        if k in cfg and isinstance(cfg[k], str):
            try:
                cfg[k] = cfg[k].lower() in ('true', '1', 'yes', 'on')
            except (ValueError, AttributeError):
                pass  # ignore
    
    return cfg 