# core/utils.py

import torch
import logging
import re
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
    """Setup partial freeze schedule (supports nested cfg['experiment'])."""
    target = cfg.get("experiment") if isinstance(cfg.get("experiment"), dict) else cfg
    fl = int(target.get("student_freeze_level", -1) or -1)

    plan = target.get("student_freeze_schedule")
    if plan is None:
        plan = [max(-1, fl - s) for s in range(num_stages)]
    # Normalize by pad/truncate instead of raising, to avoid unintended hard failure
    if len(plan) < num_stages:
        plan = plan + [plan[-1] if plan else -1] * (num_stages - len(plan))
    elif len(plan) > num_stages:
        plan = plan[:num_stages]
    target["student_freeze_schedule"] = plan

    # Auto-set student_pretrained based on freeze schedule
    if "student_pretrained" not in target:
        need_pt = any(lvl >= 0 for lvl in target["student_freeze_schedule"])
        target["student_pretrained"] = need_pt
        if target.get("debug_verbose"):
            logging.debug(
                "[Auto-cfg] student_pretrained←%s (freeze_sched=%s)",
                target["student_pretrained"],
                target["student_freeze_schedule"],
            )

    if fl >= 0 and not target.get("student_pretrained", False):
        logging.warning(
            "freeze_level ≥0 인데 student_pretrained=False ‑‑ 동결된 층이 랜덤 초기화 상태가 됩니다."
        )


def setup_safety_switches(num_stages: int):
    """Setup safety switches for partial freeze."""
    return {"student_freeze_level": -1, "teacher1_freeze_level": -1, "teacher2_freeze_level": -1, "student_freeze_schedule": [-1] * num_stages}

def setup_safety_switches_with_cfg(cfg: Dict[str, Any], num_stages: int):
    """Setup safety switches for partial freeze (supports nested cfg['experiment'])."""
    target = cfg.get("experiment") if isinstance(cfg.get("experiment"), dict) else cfg
    if not target.get("use_partial_freeze", False):
        # Respect user/anchor values; only fill defaults if missing
        if "student_freeze_level" not in target:
            target["student_freeze_level"] = -1
        if "teacher1_freeze_level" not in target:
            target["teacher1_freeze_level"] = -1
        if "teacher2_freeze_level" not in target:
            target["teacher2_freeze_level"] = -1
        if "student_freeze_schedule" not in target:
            target["student_freeze_schedule"] = [-1] * num_stages
    else:
        # If schedule exists but has mismatched length, normalize by pad/truncate
        plan = target.get("student_freeze_schedule")
        if isinstance(plan, list) and len(plan) != num_stages:
            if len(plan) < num_stages:
                plan = plan + [-1] * (num_stages - len(plan))
            else:
                plan = plan[:num_stages]
            target["student_freeze_schedule"] = plan


def auto_set_ib_mbm_query_dim(cfg: Dict[str, Any]):
    """Auto-set ib_mbm_query_dim based on student model (legacy keys removed)."""
    key = "ib_mbm_query_dim"
    if cfg.get(key, 0) in (0, None):
        cfg[key] = 512  # Default
    return cfg

def auto_set_ib_mbm_query_dim_with_model(student_model, cfg: Dict[str, Any]):
    """Auto-set ib_mbm_query_dim based on student model (legacy keys removed)."""
    key = "ib_mbm_query_dim"
    
    # distill_feat(어댑터 512)를 쿼리로 쓰는 경우 우선
    if cfg.get("feat_kd_key", "feat_2d") == "distill_feat" and cfg.get("use_distillation_adapter", False):
        qdim = int(cfg.get("distill_out_dim", 0))
        if qdim > 0:
            cfg[key] = qdim
            logging.info(f"[auto_set_ib] q_dim set to distill_out_dim={qdim}")
            return
    
    # 이하 기존 로직 유지(학생 feat 기반 추정)
    if cfg.get(key, 0) in (0, None):
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
            dummy = torch.randn(1, 3, 32, 32, device=cfg["device"])
            feat_dict, _, _ = student_model(dummy)
            qdim = feat_dict.get("distill_feat", feat_dict.get("feat_2d")).shape[-1]
            cfg[key] = int(qdim)
            if cfg.get("debug_verbose"):
                logging.debug("[Auto-cfg] %s ← %d", key, qdim)


def cast_numeric_configs(cfg: Dict[str, Any]):
    """Recursively cast string values that look like bool/int/float.

    - Booleans: "true/false/1/0/yes/no/on/off" (case-insensitive)
    - Integers: optional sign, digits only (e.g., "-3")
    - Floats: standard or scientific notation (e.g., "0.001", "1e-3")
    Leaves non-numeric strings unchanged. Works for any nested dicts.
    """

    bool_truthy = {"true", "1", "yes", "on"}
    bool_falsy = {"false", "0", "no", "off"}
    int_pattern = re.compile(r"^[+-]?\d+$")
    float_pattern = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")

    def _maybe_cast(value: Any) -> Any:
        if isinstance(value, str):
            s = value.strip()
            sl = s.lower()
            # Bool
            if sl in bool_truthy:
                return True
            if sl in bool_falsy:
                return False
            # Int
            if int_pattern.match(s):
                try:
                    return int(s)
                except ValueError:
                    return value
            # Float
            if float_pattern.match(s):
                try:
                    return float(s)
                except ValueError:
                    return value
        return value

    def _cast_in_place(d: Dict[str, Any]):
        for k, v in list(d.items()):
            if isinstance(v, dict):
                _cast_in_place(v)
            else:
                d[k] = _maybe_cast(v)

    _cast_in_place(cfg)
    return cfg