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
    """학생 모델/설정으로부터 ib_mbm_query_dim을 안전하게 결정.

    - 중첩 cfg(experiment, experiment.method) 조회 지원
    - student_model.get_feat_dim() 폴백 지원
    - distill_feat/feat_2d 미노출 시 distill_out_dim 또는 512로 폴백
    - device 자동 선택 및 출력 형태(tuple/dict) 유연 처리
    """
    key = "ib_mbm_query_dim"

    def _get(k: str, default=None):
        if k in cfg:
            return cfg[k]
        exp = cfg.get("experiment", {}) if isinstance(cfg.get("experiment"), dict) else {}
        if k in exp:
            return exp[k]
        meth = exp.get("method", {}) if isinstance(exp.get("method"), dict) else {}
        return meth.get(k, default)

    def _set_key(v: int):
        vv = int(v)
        cfg[key] = vv
        exp = cfg.get("experiment")
        if isinstance(exp, dict):
            exp[key] = vv
            meth = exp.get("method")
            if isinstance(meth, dict):
                meth[key] = vv

    # 0) 사전 지정값이 있으면 그대로 사용
    preset = int(_get(key, 0) or 0)
    if preset > 0:
        _set_key(preset)
        return

    # 0.5) 모델 API 폴백(get_feat_dim)
    if hasattr(student_model, "get_feat_dim"):
        try:
            d = int(student_model.get_feat_dim())
            if d > 0:
                _set_key(d)
                logging.info("[auto_set_ib] q_dim set via student.get_feat_dim()=%d", d)
                return
        except Exception:
            pass

    # 1) 어댑터가 설정되어 있으면 distill_out_dim을 우선
    use_da = bool(_get("use_distillation_adapter", False))
    feat_key = str(_get("feat_kd_key", "feat_2d"))
    distill_dim = int(_get("distill_out_dim", 0) or 0)
    if use_da and feat_key == "distill_feat" and distill_dim > 0:
        _set_key(distill_dim)
        logging.info("[auto_set_ib] q_dim set to distill_out_dim=%d", distill_dim)
        return

    # 2) 모델에서 직접 추정 시도 (모델 파라미터 디바이스 우선)
    dev_from_model = None
    try:
        p = next(student_model.parameters(), None)
        if p is not None:
            dev_from_model = p.device.type
    except Exception:
        pass
    device = dev_from_model or _get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        with torch.no_grad():
            # BN 안전: 현재 모드 저장 후 eval로 전환
            prev_train = getattr(student_model, "training", True)
            student_model.eval()
            try:
                # 입력 크기 결정: cfg.image_size 우선, 없으면 small_input→32, 아니면 224
                img_size = int(_get("image_size", 0) or 0)
                if img_size <= 0:
                    img_size = 32 if bool(_get("small_input", True)) else 224
                dummy = torch.randn(1, 3, img_size, img_size, device=device)
                out = student_model(dummy)
            finally:
                # 원래 모드로 복원
                try:
                    student_model.train(prev_train)
                except Exception:
                    pass
            feat_dict = None
            if isinstance(out, tuple) and len(out) > 0:
                feat_dict = out[0]
            elif isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
                feat_dict = out[0]
            elif isinstance(out, dict):
                feat_dict = out

            cand = None
            if isinstance(feat_dict, dict):
                cand = feat_dict.get(feat_key) or feat_dict.get("distill_feat") or feat_dict.get("feat_2d")
            if cand is not None and hasattr(cand, "shape") and getattr(cand, "shape", None) and cand.shape[-1] > 0:
                _set_key(int(cand.shape[-1]))
                if _get("debug_verbose"):
                    logging.debug("[auto_set_ib] %s ← %d (from model features)", key, int(cand.shape[-1]))
                return
    except Exception as e:
        if _get("debug_verbose"):
            logging.debug("[auto_set_ib] model-based qdim inference failed: %s", e)

    # 3) 최종 폴백: distill_out_dim > 0 아니면 512
    qdim_fb = distill_dim if distill_dim > 0 else 512
    _set_key(qdim_fb)
    logging.info("[auto_set_ib] q_dim fallback set to %d", qdim_fb)


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