# utils/print_cfg.py
from __future__ import annotations
# -*- coding: utf-8 -*-
from collections import defaultdict
from typing import Dict, Callable, Any, List, Tuple

# ────────────────────────────────────────
# NEW ①  공통 테이블 렌더러 (단일 그룹·가로 정렬)
# ----------------------------------------------------------------------------
def _render_table(rows, title="Hyper‑parameters", ascii_only=False, log_fn=print):
    """`rows` == List[Tuple[str, Any]]  →  pretty table 문자열으로 출력"""
    if not rows:
        return

    k_width = max(len(k) for k, _ in rows) + 2
    v_width = max(len(str(v)) for _, v in rows) + 2
    h   = "-" if ascii_only else "─"
    tl, tr, bl, br, v = ("+", "+", "+", "+", "|") if ascii_only else ("┌", "┐", "└", "┘", "│")
    hd_mid = "+" if ascii_only else "┬"

    log_fn(tl + h * (k_width + v_width + 1) + tr)
    log_fn(f"{v} {title} ({len(rows)})".ljust(k_width + v_width + 2) + v)
    log_fn(("+" if ascii_only else "├") + h * k_width + hd_mid + h * v_width + ("+" if ascii_only else "┤"))
    for k, v_ in rows:
        log_fn(f"{v} {k.ljust(k_width-1)}{v} {str(v_).ljust(v_width-1)}{v}")
    log_fn(bl + h * k_width + ("+" if ascii_only else "┴") + h * v_width + br)


# ────────────────────────────────────────
# NEW ②  그룹별 하이퍼파라미터 출력
# ----------------------------------------------------------------------------
def print_hparams_grouped(
    cfg: Dict[str, Any],
    src_map: Dict[str, str],
    log_fn: Callable[[str], None] = print,
    *,
    include_unmapped: bool = True,
    order: Tuple[str, ...] = ("base", "scenario", "method", "CLI", "others"),
) -> None:
    """
    cfg      : 최종 머지된 dict
    src_map  : {key: "base" | "scenario" | "method" | "CLI" …}
    include_unmapped : True → src_map 에 없는 key 는 자동으로 others 로
    """

    groups: Dict[str, List[Tuple[str, Any]]] = defaultdict(list)
    for k, v in cfg.items():
        label = src_map.get(k, "others" if include_unmapped else None)
        if label is None:
            continue
        groups[label].append((k, v))

    ordered_groups = list(order) + [g for g in groups if g not in order]

    for g_name in ordered_groups:
        if g_name not in groups:
            continue
        params = sorted(groups[g_name])
        width_k = max(len(k) for k, _ in params)
        width_v = max(len(str(v)) for _, v in params)
        border  = "─" * (width_k + width_v + 7)
        log_fn(f"┌{border}┐")
        log_fn(f"│ [{g_name}] Hyper‑parameters ({len(params)}) │")
        log_fn(f"├{'─'*width_k}┬{'─'*width_v}┤")
        for k, v in params:
            log_fn(f"│ {k:<{width_k}} │ {str(v):<{width_v}} │")
        log_fn(f"└{border}┘")

def print_hparams(
    cfg,
    title="Hyper‑parameters",
    ascii_only=False,
    log_fn=print,
):
    """
    # (기존 함수는 보존하되, 동작을 _render_table 로 위임)
    # ---------------------------------------------------------------------

    Parameters
    ----------
    cfg : dict | Namespace | OmegaConf | yacs CfgNode
        Configuration object. Nested structures are flattened using
        dot notation.
    title : str, optional
        Table title displayed at the top. Defaults to ``"Hyper-parameters"``.
    """
    # 1) cfg → python dict  -------------------------------------------------
    try:
        from omegaconf import OmegaConf  # ΩConf?
        if isinstance(cfg, (OmegaConf,)):
            cfg = OmegaConf.to_container(cfg, resolve=True)
    except ModuleNotFoundError:
        pass
    try:
        from yacs.config import CfgNode  # yacs?
        if isinstance(cfg, CfgNode):
            cfg = cfg.clone().freeze(False)
            cfg = cfg.dump() if isinstance(cfg, str) else dict(cfg)
    except ModuleNotFoundError:
        pass
    if not isinstance(cfg, dict):
        cfg = vars(cfg)  # Namespace -> dict

    # 2) flatten ------------------------------------------------------------
    flat = {}

    def _walk(d, prefix=""):
        for k, v in d.items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                _walk(v, key + ".")
            else:
                flat[key] = v

    _walk(cfg)

    # 3) pretty print -------------------------------------------------------
    rows = sorted(flat.items())
    _render_table(rows, title=title, ascii_only=ascii_only, log_fn=log_fn)
