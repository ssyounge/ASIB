# utils/print_cfg.py
from __future__ import annotations

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
    cfg: dict,
    src_map: dict[str, str] | None = None,
    *,
    ascii_only: bool = False,
    log_fn=print,
):
    """
    cfg     : 최종 머지된 dict
    src_map : 각 key → 'base' / 'scenario' / 'method' / 'sweep' / 'CLI' ...
              (없으면 모두 'unknown' 으로 처리)
    """
    from collections import defaultdict

    src_map = src_map or {}
    grouped: dict[str, list[tuple[str, any]]] = defaultdict(list)
    for k, v in cfg.items():
        grouped[src_map.get(k, "others")].append((k, v))

    pref_order = ["base", "scenario", "method", "sweep", "CLI", "others"]
    for grp in sorted(grouped.keys(), key=lambda g: (pref_order.index(g) if g in pref_order else 999, g)):
        rows = sorted(grouped[grp], key=lambda kv: kv[0])
        _render_table(rows, title=f"[{grp}] Hyper‑parameters", ascii_only=ascii_only, log_fn=log_fn)

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
