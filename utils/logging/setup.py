# utils/logging_setup.py
"""공통 로깅/모니터링 초기화."""

import json
import logging
import os
import pprint
import sys
from pathlib import Path

import wandb

try:
    from rich.console import Console
    from rich.table import Table
    _RICH_OK = True
except ImportError:
    _RICH_OK = False

__all__ = ["setup_logging", "log_hparams", "get_logger", "setup_logger", "init_logger"]

_LOGGERS = {}        # cache (exp-id → logger)


def init_logger(level: str | int = "INFO"):
    """Initialize basic logger with console output only.
    
    This is a safer alternative to logging.basicConfig that doesn't interfere
    with file handlers that are added later.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # 기존 핸들러 제거 (중복 방지)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 콘솔 핸들러만 추가
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)


def _ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def setup_logging(cfg: dict):
    # ── 중복 로그 핸들러 방지 ─────────────────────────────
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    level_str = (cfg.get("log_level") or "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    
    # 안전장치: results_dir이 "."이면 기본값으로 "logs" 사용
    results_dir = cfg.get("results_dir", ".")
    if results_dir == "." or results_dir == "":
        results_dir = "logs"
        print(f"[Warning] results_dir이 '.'으로 설정되어 있어서 'logs' 디렉토리로 변경합니다.")
    
    log_file = os.path.join(results_dir, cfg.get("log_filename", "train.log"))
    _ensure_dir(log_file)

    fh = logging.FileHandler(log_file, mode="a")
    ch = logging.StreamHandler(sys.stdout)

    logger = logging.getLogger()
    logger.setLevel(level)

    # ── 중복 추가 방지 ────────────────────────────────
    if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == log_file for h in logger.handlers):
        logger.addHandler(fh)

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        logger.addHandler(ch)

    # 다른 서브‑logger 로 메시지가 두 번 올라오지 않도록
    logger.propagate = False

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.info("[logging_setup] log_file => %s   (level=%s)", log_file, logging.getLevelName(level))
    return log_file


def _to_plain_dict(cfg):
    if hasattr(cfg, "__dict__"):
        return vars(cfg)
    return dict(cfg)


def log_hparams(cfg):
    cfg = _to_plain_dict(cfg)
    if not cfg.get("log_all_hparams", True):
        return
    # 1) pretty-print to log
    if _RICH_OK and sys.stdout.isatty():
        table = Table(title="All Hyper-parameters", show_lines=False)
        table.add_column("Key", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        for k in sorted(cfg.keys()):
            table.add_row(k, pprint.pformat(cfg[k], compact=True))
        Console().print(table)
    else:
        logging.info("HParams:\n%s", json.dumps(cfg, indent=2, default=str))

    # 2) 별도 JSON 사본
    dst = os.path.join(cfg.get("results_dir", "."), "hparams_full.json")
    _ensure_dir(dst)
    with open(dst, "w") as f:
        json.dump(cfg, f, indent=2, default=str)
    logging.info("[logging_setup] hparams saved => %s", dst)


def _wandb_log(text: str):
    """stream train.log line ↔️ W&B console tab"""
    if wandb.run is not None:
        wandb.log({"logs": text}, commit=False)


def get_logger(
    exp_dir: str,
    log_file: str = "train.log",
    level: str = "INFO",
    stream_level: str = "WARNING",
):
    """Return a logger that writes to file, console, and optionally W&B."""
    # 안전장치: exp_dir이 "."이면 기본값으로 "logs" 사용
    if exp_dir == "." or exp_dir == "":
        exp_dir = "logs"
        print(f"[Warning] exp_dir이 '.'으로 설정되어 있어서 'logs' 디렉토리로 변경합니다.")
    
    gkey = os.path.abspath(os.path.join(exp_dir, log_file))
    if gkey in _LOGGERS:
        return _LOGGERS[gkey]

    os.makedirs(exp_dir, exist_ok=True)
    logger = logging.getLogger(gkey)
    logger.setLevel(logging.DEBUG)

    f_hdl = logging.FileHandler(gkey, mode="a", encoding="utf-8")
    f_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    f_hdl.setFormatter(f_fmt)
    f_level = getattr(logging, (level or "INFO").upper())
    f_hdl.setLevel(f_level)
    logger.addHandler(f_hdl)

    s_hdl = logging.StreamHandler(sys.stdout)
    s_fmt = logging.Formatter("%(levelname)s | %(message)s")
    s_hdl.setFormatter(s_fmt)
    s_level = getattr(logging, (stream_level or "WARNING").upper())
    s_hdl.setLevel(s_level)
    logger.addHandler(s_hdl)

    class _WBHandler(logging.Handler):
        def emit(self, record):
            _wandb_log(self.format(record))

    if wandb.run is not None:
        wb_hdl = _WBHandler()
        wb_hdl.setFormatter(f_fmt)
        wb_hdl.setLevel(f_level)
        logger.addHandler(wb_hdl)

    _LOGGERS[gkey] = logger
    logger.propagate = False
    return logger


def setup_logger(cfg: dict):
    """Return a basic three-channel logger (train.log, run.log, console)."""
    # 안전장치: results_dir이 "."이면 기본값으로 "logs" 사용
    results_dir = cfg.get("results_dir", ".")
    if results_dir == "." or results_dir == "":
        results_dir = "logs"
        print(f"[Warning] results_dir이 '.'으로 설정되어 있어서 'logs' 디렉토리로 변경합니다.")
    
    log_dir = Path(results_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    fh_train = logging.FileHandler(log_dir / "train.log", mode="w")
    fh_train.setLevel(logging.DEBUG)

    fh_run = logging.FileHandler(log_dir / "run.log", mode="w")
    fh_run.setLevel(logging.INFO)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO if cfg.get("disable_tqdm", False) else logging.WARNING)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s │ %(levelname)-5s │ %(message)s",
        handlers=[fh_train, fh_run, ch],
        force=True,        # 중복 logger 초기화 방지
    )
    logger = logging.getLogger("KD")

    if cfg.get("log_all_hparams", False):
        logger.info("HParams:")
        logger.info(json.dumps(cfg, indent=2, default=str))
    return logger
