# utils/logging_setup.py
"""공통 로깅/모니터링 초기화."""

import logging
import os
import sys
from typing import Union, Optional, Dict, Any

try:
    import wandb
    _WANDB_OK = True
except ImportError:
    _WANDB_OK = False

try:
    from rich.console import Console
    from rich.table import Table
    _RICH_OK = True
except ImportError:
    _RICH_OK = False

__all__ = ["setup_logging", "log_hparams", "get_logger", "setup_logger", "init_logger"]

_LOGGERS = {}        # cache (exp-id → logger)


def init_logger(level: Union[str, int] = "INFO"):
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


def setup_logging(cfg: Dict[str, Any]) -> logging.Logger:
    """Setup logging for the experiment."""
    results_dir = cfg.get("results_dir", ".")
    
    # results_dir이 실제로 "." 또는 ""인 경우에만 경고 출력
    if results_dir == "." or results_dir == "" or results_dir is None:
        results_dir = "logs"
        print(f"[Warning] results_dir이 '{cfg.get('results_dir', '.')}'으로 설정되어 있어서 'logs' 디렉토리로 변경합니다.")
    
    log_file = os.path.join(results_dir, cfg.get("log_filename", "train.log"))
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Setup logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger


def log_hparams(logger: logging.Logger, cfg: Dict[str, Any]) -> None:
    """Log hyperparameters."""
    logger.info("HParams:\n%s", cfg)

    # 2) 별도 JSON 사본
    dst = os.path.join(cfg.get("results_dir", "."), "hparams_full.json")
    _ensure_dir(dst)
    with open(dst, "w") as f:
        json.dump(cfg, f, indent=2, default=str)
    logging.info("[logging_setup] hparams saved => %s", dst)


def _wandb_log(text: str):
    """stream train.log line ↔️ W&B console tab"""
    if _WANDB_OK and wandb.run is not None:
        wandb.log({"logs": text}, commit=False)


def get_logger(
    exp_dir: str,
    level: Union[str, int] = "INFO",
    stream_level: Union[str, int] = "INFO"
) -> logging.Logger:
    """Get or create a logger for the experiment."""
    results_dir = exp_dir
    
    # results_dir이 실제로 "." 또는 ""인 경우에만 경고 출력
    if results_dir == "." or results_dir == "" or results_dir is None:
        results_dir = "logs"
        print(f"[Warning] results_dir이 '{exp_dir}'으로 설정되어 있어서 'logs' 디렉토리로 변경합니다.")
    
    log_file = os.path.join(results_dir, "train.log")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Setup logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, level.upper()))
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, stream_level.upper()))
    console_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger


def setup_logger(
    exp_dir: str,
    level: Union[str, int] = "INFO",
    stream_level: Union[str, int] = "INFO"
) -> logging.Logger:
    """Setup logger for the experiment."""
    return get_logger(exp_dir, level, stream_level)
