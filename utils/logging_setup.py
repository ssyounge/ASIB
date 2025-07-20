"""공통 로깅/모니터링 초기화."""
import logging, os, sys, json, pprint
from datetime import datetime

try:
    from rich.console import Console
    from rich.table import Table
    _RICH_OK = True
except ImportError:
    _RICH_OK = False

__all__ = ["setup_logging", "log_hparams"]


def _ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def setup_logging(cfg: dict):
    # ── 중복 로그 핸들러 방지 ─────────────────────────────
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    level = getattr(logging, cfg.get("log_level", "INFO").upper(), logging.INFO)
    log_file = os.path.join(cfg.get("results_dir", "."), cfg.get("log_filename", "train.log"))
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
