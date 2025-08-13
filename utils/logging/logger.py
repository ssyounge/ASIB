# utils/logger.py

import os
import csv
import json
import time
import logging
from datetime import datetime


def _json_default(obj):
    """Fallback serialization for unsupported objects."""
    return str(obj)

def save_json(exp_dict, save_path):
    """Save ``exp_dict`` (configs + results) as a JSON file."""
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(exp_dict, f, indent=4, default=_json_default)

def save_csv_row(exp_dict, csv_path, fieldnames, write_header_if_new=True):
    """
    Write a single row from `exp_dict` into a CSV file at `csv_path`.
    If `write_header_if_new` and the file doesn't exist, write the header first.
    """
    file_exists = os.path.exists(csv_path)

    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header_if_new and not file_exists:
            writer.writeheader()

        row_data = {}
        for fn in fieldnames:
            val = exp_dict.get(fn, "")
            if isinstance(val, float):
                row_data[fn] = f"{val:.2f}"
            else:
                row_data[fn] = str(val)
        writer.writerow(row_data)

class ExperimentLogger:
    """
    Handles experiment configuration + result metrics in a single dict (self.config).
    1) Create a unique exp_id for each run
    2) Save to JSON (entire config) 
    3) Save to a unique CSV file (subset of fields), 
       and also store that csv_filename in the JSON.
    """

    def __init__(self, args, exp_name="exp"):
        """
        args can be an argparse.Namespace or a dict.
        We store it in self.config, plus generate an exp_id with a timestamp.

        :param args: namespace or dict of config
        :param exp_name: prefix for experiment id
        """
        # Convert to dict if argparse.Namespace
        if hasattr(args, "__dict__"):
            self.config = vars(args)
        elif isinstance(args, dict):
            self.config = args
        else:
            raise ValueError("args must be Namespace or dict")

        self.exp_name = exp_name

        # Generate exp_id using provided value or timestamp
        self.exp_id = self.config.get("exp_id") or self._generate_exp_id(exp_name)
        self.config["exp_id"] = self.exp_id

        # Where to save results
        self.results_dir = self.config.get("results_dir", "experiments/test/results")
        # ensure results_dir is recorded in config for downstream users
        self.config.setdefault("results_dir", self.results_dir)

        # For timing
        self.start_time = time.time()

        # ── (옵션) 모든 하이퍼파라미터 즉시 출력 ───────────────
        if self.config.get("log_all_hparams", False):
            import logging
            import pprint

            logging.getLogger().setLevel(self.config.get("log_level", "DEBUG"))
            pprint.pprint(self.config, width=120)

    def _generate_exp_id(self, exp_name="exp"):
        """
        Creates an experiment ID like 'eval_experiment_synergy_20240805_153210'
        using exp_name, eval_mode, and timestamp
        """
        eval_mode = self.config.get("eval_mode", "noeval")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{exp_name}_{eval_mode}_{ts}"

    def update_metric(self, key, value):
        """
        Save any metric (accuracy, loss, hyperparams, etc.) into self.config
        so it can be written out on finalize().
        """
        self.config[key] = value

    def get_metric(self, key, default=None):
        """
        Get a metric value from self.config.
        
        :param key: metric key
        :param default: default value if key doesn't exist
        :return: metric value or default
        """
        return self.config.get(key, default)

    def save_results(self):
        """
        Alias for finalize() method for backward compatibility.
        """
        self.finalize()

    # ------------------------------------------------------------------
    # 표준 logging.Logger 인터페이스와 동일한 시그니처를 제공하여
    # logger.info("… %d", val) 형태의 가변 인자 호출을 허용한다.
    # 필요 시 debug / warning 등도 동일 패턴으로 확장해 둔다.
    # ------------------------------------------------------------------
    def info(self, msg: str, *args, **kwargs):
        """Drop‑in for ``logging.Logger.info`` (가변 인자 지원)."""
        logging.info(msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs):
        """Drop‑in for ``logging.Logger.debug``."""
        logging.debug(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        """Drop‑in for ``logging.Logger.warning``."""
        logging.warning(msg, *args, **kwargs)

    def finalize(self):
        """
        1) Calculates total_time_sec
        2) Saves a full JSON file with self.config
        3) Saves a unique CSV for this experiment, with essential columns only
        4) Also store 'csv_filename' in self.config for cross-reference
        """
        # 1) total time
        total_time = time.time() - self.start_time
        self.config["total_time_sec"] = total_time
        # drop unserializable objects before saving
        self.config.pop("logger", None)


        # Ensure results directory exists
        os.makedirs(self.results_dir, exist_ok=True)

        # 2) JSON  ▶  exp_id.json  (+ latest.json link)
        json_path = os.path.join(self.results_dir, f"{self.exp_id}.json")

        # 3) CSV file path (fixed name)
        csv_filename = "summary.csv"
        self.config["csv_filename"] = csv_filename
        csv_path = os.path.join(self.results_dir, csv_filename)

        # Save the JSON (all info)
        save_json(self.config, json_path)
        logging.info("[ExperimentLogger] JSON saved ⇒ %s", json_path)

        # 최신 결과 가리키는 심링크/복사본
        latest_path = os.path.join(self.results_dir, "latest.json")
        try:
            if os.path.islink(latest_path) or os.path.exists(latest_path):
                os.remove(latest_path)
            if os.name != "nt":
                os.symlink(os.path.basename(json_path), latest_path)
            else:
                import shutil
                shutil.copy2(json_path, latest_path)
        except OSError:
            # 심링크가 안 되는 파일시스템이면 그만 복사
            import shutil
            shutil.copy2(json_path, latest_path)

        # 4) Write CSV
        #   - 기본 열 + 모든 ep* 또는 teacher_ep* key 자동 포함
        #   - 실제 사용하는 핵심 메타 위주로 필드 구성 (가독성↑)
        base_cols = [
            "exp_id",
            "csv_filename",
            "total_time_sec",
            "final_student_acc",
            "num_classes",
            "batch_size",
            # KD / IB / CCCP
            "ce_alpha",
            "kd_alpha",
            "use_ib",
            "ib_beta",
            "ib_epochs_per_stage",
            "use_cccp",
            # Optim
            "optimizer",
            "student_lr",
            "student_weight_decay",
        ]

        epoch_cols = [
            k for k in self.config.keys()
            if k.startswith(("student_ep", "teacher_ep"))
        ]

        fieldnames = base_cols + sorted(epoch_cols)

        save_csv_row(self.config, csv_path, fieldnames, write_header_if_new=True)
        logging.info("[ExperimentLogger] CSV saved => %s", csv_path)

    # --------------------------------------------------------------
    # Meta writer: save high-level experiment metadata to meta.json
    # --------------------------------------------------------------
    def save_meta(self, meta: dict):
        try:
            os.makedirs(self.results_dir, exist_ok=True)
            path = os.path.join(self.results_dir, "meta.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            logging.info("[ExperimentLogger] META saved ⇒ %s", path)
        except Exception as e:
            logging.warning("[ExperimentLogger] meta.json save failed: %s", e)
