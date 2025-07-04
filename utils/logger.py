# utils/logger.py

import os, sys, csv, json, time, logging
from datetime import datetime                 # ★ 추가
from typing import Optional

# ── 0) console logger 설정 (한 줄 timestamp + 색상* 지원)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)s │ %(message)s",
    datefmt="%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

def save_json(exp_dict, save_path):
    """
    Save exp_dict (configs + results) as a JSON file.
    """
    with open(save_path, 'w') as f:
        json.dump(exp_dict, f, indent=4)

def save_csv_row(exp_dict, csv_path, fieldnames, write_header_if_new=True):
    """
    Write a single row from `exp_dict` into a CSV file at `csv_path`.
    If `write_header_if_new` and the file doesn't exist, write the header first.
    """
    file_exists = os.path.exists(csv_path)

    with open(csv_path, 'a', newline='') as f:
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
        self.results_dir = self.config.get("results_dir", "results")

        self.start_time      = time.time()     # 러닝타임 측정용
        self.metric_history  = []              # step‑wise metric 로그용

    def _generate_exp_id(self, exp_name: str = "exp") -> str:
        """
        Creates an experiment ID like 'ibkd_noeval_20250704_123456'
        """
        eval_mode = self.config.get("eval_mode", "noeval")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{exp_name}_{eval_mode}_{ts}"

    def update_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """
        Save a metric and, if supplied, its step/epoch.

        Parameters
        ----------
        name: str
            Metric name (e.g. ``train_acc`` or ``loss``).
        value: float
            Metric value.
        step: Optional[int]
            Optional step/epoch associated with this metric.
        """
        self.config[name] = value
        self.metric_history.append({"name": name, "value": value, "step": step})

    # drop‑in logger
    def info(self, msg: str):
        logging.info(msg)

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


        # Ensure results directory exists
        os.makedirs(self.results_dir, exist_ok=True)

        # 2) JSON file path (per-run using exp_id)
        json_path = os.path.join(self.results_dir, f"{self.exp_id}.json")

        # metric 히스토리 포함
        self.config["metrics"] = self.metric_history

        # 3) CSV file path (fixed name)
        csv_filename = "summary.csv"
        self.config["csv_filename"] = csv_filename
        csv_path = os.path.join(self.results_dir, csv_filename)

        # Save the JSON (all info)
        save_json(self.config, json_path)
        self.info(f"JSON  ↗ {json_path}")

        # 4) Write CSV
        #   - 기본 열 + 모든 ep* 또는 teacher_ep* key 자동 포함
        base_cols = [
            "exp_id",
            "csv_filename",
            "eval_mode",
            "train_acc",
            "test_acc",
            "final_test_acc",
            "batch_size",
            "total_time_sec",
            "mbm_type",
            "mbm_r",
            "mbm_n_head",
            "mbm_learnable_q",
        ]

        # teacher / student epoch-wise metric columns also automatically included
        extra_cols = sorted(
            k for k in self.config.keys()
            if k.startswith(("teacher_ep", "student_ep"))
        )
        fieldnames = base_cols + extra_cols

        # CSV 저장
        save_csv_row(self.config, csv_path, fieldnames)
        self.info(f"CSV   ↗ {csv_path}")

        # ── 한 줄 summary ───────────────────────
        tr = self.config.get("train_acc", "-")
        te = self.config.get("final_test_acc", self.config.get("test_acc", "-"))
        self.info(
            f"SUMMARY │ {self.exp_id} │ train {tr} │ test {te} │ "
            f"time {total_time/60:.1f} min"
        )

