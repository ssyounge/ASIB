# utils/logger.py

import os
import csv
import json
import time
from datetime import datetime

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

        # For timing
        self.start_time = time.time()

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

    def info(self, msg: str):
        """
        drop-in replacement for logging.Logger.info.
        현재는 stdout으로만 출력하지만, 필요하면
        파일에 따로 쓰거나 time-stamp를 붙이는 등 확장 가능.
        """
        print(msg)

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

        # 3) CSV file path (fixed name)
        csv_filename = "summary.csv"
        self.config["csv_filename"] = csv_filename
        csv_path = os.path.join(self.results_dir, csv_filename)

        # Save the JSON (all info)
        save_json(self.config, json_path)
        print(f"[ExperimentLogger] JSON saved => {json_path}")

        # 4) Write CSV
        #   - 기본 열 + 모든 ep* 또는 teacher_ep* key 자동 포함
        base_cols = [
            "exp_id",
            "csv_filename",
            "eval_mode",
            "train_acc",
            "test_acc",
            "batch_size",
            "total_time_sec",
            "mbm_type",
            "mbm_r",
            "mbm_n_head",
            "mbm_learnable_q",
        ]

        epoch_cols = [
            k for k in self.config.keys()
            if k.startswith(("student_ep", "teacher_ep"))
        ]

        fieldnames = base_cols + sorted(epoch_cols)

        save_csv_row(self.config, csv_path, fieldnames, write_header_if_new=True)
        print(f"[ExperimentLogger] CSV saved => {csv_path}")
