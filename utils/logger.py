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

def save_csv_row(exp_dict, csv_path, fieldnames):
    """
    Append one row (exp_dict) to a CSV file.
    If the file doesn't exist, write header first.
    """
    file_exists = os.path.exists(csv_path)

    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        # Gather row data from exp_dict
        row_data = {}
        for fn in fieldnames:
            # convert to string or fallback to ""
            val = exp_dict.get(fn, "")
            if isinstance(val, float):
                # format float => '%.2f'
                row_data[fn] = f"{val:.2f}"
            else:
                row_data[fn] = str(val)
        writer.writerow(row_data)

class ExperimentLogger:
    """
    Handles experiment configuration + result metrics in a single dict (self.config).
    - final JSON dump
    - summary.csv appending
    """
    def __init__(self, args, exp_name="exp"):
        """
        args can be an argparse.Namespace or a dict.
        We store it in self.config, plus generate an exp_id with timestamp.
        """
        if hasattr(args, "__dict__"):
            self.config = vars(args)
        elif isinstance(args, dict):
            self.config = args
        else:
            raise ValueError("args must be Namespace or dict")

        # set experiment name
        self.exp_name = exp_name

        # generate experiment id
        self.exp_id = self._generate_exp_id()
        self.config["exp_id"] = self.exp_id

        # results directory
        self.results_dir = self.config.get("results_dir", "results")
        os.makedirs(self.results_dir, exist_ok=True)

        self.start_time = time.time()

    def _generate_exp_id(self):
        """
        Creates an experiment ID, e.g. 'exp_eval_single_20240509_1300'
        using self.exp_name + self.config keys + timestamp
        """
        # you can pick relevant keys or keep it simpler
        eval_mode = self.config.get("eval_mode", "noeval")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.exp_name}_{eval_mode}_{ts}"

    def update_metric(self, key, value):
        """
        Save a metric (string, float, etc.) into self.config
        """
        self.config[key] = value

    def finalize(self):
        """
        At the end: saves JSON + appends summary.csv
        """
        total_time = time.time() - self.start_time
        self.config["total_time_sec"] = total_time

        # 1) Save JSON
        json_path = os.path.join(self.results_dir, f"{self.exp_id}.json")
        save_json(self.config, json_path)
        print(f"[ExperimentLogger] JSON saved => {json_path}")

        # 2) summary.csv append
        # define fieldnames as you see fit
        fieldnames = [
            "exp_id",
            "eval_mode",
            "train_acc",
            "test_acc",
            "batch_size",
            "config",
            "total_time_sec",
            # add more if you want: teacher1_ckpt, synergy, etc.
        ]
        summary_csv = os.path.join(self.results_dir, "summary.csv")
        save_csv_row(self.config, summary_csv, fieldnames)
        print(f"[ExperimentLogger] CSV appended => {summary_csv}")
