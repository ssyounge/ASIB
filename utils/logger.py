"""
utils/logger.py

Utility to log experiment configs & results in a structured way.
- Generate a unique exp_id
- Save in JSON (or CSV) for easy analysis
"""

import os
import csv
import json
import time
from datetime import datetime

def generate_exp_id(args_dict, prefix="exp", use_timestamp=True):
    """
    Generate a unique experiment ID string based on some key arguments + timestamp.
    Example:
     "exp_asmb_resnet_eff_alpha0.6_stage2_20240509_0412"
    """
    # Build partial ID from method, teacher1, teacher2, alpha, stage...
    method = args_dict.get("method", "unknown")
    teacher1 = args_dict.get("teacher1", "T1")
    teacher2 = args_dict.get("teacher2", "T2")
    alpha   = args_dict.get("alpha", "na")
    stage   = args_dict.get("stage", "na")

    exp_id = f"{prefix}_{method}_{teacher1}_{teacher2}_a{alpha}_s{stage}"
    
    if use_timestamp:
        # e.g. "20240509_0412"
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        exp_id += f"_{stamp}"

    return exp_id

def save_json(exp_dict, save_path):
    """
    Save exp_dict (configs + results) as a JSON file.
    """
    with open(save_path, 'w') as f:
        json.dump(exp_dict, f, indent=4)

def save_csv_row(exp_dict, csv_path, fieldnames):
    """
    Append one row (exp_dict) to a CSV file. 
    fieldnames should be a list of columns in consistent order.
    """
    # Convert all values to string if needed
    row_data = {}
    for fn in fieldnames:
        row_data[fn] = exp_dict.get(fn, "")

    # Append mode
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)

class ExperimentLogger:
    """
    Unified utility to:
      - store experiment config (argparse)
      - track results (top1, top5, etc)
      - generate exp_id
      - save JSON + CSV
    """
    def __init__(self, args):
        # Convert argparse.Namespace -> dict
        if hasattr(args, "__dict__"):
            self.config = vars(args)
        elif isinstance(args, dict):
            self.config = args
        else:
            raise ValueError("args must be Namespace or dict")

        # generate experiment id
        self.exp_id = generate_exp_id(self.config, prefix="asmb", use_timestamp=True)
        self.config["exp_id"] = self.exp_id

        # create default results folder
        self.results_dir = self.config.get("results_dir", "results")
        os.makedirs(self.results_dir, exist_ok=True)

        # optional: store start_time
        self.start_time = time.time()

    def update_metric(self, key, value):
        """
        Update a result metric in self.config
        E.g. top1, top5, final_loss
        """
        self.config[key] = value

    def finalize(self):
        """
        Called at the end of experiment to save logs.
        """
        # compute total_time (optionally)
        self.config["total_time_sec"] = time.time() - self.start_time

        # 1) Save JSON
        json_path = os.path.join(self.results_dir, f"{self.exp_id}.json")
        save_json(self.config, json_path)
        print(f"[Logger] JSON saved => {json_path}")

        # 2) Optionally, append to summary.csv
        summary_csv = os.path.join(self.results_dir, "summary.csv")
        fieldnames = [
            "exp_id",
            "method",
            "teacher1",
            "teacher2",
            "student",
            "alpha",
            "stage",
            "disagreement_rate",
            "disagreement_after_adapt",
            "reg_lambda",
            "mbm_reg_lambda",
            "synergy_ce_alpha",
            "teacher_adapt_alpha_kd",
            "top1",
            "top5",
            "final_loss",
            "lr",
            "batch_size",
            "seed",
            "total_time_sec",
        ]
        save_csv_row(self.config, summary_csv, fieldnames)
        print(f"[Logger] summary.csv appended => {summary_csv}")
