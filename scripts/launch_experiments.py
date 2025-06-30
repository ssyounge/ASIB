#!/usr/bin/env python3
"""Launch experiments based on a YAML setup file.

This script reads `configs/run_setup.yaml` to determine which
ASMB teacher pairs and baseline experiments to execute.  It wraps the
existing training scripts (`main.py` and `scripts/run_single_teacher.py`).
"""
import argparse
import os
import subprocess
import yaml


DEFAULT_SETUP_PATH = os.path.join("configs", "run_setup.yaml")


def load_yaml(path: str) -> dict:
    """Load YAML file and return an empty dict if the file doesn't exist."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Setup file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_command(cmd: str):
    """Helper to print and run a shell command."""
    print(f"[launch_experiments.py] Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def run_asmb_experiments(setup: dict):
    pairs = setup.get("asmb_teacher_pairs", [])
    student_list = setup.get("student_list", ["resnet_adapter"])
    for t1, t2 in pairs:
        for student in student_list:
            cmd = (
                f"python main.py --teacher1_type {t1} --teacher2_type {t2} "
                f"--student_type {student}"
            )
            run_command(cmd)


def run_baseline_experiments(setup: dict):
    baselines = setup.get("baseline_setups", [])
    methods = setup.get("baseline_methods_to_run", [])
    for cfg in baselines:
        teacher = cfg["teacher"]
        student = cfg["student"]
        for method in methods:
            cmd = (
                "python scripts/run_single_teacher.py "
                f"--method {method} --teacher_type {teacher} "
                f"--student_type {student}"
            )
            run_command(cmd)


def parse_args():
    p = argparse.ArgumentParser(description="Launch ASMB experiments")
    p.add_argument(
        "--setup",
        type=str,
        default=DEFAULT_SETUP_PATH,
        help="Path to run_setup.yaml",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup = load_yaml(args.setup)
    run_asmb_experiments(setup)
    run_baseline_experiments(setup)
