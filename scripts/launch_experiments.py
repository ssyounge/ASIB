#!/usr/bin/env python3
"""Launch experiments based on a YAML setup file.

This script reads `configs/run_setup.yaml` to determine which
ASMB teacher pairs and baseline experiments to execute.  It wraps the
existing training scripts (`main.py` and `scripts/run_single_teacher.py`).
"""
import argparse
import os
import subprocess
import tempfile
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


def generate_temp_config(method: str) -> str:
    """Create a temporary YAML config merging defaults with hparams."""
    base_cfg_path = os.environ.get("BASE_CONFIG", os.path.join("configs", "default.yaml"))
    hparams_path = os.path.join("configs", "hparams.yaml")

    cfg = {}
    if os.path.exists(base_cfg_path):
        with open(base_cfg_path, "r", encoding="utf-8") as f:
            cfg.update(yaml.safe_load(f) or {})
    if os.path.exists(hparams_path):
        with open(hparams_path, "r", encoding="utf-8") as f:
            cfg.update(yaml.safe_load(f) or {})

    cfg["method"] = method

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.safe_dump(cfg, tmp)
    tmp.close()
    return tmp.name


def collect_teachers(setup: dict) -> set:
    """Collect unique teacher types from the setup dictionary."""
    teachers = set()
    for t1, t2 in setup.get("asmb_teacher_pairs", []):
        teachers.add(t1)
        teachers.add(t2)
    for cfg in setup.get("baseline_setups", []):
        teachers.add(cfg.get("teacher"))
    return teachers


def finetune_teachers(setup: dict):
    """Fine-tune all required teachers before running distillation."""
    teachers = collect_teachers(setup)
    if not teachers:
        return
    print(">>> [Phase 1] Starting Teacher Fine-tuning...")
    os.makedirs("checkpoints", exist_ok=True)
    for teacher in teachers:
        ckpt_path = f"checkpoints/{teacher}_ft.pth"
        if os.path.exists(ckpt_path):
            print(
                f"[launch_experiments.py] Checkpoint for {teacher} already exists. Skipping fine-tuning."
            )
            continue
        cmd = (
            f"python scripts/fine_tuning.py --config configs/hparams.yaml "
            f"--teacher_type {teacher} --finetune_ckpt_path {ckpt_path}"
        )
        run_command(cmd)
    print(">>> [Phase 1] Teacher Fine-tuning complete.\n")

def run_asmb_experiments(setup: dict):
    pairs = setup.get("asmb_teacher_pairs", [])
    student_list = setup.get("student_list", ["resnet_adapter"])
    for t1, t2 in pairs:
        for student in student_list:
            cfg_path = generate_temp_config("asmb")
            cmd = (
                f"python main.py --config {cfg_path} "
                f"--teacher1_type {t1} --teacher2_type {t2} "
                f"--teacher1_ckpt checkpoints/{t1}_ft.pth "
                f"--teacher2_ckpt checkpoints/{t2}_ft.pth "
                f"--student_type {student}"
            )
            try:
                run_command(cmd)
            finally:
                if os.path.exists(cfg_path):
                    os.remove(cfg_path)


def run_baseline_experiments(setup: dict):
    baselines = setup.get("baseline_setups", [])
    methods = setup.get("baseline_methods_to_run", [])
    for cfg in baselines:
        teacher = cfg["teacher"]
        student = cfg["student"]
        for method in methods:
            cfg_path = generate_temp_config(method)
            cmd = (
                "python scripts/run_single_teacher.py "
                f"--config {cfg_path} --method {method} "
                f"--teacher_type {teacher} "
                f"--teacher_ckpt checkpoints/{teacher}_ft.pth "
                f"--student_type {student}"
            )
            try:
                run_command(cmd)
            finally:
                if os.path.exists(cfg_path):
                    os.remove(cfg_path)


def parse_args():
    p = argparse.ArgumentParser(description="Launch ASMB experiments")
    p.add_argument(
        "--setup",
        type=str,
        default=DEFAULT_SETUP_PATH,
        help="Path to run_setup.yaml",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup = load_yaml(args.setup)

    # Phase 1: teacher fine-tuning
    finetune_teachers(setup)

    # Phase 2: ASMB multi-teacher distillation
    print(">>> [Phase 2] Starting ASMB Distillation...")
    run_asmb_experiments(setup)

    # Baseline single-teacher experiments
    run_baseline_experiments(setup)


if __name__ == "__main__":
    main()
