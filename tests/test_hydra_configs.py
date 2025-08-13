#!/usr/bin/env python3
"""Hydra composition and schema validation tests for configs/*

These tests validate that:
- Experiment configs under configs/experiment can be composed by Hydra
- Overrides used by run scripts apply correctly (experiment.* keys)
- Model group files (teacher/student) use bare mappings (no wrappers)
- No deprecated suffixes in config files
- Hydra run.dir policy references experiment.results_dir
"""

from pathlib import Path
import os
import io
import yaml
import pytest
from omegaconf import OmegaConf


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def configs_dir() -> Path:
    return _project_root() / "configs"


@pytest.fixture(scope="module")
def experiment_dir(configs_dir: Path) -> Path:
    return configs_dir / "experiment"


def _compose(config_name: str, overrides: list[str] | None = None):
    # Use Hydra's programmatic API with isolated global state
    from hydra import initialize, compose
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    # Hydra requires a relative config_path (relative to this test file's dir)
    rel_config_path = os.path.relpath(str(_project_root() / "configs"), start=str(Path(__file__).parent))
    with initialize(version_base=None, config_path=rel_config_path, job_name="test_hydra_compose"):
        cfg = compose(config_name=config_name, overrides=overrides or [])
    return cfg


class TestHydraExperimentCompose:
    def test_compose_all_experiment_configs(self, experiment_dir: Path):
        for cfg_file in sorted(experiment_dir.glob("*.yaml")):
            if cfg_file.name == "_template.yaml":
                continue
            cfg = _compose(config_name=f"experiment/{cfg_file.stem}")

            # Drill down to the effective experiment node (handles nested 'experiment' keys)
            def get_exp_node(c):
                node = c
                for _ in range(3):
                    if "experiment" in node:
                        node = node["experiment"]
                    else:
                        break
                return node

            assert "experiment" in cfg
            exp = get_exp_node(cfg)
            assert exp is not None
            assert "dataset" in exp
            assert "batch_size" in exp["dataset"]
            assert int(exp["dataset"]["batch_size"]) > 0
            assert "num_stages" in exp
            assert int(exp["num_stages"]) > 0

            # IB_MBM keys present when use_ib
            if cfg.experiment.get("use_ib", False):
                for k in ("ib_mbm_query_dim", "ib_mbm_out_dim", "ib_mbm_n_head"):
                    assert k in cfg.experiment, f"missing {k} in {cfg_file.name}"

    def test_overrides_apply_correctly(self):
        # Mirror run script overrides
        # Use nested path (experiment.experiment.*) to match composed structure
        overrides = [
            "experiment.experiment.dataset.batch_size=16",
            "experiment.experiment.student_epochs_per_stage=[5]",
            "experiment.experiment.num_stages=1",
        ]
        cfg = _compose(config_name="experiment/ablation_baseline", overrides=overrides)
        # Resolve effective experiment node
        def get_exp_node(c):
            node = c
            for _ in range(3):
                if "experiment" in node:
                    node = node["experiment"]
                else:
                    break
            return node

        exp = get_exp_node(cfg)
        assert int(exp["dataset"]["batch_size"]) == 16
        assert list(exp["student_epochs_per_stage"]) == [5]
        assert int(exp["num_stages"]) == 1

    def test_hydra_run_dir_policy_references_results_dir(self):
        cfg = _compose(config_name="experiment/ablation_baseline")
        if "hydra" in cfg and "run" in cfg.hydra:
            run_dir = str(cfg.hydra.run.dir)
            # We don't force resolver evaluation; just ensure reference to experiment.results_dir
            assert "experiment.results_dir" in run_dir or run_dir.startswith("${experiment.results_dir}")


class TestModelGroupFiles:
    def test_teacher_group_files_are_bare_mappings(self, configs_dir: Path):
        teacher_dir = configs_dir / "model" / "teacher"
        for f in sorted(teacher_dir.glob("*.yaml")):
            data = yaml.safe_load(f.read_text(encoding="utf-8"))
            assert isinstance(data, dict), f"{f} must be a mapping"
            # No wrapper keys like model:/teacher:
            assert "model" not in data and "teacher" not in data, f"{f} should be bare mapping"
            assert "name" in data and "pretrained" in data, f"{f} must contain name and pretrained"

    def test_student_group_files_are_bare_mappings(self, configs_dir: Path):
        student_dir = configs_dir / "model" / "student"
        for f in sorted(student_dir.glob("*.yaml")):
            data = yaml.safe_load(f.read_text(encoding="utf-8"))
            assert isinstance(data, dict), f"{f} must be a mapping"
            # No wrapper keys like model:/student:
            assert "model" not in data and "student" not in data, f"{f} should be bare mapping"
            assert "name" in data and "pretrained" in data, f"{f} must contain name and pretrained"


class TestNoDeprecatedSuffixes:
    def test_no_deprecated_suffixes_in_config_files(self, configs_dir: Path):
        # Scan experiment and finetune configs
        scan_dirs = [configs_dir / "experiment", configs_dir / "finetune"]
        for d in scan_dirs:
            if not d.exists():
                continue
            for f in sorted(d.glob("*.yaml")):
                content = f.read_text(encoding="utf-8")
                
                # Skip valid keys that contain _student or _teacher
                valid_keys = [
                    'compute_teacher_eval',  # Valid configuration key
                    'teacher_adapt_epochs',  # Valid configuration key
                    'teacher_lr',           # Valid configuration key
                    'teacher_weight_decay', # Valid configuration key
                    'teacher1_ckpt',        # Valid configuration key
                    'teacher2_ckpt',        # Valid configuration key
                    'teacher1_freeze_level', # Valid configuration key
                    'teacher2_freeze_level', # Valid configuration key
                    'teacher1_freeze_bn',   # Valid configuration key
                    'teacher2_freeze_bn',   # Valid configuration key
                ]
                
                # Check for actual deprecated suffixes, not valid keys
                # Only fail if we find actual deprecated patterns, not valid configuration keys
                deprecated_patterns = [
                    'resnet50_student',     # Deprecated: should be resnet50_scratch
                    'resnet101_student',    # Deprecated: should be resnet101_scratch
                    'convnext_s_teacher',   # Deprecated: should be convnext_s
                    'resnet152_teacher',    # Deprecated: should be resnet152
                ]
                
                for pattern in deprecated_patterns:
                    if pattern in content:
                        pytest.fail(f"{f} contains deprecated pattern '{pattern}'")


