import pytest
import yaml
from pathlib import Path
from omegaconf import OmegaConf

class TestExperimentConfigs:
    """Test all experiment configurations"""

    @pytest.fixture(scope="class")
    def config_dir(self):
        return Path("configs/experiment")

    @pytest.fixture(scope="class")
    def experiment_configs(self, config_dir):
        """Load all experiment config files"""
        configs = {}
        for config_file in config_dir.glob("*.yaml"):
            if config_file.name != "_template.yaml":
                with open(config_file, 'r', encoding='utf-8') as f:
                    configs[config_file.stem] = yaml.safe_load(f)
        return configs

    def test_all_experiment_configs_exist(self, config_dir):
        """Test that all expected experiment configs exist"""
        expected_configs = [
            "L0_baseline",
            "L1_ib",
            "L2_cccp", 
            "L3_asib_cccp",
            "L4_full",
            "overlap_100",
            "sota_scenario_a"
        ]
        
        for config_name in expected_configs:
            config_file = config_dir / f"{config_name}.yaml"
            assert config_file.exists(), f"Config file {config_name}.yaml does not exist"

    def test_experiment_configs_structure(self, experiment_configs):
        """Test that all experiment configs have required structure"""
        required_sections = ["defaults"]

        for config_name, config in experiment_configs.items():
            for section in required_sections:
                assert section in config, f"Config {config_name} missing {section} section"

    def test_dataset_configs(self, experiment_configs):
        """Test dataset configurations"""
        for config_name, config in experiment_configs.items():
            # Check batch size exists
            batch_size = (
                config.get("experiment", {})
                .get("dataset", {})
                .get("batch_size")
            )
            assert isinstance(batch_size, int) and batch_size > 0, f"Invalid batch_size in {config_name}"

    def test_method_configs(self, experiment_configs):
        """Test method configurations"""
        for config_name, config in experiment_configs.items():
            # Check that config has required fields
            assert "defaults" in config, f"Config {config_name} missing defaults section"
            assert "experiment" in config, f"Config {config_name} missing experiment section"
            assert "dataset" in config["experiment"], f"Config {config_name} missing experiment.dataset"
            assert "batch_size" in config["experiment"]["dataset"], f"Config {config_name} missing batch_size under experiment.dataset"

    def test_model_configs(self, experiment_configs):
        """Test model configurations"""
        for config_name, config in experiment_configs.items():
            # Check that config has defaults section
            assert "defaults" in config, f"Config {config_name} missing defaults section"

    def test_hyperparameter_configs(self, experiment_configs):
        """Test hyperparameter configurations"""
        for config_name, config in experiment_configs.items():
            # Check learning rates
            a_step_lr = config.get("experiment", {}).get("a_step_lr")
            b_step_lr = config.get("experiment", {}).get("b_step_lr")
            assert isinstance(a_step_lr, (int, float)) and a_step_lr > 0, f"Invalid a_step_lr in {config_name}"
            assert isinstance(b_step_lr, (int, float)) and b_step_lr > 0, f"Invalid b_step_lr in {config_name}"

            # Check num_stages
            num_stages = config.get("experiment", {}).get("num_stages")
            assert isinstance(num_stages, int) and num_stages > 0, f"Invalid num_stages in {config_name}"

    def test_mbm_configs(self, experiment_configs):
        """Test IB_MBM-specific configurations (legacy keys removed)."""
        for config_name, config in experiment_configs.items():
            exp_cfg = config.get("experiment", {})
            qd = exp_cfg.get("ib_mbm_query_dim")
            od = exp_cfg.get("ib_mbm_out_dim")
            nh = exp_cfg.get("ib_mbm_n_head")
            assert isinstance(qd, int) and qd > 0, f"Invalid ib_mbm_query_dim in {config_name}"
            assert isinstance(od, int) and od > 0, f"Invalid ib_mbm_out_dim in {config_name}"
            assert isinstance(nh, int) and nh > 0, f"Invalid ib_mbm_n_head in {config_name}"

    def test_configs_can_be_loaded_with_hydra(self, experiment_configs):
        """Test that configs can be loaded with Hydra"""
        for config_name, config in experiment_configs.items():
            # Convert to OmegaConf
            omega_config = OmegaConf.create(config)

            # Check that we can access basic values
            batch_size = OmegaConf.select(omega_config, "experiment.dataset.batch_size")
            assert batch_size is not None and batch_size > 0

            num_stages = OmegaConf.select(omega_config, "experiment.num_stages")
            assert num_stages is not None and num_stages > 0

    def test_no_deprecated_suffixes(self, experiment_configs):
        """Test that no deprecated _student or _teacher suffixes are used"""
        for config_name, config in experiment_configs.items():
            config_str = str(config)
            
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
                if pattern in config_str:
                    pytest.fail(f"Config {config_name} contains deprecated pattern '{pattern}'") 