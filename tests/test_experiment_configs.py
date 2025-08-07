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
                with open(config_file, 'r') as f:
                    configs[config_file.stem] = yaml.safe_load(f)
        return configs

    def test_all_experiment_configs_exist(self, config_dir):
        """Test that all expected experiment configs exist"""
        expected_configs = [
            "ablation_baseline",
            "ablation_cccp", 
            "ablation_full",
            "ablation_ib",
            "ablation_tadapt",
            "overlap_100",
            "sota_scenario_a"
        ]
        
        for config_name in expected_configs:
            config_file = config_dir / f"{config_name}.yaml"
            assert config_file.exists(), f"Config file {config_name}.yaml does not exist"

    def test_experiment_configs_structure(self, experiment_configs):
        """Test that all experiment configs have required structure"""
        required_sections = ["defaults"]  # Only check for defaults section

        for config_name, config in experiment_configs.items():
            for section in required_sections:
                assert section in config, f"Config {config_name} missing {section} section"

    def test_dataset_configs(self, experiment_configs):
        """Test dataset configurations"""
        for config_name, config in experiment_configs.items():
            # Check batch size exists
            batch_size = config.get("batch_size")
            assert isinstance(batch_size, int) and batch_size > 0, f"Invalid batch_size in {config_name}"

    def test_method_configs(self, experiment_configs):
        """Test method configurations"""
        for config_name, config in experiment_configs.items():
            # Check that config has required fields
            assert "defaults" in config, f"Config {config_name} missing defaults section"
            assert "batch_size" in config, f"Config {config_name} missing batch_size"

    def test_model_configs(self, experiment_configs):
        """Test model configurations"""
        for config_name, config in experiment_configs.items():
            # Check that config has defaults section
            assert "defaults" in config, f"Config {config_name} missing defaults section"

    def test_hyperparameter_configs(self, experiment_configs):
        """Test hyperparameter configurations"""
        for config_name, config in experiment_configs.items():
            # Check learning rates
            student_lr = config.get("student_lr")
            assert isinstance(student_lr, (int, float)) and student_lr > 0, f"Invalid student_lr in {config_name}"

            teacher_lr = config.get("teacher_lr")
            assert isinstance(teacher_lr, (int, float)) and teacher_lr >= 0, f"Invalid teacher_lr in {config_name}"

            # Check num_stages
            num_stages = config.get("num_stages")
            assert isinstance(num_stages, int) and num_stages > 0, f"Invalid num_stages in {config_name}"

    def test_mbm_configs(self, experiment_configs):
        """Test MBM-specific configurations"""
        for config_name, config in experiment_configs.items():
            # Check mbm_query_dim
            mbm_query_dim = config.get("mbm_query_dim")
            assert isinstance(mbm_query_dim, int) and mbm_query_dim > 0, f"Invalid mbm_query_dim in {config_name}"
            
            # Check mbm_out_dim
            mbm_out_dim = config.get("mbm_out_dim")
            assert isinstance(mbm_out_dim, int) and mbm_out_dim > 0, f"Invalid mbm_out_dim in {config_name}"
            
            # Check mbm_n_head
            mbm_n_head = config.get("mbm_n_head")
            assert isinstance(mbm_n_head, int) and mbm_n_head > 0, f"Invalid mbm_n_head in {config_name}"

    def test_configs_can_be_loaded_with_hydra(self, experiment_configs):
        """Test that configs can be loaded with Hydra"""
        for config_name, config in experiment_configs.items():
            # Convert to OmegaConf
            omega_config = OmegaConf.create(config)

            # Check that we can access basic values
            batch_size = OmegaConf.select(omega_config, "batch_size")
            assert batch_size is not None and batch_size > 0

            num_stages = OmegaConf.select(omega_config, "num_stages")
            assert num_stages is not None and num_stages > 0

    def test_no_deprecated_suffixes(self, experiment_configs):
        """Test that no deprecated _student or _teacher suffixes are used"""
        for config_name, config in experiment_configs.items():
            config_str = str(config)
            
            # Check for deprecated suffixes
            assert "_student" not in config_str, f"Config {config_name} contains deprecated '_student' suffix"
            assert "_teacher" not in config_str, f"Config {config_name} contains deprecated '_teacher' suffix" 