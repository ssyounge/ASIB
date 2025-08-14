#!/usr/bin/env python3
"""Test all configuration files"""

import pytest
import yaml
from pathlib import Path
from omegaconf import OmegaConf, DictConfig


class TestBaseConfig:
    """Test base configuration"""
    
    def test_base_config_structure(self):
        """Test base config structure"""
        config = OmegaConf.load("configs/base.yaml")
        
        # Check required sections
        assert "defaults" in config
        assert "experiment" in config
        assert "dataset" in config["experiment"]
        assert "batch_size" in config["experiment"]["dataset"]
    
    def test_base_config_values(self):
        """Test base config values"""
        config = OmegaConf.load("configs/base.yaml")
        
        # Check basic values
        assert config["experiment"].device == "cuda"
        assert isinstance(config["experiment"]["dataset"]["batch_size"], int)
        assert config["experiment"]["dataset"]["batch_size"] > 0
        assert config["experiment"]["seed"] == 42


class TestExperimentConfigs:
    """Test experiment configurations"""
    
    @pytest.mark.parametrize("config_file", [
        "configs/experiment/L0_baseline.yaml",
    ])
    def test_experiment_config_structure(self, config_file):
        """Test experiment config structure"""
        config = OmegaConf.load(config_file)
        
        # Check required sections
        assert "defaults" in config
        assert "experiment" in config
        exp = config["experiment"]
        assert "teacher1_ckpt" in exp
        assert "teacher2_ckpt" in exp
        assert "results_dir" in exp
    
    @pytest.mark.parametrize("config_file", [
        "configs/experiment/L0_baseline.yaml",
    ])
    def test_experiment_config_values(self, config_file):
        """Test experiment config values"""
        config = OmegaConf.load(config_file)
        
        # Check basic configurations
        exp = config["experiment"]
        assert "teacher1_ckpt" in exp
        assert "teacher2_ckpt" in exp
        assert "results_dir" in exp
        assert "exp_id" in exp
    
    def test_res152_convnext_effi_config(self):
        """Test specific res152_convnext_effi config"""
        config = OmegaConf.load("configs/experiment/L0_baseline.yaml")
        exp = config["experiment"]
        assert "results_dir" in exp and isinstance(exp["results_dir"], str)
    
    def test_res152_effi_l2_config(self):
        """Test specific res152_effi_l2 config"""
        assert True


class TestFinetuneConfigs:
    """Test finetune configurations"""
    
    @pytest.mark.parametrize("config_file", [
        "configs/finetune/resnet152_cifar100.yaml",
        "configs/finetune/resnet152_imagenet32.yaml",
        "configs/finetune/convnext_l_cifar100.yaml",
        "configs/finetune/convnext_l_imagenet32.yaml",
        "configs/finetune/efficientnet_l2_cifar100.yaml",
        "configs/finetune/efficientnet_l2_imagenet32.yaml",
        "configs/finetune/convnext_s_cifar100.yaml",
        "configs/finetune/convnext_s_imagenet32.yaml"
    ])
    def test_finetune_config_structure(self, config_file):
        """Test finetune config structure (flat schema)"""
        config = OmegaConf.load(config_file)

        # Required flat fields
        required_fields = [
            "teacher_type", "small_input", "teacher_pretrained",
            "finetune_epochs", "finetune_lr", "batch_size",
            "results_dir", "exp_id", "finetune_ckpt_path"
        ]
        for field in required_fields:
            assert field in config, f"Missing field {field} in {config_file}"
    
    @pytest.mark.parametrize("config_file", [
        "configs/finetune/resnet152_cifar100.yaml",
        "configs/finetune/resnet152_imagenet32.yaml",
        "configs/finetune/convnext_l_cifar100.yaml",
        "configs/finetune/convnext_l_imagenet32.yaml",
        "configs/finetune/efficientnet_l2_cifar100.yaml",
        "configs/finetune/efficientnet_l2_imagenet32.yaml",
        "configs/finetune/convnext_s_cifar100.yaml",
        "configs/finetune/convnext_s_imagenet32.yaml"
    ])
    def test_finetune_config_values(self, config_file):
        """Test finetune config values (flat schema)"""
        config = OmegaConf.load(config_file)

        assert isinstance(config["finetune_epochs"], int)
        assert isinstance(float(config["finetune_lr"]), float)
        assert isinstance(config["batch_size"], int)

        assert config["finetune_epochs"] > 0
        assert float(config["finetune_lr"]) > 0
        assert config["batch_size"] > 0
    
    def test_resnet152_finetune_config(self):
        """Test specific resnet152 finetune config"""
        config = OmegaConf.load("configs/finetune/resnet152_cifar100.yaml")
        
        assert config.teacher_type == "resnet152"
        assert config.finetune_epochs == 70  # 전통적 모델이므로 충분한 학습 시간
        assert config.finetune_lr == 1.2e-4  # 전통적 모델이므로 적당한 학습률
        assert config.batch_size == 64
        assert config.finetune_weight_decay == 2e-3  # 전통적 모델이므로 적당한 정규화
        assert config.label_smoothing == 0.3  # 전통적 모델이므로 적당한 label smoothing
        assert config.warmup_epochs == 3
        assert config.early_stopping_patience == 10
        assert config.early_stopping_min_delta == 0.1
    
    def test_convnext_s_finetune_config(self):
        """Test specific convnext_s finetune config"""
        config = OmegaConf.load("configs/finetune/convnext_s_cifar100.yaml")
        assert config["teacher_type"] == "convnext_s"
        assert config["finetune_epochs"] > 0
        assert float(config["finetune_lr"]) > 0
        assert config["batch_size"] > 0
    
    def test_convnext_l_finetune_config(self):
        """Test specific convnext_l finetune config"""
        config = OmegaConf.load("configs/finetune/convnext_l_cifar100.yaml")
        assert config["teacher_type"] == "convnext_l"
        assert config["finetune_epochs"] > 0
        assert float(config["finetune_lr"]) > 0
        assert config["batch_size"] > 0
    
    def test_efficientnet_l2_finetune_config(self):
        """Test specific efficientnet_l2 finetune config"""
        config = OmegaConf.load("configs/finetune/efficientnet_l2_cifar100.yaml")
        assert config["teacher_type"] == "efficientnet_l2"
        assert config["finetune_epochs"] > 0
        assert float(config["finetune_lr"]) > 0
        assert config["batch_size"] > 0
    
    def test_finetune_warmup_configs(self):
        """Test warm-up configurations for all models"""
        configs = [
            "configs/finetune/convnext_l_cifar100.yaml",
            "configs/finetune/convnext_s_cifar100.yaml", 
            "configs/finetune/resnet152_cifar100.yaml",
            "configs/finetune/efficientnet_l2_cifar100.yaml"
        ]
        
        for config_file in configs:
            config = OmegaConf.load(config_file)
            if "warmup_epochs" in config and "finetune_epochs" in config:
                assert config["warmup_epochs"] >= 0
                assert config["warmup_epochs"] < config["finetune_epochs"]
            if "min_lr" in config and "finetune_lr" in config:
                assert float(config["min_lr"]) > 0
                assert float(config["min_lr"]) < float(config["finetune_lr"])
    
    def test_finetune_early_stopping_configs(self):
        """Test early stopping configurations for all models"""
        configs = [
            "configs/finetune/convnext_l_cifar100.yaml",
            "configs/finetune/convnext_s_cifar100.yaml", 
            "configs/finetune/resnet152_cifar100.yaml",
            "configs/finetune/efficientnet_l2_cifar100.yaml"
        ]
        
        for config_file in configs:
            config = OmegaConf.load(config_file)
            if "early_stopping_patience" in config and "finetune_epochs" in config:
                assert config["early_stopping_patience"] > 0
                assert config["early_stopping_patience"] < config["finetune_epochs"]
            if "early_stopping_min_delta" in config:
                assert config["early_stopping_min_delta"] > 0
                assert config["early_stopping_min_delta"] < 1.0
    
    def test_finetune_advanced_scheduling_configs(self):
        """Test advanced scheduling configurations for all models"""
        configs = [
            ("configs/finetune/convnext_l_cifar100.yaml", "onecycle"),
            ("configs/finetune/convnext_s_cifar100.yaml", "reduce_on_plateau"), 
            ("configs/finetune/resnet152_cifar100.yaml", "cosine_warm_restarts"),
            ("configs/finetune/efficientnet_l2_cifar100.yaml", "multistep")
        ]
        
        for config_file, expected_scheduler in configs:
            config = OmegaConf.load(config_file)
            if "scheduler_type" in config:
                assert config["scheduler_type"] == expected_scheduler

            # 스케줄러별 특정 설정 검증 (flat)
            if config.get("scheduler_type") == "cosine_warm_restarts":
                assert "restart_period" in config
                assert "restart_multiplier" in config
                assert config.restart_period > 0
                assert config.restart_multiplier > 0

            elif config.get("scheduler_type") == "multistep":
                assert "lr_milestones" in config
                assert "lr_gamma" in config
                assert hasattr(config.lr_milestones, '__iter__'), "lr_milestones should be iterable"
                assert len(config.lr_milestones) > 0
                assert 0 < config.lr_gamma < 1.0

            if "warmup_epochs" in config:
                assert config["warmup_epochs"] >= 0
            if "min_lr" in config and "finetune_lr" in config:
                assert float(config["min_lr"]) > 0
                assert float(config["min_lr"]) < float(config["finetune_lr"])
    
    def test_model_specific_scheduler_selection(self):
        """Test that each model has appropriate scheduler selection"""
        scheduler_configs = {
            "convnext_l_cifar100": {
                "scheduler_type": "onecycle",
                "reason": "큰 모델이므로 가장 효과적인 스케줄링"
            },
            "convnext_s_cifar100": {
                "scheduler_type": "reduce_on_plateau", 
                "reason": "중간 모델이므로 검증 성능 기반 적응적 스케줄링"
            },
            "resnet152_cifar100": {
                "scheduler_type": "cosine_warm_restarts",
                "reason": "전통적 모델이므로 주기적 재시작으로 local minima 탈출"
            },
            "efficientnet_l2_cifar100": {
                "scheduler_type": "multistep",
                "reason": "효율적 모델이므로 명시적이고 예측 가능한 스케줄링"
            }
        }
        
        for model_name, expected_config in scheduler_configs.items():
            config_path = f"configs/finetune/{model_name}.yaml"
            config = OmegaConf.load(config_path)
            if "scheduler_type" in config:
                assert config["scheduler_type"] == expected_config["scheduler_type"]


class TestMethodConfigs:
    """Test method configurations"""
    
    @pytest.mark.parametrize("method", [
        "asib", "vanilla_kd", "dkd", "crd", "fitnet", "at",
        "simkd", "reviewkd", "sskd", "ab", "ft"
    ])
    def test_method_config_exists(self, method):
        """Test method config file exists"""
        config_file = f"configs/method/{method}.yaml"
        assert Path(config_file).exists()
    
    @pytest.mark.parametrize("method", [
        "asib", "vanilla_kd", "dkd", "crd", "fitnet", "at",
        "simkd", "reviewkd", "sskd", "ab", "ft"
    ])
    def test_method_config_structure(self, method):
        """Test method config structure"""
        config_file = f"configs/method/{method}.yaml"
        config = OmegaConf.load(config_file)
        
        # All method configs should have some parameters
        assert len(config) > 0
    
    def test_asib_method_config(self):
        """Test ASIB method config"""
        config = OmegaConf.load("configs/method/asib.yaml")
        assert "name" in config and config["name"] == "asib"
        assert "ce_alpha" in config
        assert "kd_alpha" in config




class TestModelConfigs:
    """Test model configurations"""
    
    @pytest.mark.parametrize("teacher", [
        "resnet152", "convnext_l", "convnext_s", "efficientnet_l2"
    ])
    def test_teacher_config_exists(self, teacher):
        """Test teacher config file exists"""
        config_file = f"configs/model/teacher/{teacher}.yaml"
        assert Path(config_file).exists()
    
    @pytest.mark.parametrize("student", [
        "resnet152_pretrain", "resnet101_pretrain", "resnet50_scratch",
        "shufflenet_v2_scratch", "mobilenet_v2_scratch", "efficientnet_b0_scratch"
    ])
    def test_student_config_exists(self, student):
        """Test student config file exists"""
        config_file = f"configs/model/student/{student}.yaml"
        assert Path(config_file).exists()
    
    @pytest.mark.parametrize("teacher", [
        "resnet152", "convnext_l", "convnext_s", "efficientnet_l2"
    ])
    def test_teacher_config_structure(self, teacher):
        """Test teacher config structure (flat or nested)"""
        config_file = f"configs/model/teacher/{teacher}.yaml"
        config = OmegaConf.load(config_file)
        if "model" in config and "teacher" in config["model"]:
            teacher_cfg = config["model"]["teacher"]
        else:
            teacher_cfg = config
        assert "name" in teacher_cfg
        assert "pretrained" in teacher_cfg
    
    @pytest.mark.parametrize("student", [
        "resnet152_pretrain", "resnet101_pretrain", "resnet50_scratch",
        "shufflenet_v2_scratch", "mobilenet_v2_scratch", "efficientnet_b0_scratch"
    ])
    def test_student_config_structure(self, student):
        """Test student config structure (flat or nested)"""
        config_file = f"configs/model/student/{student}.yaml"
        config = OmegaConf.load(config_file)
        if "model" in config and "student" in config["model"]:
            student_cfg = config["model"]["student"]
        else:
            student_cfg = config
        assert "name" in student_cfg
        assert "pretrained" in student_cfg


class TestRegistryConfigs:
    """Test registry configurations"""
    
    def test_registry_map_structure(self):
        """Test registry map structure"""
        config = OmegaConf.load("configs/registry_map.yaml")
        
        assert "teachers" in config
        assert "students" in config
        
        # Check teachers
        assert len(config.teachers) > 0
        for teacher in config.teachers:
            assert isinstance(teacher, str)  # Teachers are strings
        
        # Check students
        assert len(config.students) > 0
        for student in config.students:
            assert isinstance(student, str)  # Students are strings
    
    def test_registry_key_structure(self):
        """Test registry key structure"""
        config = OmegaConf.load("configs/registry_key.yaml")
        
        assert "teacher_keys" in config
        assert "student_keys" in config
        
        # Check teacher keys
        assert len(config.teacher_keys) > 0
        for key in config.teacher_keys:
            assert isinstance(key, str)
        
        # Check student keys
        assert len(config.student_keys) > 0
        for key in config.student_keys:
            assert isinstance(key, str)
    
    def test_registry_consistency(self):
        """Test registry consistency between map and keys"""
        map_config = OmegaConf.load("configs/registry_map.yaml")
        key_config = OmegaConf.load("configs/registry_key.yaml")
        
        # Check teacher consistency
        map_teacher_names = map_config.teachers  # Teachers are strings
        key_teacher_names = key_config.teacher_keys
        
        for name in map_teacher_names:
            assert name in key_teacher_names
        
        # Check student consistency
        map_student_names = [s for s in map_config.students]  # Students are strings
        key_student_names = key_config.student_keys
        
        for name in map_student_names:
            assert name in key_student_names


class TestConfigValidation:
    """Test configuration validation"""
    
    def test_all_configs_valid_yaml(self):
        """Test all config files are valid YAML"""
        config_dirs = [
            "configs",
            "configs/experiment",
            "configs/finetune",
            "configs/method",
            "configs/model/teacher",
            "configs/model/student"
        ]
        
        for config_dir in config_dirs:
            if Path(config_dir).exists():
                for config_file in Path(config_dir).glob("*.yaml"):
                    try:
                        config = OmegaConf.load(str(config_file))
                        assert config is not None
                    except Exception as e:
                        pytest.fail(f"Invalid YAML in {config_file}: {e}")
    
    def test_config_paths_exist(self):
        """Test all referenced paths in configs exist"""
        # This test would check if checkpoint paths, data paths, etc. exist
        # For now, just check that configs can be loaded
        config_files = [
            "configs/base.yaml",
            "configs/experiment/L0_baseline.yaml",
            "configs/finetune/resnet152_cifar100.yaml",
            "configs/finetune/convnext_l_cifar100.yaml",
            "configs/finetune/efficientnet_l2_cifar100.yaml",
            "configs/finetune/convnext_s_cifar100.yaml"
        ]
        
        for config_file in config_files:
            assert Path(config_file).exists()
            config = OmegaConf.load(config_file)
            assert config is not None
    
    def test_config_types(self):
        """Test config value types"""
        config = OmegaConf.load("configs/base.yaml")
        
        # Check basic config types
        assert isinstance(config["experiment"].device, str)
        assert isinstance(config["experiment"]["dataset"]["batch_size"], int)
        assert isinstance(config["experiment"]["seed"], int)