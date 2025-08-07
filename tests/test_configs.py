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
        assert "device" in config
        assert "batch_size" in config
    
    def test_base_config_values(self):
        """Test base config values"""
        config = OmegaConf.load("configs/base.yaml")
        
        # Check basic values
        assert config.device == "cuda"
        assert config.batch_size == 128
        assert config.seed == 42


class TestExperimentConfigs:
    """Test experiment configurations"""
    
    @pytest.mark.parametrize("config_file", [
        "configs/experiment/legacy/res152_convnext_effi.yaml",
        "configs/experiment/legacy/res152_effi_l2.yaml"
    ])
    def test_experiment_config_structure(self, config_file):
        """Test experiment config structure"""
        config = OmegaConf.load(config_file)
        
        # Check required sections
        assert "defaults" in config
        assert "teacher1_ckpt" in config
        assert "teacher2_ckpt" in config
        assert "results_dir" in config
    
    @pytest.mark.parametrize("config_file", [
        "configs/experiment/legacy/res152_convnext_effi.yaml",
        "configs/experiment/legacy/res152_effi_l2.yaml"
    ])
    def test_experiment_config_values(self, config_file):
        """Test experiment config values"""
        config = OmegaConf.load(config_file)
        
        # Check basic configurations
        assert "teacher1_ckpt" in config
        assert "teacher2_ckpt" in config
        assert "results_dir" in config
        assert "exp_id" in config
    
    def test_res152_convnext_effi_config(self):
        """Test specific res152_convnext_effi config"""
        config = OmegaConf.load("configs/experiment/legacy/res152_convnext_effi.yaml")
        
        # Check specific values
        assert "convnext_l_cifar100.pth" in config.teacher1_ckpt
        assert "efficientnet_l2_cifar32.pth" in config.teacher2_ckpt
        assert config.exp_id == "res152_convnext_effi"
    
    def test_res152_effi_l2_config(self):
        """Test specific res152_effi_l2 config"""
        config = OmegaConf.load("configs/experiment/legacy/res152_effi_l2.yaml")
        
        # Check specific values
        assert "resnet152_cifar32.pth" in config.teacher1_ckpt
        assert "efficientnet_l2_cifar32.pth" in config.teacher2_ckpt
        assert config.exp_id == "res152_effi_l2"


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
        """Test finetune config structure"""
        config = OmegaConf.load(config_file)
        
        # Check required sections
        assert "teacher_type" in config
        assert "finetune_epochs" in config
        assert "finetune_lr" in config
        assert "batch_size" in config
        assert "results_dir" in config
        assert "finetune_ckpt_path" in config
    
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
        """Test finetune config values"""
        config = OmegaConf.load(config_file)
        
        # Check numeric values
        assert isinstance(config.finetune_epochs, int)
        assert isinstance(config.finetune_lr, float)
        assert isinstance(config.batch_size, int)
        
        # Check positive values
        assert config.finetune_epochs > 0
        assert config.finetune_lr > 0
        assert config.batch_size > 0
    
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
        
        assert config.teacher_type == "convnext_s"
        assert config.finetune_epochs == 80  # 중간 모델이므로 충분한 학습 시간
        assert config.finetune_lr == 1.5e-4  # 중간 모델이므로 적당한 학습률
        assert config.batch_size == 128
        assert config.finetune_weight_decay == 6e-3  # 과적합 방지를 위해 증가된 정규화
        assert config.label_smoothing == 0.5  # 과적합 방지를 위해 증가된 label smoothing
        assert config.warmup_epochs == 4
        assert config.early_stopping_patience == 10
        assert config.early_stopping_min_delta == 0.1
    
    def test_convnext_l_finetune_config(self):
        """Test specific convnext_l finetune config"""
        config = OmegaConf.load("configs/finetune/convnext_l_cifar100.yaml")
        
        assert config.teacher_type == "convnext_l"
        assert config.finetune_epochs == 60  # 큰 모델이므로 충분한 학습 시간
        assert config.finetune_lr == 8e-5  # 큰 모델이므로 더 낮은 학습률
        assert config.batch_size == 64  # ConvNeXt-L은 메모리 제약으로 작은 배치
        assert config.finetune_weight_decay == 8e-3  # 큰 모델이므로 강한 정규화
        assert config.label_smoothing == 0.5  # 큰 모델이므로 강한 label smoothing
        assert config.warmup_epochs == 5
        assert config.early_stopping_patience == 15
        assert config.early_stopping_min_delta == 0.05
    
    def test_efficientnet_l2_finetune_config(self):
        """Test specific efficientnet_l2 finetune config"""
        config = OmegaConf.load("configs/finetune/efficientnet_l2_cifar100.yaml")
        
        assert config.teacher_type == "efficientnet_l2"
        assert config.finetune_epochs == 65  # 효율적 모델이므로 적당한 학습 시간
        assert config.finetune_lr == 1.8e-4  # 효율적 모델이므로 약간 높은 학습률
        assert config.batch_size == 32  # A6000 GPU에서 EfficientNet-L2용
        assert config.finetune_weight_decay == 3e-3  # 과적합 방지를 위해 증가된 정규화
        assert config.label_smoothing == 0.4  # 과적합 방지를 위해 증가된 label smoothing
        assert config.warmup_epochs == 3
        assert config.early_stopping_patience == 6
        assert config.early_stopping_min_delta == 0.15
    
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
            
            # Warm-up 설정 검증
            assert "warmup_epochs" in config
            assert config.warmup_epochs >= 0
            assert config.warmup_epochs < config.finetune_epochs  # warmup < total epochs
            
            # min_lr 설정 검증
            assert "min_lr" in config
            assert config.min_lr > 0
            assert config.min_lr < config.finetune_lr  # min_lr < max_lr
    
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
            
            # Early stopping 설정 검증
            assert "early_stopping_patience" in config
            assert config.early_stopping_patience > 0
            assert config.early_stopping_patience < config.finetune_epochs
            
            assert "early_stopping_min_delta" in config
            assert config.early_stopping_min_delta > 0
            assert config.early_stopping_min_delta < 1.0  # 1% 미만이어야 함
    
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
            
            # 스케줄러 타입 검증
            assert "scheduler_type" in config
            assert config.scheduler_type == expected_scheduler
            
            # 스케줄러별 특정 설정 검증
            if config.scheduler_type == "cosine_warm_restarts":
                assert "restart_period" in config
                assert "restart_multiplier" in config
                assert config.restart_period > 0
                assert config.restart_multiplier > 0
                
            elif config.scheduler_type == "multistep":
                assert "lr_milestones" in config
                assert "lr_gamma" in config
                # OmegaConf에서는 리스트가 ListConfig 타입으로 변환됨
                assert hasattr(config.lr_milestones, '__iter__'), "lr_milestones should be iterable"
                assert len(config.lr_milestones) > 0
                assert config.lr_gamma > 0 and config.lr_gamma < 1.0
                
            # 모든 스케줄러에 공통적으로 필요한 설정
            assert "warmup_epochs" in config
            assert "min_lr" in config
            assert config.warmup_epochs >= 0
            assert config.min_lr > 0
            assert config.min_lr < config.finetune_lr
    
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
            
            assert config.scheduler_type == expected_config["scheduler_type"], \
                f"{model_name}: Expected {expected_config['scheduler_type']}, got {config.scheduler_type}. Reason: {expected_config['reason']}"


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
        
        # Check ASIB specific parameters
        assert "method" in config
        assert "ce_alpha" in config.method
        assert "kd_alpha" in config.method




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
        """Test teacher config structure"""
        config_file = f"configs/model/teacher/{teacher}.yaml"
        config = OmegaConf.load(config_file)
        
        assert "model" in config
        assert "teacher" in config.model
        assert "name" in config.model.teacher
        assert "pretrained" in config.model.teacher
    
    @pytest.mark.parametrize("student", [
        "resnet152_pretrain", "resnet101_pretrain", "resnet50_scratch",
        "shufflenet_v2_scratch", "mobilenet_v2_scratch", "efficientnet_b0_scratch"
    ])
    def test_student_config_structure(self, student):
        """Test student config structure"""
        config_file = f"configs/model/student/{student}.yaml"
        config = OmegaConf.load(config_file)
        
        assert "model" in config
        assert "student" in config.model
        assert "name" in config.model.student
        assert "pretrained" in config.model.student


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
            "configs/experiment/legacy/res152_convnext_effi.yaml",
            "configs/experiment/legacy/res152_effi_l2.yaml",
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
        assert isinstance(config.device, str)
        assert isinstance(config.batch_size, int)
        assert isinstance(config.seed, int) 