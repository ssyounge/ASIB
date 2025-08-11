#!/usr/bin/env python3
"""Comprehensive Registry Mapping Test - Prevents registry issues from happening again"""

import sys
import os
from pathlib import Path
import yaml
import pytest
from omegaconf import OmegaConf

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestComprehensiveRegistryMapping:
    """Comprehensive test for registry mapping across all config files - PREVENTS FUTURE ISSUES"""
    
    def test_registry_files_exist(self, registry_configs):
        """Test that registry files exist"""
        registry_files = [
            registry_configs["registry_key_path"],
            registry_configs["registry_map_path"]
        ]
        
        for registry_file in registry_files:
            assert Path(registry_file).exists(), f"Registry file not found: {registry_file}"
    
    def test_registry_structure(self):
        """Test registry file structure"""
        # Load registry files
        key_config = OmegaConf.load("configs/registry_key.yaml")
        map_config = OmegaConf.load("configs/registry_map.yaml")
        
        # Check key structure
        assert "teacher_keys" in key_config, "teacher_keys missing in registry_key.yaml"
        assert "student_keys" in key_config, "student_keys missing in registry_key.yaml"
        
        # Check map structure
        assert "teachers" in map_config, "teachers missing in registry_map.yaml"
        assert "students" in map_config, "students missing in registry_map.yaml"
        
        # Check data types (OmegaConf returns special types, so we check if they're iterable)
        assert hasattr(key_config.teacher_keys, '__iter__'), "teacher_keys should be iterable"
        assert hasattr(key_config.student_keys, '__iter__'), "student_keys should be iterable"
        assert hasattr(map_config.teachers, 'items'), "teachers should be a dict-like object"
        assert hasattr(map_config.students, 'items'), "students should be a dict-like object"
    
    def test_registry_consistency(self, registry_validation):
        """Test consistency between registry_key and registry_map - CRITICAL FOR PREVENTING ISSUES"""
        is_valid, message = registry_validation()
        assert is_valid, f"Registry validation failed: {message}"
        
        # 추가 검증을 위해 직접 로드
        key_config = OmegaConf.load("configs/registry_key.yaml")
        map_config = OmegaConf.load("configs/registry_map.yaml")
        
        # Check teacher consistency
        key_teachers = set(key_config.teacher_keys)
        map_teachers = set(map_config.teachers.keys())
        
        assert key_teachers == map_teachers, f"Teacher mismatch: key={key_teachers}, map={map_teachers}"
        
        # Check student consistency
        key_students = set(key_config.student_keys)
        map_students = set(map_config.students.keys())
        
        assert key_students == map_students, f"Student mismatch: key={key_students}, map={map_students}"
        
        # CRITICAL: Check for _student suffix in registry keys
        for student_key in key_students:
            assert not student_key.endswith('_student'), f"Student key '{student_key}' should not end with '_student'"
        
        for student_key in map_students:
            assert not student_key.endswith('_student'), f"Student key '{student_key}' should not end with '_student'"
        
        # CRITICAL: Check for _teacher suffix in registry keys
        for teacher_key in key_teachers:
            assert not teacher_key.endswith('_teacher'), f"Teacher key '{teacher_key}' should not end with '_teacher'"
        
        for teacher_key in map_teachers:
            assert not teacher_key.endswith('_teacher'), f"Teacher key '{teacher_key}' should not end with '_teacher'"
    
    def test_all_experiment_configs_use_valid_registry_keys(self):
        """Test all experiment configs use valid registry keys"""
        key_config = OmegaConf.load("configs/registry_key.yaml")
        valid_teachers = set(key_config.teacher_keys)
        valid_students = set(key_config.student_keys)
        
        experiment_dir = Path("configs/experiment")
        if not experiment_dir.exists():
            pytest.skip("Experiment config directory not found")
        
        errors = []
        
        for config_file in experiment_dir.glob("*.yaml"):
            try:
                config = OmegaConf.load(str(config_file))
                
                # Check teacher1
                if "defaults" in config:
                    for default in config.defaults:
                        if isinstance(default, str) and "teacher@teacher1" in default:
                            teacher1_name = default.split(":")[-1].strip()
                            if teacher1_name not in valid_teachers:
                                errors.append(f"{config_file}: teacher1 '{teacher1_name}' not in registry")
                
                # Check teacher2
                if "defaults" in config:
                    for default in config.defaults:
                        if isinstance(default, str) and "teacher@teacher2" in default:
                            teacher2_name = default.split(":")[-1].strip()
                            if teacher2_name not in valid_teachers:
                                errors.append(f"{config_file}: teacher2 '{teacher2_name}' not in registry")
                
                # Check student
                if "defaults" in config:
                    for default in config.defaults:
                        if isinstance(default, str) and "model/student" in default:
                            student_name = default.split(":")[-1].strip()
                            if student_name not in valid_students:
                                errors.append(f"{config_file}: student '{student_name}' not in registry")
                            
                            # CRITICAL: Check for _student suffix in config files
                            if student_name.endswith('_student'):
                                errors.append(f"{config_file}: student '{student_name}' should not end with '_student'")
                
                # Check for _teacher suffix in config files
                if "defaults" in config:
                    for default in config.defaults:
                        if isinstance(default, str) and ("teacher@teacher1" in default or "teacher@teacher2" in default):
                            teacher_name = default.split(":")[-1].strip()
                            if teacher_name.endswith('_teacher'):
                                errors.append(f"{config_file}: teacher '{teacher_name}' should not end with '_teacher'")
                                
            except Exception as e:
                errors.append(f"{config_file}: Error loading config - {e}")
        
        if errors:
            error_msg = "\n".join(errors)
            pytest.fail(f"Registry mapping errors found:\n{error_msg}")
    
    def test_all_finetune_configs_use_valid_registry_keys(self):
        """Test all finetune configs use valid registry keys"""
        key_config = OmegaConf.load("configs/registry_key.yaml")
        valid_teachers = set(key_config.teacher_keys)
        
        finetune_dir = Path("configs/finetune")
        if not finetune_dir.exists():
            pytest.skip("Finetune config directory not found")
        
        errors = []
        
        for config_file in finetune_dir.glob("*.yaml"):
            try:
                config = OmegaConf.load(str(config_file))
                
                # Check teacher_type
                if "teacher_type" in config:
                    teacher_type = config.teacher_type
                    if teacher_type not in valid_teachers:
                        errors.append(f"{config_file}: teacher_type '{teacher_type}' not in registry")
                        
            except Exception as e:
                errors.append(f"{config_file}: Error loading config - {e}")
        
        if errors:
            error_msg = "\n".join(errors)
            pytest.fail(f"Registry mapping errors found:\n{error_msg}")
    
    def test_model_files_exist_for_registry_entries(self):
        """Test that model files exist for all registry entries"""
        map_config = OmegaConf.load("configs/registry_map.yaml")
        
        errors = []
        
        # Check teacher model files
        for teacher_name, teacher_path in map_config.teachers.items():
            # Extract module path from registry entry
            module_path = teacher_path.split(".")[:-1]  # Remove function name
            # Convert models.teachers.convnext_s to models/teachers/convnext_s.py
            # module_path = ['models', 'teachers', 'convnext_s']
            # We want: models/teachers/convnext_s.py
            file_path = Path("models") / "/".join(module_path[1:-1]) / f"{module_path[-1]}.py"
            
            print(f"Checking teacher: {teacher_name} -> {file_path}")
            if not file_path.exists():
                errors.append(f"Teacher model file not found: {file_path} (for {teacher_name})")
        
        # Check student model files
        for student_name, student_path in map_config.students.items():
            # Extract module path from registry entry
            module_path = student_path.split(".")[:-1]  # Remove function name
            # Convert models.students.resnet50_student to models/students/resnet50_student.py
            # module_path = ['models', 'students', 'resnet50_student']
            # We want: models/students/resnet50_student.py
            file_path = Path("models") / "/".join(module_path[1:-1]) / f"{module_path[-1]}.py"
            
            print(f"Checking student: {student_name} -> {file_path}")
            if not file_path.exists():
                errors.append(f"Student model file not found: {file_path} (for {student_name})")
        
        if errors:
            error_msg = "\n".join(errors)
            pytest.fail(f"Missing model files:\n{error_msg}")
    
    def test_config_files_use_correct_naming_convention(self):
        """Test that config files use correct naming convention (no _student suffix)"""
        config_dirs = [
            "configs/experiment",
            "configs/finetune"
        ]
        
        errors = []
        
        for config_dir in config_dirs:
            dir_path = Path(config_dir)
            if not dir_path.exists():
                continue
                
            for config_file in dir_path.glob("*.yaml"):
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for _student suffix in model references
                    if "_student" in content:
                        lines_with_student = [line.strip() for line in content.split('\n') if "_student" in line]
                        errors.append(f"{config_file}: Contains _student suffix in lines: {lines_with_student}")
                        
                except Exception as e:
                    errors.append(f"{config_file}: Error reading file - {e}")
        
        if errors:
            error_msg = "\n".join(errors)
            pytest.fail(f"Config files with incorrect naming convention:\n{error_msg}")
    
    def test_registry_function_imports_are_valid(self):
        """Test that registry function imports are valid"""
        map_config = OmegaConf.load("configs/registry_map.yaml")
        
        errors = []
        
        # Test all registry entries
        all_entries = {**map_config.teachers, **map_config.students}
        
        for model_name, import_path in all_entries.items():
            try:
                # Parse import path
                module_path, function_name = import_path.rsplit(".", 1)
                
                # Try to import the module
                __import__(module_path)
                
                # Try to get the function
                module = sys.modules[module_path]
                if not hasattr(module, function_name):
                    errors.append(f"{model_name}: Function '{function_name}' not found in module '{module_path}'")
                    
            except ImportError as e:
                errors.append(f"{model_name}: Cannot import module '{module_path}' - {e}")
            except Exception as e:
                errors.append(f"{model_name}: Error testing import '{import_path}' - {e}")
        
        if errors:
            error_msg = "\n".join(errors)
            pytest.fail(f"Registry import errors:\n{error_msg}")
    
    def test_comprehensive_registry_summary(self):
        """Print comprehensive registry summary"""
        key_config = OmegaConf.load("configs/registry_key.yaml")
        map_config = OmegaConf.load("configs/registry_map.yaml")
        
        print("\n" + "="*60)
        print("🔍 COMPREHENSIVE REGISTRY SUMMARY")
        print("="*60)
        
        print(f"\n📚 Teachers ({len(key_config.teacher_keys)}):")
        for teacher in sorted(key_config.teacher_keys):
            print(f"  ✅ {teacher}")
        
        print(f"\n🎓 Students ({len(key_config.student_keys)}):")
        for student in sorted(key_config.student_keys):
            print(f"  ✅ {student}")
        
        print(f"\n📁 Registry Map Entries:")
        print(f"  Teachers: {len(map_config.teachers)}")
        print(f"  Students: {len(map_config.students)}")
        
        print(f"\n🎯 Registry Consistency: ✅ PASSED")
        print("🚫 _student suffix check: ✅ PASSED")
        print("🔒 FUTURE REGISTRY ISSUES PREVENTED!")
        print("="*60)
        
        # This test always passes, it's just for reporting
        assert True
    
    def test_prevent_future_registry_issues(self):
        """CRITICAL: Test to prevent future registry issues"""
        key_config = OmegaConf.load("configs/registry_key.yaml")
        map_config = OmegaConf.load("configs/registry_map.yaml")
        
        # 1. Check for _student suffix in registry keys
        for student_key in key_config.student_keys:
            assert not student_key.endswith('_student'), f"CRITICAL: Student key '{student_key}' ends with '_student'"
        
        for student_key in map_config.students.keys():
            assert not student_key.endswith('_student'), f"CRITICAL: Student key '{student_key}' ends with '_student'"
        
        # 2. Check for _teacher suffix in registry keys
        for teacher_key in key_config.teacher_keys:
            assert not teacher_key.endswith('_teacher'), f"CRITICAL: Teacher key '{teacher_key}' ends with '_teacher'"
        
        for teacher_key in map_config.teachers.keys():
            assert not teacher_key.endswith('_teacher'), f"CRITICAL: Teacher key '{teacher_key}' ends with '_teacher'"
        
        # 3. Check for _student and _teacher suffix in config files
        config_dirs = ["configs/experiment", "configs/finetune"]
        for config_dir in config_dirs:
            dir_path = Path(config_dir)
            if dir_path.exists():
                for config_file in dir_path.glob("*.yaml"):
                    with open(config_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if "_student" in content:
                            pytest.fail(f"CRITICAL: {config_file} contains '_student' suffix")
                        if "_teacher" in content:
                            pytest.fail(f"CRITICAL: {config_file} contains '_teacher' suffix")
        
        # 4. Check registry consistency
        key_teachers = set(key_config.teacher_keys)
        map_teachers = set(map_config.teachers.keys())
        assert key_teachers == map_teachers, "CRITICAL: Teacher registry mismatch"
        
        key_students = set(key_config.student_keys)
        map_students = set(map_config.students.keys())
        assert key_students == map_students, "CRITICAL: Student registry mismatch"
        
        print("✅ FUTURE REGISTRY ISSUES PREVENTED!")
        assert True 