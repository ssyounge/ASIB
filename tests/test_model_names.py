#!/usr/bin/env python3
"""
Test Model Names Consistency

모델 이름과 config 파일들이 일치하는지 확인하는 테스트
- Registry에 등록된 모델 이름
- Config 파일에서 사용하는 모델 이름
- 파인튜닝 config 파일들
- 실험 config 파일들
"""

import os
import yaml
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.common import registry

def test_registry_models():
    """Registry에 등록된 모든 모델들을 확인"""
    print("🔍 Testing Registry Models...")
    
    registry.ensure_scanned()
    
    # Expected teacher models
    expected_teachers = [
        "resnet152_teacher",
        "convnext_s_teacher", 
        "convnext_l_teacher",
        "efficientnet_l2_teacher"
    ]
    
    # Expected student models
    expected_students = [
        "resnet152_pretrain_student",
        "resnet101_pretrain_student", 
        "resnet50_scratch_student",
        "shufflenet_v2_scratch_student",
        "mobilenet_v2_scratch_student",
        "efficientnet_b0_scratch_student"
    ]
    
    # Check teacher models
    print("\n📚 Teacher Models:")
    for teacher in expected_teachers:
        if teacher in registry.MODEL_REGISTRY:
            print(f"  ✅ {teacher}")
        else:
            print(f"  ❌ {teacher} - NOT FOUND")
            return False
    
    # Check student models
    print("\n🎓 Student Models:")
    for student in expected_students:
        if student in registry.MODEL_REGISTRY:
            print(f"  ✅ {student}")
        else:
            print(f"  ❌ {student} - NOT FOUND")
            return False
    
    return True

def test_finetune_configs():
    """파인튜닝 config 파일들의 teacher_type 확인"""
    print("\n🔧 Testing Finetune Configs...")
    
    finetune_dir = Path("configs/finetune")
    if not finetune_dir.exists():
        print(f"  ❌ Finetune config directory not found: {finetune_dir}")
        return False
    
    config_files = list(finetune_dir.glob("*.yaml"))
    print(f"  Found {len(config_files)} finetune config files")
    
    # Expected teacher_type values (registry names)
    expected_teacher_types = [
        "resnet152_teacher",
        "convnext_s_teacher", 
        "convnext_l_teacher",
        "efficientnet_l2_teacher"
    ]
    
    for config_file in config_files:
        print(f"\n  📄 {config_file.name}:")
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check if teacher_type exists
            if 'teacher_type' not in config:
                print(f"    ❌ No teacher_type found")
                return False
            
            teacher_type = config['teacher_type']
            print(f"    teacher_type: {teacher_type}")
            
            # Check if teacher_type matches registry
            if teacher_type in expected_teacher_types:
                print(f"    ✅ Matches registry")
            else:
                print(f"    ❌ Does NOT match registry")
                print(f"    Expected one of: {expected_teacher_types}")
                return False
                
        except Exception as e:
            print(f"    ❌ Error reading config: {e}")
            return False
    
    return True

def test_experiment_configs():
    """실험 config 파일들의 teacher/student 이름 확인"""
    print("\n🧪 Testing Experiment Configs...")
    
    experiment_dir = Path("configs/experiment")
    if not experiment_dir.exists():
        print(f"  ❌ Experiment config directory not found: {experiment_dir}")
        return False
    
    config_files = list(experiment_dir.glob("*.yaml"))
    print(f"  Found {len(config_files)} experiment config files")
    
    # Expected model names
    expected_teachers = [
        "resnet152_teacher",
        "convnext_s_teacher", 
        "convnext_l_teacher",
        "efficientnet_l2_teacher"
    ]
    
    expected_students = [
        "resnet152_pretrain_student",
        "resnet101_pretrain_student", 
        "resnet50_scratch_student",
        "shufflenet_v2_scratch_student",
        "mobilenet_v2_scratch_student",
        "efficientnet_b0_scratch_student"
    ]
    
    for config_file in config_files:
        print(f"\n  📄 {config_file.name}:")
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check teacher1
            if 'teacher1' in config:
                teacher1_name = config['teacher1'].get('model', {}).get('teacher', {}).get('name')
                if teacher1_name:
                    print(f"    teacher1: {teacher1_name}")
                    if teacher1_name in expected_teachers:
                        print(f"      ✅ Matches registry")
                    else:
                        print(f"      ❌ Does NOT match registry")
                        return False
                else:
                    print(f"    ❌ No teacher1 name found")
                    return False
            
            # Check teacher2
            if 'teacher2' in config:
                teacher2_name = config['teacher2'].get('model', {}).get('teacher', {}).get('name')
                if teacher2_name:
                    print(f"    teacher2: {teacher2_name}")
                    if teacher2_name in expected_teachers:
                        print(f"      ✅ Matches registry")
                    else:
                        print(f"      ❌ Does NOT match registry")
                        return False
                else:
                    print(f"    ❌ No teacher2 name found")
                    return False
            
            # Check student
            if 'student' in config:
                student_name = config['student'].get('model', {}).get('student', {}).get('name')
                if student_name:
                    print(f"    student: {student_name}")
                    if student_name in expected_students:
                        print(f"      ✅ Matches registry")
                    else:
                        print(f"      ❌ Does NOT match registry")
                        return False
                else:
                    print(f"    ❌ No student name found")
                    return False
                    
        except Exception as e:
            print(f"    ❌ Error reading config: {e}")
            return False
    
    return True

def test_model_creation():
    """실제로 모델을 생성해서 테스트"""
    print("\n🏗️ Testing Model Creation...")
    
    from core.builder import create_teacher_by_name, create_student_by_name
    
    # Test teacher models
    print("\n📚 Creating Teacher Models:")
    teachers = [
        "resnet152_teacher",
        "convnext_s_teacher", 
        "convnext_l_teacher",
        "efficientnet_l2_teacher"
    ]
    
    for teacher_name in teachers:
        try:
            model = create_teacher_by_name(
                teacher_name=teacher_name,
                num_classes=100,
                pretrained=False,  # Use random weights for testing
                small_input=True
            )
            param_count = sum(p.numel() for p in model.parameters())
            print(f"  ✅ {teacher_name} - {param_count:,} parameters")
        except Exception as e:
            print(f"  ❌ {teacher_name} - Error: {e}")
            return False
    
    # Test student models
    print("\n🎓 Creating Student Models:")
    students = [
        "resnet152_pretrain_student",
        "resnet101_pretrain_student", 
        "resnet50_scratch_student",
        "shufflenet_v2_scratch_student",
        "mobilenet_v2_scratch_student",
        "efficientnet_b0_scratch_student"
    ]
    
    for student_name in students:
        try:
            model = create_student_by_name(
                student_name=student_name,
                num_classes=100,
                pretrained=False,  # Use random weights for testing
                small_input=True
            )
            param_count = sum(p.numel() for p in model.parameters())
            print(f"  ✅ {student_name} - {param_count:,} parameters")
        except Exception as e:
            print(f"  ❌ {student_name} - Error: {e}")
            return False
    
    return True

def test_config_loading():
    """Config 파일들이 제대로 로드되는지 테스트"""
    print("\n📋 Testing Config Loading...")
    
    import hydra
    from omegaconf import DictConfig
    
    # Test finetune configs
    finetune_configs = [
        "finetune/resnet152_cifar100",
        "finetune/convnext_s_cifar100",
        "finetune/convnext_l_cifar100", 
        "finetune/efficientnet_l2_cifar100"
    ]
    
    for config_name in finetune_configs:
        print(f"\n  📄 {config_name}:")
        try:
            # Simulate Hydra config loading
            config_path = Path("configs") / f"{config_name}.yaml"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                if 'teacher_type' in config:
                    teacher_type = config['teacher_type']
                    print(f"    teacher_type: {teacher_type}")
                    
                    # Check if it's a valid registry name
                    registry.ensure_scanned()
                    if teacher_type in registry.MODEL_REGISTRY:
                        print(f"    ✅ Valid registry name")
                    else:
                        print(f"    ❌ Invalid registry name")
                        return False
                else:
                    print(f"    ❌ No teacher_type found")
                    return False
            else:
                print(f"    ❌ Config file not found")
                return False
                
        except Exception as e:
            print(f"    ❌ Error loading config: {e}")
            return False
    
    return True

def main():
    """Run all tests"""
    print("🎯 Model Names Consistency Test")
    print("=" * 50)
    
    tests = [
        ("Registry Models", test_registry_models),
        ("Finetune Configs", test_finetune_configs),
        ("Experiment Configs", test_experiment_configs),
        ("Model Creation", test_model_creation),
        ("Config Loading", test_config_loading)
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            if result:
                print(f"✅ {test_name} - PASSED")
            else:
                print(f"❌ {test_name} - FAILED")
                all_passed = False
        except Exception as e:
            print(f"❌ {test_name} - ERROR: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ All model names are consistent across registry and configs")
    else:
        print("💥 SOME TESTS FAILED!")
        print("❌ Please fix the inconsistencies above")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

# pytest에서 실행되지 않도록 마킹
pytest_plugins = [] 