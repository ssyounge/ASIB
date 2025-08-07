# Tests Directory

This directory contains comprehensive tests for the ASMB-KD project, ensuring all components work correctly and reliably.

## 📁 Test Structure

### 🔧 Core Functionality Tests
- **`test_core.py`** - Core builder, trainer, and utility functions
- **`test_core_utils.py`** - Core utility functions (renorm_ce_kd, setup_partial_freeze_schedule, etc.)
- **`test_utils_common.py`** - Common utility functions (set_random_seed, check_label_range, etc.)

### 🧠 Model & Module Tests
- **`test_models.py`** - Basic model creation and functionality
- **`test_models_advanced.py`** - Advanced model features and integration
- **`test_modules.py`** - Module functionality (trainers, losses, disagreement)
- **`test_modules_partial_freeze.py`** - Partial freeze functionality

### 📊 Data & Configuration Tests
- **`test_data.py`** - Dataset loading and data processing
- **`test_configs.py`** - Configuration file validation
- **`test_finetune_configs.py`** - Fine-tuning configuration tests
- **`test_experiment_configs.py`** - Experiment configuration tests

### 🔄 Training & Experiment Tests
- **`test_main.py`** - Main entry point functionality
- **`test_main_training.py`** - Main training pipeline
- **`test_asib_step.py`** - ASIB step-by-step functionality
- **`test_asib_cl.py`** - ASIB continual learning
- **`test_cl_experiments.py`** - Continual learning experiments
- **`test_training_pipeline.py`** - Training pipeline components

### 🧪 Integration & Validation Tests
- **`test_integration.py`** - End-to-end integration tests
- **`test_final_validation.py`** - Final validation scenarios
- **`test_experiment_execution.py`** - Experiment execution validation
- **`test_framework_robustness.py`** - Framework robustness and edge cases

### 🔍 Analysis & Script Tests
- **`test_scripts.py`** - Analysis and utility scripts
- **`test_pycil_integration.py`** - PyCIL framework integration
- **`test_pycil_models.py`** - PyCIL model functionality

### 🛡️ Error Prevention & Edge Cases
- **`test_error_prevention.py`** - Error handling and prevention
- **`test_overlap_dataset.py`** - Class overlap dataset functionality

### 📐 Specialized Component Tests
- **`test_mbm_tensor_shapes.py`** - MBM tensor shape validation
- **`test_ib_mbm_shapes.py`** - IB-MBM specific shape tests
- **`test_kd_methods.py`** - Knowledge distillation methods
- **`test_registry_comprehensive.py`** - Model registry functionality

### 🧩 Utility Tests
- **`test_utils.py`** - General utility functions
- **`test_disagreement.py`** - Disagreement computation

## 🚀 Running Tests

### Run All Tests
```bash
./run/run_test.sh
```

### Run Specific Test Categories
```bash
# Core functionality
pytest tests/test_core.py tests/test_core_utils.py tests/test_utils_common.py -v

# Model tests
pytest tests/test_models*.py tests/test_modules*.py -v

# Data and config tests
pytest tests/test_data.py tests/test_configs.py -v

# Training tests
pytest tests/test_main*.py tests/test_asib*.py tests/test_training*.py -v

# Integration tests
pytest tests/test_integration.py tests/test_final_validation.py -v
```

### Run Tests with Coverage
```bash
pytest --cov=. --cov-report=html tests/
```

## 📋 Test Coverage

### ✅ Main.py Functions Covered
- `create_student_by_name()` - ✅ `test_core.py`, `test_models.py`
- `create_teacher_by_name()` - ✅ `test_core.py`, `test_models.py`
- `run_training_stages()` - ✅ `test_core.py`, `test_training_pipeline.py`
- `run_continual_learning()` - ✅ `test_core.py`, `test_cl_experiments.py`
- `renorm_ce_kd()` - ✅ `test_core_utils.py`
- `setup_partial_freeze_schedule_with_cfg()` - ✅ `test_core_utils.py`
- `setup_safety_switches_with_cfg()` - ✅ `test_core_utils.py`
- `auto_set_mbm_query_dim_with_model()` - ✅ `test_core_utils.py`
- `cast_numeric_configs()` - ✅ `test_core_utils.py`

### ✅ Utils Functions Covered
- `set_random_seed()` - ✅ `test_utils_common.py`
- `check_label_range()` - ✅ `test_utils_common.py`
- `get_model_num_classes()` - ✅ `test_utils_common.py`
- `count_trainable_parameters()` - ✅ `test_utils_common.py`
- `get_amp_components()` - ✅ `test_utils_common.py`
- `mixup_data()`, `cutmix_data()` - ✅ `test_utils_common.py`

### ✅ Partial Freeze Functions Covered
- `apply_partial_freeze()` - ✅ `test_modules_partial_freeze.py`
- `partial_freeze_teacher_resnet()` - ✅ `test_modules_partial_freeze.py`
- `partial_freeze_teacher_efficientnet()` - ✅ `test_modules_partial_freeze.py`
- `partial_freeze_student_resnet()` - ✅ `test_modules_partial_freeze.py`

### ✅ Data Loading Functions Covered
- `get_cifar100_loaders()` - ✅ `test_data.py`
- `get_imagenet32_loaders()` - ✅ `test_data.py`
- Overlap dataset functionality - ✅ `test_overlap_dataset.py`

### ✅ Model Registry & Creation Covered
- Model registry functionality - ✅ `test_registry_comprehensive.py`
- Teacher/Student model creation - ✅ `test_models.py`, `test_models_advanced.py`
- MBM and synergy head creation - ✅ `test_mbm_tensor_shapes.py`

## 🎯 Test Quality Standards

### ✅ All Tests Must:
- **Be Independent**: Each test should run independently
- **Be Deterministic**: Same input should produce same output
- **Have Clear Assertions**: Explicit checks for expected behavior
- **Handle Edge Cases**: Test boundary conditions and error scenarios
- **Use Mock Data**: Avoid real data loading when possible
- **Be Fast**: Complete in reasonable time (< 1 second per test)

### ✅ Integration Tests Must:
- **Test Real Workflows**: End-to-end functionality
- **Validate Configurations**: Ensure configs work correctly
- **Check Error Handling**: Verify graceful failure modes
- **Test Performance**: Ensure reasonable memory/time usage

## 🔧 Test Utilities

### Fixtures (conftest.py)
- **`temp_config_file`** - Temporary configuration files
- **`dummy_teachers`** - Mock teacher models
- **`dummy_student`** - Mock student model
- **`dummy_mbm`** - Mock MBM component
- **`dummy_synergy_head`** - Mock synergy head

### Mock Classes
- **`MockDataset`** - Dataset simulation
- **`MockDataLoader`** - DataLoader simulation
- **`MockModel`** - Model simulation for testing

## 📊 Test Statistics

- **Total Test Files**: 25
- **Total Test Functions**: ~200+
- **Coverage**: Core functionality, models, data, training, integration
- **Execution Time**: ~30-60 seconds for full suite

## 🐛 Debugging Tests

### Common Issues
1. **CUDA Device Errors**: Tests use `.cuda()` - ensure CUDA available
2. **Import Errors**: Check PYTHONPATH includes project root
3. **Memory Issues**: Tests use small batch sizes and models
4. **Path Issues**: All paths use relative paths from project root

### Debug Commands
```bash
# Run single test with verbose output
pytest tests/test_core.py::TestCoreBuilder::test_create_student_by_name -v -s

# Run with print statements
pytest tests/test_core.py -v -s

# Run with debugger
pytest tests/test_core.py --pdb
```

## 📝 Adding New Tests

### Guidelines
1. **Follow Naming Convention**: `test_*.py` for files, `test_*` for functions
2. **Use Descriptive Names**: Clear test function names
3. **Add Documentation**: Docstrings for test classes and functions
4. **Use Appropriate Fixtures**: Leverage existing fixtures when possible
5. **Test Edge Cases**: Include boundary conditions and error scenarios
6. **Keep Tests Fast**: Use small models and datasets

### Example Test Structure
```python
def test_function_name():
    """Test description of what this test validates"""
    # Setup
    input_data = create_test_data()
    
    # Execute
    result = function_under_test(input_data)
    
    # Assert
    assert result is not None
    assert result.shape == expected_shape
    assert result.dtype == expected_dtype
```

## 🎉 Test Results

All tests should pass before merging any changes. The test suite ensures:
- ✅ All core functions work correctly
- ✅ Model creation and training pipelines function properly
- ✅ Data loading and processing work as expected
- ✅ Configuration handling is robust
- ✅ Error handling is graceful
- ✅ Integration workflows function end-to-end 