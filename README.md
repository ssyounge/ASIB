# ASIB Knowledge Distillation Framework

**ASIB** (Adaptive Synergy Information-Bottleneck) is a multi-stage knowledge distillation framework that uses the Information‑Bottleneck Manifold‑Bridging Module (IB_MBM) to create synergistic knowledge from multiple teachers.

## 🎯 **Latest Updates**

### ✅ **Project Cleanup & Organization**
- **Test Files**: All test files moved to `tests/` folder and organized
- **Checkpoints**: Model checkpoints now saved in `checkpoints/` folder
- **Path Standardization**: All absolute paths converted to relative paths for portability
- **Comprehensive Testing**: All 40+ test files now covered in automated test suite

### 🧪 **Complete Test Coverage**
- **42 Test Files**: Comprehensive test suite covering all components
- **Automated Testing**: `run/run_test.sh` runs all tests in parallel
- **Test Categories**: 
  - Core ASIB tests, PyCIL integration, Data & Utils
  - Model tests, Config & Experiment tests, Script & Integration
  - KD & Special tests, Framework robustness, Main integration

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/YourName/ASIB.git
cd ASIB

# Note: All paths are relative - no absolute path dependencies!

# Create conda environment
conda env create -f environment.yml
conda activate asib

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run all tests (comprehensive test suite)
sbatch run/run_test.sh

# Run complete ablation study
sbatch run/run_ablation_study.sh

# Run teacher fine-tuning
sbatch run/run_finetune_single.sh convnext_s_cifar100

# Run beta sensitivity analysis
python scripts/analysis/beta_sensitivity.py

# Run comprehensive analysis
python scripts/analysis/comprehensive_analysis.py
```

## 📁 Project Structure

```
ASIB/
├── main.py                 # Main training script
├── eval.py                 # Model evaluation
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── environment.yml        # Conda environment
├── setup.py              # Package setup
├── pytest.ini           # Test configuration
├── .gitignore           # Git ignore rules
├── LICENSE              # License file
├── docs/                # Documentation
│   ├── API.md          # API documentation
│   ├── CONFIGURATION.md # Configuration guide
│   ├── INSTALLATION.md  # Installation guide
│   └── reports/        # Experiment reports
│       ├── EXPERIMENT_PLAN.md
│       ├── IMPROVED_ABLATION_STUDY_REPORT.md
│       └── ABLATION_STUDY_REPORT.md
├── configs/             # Configuration files
│   ├── base.yaml       # Base configuration
│   ├── experiment/     # Experiment configs
│   │   ├── ablation_*.yaml  # Ablation study configs
│   │   ├── sota_*.yaml      # SOTA comparison configs
│   │   └── overlap_*.yaml   # Overlap analysis configs
│   ├── finetune/       # Fine-tuning configs
│   └── model/          # Model configs
├── scripts/            # Utility scripts
│   ├── analysis/       # Analysis scripts
│   │   ├── beta_sensitivity.py
│   │   ├── information_plane_analysis.py
│   │   ├── cccp_stability_analysis.py
│   │   ├── teacher_adaptation_analysis.py
│   │   ├── pf_efficiency_analysis.py
│   │   └── comprehensive_analysis.py
│   ├── training/       # Training scripts
│   │   └── fine_tuning.py
│   └── setup/          # Setup scripts
├── run/                # SLURM execution scripts
│   ├── run_test.sh              # Comprehensive test suite
│   ├── run_asib_ablation_study.sh
│   ├── run_asib_sota_comparison.sh
│   ├── run_finetune_single.sh
│   └── run_finetune_all_teachers.sh
├── models/             # Model definitions (IB_MBM)
├── data/               # Data loading utilities
├── utils/              # Utility modules
├── core/               # Core functionality
├── methods/            # Distillation methods
├── modules/            # Training modules
├── tests/              # Comprehensive test suite (42 files)
│   ├── conftest.py     # Common test fixtures
│   ├── test_asib_*.py  # Core ASIB tests
│   ├── test_pycil_*.py # PyCIL integration tests
│   ├── test_*.py       # All other test files
│   └── README.md       # Test documentation
├── experiments/        # 🧪 All experiments (logs + results integrated)
│   ├── test/           # Test experiments
│   │   ├── logs/       # Test logs
│   │   └── results/    # Test results
│   ├── ablation/       # Ablation study experiments
│   │   ├── baseline/   # Baseline experiments
│   │   ├── cccp/       # CCCP experiments
│   │   ├── ib/         # Information Bottleneck experiments
│   │   ├── tadapt/     # Teacher adaptation experiments
│   │   └── full/       # Full ASIB experiments
│   ├── overlap/        # Class overlap experiments
│   │   ├── logs/       # Overlap logs
│   │   └── results/    # Overlap results
│   ├── sota/           # SOTA comparison experiments
│   │   ├── asib_cl/    # ASIB-CL experiments
│   │   ├── finetune/   # Fine-tuning experiments
│   │   ├── ewc/        # EWC experiments
│   │   ├── lwf/        # LwF experiments
│   │   ├── icarl/      # iCaRL experiments
│   │   └── der/        # DER experiments
│   └── finetune/       # Fine-tuning experiments
│       ├── {model_name}/
│       │   ├── logs/   # Training logs
│       │   └── results/ # Training results
├── checkpoints/        # 💾 Model checkpoints
│   ├── teachers/       # Teacher model checkpoints
│   ├── students/       # Student model checkpoints
│   └── finetuned/      # Fine-tuned model checkpoints
├── analysis/           # 📊 Analysis and visualization results
│   ├── plots/          # Graphs and charts
│   ├── reports/        # Analysis reports
│   ├── sensitivity_analysis/     # Sensitivity analysis
│   ├── overlap_analysis/         # Overlap analysis
│   ├── teacher_adaptation/       # Teacher adaptation analysis
│   ├── cccp_stability/           # CCCP stability analysis
│   ├── information_plane/        # Information plane analysis
│   ├── pf_efficiency/            # Partial freezing efficiency
│   └── beta_sensitivity/         # Beta sensitivity analysis
├── PyCIL/              # PyCIL framework integration
└── .github/            # GitHub workflows
```

## 📋 **Directory Structure Overview**

### 🧪 **`experiments/` - All Experiments (Integrated)**
- **Unified Structure**: Each experiment has both `logs/` and `results/` in one place
- **Easy Navigation**: Find all experiment data in one location
- **Consistent Organization**: All experiments follow the same structure

**Example Structure:**
```
experiments/
├── test/                    # Test experiments
│   ├── logs/               # Test execution logs
│   └── results/            # Test results and summaries
├── ablation/baseline/       # Baseline ablation study
│   ├── logs/               # Training logs
│   └── results/            # Performance results
├── sota/asib_cl/           # ASIB-CL SOTA comparison
│   ├── logs/               # Training logs
│   └── results/            # Comparison results
└── finetune/convnext_s_cifar100/  # Fine-tuning experiment
    ├── logs/               # Training logs
    └── results/            # Fine-tuning results
```

### 💾 **`checkpoints/` - Model Storage**
- **teachers/**: Pre-trained teacher model checkpoints
- **students/**: Student model checkpoints during training
- **finetuned/**: Fine-tuned model checkpoints

### 📊 **`analysis/` - Analysis & Visualization**
- **plots/**: Graphs, charts, and visualizations
- **reports/**: Analysis reports and summaries
- **{analysis_type}/**: Specific analysis results (sensitivity, overlap, etc.)

## 🚀 **Usage Examples**

### Running Experiments
```bash
# Run comprehensive test suite
sbatch run/run_test.sh
# Results: experiments/test/logs/ + experiments/test/results/

# Run ablation study
sbatch run/run_asib_ablation_study.sh
# Results: experiments/ablation/baseline/logs/ + experiments/ablation/baseline/results/

# Run SOTA comparison
sbatch run/run_asib_sota_comparison.sh
# Results: experiments/sota/asib_cl/logs/ + experiments/sota/asib_cl/results/

# Run fine-tuning
sbatch run/run_finetune_single.sh convnext_s_cifar100
# Results: experiments/finetune/convnext_s_cifar100/logs/ + experiments/finetune/convnext_s_cifar100/results/
```

### Checking Results
```bash
# View test results
cat experiments/test/results/summary.log

# Check ablation study logs
ls experiments/ablation/baseline/logs/

# View SOTA comparison results
ls experiments/sota/asib_cl/results/

# Access analysis results
ls analysis/plots/
ls analysis/reports/
```

## 🔧 Key Features

### 🎯 **ASIB Method**
- **Multi-Stage Distillation**: Teacher ↔ Student updates in phases
- **Information‑Bottleneck IB_MBM (IB_MBM)**: Fuses teacher features using IB principles (VIB is applied inside IB_MBM; head is a plain MLP)
- **Adaptive Synergy**: Creates synergistic knowledge from multiple teachers

### 🧊 **Partial Freezing**
- **Efficient Training**: Freeze backbone, adapt BN/Heads/IB_MBM
- **Flexible Levels**: -1 (no freeze) to N (freeze N blocks)
- **Auto-Scheduling**: Stage-wise freeze level progression

### 🎨 **Multiple KD Methods**
- **ASIB**: Our proposed method (default)
- **Vanilla KD**: Traditional knowledge distillation
- **FitNet**: Feature-level distillation
- **CRD**: Contrastive representation distillation
- **AT**: Attention transfer
- **DKD**: Decoupled knowledge distillation

### 📊 **Supported Datasets**
- **CIFAR-100**: 100-class image classification
- **ImageNet-32**: Downsampled ImageNet
- **Custom**: Easy to extend

## 🛠️ Configuration

### Basic Configuration

```yaml
# configs/experiment/ablation_baseline.yaml
defaults:
  - /base
  - /model/teacher@teacher1: convnext_s_teacher
  - /model/teacher@teacher2: resnet152_teacher
  - /model/student: resnet50_scratch_student
  - _self_

# Model settings
teacher1_ckpt: checkpoints/convnext_s_cifar32.pth
teacher2_ckpt: checkpoints/resnet152_cifar32.pth

# Training settings
student_lr: 0.1
batch_size: 64
num_epochs: 200

# ASIB settings
use_ib: false
use_cccp: false
use_partial_freeze: false
```

## 🧪 Experiment Plan

### Phase 1: Ablation Study
```bash
# Run complete ablation study (5 experiments)
sbatch run/run_ablation_study.sh
```

**Experiments:**
1. **Baseline**: IB_MBM + E2E + Fixed Teachers
2. **+IB**: Information Bottleneck
3. **+CCCP**: Stage-wise learning
4. **+T-Adapt**: Teacher Adaptation
5. **ASIB Full**: Progressive Partial Freezing

### Phase 2: SOTA Comparison
```bash
# Run SOTA comparison experiments
python main.py --config-name experiment/sota_scenario_a
```

### Phase 3: Overlap Analysis
```bash
# Run overlap analysis experiments
python main.py --config-name experiment/overlap_100
```

## 📊 Analysis Tools

### Beta Sensitivity Analysis
```bash
python scripts/analysis/beta_sensitivity.py
```

### Comprehensive Analysis
```bash
python scripts/analysis/comprehensive_analysis.py
```

**Analysis Components:**
- Information Plane Analysis (IB Theory Connection)
- CCCP Stability Analysis (Learning Curve Comparison)
- Teacher Adaptation Analysis (Performance Preservation)
- PF Efficiency Analysis (Memory & Time Optimization)

## 🧪 Testing

### **Comprehensive Test Suite**
```bash
# Run all tests (recommended - parallel execution)
sbatch run/run_test.sh

# Run all tests directly
python -m pytest tests/ -v

# Run specific test category
python -m pytest tests/test_asib_*.py -v  # Core ASIB tests
python -m pytest tests/test_pycil_*.py -v # PyCIL integration tests

# Run specific test file
python -m pytest tests/test_integration.py -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

### **Test Categories**
- **Core ASIB Tests**: `test_asib_cl.py`, `test_asib_step.py`
- **PyCIL Integration**: `test_pycil_integration.py`, `test_pycil_models.py`
- **Data & Utils**: `test_data.py`, `test_utils.py`, `test_core.py`
- **Models**: `test_models.py`, `test_models_advanced.py`
- **Configs & Experiments**: `test_configs.py`, `test_experiment_*.py`
- **Framework Robustness**: `test_framework_robustness.py`, `test_error_prevention.py`
- **Main Integration**: `test_main.py`, `test_main_training.py`, `test_training_simple.py`

### **Test Results**
- **Summary**: `experiments/test_results/summary.log`
- **Individual Logs**: `experiments/test_results/*.log`
- **Coverage Report**: `htmlcov/index.html`

## 📚 Documentation

- **API Documentation**: `docs/API.md`
- **Configuration Guide**: `docs/CONFIGURATION.md`
- **Installation Guide**: `docs/INSTALLATION.md`
- **Experiment Reports**: `docs/reports/`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Information Bottleneck theory
- Multi-teacher knowledge distillation
- Progressive partial freezing techniques
