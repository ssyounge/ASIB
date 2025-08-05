# ASIB Knowledge Distillation Framework

**ASIB** (Adaptive Synergy Information-Bottleneck) is a multi-stage knowledge distillation framework that uses Information-Bottleneck Manifold Bridging Module (IB-MBM) to create synergistic knowledge from multiple teachers.

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/YourName/ASIB-KD.git
cd ASIB-KD

# Create conda environment
conda env create -f environment.yml
conda activate asib

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run complete ablation study
sbatch run/run_ablation_study.sh

# Run teacher fine-tuning
sbatch run/run_finetune_single.sh convnext_s_cifar32

# Run beta sensitivity analysis
python scripts/analysis/beta_sensitivity.py

# Run comprehensive analysis
python scripts/analysis/comprehensive_analysis.py
```

## 📁 Project Structure

```
ASIB-KD/
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
│   ├── run_ablation_study.sh
│   ├── run_finetune_single.sh
│   └── run_finetune_all_teachers.sh
├── models/             # Model definitions
├── data/               # Data loading utilities
├── utils/              # Utility modules
├── core/               # Core functionality
├── methods/            # Distillation methods
├── modules/            # Training modules
├── tests/              # Test files
├── outputs/            # Experiment outputs
├── checkpoints/        # Model checkpoints
├── experiments/        # Experiment results
└── .github/            # GitHub workflows
```

## 🔧 Key Features

### 🎯 **ASIB Method**
- **Multi-Stage Distillation**: Teacher ↔ Student updates in phases
- **Information-Bottleneck MBM**: Fuses teacher features using IB principles
- **Adaptive Synergy**: Creates synergistic knowledge from multiple teachers

### 🧊 **Partial Freezing**
- **Efficient Training**: Freeze backbone, adapt BN/Heads/MBM
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
1. **Baseline**: MBM + E2E + Fixed Teachers
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

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_integration.py

# Run with coverage
python -m pytest tests/ --cov=.
```

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
