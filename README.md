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
# Run default experiment
sbatch run/run.sh

# Run with custom config
python main.py experiment=res152_convnext_effi

# Run sensitivity analysis
sbatch run/run_sensitivity.sh

# Run overlap analysis
sbatch run/run_overlap.sh

# Run teacher fine-tuning
sbatch run/run_finetune_clean.sh
```

## 📁 Project Structure

```
ASIB-KD/
├── main.py                 # Main training script (simplified)
├── eval.py                 # Model evaluation
├── core/                   # Core functionality
│   ├── builder.py         # Model creation utilities
│   ├── trainer.py         # Training logic
│   └── utils.py           # Configuration utilities
├── configs/               # Configuration files
│   ├── base.yaml         # Base configuration
│   ├── method/asib.yaml  # ASIB method config
│   ├── experiment/       # Experiment configs
│   └── model/           # Model configs
├── utils/                # Utilities (reorganized)
│   ├── logging/         # Logging utilities
│   ├── data/           # Data utilities
│   ├── training/       # Training utilities
│   └── common/         # Common utilities
├── modules/             # Training modules
├── methods/            # Distillation methods
├── models/             # Model definitions
├── scripts/            # Utility scripts (organized)
│   ├── analysis/       # Analysis scripts
│   ├── training/       # Training scripts
│   └── setup/          # Setup scripts
└── run/                # SLURM execution scripts
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
# configs/base.yaml
method: asib
num_stages: 4
teacher_lr: 0.0002
student_lr: 0.001
use_ib: true
ib_beta: 0.001
use_cccp: true  # Concave-Convex Procedure
```

### Model Configuration

```yaml
# configs/experiment/res152_convnext_effi.yaml
teacher1:
  model:
    teacher:
      name: convnext_l_teacher
teacher2:
  model:
    teacher:
      name: efficientnet_l2_teacher
model:
  student:
    model:
      student:
        name: resnet152_pretrain_student
```

### Training Configuration

```yaml
# Partial freezing
use_partial_freeze: true
student_freeze_schedule: [-1, 2, 1, 0]  # Stage-wise freeze levels

# MBM settings
mbm_query_dim: 1024
mbm_out_dim: 1024
mbm_n_head: 8

# Loss weights
ce_alpha: 0.3
kd_alpha: 0.7
ib_beta: 0.001
```

## 🚀 Usage Examples

### 1. Standard Training

```bash
# Run with default settings
python main.py

# Run with custom experiment
python main.py --config-name experiment/res152_convnext_effi

# Override parameters
python main.py student_lr=0.0005 ib_beta=0.01
```

### 2. Teacher Fine-tuning

```bash
# Fine-tune teachers before distillation
python scripts/fine_tuning.py --config-name base \
  +teacher_type=resnet152 +finetune_epochs=100
```

### 3. Student Baseline

```bash
# Train student alone for baseline
python scripts/train_student_baseline.py --config-name base
```

### 4. Evaluation

```bash
# Evaluate single model
python eval.py +eval_mode=single +ckpt_path=./results/student_final.pth

# Evaluate synergy model
python eval.py +eval_mode=synergy
```

## 🔬 Advanced Features

### CCCP (Concave-Convex Procedure)

Enable CCCP for stable teacher updates:

```yaml
use_cccp: true
tau: 4.0  # Temperature for CCCP
```

### Continual Learning

```bash
# Enable continual learning mode
python main.py cl_mode=true num_tasks=5
```

### Data Augmentation

```bash
# Enable/disable augmentation
python main.py data_aug=true
python main.py data_aug=false

# MixUp and CutMix
python main.py mixup_alpha=0.2 cutmix_alpha_distill=0.3
```

### Automatic Mixed Precision

```yaml
use_amp: true
amp_dtype: float16  # or bfloat16
```

## 📈 Results

Results are saved in the `outputs/` directory:

```
outputs/
├── experiment_name/
│   ├── train.log          # Training logs
│   ├── metrics.csv        # Performance metrics
│   ├── config.yaml        # Final configuration
│   └── checkpoints/       # Model checkpoints
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_asib_step.py
```

## 📚 API Reference

### Core Functions

```python
from core import (
    create_student_by_name,
    create_teacher_by_name,
    run_training_stages,
    run_continual_learning
)

# Create models
student = create_student_by_name("resnet152_pretrain_student")
teacher1 = create_teacher_by_name("convnext_l_teacher")

# Run training
final_acc = run_training_stages(
    teacher_wrappers=[teacher1, teacher2],
    mbm=mbm,
    synergy_head=synergy_head,
    student_model=student,
    train_loader=train_loader,
    test_loader=test_loader,
    cfg=cfg,
    exp_logger=exp_logger,
    num_stages=4
)
```

### Configuration Utilities

```python
from core.utils import (
    setup_partial_freeze_schedule,
    auto_set_mbm_query_dim,
    cast_numeric_configs
)

# Setup training configuration
setup_partial_freeze_schedule(cfg, num_stages)
auto_set_mbm_query_dim(student_model, cfg)
cast_numeric_configs(cfg)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

If you use this framework in your research, please cite:

```bibtex
@misc{asib-kd,
  title={ASIB: Adaptive Synergy Information-Bottleneck Knowledge Distillation},
  author={Suyoung Yang},
  year={2024},
  howpublished={\url{https://github.com/YourName/ASIB-KD}}
}
```

## 📞 Contact

- **Email**: suyoung425@yonsei.ac.kr
- **GitHub Issues**: [Create an issue](https://github.com/YourName/ASIB-KD/issues)

---

## 🔄 Recent Updates

### v2.0.0 (Latest)
- ✅ **Code Refactoring**: Modular structure with `core/` package
- ✅ **Utils Reorganization**: Functional subfolders in `utils/`
- ✅ **ASMB → ASIB**: Renamed to reflect Information-Bottleneck approach
- ✅ **CCCP Integration**: Added Concave-Convex Procedure for stability
- ✅ **Main.py Simplification**: Reduced from 955 to 329 lines (66% reduction)

### v1.0.0
- 🎯 Initial release with ASMB framework
- 🧊 Partial freezing mechanism
- 📊 Multi-teacher distillation
- 🎨 Multiple KD methods support
