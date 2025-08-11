# Configuration Guide

## Overview

ASIB uses Hydra for configuration management. All configurations are stored in YAML files under the `configs/` directory.

## Configuration Structure

```
configs/
├── base.yaml              # Base configuration
├── method/
│   └── asib.yaml         # ASIB method configuration
├── experiment/
│   ├── res152_convnext_effi.yaml
│   └── ...
├── model/
│   ├── teacher/
│   │   ├── convnext_l.yaml
│   │   └── efficientnet_l2.yaml
│   └── student/
│       └── resnet152_pretrain.yaml
└── dataset/
    └── cifar100.yaml
```

## Base Configuration

### Core Settings

```yaml
# configs/base.yaml
defaults:
  - method: asib
  - dataset: cifar100
  - _self_

# Training settings
num_stages: 4
teacher_lr: 0.0002
student_lr: 0.001
teacher_weight_decay: 0.0001
student_weight_decay: 0.0003

# Device and optimization
device: cuda
use_amp: true
amp_dtype: float16
grad_clip_norm: 1.0

# ASIB specific
use_ib: true
ib_beta: 0.001
ib_beta_warmup_epochs: 3
use_cccp: true
tau: 4.0

# Partial freezing
use_partial_freeze: true
student_freeze_schedule: [-1, 2, 1, 0]
teacher1_freeze_level: 1
teacher2_freeze_level: 1

# IB‑MBM settings
ib_mbm_query_dim: 1024
ib_mbm_out_dim: 1024
ib_mbm_n_head: 8
ib_mbm_dropout: 0.0
ib_mbm_learnable_q: false

# Loss weights
ce_alpha: 0.3
kd_alpha: 0.7
kd_ens_alpha: 0.7

# Data augmentation
data_aug: true
mixup_alpha: 0.0
cutmix_alpha_distill: 0.3
```

## Method Configuration

### ASIB Method

```yaml
# configs/method/asib.yaml
name: asib
ce_alpha: 0.3
kd_alpha: 0.0
kd_ens_alpha: 0.7

# Information Bottleneck
use_ib: true
ib_beta: 0.001
ib_beta_warmup_epochs: 3

# CCCP for teacher stability
use_cccp: true
tau: 4.0

# Disagreement weighting
use_disagree_weight: true
disagree_mode: both_wrong
disagree_lambda_high: 1.5
disagree_lambda_low: 1.0
```

### Other Methods

```yaml
# configs/method/vanilla_kd.yaml
name: vanilla_kd
ce_alpha: 0.3
kd_alpha: 0.7
temperature: 4.0

# configs/method/fitnet.yaml
name: fitnet
ce_alpha: 0.3
feat_kd_alpha: 0.7
feat_kd_key: feat_2d
```

## Model Configuration

### Teacher Models

```yaml
# configs/model/teacher/convnext_l.yaml
name: convnext_l_teacher
pretrained: true
num_classes: 100
small_input: true

# Fine-tuning settings
finetune_epochs: 100
finetune_lr: 0.0005
finetune_use_cutmix: true
finetune_alpha: 1.0

# Partial freezing
freeze_level: 1
freeze_bn: true
use_adapter: false
```

### Student Models

```yaml
# configs/model/student/resnet152_pretrain.yaml
name: resnet152_pretrain_student
pretrained: true
num_classes: 100
small_input: true
use_adapter: true

# Partial freezing
freeze_level: -1
freeze_bn: false
```

## Experiment Configuration

### Example Experiment

```yaml
# configs/experiment/res152_convnext_effi.yaml
defaults:
  - base
  - _self_

# Model selection
teacher1:
  model:
    teacher:
      name: convnext_l_teacher
      pretrained: true

teacher2:
  model:
    teacher:
      name: efficientnet_l2_teacher
      pretrained: true

model:
  student:
    model:
      student:
        name: resnet152_pretrain_student
        pretrained: true
        use_adapter: true

# Checkpoints
teacher1_ckpt: checkpoints/convnext_l_cifar100.pth
teacher2_ckpt: checkpoints/efficientnet_l2_cifar32.pth

# Experiment settings
results_dir: outputs/res152_convnext_effi
exp_id: res152_convnext_effi

# Hyperparameters
student_lr: 0.0005
ib_beta: 0.001
ib_mbm_query_dim: 1024
ib_mbm_out_dim: 1024
grad_clip_norm: 1.0
```

## Dataset Configuration

### CIFAR-100

```yaml
# configs/dataset/cifar100.yaml
dataset:
  name: cifar100
  root: ./data
  small_input: true
  data_aug: 1

batch_size: 64
num_workers: 2
```

### ImageNet-32

```yaml
# configs/dataset/imagenet32.yaml
dataset:
  name: imagenet32
  root: ./data
  small_input: true

batch_size: 32
num_workers: 4
```

## Command Line Overrides

### Basic Overrides

```bash
# Override learning rates
python main.py teacher_lr=0.0001 student_lr=0.0005

# Override model settings
python main.py model.student.model.student.name=resnet50_pretrain_student

# Override training parameters
python main.py num_stages=3 batch_size=32

# Override ASIB parameters
python main.py ib_beta=0.01 use_cccp=false
```

### Complex Overrides

```bash
# Override nested configurations
python main.py \
  teacher1.model.teacher.name=convnext_l_teacher \
  teacher2.model.teacher.name=efficientnet_l2_teacher \
  model.student.model.student.name=resnet152_pretrain_student

# Override multiple parameters
python main.py \
  student_lr=0.0005 \
  ib_beta=0.001 \
  ib_mbm_query_dim=1024 \
  grad_clip_norm=1.0 \
  use_amp=true
```

## Environment Variables

### Setting Environment Variables

```bash
# Dataset path
export DATA_ROOT=/path/to/datasets

# Weights & Biases
export WANDB_ENTITY=your_entity
export WANDB_PROJECT=your_project

# CUDA settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Using in Configuration

```yaml
# configs/base.yaml
dataset:
  root: ${oc.env:DATA_ROOT,./data}

# Weights & Biases
use_wandb: true
wandb_entity: ${oc.env:WANDB_ENTITY,}
wandb_project: ${oc.env:WANDB_PROJECT,asib}
```

## Advanced Configuration

### Custom Loss Functions

```yaml
# configs/method/custom.yaml
name: custom
ce_alpha: 0.3
kd_alpha: 0.7

# Custom loss weights
custom_loss_alpha: 0.1
custom_loss_weight: 1.0
```

### Multi-GPU Training

```yaml
# configs/base.yaml
device: cuda
num_gpus: 2
distributed: true
backend: nccl
```

### Continual Learning

```yaml
# configs/experiment/cl_experiment.yaml
cl_mode: true
num_tasks: 5
replay_ratio: 0.5
replay_capacity: 2000
lambda_ewc: 0.4
```

## Configuration Validation

### Schema Validation

```python
# configs/schema.yaml
@dataclass
class Config:
    num_stages: int = 4
    teacher_lr: float = 0.0002
    student_lr: float = 0.001
    use_ib: bool = True
    ib_beta: float = 0.001
```

### Validation Script

```python
# scripts/validate_config.py
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../configs", config_name="base")
def validate_config(cfg: DictConfig):
    # Validation logic
    assert cfg.num_stages > 0
    assert 0 < cfg.teacher_lr < 1
    assert 0 < cfg.student_lr < 1
    print("Configuration is valid!")

if __name__ == "__main__":
    validate_config()
```

## Best Practices

### 1. Use Defaults

```yaml
# configs/experiment/my_experiment.yaml
defaults:
  - base
  - method: asib
  - dataset: cifar100
  - _self_
```

### 2. Organize by Function

```yaml
# Group related settings
training:
  num_stages: 4
  teacher_lr: 0.0002
  student_lr: 0.001

model:
  teacher1:
    name: convnext_l_teacher
  teacher2:
    name: efficientnet_l2_teacher
  student:
    name: resnet152_pretrain_student
```

### 3. Use Variables

```yaml
# configs/base.yaml
# Define common values
&common
  num_classes: 100
  small_input: true
  pretrained: true

teacher1:
  <<: *common
  name: convnext_l_teacher

teacher2:
  <<: *common
  name: efficientnet_l2_teacher
```

### 4. Document Changes

```yaml
# configs/experiment/experiment_v2.yaml
# Version 2: Reduced learning rates for stability
# Changes:
# - teacher_lr: 0.0002 -> 0.0001
# - student_lr: 0.001 -> 0.0005
# - ib_beta: 0.01 -> 0.001

teacher_lr: 0.0001
student_lr: 0.0005
ib_beta: 0.001
```

## Troubleshooting

### Common Issues

1. **Configuration not found**:
   ```bash
   # Check config path
   python main.py --help
   
   # List available configs
   find configs/ -name "*.yaml"
   ```

2. **Type errors**:
   ```bash
   # Validate types
   python -c "from omegaconf import OmegaConf; cfg = OmegaConf.load('configs/base.yaml'); print(cfg)"
   ```

3. **Missing dependencies**:
   ```bash
   # Check Hydra installation
   python -c "import hydra; print(hydra.__version__)"
   ``` 