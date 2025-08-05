# Installation Guide

## Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU training)
- 16GB+ RAM
- 24GB+ GPU memory (for large models)

## Quick Installation

### 1. Clone Repository

```bash
git clone https://github.com/YourName/ASIB-KD.git
cd ASIB-KD
```

### 2. Create Conda Environment

```bash
# Create environment from YAML
conda env create -f environment.yml
conda activate asib

# Or create manually
conda create -n asib python=3.9
conda activate asib
```

### 3. Install Dependencies

```bash
# Install PyTorch (adjust version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

## Detailed Installation

### Environment Setup

The `environment.yml` file includes:

```yaml
name: asib
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - pytorch=2.0.0
  - torchvision=0.15.0
  - pytorch-cuda=11.8
  - cudatoolkit=11.8
  - pip
  - pip:
    - hydra-core>=1.3.0
    - omegaconf>=2.3.0
    - timm>=0.9.0
    - pandas>=1.5.0
    - matplotlib>=3.6.0
    - seaborn>=0.12.0
    - wandb>=0.15.0
    - pytest>=7.0.0
```

### Manual Installation

If you prefer manual installation:

```bash
# Create environment
conda create -n asib python=3.9
conda activate asib

# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install core dependencies
pip install hydra-core>=1.3.0
pip install omegaconf>=2.3.0
pip install timm>=0.9.0

# Install optional dependencies
pip install pandas matplotlib seaborn
pip install wandb  # for experiment tracking
pip install pytest  # for testing
```

### GPU Setup

#### CUDA Installation

1. **Check CUDA version**:
   ```bash
   nvidia-smi
   ```

2. **Install matching PyTorch version**:
   ```bash
   # For CUDA 11.8
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   
   # For CUDA 12.1
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
   ```

3. **Verify installation**:
   ```python
   import torch
   print(f"PyTorch version: {torch.__version__}")
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA version: {torch.version.cuda}")
   ```

#### Memory Optimization

For large models, consider:

```bash
# Set CUDA memory allocation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Or in Python
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
```

## Dataset Setup

### CIFAR-100

```bash
# Download automatically (default)
python main.py

# Or download manually
mkdir -p data
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar -xzf cifar-100-python.tar.gz -C data/
```

### ImageNet-32

```bash
# Download from official source
mkdir -p data/imagenet32
# Follow ImageNet download instructions
# Place in data/imagenet32/train/ and data/imagenet32/val/
```

### Custom Dataset

Create a custom dataset class:

```python
# data/custom.py
from torch.utils.data import Dataset
from utils.data import ClassInfoMixin

class CustomDataset(ClassInfoMixin, Dataset):
    def __init__(self, root, transform=None):
        # Implementation here
        pass
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Return (image, label)
        pass
```

## Checkpoint Setup

### Download Pre-trained Models

```bash
mkdir -p checkpoints

# Download teacher checkpoints
wget -O checkpoints/convnext_l_cifar100.pth <URL>
wget -O checkpoints/efficientnet_l2_cifar32.pth <URL>

# Or use your own fine-tuned models
cp /path/to/your/model.pth checkpoints/
```

### Model Registry

Add custom models to the registry:

```python
# models/common/registry.py
from models.common.base_wrapper import MODEL_REGISTRY

def register_custom_model():
    MODEL_REGISTRY['custom_model'] = CustomModelBuilder

# Or in configs/registry_map.yaml
custom_model: models.custom.CustomModelBuilder
```

## Verification

### Test Installation

```bash
# Run basic tests
pytest tests/ -v

# Test specific components
python -c "import torch; print('PyTorch OK')"
python -c "import hydra; print('Hydra OK')"
python -c "import timm; print('TIMM OK')"
```

### Quick Training Test

```bash
# Run a small experiment
python main.py \
  --config-name experiment/res152_convnext_effi \
  num_stages=1 \
  student_epochs_per_stage=1 \
  batch_size=8
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**:
   ```bash
   # Reduce batch size
   python main.py batch_size=16
   
   # Enable gradient checkpointing
   python main.py use_checkpointing=true
   ```

2. **Import errors**:
   ```bash
   # Check Python path
   python -c "import sys; print(sys.path)"
   
   # Reinstall in development mode
   pip install -e .
   ```

3. **Hydra configuration errors**:
   ```bash
   # Check config structure
   python main.py --help
   
   # Validate config
   python -c "from omegaconf import OmegaConf; OmegaConf.load('configs/base.yaml')"
   ```

### Performance Optimization

```bash
# Enable mixed precision
python main.py use_amp=true amp_dtype=float16

# Use multiple workers
python main.py num_workers=4

# Enable pin memory
python main.py pin_memory=true
```

## Development Setup

### Install in Development Mode

```bash
# Install package in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt
```

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Setup hooks
pre-commit install
```

### Code Formatting

```bash
# Format code
black .
isort .

# Check formatting
black --check .
isort --check-only .
``` 