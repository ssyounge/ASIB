# API Reference

## Core Module

### Model Building

#### `core.builder.build_model`

```python
def build_model(name: str, **kwargs) -> torch.nn.Module
```

Build a model from the registry.

**Parameters:**
- `name` (str): Model name in registry
- `**kwargs`: Additional arguments passed to model constructor

**Returns:**
- `torch.nn.Module`: Constructed model

**Example:**
```python
from core.builder import build_model

model = build_model("resnet152_pretrain_student", num_classes=100)
```

#### `core.builder.create_student_by_name`

```python
def create_student_by_name(
    student_name: str,
    pretrained: bool = True,
    small_input: bool = False,
    num_classes: int = 100,
    cfg: Optional[dict] = None,
) -> torch.nn.Module
```

Create a student model from registry.

**Parameters:**
- `student_name` (str): Student model name
- `pretrained` (bool): Use pretrained weights
- `small_input` (bool): Use small input adaptation
- `num_classes` (int): Number of classes
- `cfg` (Optional[dict]): Configuration dictionary

**Returns:**
- `torch.nn.Module`: Student model

**Example:**
```python
from core.builder import create_student_by_name

student = create_student_by_name(
    "resnet152_pretrain_student",
    pretrained=True,
    num_classes=100
)
```

#### `core.builder.create_teacher_by_name`

```python
def create_teacher_by_name(
    teacher_name: str,
    num_classes: int = 100,
    pretrained: bool = True,
    small_input: bool = False,
    cfg: Optional[dict] = None,
) -> torch.nn.Module
```

Create a teacher model from registry.

**Parameters:**
- `teacher_name` (str): Teacher model name
- `num_classes` (int): Number of classes
- `pretrained` (bool): Use pretrained weights
- `small_input` (bool): Use small input adaptation
- `cfg` (Optional[dict]): Configuration dictionary

**Returns:**
- `torch.nn.Module`: Teacher model

### Training

#### `core.trainer.create_optimizers_and_schedulers`

```python
def create_optimizers_and_schedulers(
    teacher_wrappers: List[torch.nn.Module],
    mbm: torch.nn.Module,
    synergy_head: torch.nn.Module,
    student_model: torch.nn.Module,
    cfg: Dict[str, Any],
    num_stages: int,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, 
           torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]
```

Create optimizers and schedulers for training.

**Parameters:**
- `teacher_wrappers` (List[torch.nn.Module]): List of teacher models
- `mbm` (torch.nn.Module): IB‑MBM (Information‑Bottleneck Manifold Bridging Module)
- `synergy_head` (torch.nn.Module): Synergy head
- `student_model` (torch.nn.Module): Student model
- `cfg` (Dict[str, Any]): Configuration dictionary
- `num_stages` (int): Number of training stages

**Returns:**
- `Tuple`: (teacher_optimizer, teacher_scheduler, student_optimizer, student_scheduler)

**Example:**
```python
from core.trainer import create_optimizers_and_schedulers

teacher_opt, teacher_sched, student_opt, student_sched = \
    create_optimizers_and_schedulers(
        teacher_wrappers=[teacher1, teacher2],
        mbm=mbm,
        synergy_head=synergy_head,
        student_model=student,
        cfg=cfg,
        num_stages=4
    )
```

#### `core.trainer.run_training_stages`

```python
def run_training_stages(
    teacher_wrappers: List[torch.nn.Module],
    mbm: torch.nn.Module,
    synergy_head: torch.nn.Module,
    student_model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    cfg: Dict[str, Any],
    exp_logger: ExperimentLogger,
    num_stages: int,
) -> float
```

Run the main training stages.

**Parameters:**
- `teacher_wrappers` (List[torch.nn.Module]): List of teacher models
- `mbm` (torch.nn.Module): IB‑MBM
- `synergy_head` (torch.nn.Module): Synergy head
- `student_model` (torch.nn.Module): Student model
- `train_loader` (DataLoader): Training data loader
- `test_loader` (DataLoader): Test data loader
- `cfg` (Dict[str, Any]): Configuration dictionary
- `exp_logger` (ExperimentLogger): Experiment logger
- `num_stages` (int): Number of training stages

**Returns:**
- `float`: Final student accuracy

**Example:**
```python
from core.trainer import run_training_stages

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

### Utilities

#### `core.utils.setup_partial_freeze_schedule`

```python
def setup_partial_freeze_schedule(cfg: Dict[str, Any], num_stages: int) -> None
```

Setup partial freeze schedule.

**Parameters:**
- `cfg` (Dict[str, Any]): Configuration dictionary
- `num_stages` (int): Number of training stages

**Example:**
```python
from core.utils import setup_partial_freeze_schedule

setup_partial_freeze_schedule(cfg, num_stages=4)
```

#### `core.utils.auto_set_mbm_query_dim`

```python
def auto_set_mbm_query_dim(student_model: torch.nn.Module, cfg: Dict[str, Any]) -> None
```

Auto-set MBM query dimension based on student model.

**Parameters:**
- `student_model` (torch.nn.Module): Student model
- `cfg` (Dict[str, Any]): Configuration dictionary

**Example:**
```python
from core.utils import auto_set_mbm_query_dim

auto_set_mbm_query_dim(student_model, cfg)
```

## Utils Module

### Logging

#### `utils.logging.init_logger`

```python
def init_logger(level: str = "INFO") -> None
```

Initialize logging configuration.

**Parameters:**
- `level` (str): Logging level

**Example:**
```python
from utils.logging import init_logger

init_logger("INFO")
```

#### `utils.logging.get_logger`

```python
def get_logger(
    exp_dir: str,
    level: str = "INFO",
    stream_level: str = "INFO",
) -> logging.Logger
```

Get configured logger.

**Parameters:**
- `exp_dir` (str): Experiment directory
- `level` (str): File logging level
- `stream_level` (str): Stream logging level

**Returns:**
- `logging.Logger`: Configured logger

**Example:**
```python
from utils.logging import get_logger

logger = get_logger("outputs/experiment", level="INFO")
```

#### `utils.logging.ExperimentLogger`

```python
class ExperimentLogger:
    def __init__(self, cfg: Dict[str, Any], exp_name: str = "experiment"):
        """
        Initialize experiment logger.
        
        Parameters:
        - cfg: Configuration dictionary
        - exp_name: Experiment name
        """
    
    def update_metric(self, key: str, value: Any) -> None:
        """Update metric value."""
    
    def save_results(self) -> None:
        """Save results to file."""
    
    def finalize(self) -> None:
        """Finalize logging."""
```

### Data

#### `utils.data.get_split_cifar100_loaders`

```python
def get_split_cifar100_loaders(
    num_tasks: int = 5,
    batch_size: int = 128,
    augment: bool = True,
    root: str = "./data",
) -> List[Tuple[DataLoader, DataLoader]]
```

Get split CIFAR-100 data loaders for continual learning.

**Parameters:**
- `num_tasks` (int): Number of tasks
- `batch_size` (int): Batch size
- `augment` (bool): Use data augmentation
- `root` (str): Data root directory

**Returns:**
- `List[Tuple[DataLoader, DataLoader]]`: List of (train_loader, val_loader) pairs

**Example:**
```python
from utils.data import get_split_cifar100_loaders

task_loaders = get_split_cifar100_loaders(
    num_tasks=5,
    batch_size=64,
    augment=True
)
```

### Training

#### `utils.training.compute_accuracy`

```python
def compute_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float
```

Compute classification accuracy.

**Parameters:**
- `outputs` (torch.Tensor): Model outputs
- `targets` (torch.Tensor): Ground truth labels

**Returns:**
- `float`: Accuracy percentage

**Example:**
```python
from utils.training import compute_accuracy

acc = compute_accuracy(outputs, targets)
```

#### `utils.training.get_tau`

```python
def get_tau(epoch: int, cfg: Dict[str, Any]) -> float
```

Get temperature for knowledge distillation.

**Parameters:**
- `epoch` (int): Current epoch
- `cfg` (Dict[str, Any]): Configuration dictionary

**Returns:**
- `float`: Temperature value

**Example:**
```python
from utils.training import get_tau

tau = get_tau(epoch=10, cfg=cfg)
```

#### `utils.training.get_beta`

```python
def get_beta(epoch: int, cfg: Dict[str, Any]) -> float
```

Get beta value for Information Bottleneck.

**Parameters:**
- `epoch` (int): Current epoch
- `cfg` (Dict[str, Any]): Configuration dictionary

**Returns:**
- `float`: Beta value

**Example:**
```python
from utils.training import get_beta

beta = get_beta(epoch=10, cfg=cfg)
```

### Common

#### `utils.common.set_random_seed`

```python
def set_random_seed(seed: int, deterministic: bool = True) -> None
```

Set random seed for reproducibility.

**Parameters:**
- `seed` (int): Random seed
- `deterministic` (bool): Use deterministic algorithms

**Example:**
```python
from utils.common import set_random_seed

set_random_seed(42, deterministic=True)
```

#### `utils.common.count_trainable_parameters`

```python
def count_trainable_parameters(model: torch.nn.Module) -> int
```

Count trainable parameters in model.

**Parameters:**
- `model` (torch.nn.Module): PyTorch model

**Returns:**
- `int`: Number of trainable parameters

**Example:**
```python
from utils.common import count_trainable_parameters

num_params = count_trainable_parameters(model)
```

#### `utils.common.smart_tqdm`

```python
def smart_tqdm(iterable, **kwargs) -> tqdm
```

Smart progress bar that hides when not in TTY.

**Parameters:**
- `iterable`: Iterable to wrap
- `**kwargs`: Additional tqdm arguments

**Returns:**
- `tqdm`: Progress bar object

**Example:**
```python
from utils.common import smart_tqdm

for i in smart_tqdm(range(100)):
    # Process
    pass
```

## Modules

### Trainer Student

#### `modules.trainer_student.student_distillation_update`

```python
def student_distillation_update(
    teacher_wrappers: List[torch.nn.Module],
    mbm: torch.nn.Module,
    synergy_head: torch.nn.Module,
    student_model: torch.nn.Module,
    trainloader: DataLoader,
    testloader: DataLoader,
    cfg: Dict[str, Any],
    logger: ExperimentLogger,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    global_ep: int = 0,
) -> float
```

Perform student distillation update.

**Parameters:**
- `teacher_wrappers` (List[torch.nn.Module]): List of teacher models
- `mbm` (torch.nn.Module): Manifold Bridging Module
- `synergy_head` (torch.nn.Module): Synergy head
- `student_model` (torch.nn.Module): Student model
- `trainloader` (DataLoader): Training data loader
- `testloader` (DataLoader): Test data loader
- `cfg` (Dict[str, Any]): Configuration dictionary
- `logger` (ExperimentLogger): Experiment logger
- `optimizer` (Optimizer): Student optimizer
- `scheduler` (_LRScheduler): Student scheduler
- `global_ep` (int): Global epoch counter

**Returns:**
- `float`: Final student accuracy

### Trainer Teacher

#### `modules.trainer_teacher.teacher_adaptive_update`

```python
def teacher_adaptive_update(
    teacher_wrappers: List[torch.nn.Module],
    mbm: torch.nn.Module,
    synergy_head: torch.nn.Module,
    student_model: torch.nn.Module,
    trainloader: DataLoader,
    testloader: DataLoader,
    cfg: Dict[str, Any],
    logger: ExperimentLogger,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    global_ep: int = 0,
) -> float
```

Perform teacher adaptive update.

**Parameters:**
- `teacher_wrappers` (List[torch.nn.Module]): List of teacher models
- `mbm` (torch.nn.Module): Manifold Bridging Module
- `synergy_head` (torch.nn.Module): Synergy head
- `student_model` (torch.nn.Module): Student model
- `trainloader` (DataLoader): Training data loader
- `testloader` (DataLoader): Test data loader
- `cfg` (Dict[str, Any]): Configuration dictionary
- `logger` (ExperimentLogger): Experiment logger
- `optimizer` (Optimizer): Teacher optimizer
- `scheduler` (_LRScheduler): Teacher scheduler
- `global_ep` (int): Global epoch counter

**Returns:**
- `float`: Teacher accuracy

### Losses

#### `modules.losses.ib_loss`

```python
def ib_loss(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor
```

Compute Information Bottleneck loss.

**Parameters:**
- `mu` (torch.Tensor): Mean of latent distribution
- `logvar` (torch.Tensor): Log variance of latent distribution
- `beta` (float): Beta weight for KL divergence

**Returns:**
- `torch.Tensor`: IB loss value

**Example:**
```python
from modules.losses import ib_loss

loss = ib_loss(mu, logvar, beta=0.001)
```

#### `modules.losses.kd_loss_fn`

```python
def kd_loss_fn(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 4.0,
) -> torch.Tensor
```

Compute knowledge distillation loss.

**Parameters:**
- `student_logits` (torch.Tensor): Student model logits
- `teacher_logits` (torch.Tensor): Teacher model logits
- `temperature` (float): Temperature for softmax

**Returns:**
- `torch.Tensor`: KD loss value

**Example:**
```python
from modules.losses import kd_loss_fn

loss = kd_loss_fn(student_logits, teacher_logits, temperature=4.0)
```

## Methods

### ASIB Distiller

#### `methods.asib.ASIBDistiller`

```python
class ASIBDistiller(nn.Module):
    def __init__(
        self,
        teacher1: nn.Module,
        teacher2: nn.Module,
        student: nn.Module,
        mbm: nn.Module,
        synergy_head: nn.Module,
        cfg: Dict[str, Any],
    ):
        """
        Initialize ASIB distiller.
        
        Parameters:
        - teacher1: First teacher model
        - teacher2: Second teacher model
        - student: Student model
        - mbm: Manifold Bridging Module
        - synergy_head: Synergy head
        - cfg: Configuration dictionary
        """
    
    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        epoch: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Parameters:
        - x: Input images
        - labels: Ground truth labels
        - epoch: Current epoch
        
        Returns:
        - Dict[str, torch.Tensor]: Loss dictionary
        """
```

## Configuration

### Base Configuration

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

# ASIB specific
use_ib: true
ib_beta: 0.001
use_cccp: true
tau: 4.0

# Partial freezing
use_partial_freeze: true
student_freeze_schedule: [-1, 2, 1, 0]

# IB‑MBM settings
ib_mbm_query_dim: 1024
ib_mbm_out_dim: 1024
ib_mbm_n_head: 8
```

### Method Configuration

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

# CCCP
use_cccp: true
tau: 4.0

# Disagreement weighting
use_disagree_weight: true
disagree_mode: both_wrong
disagree_lambda_high: 1.5
disagree_lambda_low: 1.0
```

## Examples

### Basic Training

```python
import hydra
from omegaconf import DictConfig
from core import run_training_stages
from utils.logging import ExperimentLogger

@hydra.main(config_path="configs", config_name="base")
def main(cfg: DictConfig):
    # Setup
    exp_logger = ExperimentLogger(cfg, exp_name="asib")
    
    # Create models
    from core.builder import create_student_by_name, create_teacher_by_name
    student = create_student_by_name("resnet152_pretrain_student")
    teacher1 = create_teacher_by_name("convnext_l_teacher")
    teacher2 = create_teacher_by_name("efficientnet_l2_teacher")
    
# Create IB‑MBM and synergy head
from models import build_ib_mbm_from_teachers as build_from_teachers
mbm, synergy_head = build_from_teachers([teacher1, teacher2], cfg)
    
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
    
    print(f"Final accuracy: {final_acc:.2f}%")

if __name__ == "__main__":
    main()
```

### Custom Loss Function

```python
from modules.losses import ib_loss, kd_loss_fn
import torch.nn.functional as F

def custom_loss(student_logits, teacher_logits, labels, mu, logvar, cfg):
    # Cross-entropy loss
    ce_loss = F.cross_entropy(student_logits, labels)
    
    # Knowledge distillation loss
    kd_loss = kd_loss_fn(student_logits, teacher_logits, temperature=4.0)
    
    # Information bottleneck loss
    ib_loss_val = ib_loss(mu, logvar, beta=cfg.ib_beta)
    
    # Total loss
    total_loss = (
        cfg.ce_alpha * ce_loss +
        cfg.kd_alpha * kd_loss +
        cfg.ib_beta * ib_loss_val
    )
    
    return total_loss, {
        'ce_loss': ce_loss.item(),
        'kd_loss': kd_loss.item(),
        'ib_loss': ib_loss_val.item(),
        'total_loss': total_loss.item()
    }
```

### Custom Dataset

```python
from torch.utils.data import Dataset
from utils.data import ClassInfoMixin

class CustomDataset(ClassInfoMixin, Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.classes = ['class1', 'class2', 'class3']  # Define your classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image, label = self.data[idx], self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
``` 