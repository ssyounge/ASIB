# ASMB Knowledge Distillation Framework

This repository provides an **Adaptive Synergy Manifold Bridging (ASMB)** multi-stage knowledge distillation framework, along with various KD methods (FitNet, CRD, AT, DKD, VanillaKD, etc.) and a partial-freeze mechanism for large models.

---

## Features

- **Multi-Stage Distillation**: Teacher ↔ Student updates in a phased (block-wise) manner  
- **ASMB** (Adaptive Synergy Manifold Bridging): Uses a Manifold Bridging Module (MBM) to fuse two Teacher feature maps into synergy logits  
- **Partial Freeze**: Freeze backbone parameters, adapt BN/Heads/MBM for efficiency  
- **Multiple KD Methods**: FitNet, CRD, AT, DKD, VanillaKD, plus custom `asmb.py`
- **CIFAR-100 / ImageNet100** dataset support
- **Automatic class count detection**: number of classes is inferred from the
  training loader when using `ImageFolder` or CIFAR datasets
- **Configurable Data Augmentation**: toggle with `--data_aug` (1/0)
- **MixUp, CutMix & Label Smoothing**: enable with `--mixup_alpha`,
  `--cutmix_alpha_distill` and `--label_smoothing`
- **MBM Dropout**: set `mbm_dropout` in configs to add dropout within the
  Manifold Bridging Module
- **Gradient Clipping**: enable by setting `grad_clip_norm` (>0) in configs
- **Teacher Weight Decay Override**: set `--teacher_weight_decay` to control
  the Adam weight decay for teacher updates
- **Learnable MBM Query**: set `mbm_learnable_q: true` to use a global learnable
  token instead of the student feature as attention query
- **Feature-Level KD**: align student features with the synergy representation.
  A nonzero `feat_kd_alpha` enables feature alignment during teacher and student
  updates. Example `hparams.yaml` snippet:

  ```yaml
  feat_kd_alpha: 1.0
  feat_kd_key: "feat_2d"
  feat_kd_norm: "none"
  ```

  The same values can be overridden via CLI using
  `--feat_kd_alpha 1.0 --feat_kd_key feat_2d --feat_kd_norm none`.
- **Hybrid Guidance**: set `hybrid_beta` (>0) to blend vanilla KD from the
  average teacher logits with the default ASMB loss.
- **Custom MBM Query Dim**: `mbm_query_dim` controls the dimension of the
  student features used as the attention query in `LightweightAttnMBM`.
  When omitted or set to `0`, the script automatically falls back to the
  feature dimension reported by the student model (if available).
  The MBM output dimension (`mbm_out_dim`) now defaults to this student
  feature size as well.
- **Requires Student Features**: `LightweightAttnMBM` needs the student
  features as the attention query, so the student model must be provided
  when using this module.
  Common student feature dimensions are:

  | Student model | Feature dim |
  |---------------|-------------|
  | `convnext_tiny` | 768 |
*This minimal repository only ships `convnext_tiny` as a ready-made student.*
- **Swin Adapter Dim**: `swin_adapter_dim` sets the hidden size of the MLP
  adapter used by a custom `student_swin_adapter` (default `64`)
- **Student Projection Normalization**: set `proj_normalize` (default `true`) to
  apply L2 normalization on features before the student projection head
  Set `proj_use_bn: true` to insert a scale-free BatchNorm layer after the
  projection for better matching with teacher features.
- **Smart Progress Bars**: progress bars hide automatically when stdout isn't a TTY
- **CIFAR-friendly ResNet/EfficientNet stem**: use `--small_input 1` when
  fine-tuning or evaluating models that modify the conv stem for 32x32 inputs
  (and remove max-pool for ResNet)
- **Disagreement Metrics**: `compute_disagreement_rate` now accepts
  `mode="pred"` to measure prediction mismatch or `mode="both_wrong"` for
  cross-error

---

## Installation

1. **Clone** this repo:
   ```bash
   git clone https://github.com/YourName/ASMB-KD.git
   cd ASMB-KD
```
2. *(Optional)* **Create and activate a Conda environment**:
```bash
conda create -n facil_env python=3.9
conda activate facil_env
```
3. **Install dependencies**:
```bash
pip install -r requirements.txt  # includes pandas for analysis
```

> **Note**
> This minimal repository only includes the `convnext_tiny` student.
> To use other architectures, provide custom modules under `models/students/`.
> Each student should implement a `create_*` factory returning a model whose
> `forward` yields `(feature_dict, logits, extra)` similar to the teacher wrappers.


## Testing

Run the helper script to install **PyTorch** and all remaining dependencies,
then invoke `pytest`:

```bash
bash scripts/setup_tests.sh
pytest
```

Unit tests are skipped unless **PyTorch** is installed.

> **Note**
> The training and evaluation scripts now call `torch.load(..., weights_only=True)`
> when loading checkpoints. Make sure you have **PyTorch&nbsp;2.1** or newer
> installed, otherwise loading state dictionaries will fail.

---

Usage

### Typical Training Flow

> **Note**
> Only a `convnext_tiny` student is provided. Add your own modules under `models/students/` to try other architectures.

1. **Fine-tune the teachers**
```bash
python scripts/train_teacher.py --config configs/minimal.yaml --teacher_type resnet152
python scripts/train_teacher.py --config configs/minimal.yaml --teacher_type efficientnet_b2
```

2. **Run IB-KD**
```bash
bash scripts/run_ibkd.sh --config configs/minimal.yaml
```
`run_ibkd.sh` loads the fine-tuned checkpoints and performs multi-stage distillation.

### Evaluation
Use `utils.eval.evaluate_acc` to compute accuracy for your model.

### Data Augmentation

Use the `--data_aug` flag to control dataset transforms. When set to `1` (default), the loaders apply `RandomCrop`, `RandomHorizontalFlip` and `RandAugment` for stronger augmentation. Passing `--data_aug 0` disables these operations and only performs normalization/resizing.

```bash
python main.py --config configs/default.yaml --data_aug 0
```

Set `num_workers` in your YAML file to control how many processes each
`DataLoader` uses (defaults to `2`).

| Flag | Purpose |
| ---- | ------- |
| `--mixup_alpha` | MixUp alpha for distillation |
| `--cutmix_alpha` | CutMix alpha during fine-tuning |
| `--cutmix_alpha_distill` | CutMix alpha during distillation (YAML key `cutmix_alpha_distill`, default `0.0`) |

CutMix takes priority over MixUp when both are > 0. MixUp is used when only `--mixup_alpha` > 0, otherwise no mixing is applied.

### MixUp, CutMix & Label Smoothing

Control MixUp or CutMix augmentation and label smoothing via CLI flags:

```bash
python main.py --mixup_alpha 0.2 --cutmix_alpha_distill 0.5 --label_smoothing 0.1
```

### Automatic Mixed Precision (AMP)

Add the following keys to your config to enable AMP:

```yaml
use_amp: true
amp_dtype: float16  # or bfloat16
grad_scaler_init_scale: 1024
```

Example usage:

```python
from utils.misc import get_amp_components

autocast_ctx, scaler = get_amp_components(cfg)
with autocast_ctx:
    out = model(x)
    loss = criterion(out, target)
if scaler:
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
else:
    loss.backward()
    optimizer.step()
```

```bash
python main.py --use_amp 1 --amp_dtype bfloat16
```

### Small Input Checkpoints

Models fine-tuned with `--small_input 1` replace their conv stems for small
images. When distilling or evaluating such checkpoints you must pass the same
`--small_input 1` flag to `main.py` or when using `utils.eval.evaluate_acc` so the architectures match.
The flag now also configures **all student adapters** (ResNet, EfficientNet and
Swin) to expect 32×32 inputs.

### Teacher Fine-Tuning

Run the helper script to fine-tune a teacher before distillation:

```bash
python scripts/train_teacher.py --config configs/minimal.yaml --teacher_type resnet152
```

Adjust the configuration file to control hyperparameters such as epochs and learning rate.




---
```plaintext
Folder Structure

(Repo Root)
├── main.py               # Main training script
├── requirements.txt
├── README.md
├── LICENSE
├── analysis/
├── configs
│   └── minimal.yaml
├── data
│   ├── cifar100.py
│   ├── imagenet100.py
│   └── __init__.py
├── methods
│   ├── asmb.py
│   ├── fitnet.py
│   ├── crd.py
│   ├── dkd.py
│   ├── at.py
│   └── vanilla_kd.py
├── models
│   ├── ib
│   │   └── vib_mbm.py
│   ├── students
│   │   └── student_convnext.py
│   └── teachers
│       ├── teacher_efficientnet.py
│       ├── teacher_resnet.py
│       └── teacher_swin.py
├── modules
│   ├── trainer_student.py
│   ├── trainer_teacher.py
│   └── losses.py
├── scripts
│   ├── train_teacher.py
│   └── run_ibkd.sh
└── utils
    ├── eval.py
    └── misc.py
```
---
---

Results

Experiment outputs (CSV, logs, etc.) reside in the results/ folder.
You can run analysis/compare_ablation.py or analysis/plot_results.ipynb to analyze or visualize them. These scripts require the pandas library, which is installed via `requirements.txt`.

---

License

This project is distributed under the MIT License.

MIT License

Copyright (c) 2024 Suyoung Yang

Permission is hereby granted, free of charge, ...


---

Citation

If you use this framework, please cite:

@misc{ASMB-KD,
  title   = {ASMB Knowledge Distillation Framework},
  author  = {Suyoung Yang},
  year    = {2024},
  howpublished = {\url{https://github.com/YourName/ASMB-KD}}
}


---

Contact

For questions or issues, please open a GitHub issue or email suyoung425@yonsei.ac.kr.

