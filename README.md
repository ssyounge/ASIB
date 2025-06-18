# ASMB Knowledge Distillation Framework

This repository provides an **Adaptive Synergy Manifold Bridging (ASMB)** multi-stage knowledge distillation framework, along with various KD methods (FitNet, CRD, AT, DKD, VanillaKD, etc.) and a partial-freeze mechanism for large models.

---

## Features

- **Multi-Stage Distillation**: Teacher ↔ Student updates in a phased (block-wise) manner  
- **ASMB** (Adaptive Synergy Manifold Bridging): Uses a Manifold Bridging Module (MBM) to fuse two Teacher feature maps into synergy logits  
- **Partial Freeze**: Freeze backbone parameters, adapt BN/Heads/MBM for efficiency  
- **Multiple KD Methods**: FitNet, CRD, AT, DKD, VanillaKD, plus custom `asmb.py`
- **CIFAR-100 / ImageNet100** dataset support
- **Configurable Data Augmentation**: toggle with `--data_aug` (1/0)
- **MixUp, CutMix & Label Smoothing**: enable with `--mixup_alpha`,
  `--cutmix_alpha_distill` and `--label_smoothing`
- **MBM Dropout**: set `mbm_dropout` in configs to add dropout within the
  Manifold Bridging Module
- **Gradient Clipping**: enable by setting `grad_clip_norm` (>0) in configs
- **Custom MBM Query Dim**: `mbm_query_dim` controls the dimension of the
  student features used as the attention query in `LightweightAttnMBM`.
  When omitted or set to `0`, the script automatically falls back to the
  feature dimension reported by the student model (if available).
  Common student feature dimensions are:

  | Student model                | Feature dim |
  |------------------------------|-------------|
  | `student_efficientnet_adapter` | 1408        |
  | `student_resnet_adapter`       | 2048        |
  | `student_swin_adapter`         | 768         |
- **Smart Progress Bars**: progress bars hide automatically when stdout isn't a TTY
- **CIFAR-friendly ResNet/EfficientNet stem**: use `--small_input 1` when
  fine-tuning or evaluating models that modify the conv stem for 32x32 inputs
  (and remove max-pool for ResNet)

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

The unified script `run_experiments.sh` (used by `run_many.sh` and
`run_sweep.sh`) automatically tries to activate a Conda environment named
`facil_env`. If you use a different environment name, set the
`CONDA_ENV` variable accordingly. You can also skip activation entirely by
exporting `USE_CONDA=0` before running the script.

The base config merged by `generate_config.py` defaults to
`configs/default.yaml`. Override it by setting the `BASE_CONFIG`
environment variable:

```bash
BASE_CONFIG=configs/partial_freeze.yaml bash scripts/run_experiments.sh --mode loop
```

Edit `configs/hparams.yaml` before running `bash scripts/run_experiments.sh --mode loop` or
`bash scripts/run_experiments.sh --mode sweep` to customize the default hyperparameters.

### Batch scripts & hyperparameter overrides

`configs/hparams.yaml` stores the default hyperparameters used by
`run_experiments.sh`. The selected YAML file (via
`BASE_CONFIG`) supplies the base settings such as model types and
partial‑freeze options. When you run a script, the values are merged in
the following order:

1. YAML file specified by `BASE_CONFIG`
2. Variables from `configs/hparams.yaml` (unless overridden)
3. Command-line overrides passed to `generate_config.py` or `main.py`

You can override any variable by exporting it before calling the script.
For example, run the batch script with the partial-freeze configuration
and a different teacher learning rate:

```bash
T_LR=0.0002 BASE_CONFIG=configs/partial_freeze.yaml bash scripts/run_experiments.sh --mode loop
```

## Testing

Install **PyTorch** and the remaining dependencies using the helper script and
then run `pytest`:

```bash
bash scripts/setup_tests.sh
pytest
```

PyTorch must be available for the unit tests to run.

> **Note**
> The training and evaluation scripts now call `torch.load(..., weights_only=True)`
> when loading checkpoints. Make sure you have **PyTorch&nbsp;2.1** or newer
> installed, otherwise loading state dictionaries will fail.

---

Usage

1) Multi-Stage Distillation (main.py)

python main.py --config configs/partial_freeze.yaml --device cuda \
  --teacher1_ckpt teacher1.pth --teacher2_ckpt teacher2.pth \
  --mbm_type LA --mbm_r 4 --mbm_n_head 1 --mbm_learnable_q 0
  # mbm_query_dim is automatically set to the student feature dimension
	•	Adjust hyperparameters in configs/*.yaml (partial freeze, learning rates, etc.).
	•	Optionally load pre-finetuned teacher checkpoints via `--teacher1_ckpt` and `--teacher2_ckpt`.
        •       Optimizers and schedulers are instantiated once before stages and reset before each stage.

2) Evaluation (eval.py)

Evaluate a single model or a synergy model (Teacher1 + Teacher2 + MBM + synergy head):

# Single model
python eval.py --eval_mode single \
  --ckpt_path ./results/single_model.pth

# Synergy model
python eval.py --eval_mode synergy \
  --teacher1_ckpt teacher1.pth \
  --teacher2_ckpt teacher2.pth \
  --mbm_ckpt mbm.pth \
  --head_ckpt synergy_head.pth \
  --mbm_type LA --mbm_r 4 --mbm_n_head 1 --mbm_learnable_q 0
  # mbm_query_dim is automatically set to the student feature dimension

	•	Prints Train/Test accuracy, optionally logs to CSV if configured.

### Data Augmentation

Use the `--data_aug` flag to control dataset transforms. When set to `1` (default), the loaders apply `RandomCrop`, `RandomHorizontalFlip` and `RandAugment` for stronger augmentation. Passing `--data_aug 0` disables these operations and only performs normalization/resizing.

```bash
python main.py --config configs/default.yaml --data_aug 0
```

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
`--small_input 1` flag to `main.py` or `eval.py` so the architectures match.
The flag now also configures the student Swin adapter to expect 32×32 inputs.

### Teacher Fine-Tuning

Fine-tune the individual teachers before running the distillation stages.
Adjust the parameters in `configs/hparams.yaml`:

```bash
# configs/hparams.yaml
FT_EPOCHS=100   # number of fine-tuning epochs
FT_LR=0.0005    # learning rate
CUTMIX_ALPHA=0  # set to 0 to disable CutMix
```

Alternatively edit the YAML file used by `scripts/fine_tuning.py`:

```yaml
# configs/fine_tune.yaml
teacher_type: resnet101
finetune_epochs: 100
finetune_lr: 0.0005
use_cutmix: false
efficientnet_dropout: 0.3  # dropout probability for EfficientNet teachers
```

Set `efficientnet_dropout` to control the dropout rate used in EfficientNet
teachers. The default value is **0.3**. You can override it on the command line:

```bash
python scripts/fine_tuning.py --config configs/fine_tune.yaml \
  --teacher_type resnet101 --dropout_p 0.5
```

For partial freezing with EfficientNet, a new freeze scope
`features_classifier` unfreezes the feature extractor and classifier modules
along with the MBM:

```yaml
# configs/partial_freeze.yaml
teacher2_freeze_scope: "features_classifier"
```

After saving the changes, re-run the batch script to generate new teacher
checkpoints and continue with distillation:

Edit `configs/hparams.yaml` if you want to tweak the default hyperparameters.

```bash
bash scripts/run_experiments.sh --mode loop
```

### Teacher Adapter & BN-Head-Only Options

With partial freezing you can further restrict the teachers to small
adapters or only update their batch-norm layers and classifier heads.
Set the following keys in your config:

```yaml
# configs/partial_freeze.yaml
teacher1_use_adapter: true
teacher1_bn_head_only: true
teacher2_use_adapter: true
teacher2_bn_head_only: false
```

These map to `TEACHER1_USE_ADAPTER`, `TEACHER1_BN_HEAD_ONLY`,
`TEACHER2_USE_ADAPTER` and `TEACHER2_BN_HEAD_ONLY` in
`configs/hparams.yaml`. `run_experiments.sh` reads the values from this
file so you can toggle them for sweeps or batch runs without editing
every config file.



---
```plaintext
Folder Structure

(Repo Root)
├── main.py               # Main training script (ASMB, partial freeze)
├── eval.py               # Evaluation script (single vs synergy)
├── requirements.txt      # Dependencies
├── README.md             # Project info
├── LICENSE               # MIT License

├── analysis
│   ├── compare_ablation.py
│   └── plot_results.ipynb

├── configs
│   ├── default.yaml
│   ├── fine_tune.yaml
│   └── partial_freeze.yaml

├── data
│   ├── cifar100.py
│   ├── imagenet100.py
│   └── __init__.py

├── methods              # Various KD algorithms
│   ├── asmb.py
│   ├── fitnet.py
│   ├── crd.py
│   ├── dkd.py
│   ├── at.py
│   ├── vanilla_kd.py
│   └── __init__.py

├── models
│   ├── __init__.py
│   ├── mbm.py
│   ├── students
│   │   ├── __init__.py
│   │   ├── student_efficientnet_adapter.py
│   │   ├── student_resnet_adapter.py
│   │   └── student_swin_adapter.py
│   └── teachers
│       ├── __init__.py
│       ├── teacher_efficientnet.py
│       ├── teacher_resnet.py
│       └── teacher_swin.py

├── modules
│   ├── trainer_student.py
│   ├── trainer_teacher.py
│   ├── cutmix_finetune_teacher.py
│   ├── disagreement.py
│   ├── partial_freeze.py
│   ├── losses.py
│   └── __init__.py

├── results
│   ├── cifar100_asmb.csv
│   └── summary.csv

├── scripts
│   ├── fine_tuning.py
│   ├── run_experiments.sh
│   ├── run_many.sh
│   └── run_sweep.sh

└── utils
    ├── logger.py
    └── misc.py

	• analysis/: Scripts or notebooks for comparing experiments (compare_ablation.py, plot_results.ipynb)
	• configs/: YAML config files for partial-freeze settings, hyperparameters
	• methods/: KD implementations (ASMB, FitNet, CRD, DKD, etc.)
	• modules/: Partial freeze utility, trainers, custom losses
	• results/: CSV logs, outputs from training/evaluation
        • scripts/: Shell scripts for multiple or batch experiments
            ◦ Edit `configs/hparams.yaml` to change common hyperparameters

```
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

