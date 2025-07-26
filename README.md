# ASIB-CL : Information-Bottleneck KD with Continual-Learning

**ASIB-CL** 확장은 기존 ASMB_KD 프레임워크에  
*Information-Bottleneck Manifold Bridging Module* (IB-MBM)과  
*Continual-Learning* (Replay + EWC) 지원을 추가합니다.

## Quick-start
```bash
# IID 학습
python main.py

# Continual-Learning (Split-CIFAR 5 tasks 예시)
python main.py cl=split_cifar
```

`--cl_mode 1` 활성화 후 `--num_tasks` 값을 지정하면 별도의 CL 전용 YAML 없이 연속 학습을 수행할 수 있습니다.

## 주요 config 플래그
* `mbm_type` (ignored) : always uses **ib_mbm**
* `use_ib`   : true / false  (IB ablation)
* `ib_beta_warmup_epochs` : ramp-up epochs for the IB KL weight
* `cl_mode`  : true → CL 활성화
* `num_tasks`, `replay_ratio`, `lambda_ewc`
* `teacher1_type`, `teacher2_type` : teacher architecture names (default `resnet152`)

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
- **IB β Warmup**: `ib_beta_warmup_epochs` linearly scales the KL weight from
  0 to `ib_beta` over the specified epochs
- **Feature-Level KD (disabled)**: set `feat_kd_alpha` > 0 to align student
  features with the synergy representation during teacher and student updates.
  Example model config snippet:

  ```yaml
  feat_kd_alpha: 0.0  # disabled by default
  feat_kd_key: "feat_2d"
  feat_kd_norm: "none"
  ```

  The same values can be overridden via CLI. Set `feat_kd_alpha` to a
  positive value to enable this loss:
  `--feat_kd_alpha 1.0 --feat_kd_key feat_2d --feat_kd_norm none`.
- **Hybrid Guidance**: set `hybrid_beta` (>0) to blend vanilla KD from the
  average teacher logits with the default ASMB loss.
- **Custom MBM Query Dim**: `mbm_query_dim` sets the dimension of the
  student features used as the attention query in `ib_mbm`.
  When omitted or set to `0`, the script falls back to the feature
  dimension reported by the student model (if available).
  The MBM output dimension (`mbm_out_dim`) now defaults to this student
  feature size as well.
- **Requires Student Features**: `ib_mbm` relies on student features as the
  attention query, so the student model must be provided.
  Common student feature dimensions are:

  | Student model                | Feature dim |
  |------------------------------|-------------|
  | `resnet101_student`            | 2048        |
- **Smart Progress Bars**: progress bars hide automatically when stdout isn't a TTY
- **CIFAR-friendly ResNet/EfficientNet stem**: use `--small_input 1` when
  fine-tuning or evaluating models that modify the conv stem for 32x32 inputs
  (and remove max-pool for ResNet)
- **Distillation Adapter**: set `use_distillation_adapter: true` to enable
  lightweight adapters on each teacher. `distill_out_dim` controls the common
 feature dimension used for synergy (default `512`). Set `debug_verbose: true` to print the teacher and student feature shapes every batch when troubleshooting (the example config `configs/experiment/res152_effi_l2.yaml` leaves this `false` by default).
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
conda env create -f environment.yml
conda activate tlqkf
```
3. *(Optional)* **Install dependencies manually**:
```bash
pip install -r requirements.txt  # includes pandas for analysis
```
> **주의** : PyTorch 1.12 – 2.2 는 NumPy 2.x 와 ABI 불일치가 있어
> `numpy<2.0`(예: 1.26.4) 로 고정해야 합니다.
4. **Download pretrained checkpoints**:
```bash
mkdir checkpoints
wget -O checkpoints/resnet152_ft.pth <링크>
wget -O checkpoints/efficientnet_l2_ft.pth <링크>  # optional EfficientNet-L2
```
5. **Run a single experiment**:
```bash
python main.py --config-name experiment/res152_effi_l2  # EfficientNet-L2 teacher
```

The unified script `run_experiments.sh` automatically tries to activate a
Conda environment named `tlqkf`. If you use a different environment name,
set the `CONDA_ENV` variable accordingly. You can also skip activation
entirely by exporting `USE_CONDA=0` before running the script. Run experiments
directly with `bash scripts/run_experiments.sh --mode {loop,sweep}`.

Set the distillation method via the `METHOD` variable or provide a
space‑separated list using `METHOD_LIST`. The default `asmb` runs the
multi‑teacher pipeline in `main.py`. Specify `vanilla_kd`, `fitnet`, `dkd`,
`at` or `crd` to launch the single‑teacher runner. With `METHOD_LIST` you can
execute several methods sequentially:

The main training script accepts the same flag via `--method` (default `asmb`)
so `run_experiments.sh` can pass it uniformly.

```bash
METHOD_LIST="asmb fitnet vanilla_kd" bash scripts/run_experiments.sh --mode loop
```

Hydra now composes configs directly. Run the default experiment with:

```bash
python main.py --config-name base
```

Freeze levels and other hyperparameters are stored in `configs/model/` YAML files.
Override any value via the command line, e.g.

```bash
python main.py model.teacher.freeze_level=1 model.student.lr=0.0008
```

Batch scripts like `run_experiments.sh` simply forward arguments to `main.py`.

When these scripts are submitted via `sbatch`, SLURM sets the environment
variable `SLURM_SUBMIT_DIR` to the directory from which the job was launched.
`run.sh` and `run_finetune_clean.sh` now use this variable to `cd` back to the
repository root so relative paths (such as `scripts/fine_tuning.py`) resolve
correctly.

Logging options live under the `log:` section in each YAML.  The helper
function `flatten_hydra_config` copies `log.level` and `log.filename` to the
top‑level keys `log_level` and `log_filename` so older scripts continue to work.


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

1. Fine-tune each teacher (optional but recommended).
2. For each stage, perform a teacher adaptive update followed by student knowledge distillation.
3. Repeat for the configured number of stages.

Baseline runs (e.g., `vanilla_kd`) produce their own logs such as `VanillaKD => ...`.

1) Multi-Stage Distillation (main.py)

python main.py --config-name base \
  device=cuda mbm_r=4 mbm_n_head=1 mbm_learnable_q=1
  # Freeze levels are defined in the model YAMLs under configs/model/
  # mbm_query_dim and mbm_out_dim are automatically set to the student feature dimension
        •       Adjust model settings in `configs/model/*` or pass Hydra overrides.
        •       Set `LR_SCHEDULE` to "step" or "cosine" to choose the learning rate scheduler.
	•	Teacher checkpoints load automatically from `checkpoints/{teacher_type}_ft.pth` when available.
        •       Each trainer creates its own optimizer and scheduler at the start of every stage.

2) Single-Teacher Distillation (run_single_teacher.py)

```bash
python scripts/run_single_teacher.py --config-name base \
  +method=vanilla_kd +teacher_type=resnet152 +teacher_ckpt=teacher.pth \
  +student_type=resnet +epochs=40 \
  dataset=imagenet32
```

The `method` option selects one of `vanilla_kd`, `fitnet`, `dkd`, `at` or `crd`.
Override the dataset by passing `dataset=imagenet32` or `dataset=cifar100` on
the command line.
Partial freezing is automatically turned off for these methods—`run_single_teacher.py`
sets `use_partial_freeze: false` when the selected `method` is not `asmb`.

3) Student Baseline (train_student_baseline.py)

### Student Baseline

Run the student alone using the same partial-freeze settings to gauge its standalone performance:

```bash
python scripts/train_student_baseline.py --config-name base \
  +student_type=resnet +epochs=40 dataset=cifar100
# Freeze levels come from the student YAML under configs/model/
```

The script uses the same optimizer and scheduler configuration as the distillation runs. The resulting accuracy serves as the reference for all distillation experiments and is saved under `results/`.


4) Evaluation (eval.py)

Evaluate a single model or a synergy model (Teacher1 + Teacher2 + MBM + synergy head):

# Single model
python eval.py +eval_mode=single \
  +ckpt_path=./results/single_model.pth

# Synergy model
python eval.py +eval_mode=synergy \
  # uses checkpoints/{teacher_type}_ft.pth if available \
  +mbm_ckpt=mbm.pth \
  +head_ckpt=synergy_head.pth \
  +student_type=resnet \
  +student_ckpt=student.pth \
  +mbm_r=4 +mbm_n_head=1 +mbm_learnable_q=1
  # mbm_query_dim and mbm_out_dim are automatically set to the student feature dimension

	•	Prints Train/Test accuracy, optionally logs to CSV if configured.

### Data Augmentation

Use the `--data_aug` flag to control dataset transforms. When set to `1` (default), the loaders apply `RandomCrop`, `RandomHorizontalFlip` and `RandAugment` for stronger augmentation. Passing `--data_aug 0` disables these operations and only performs normalization/resizing.

```bash
python main.py --config-name base --data_aug 0
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
`--small_input 1` flag to `main.py` or `eval.py` so the architectures match.
The flag now also configures **all student adapters** (ResNet, EfficientNet and
Swin) to expect 32×32 inputs.

### Teacher Fine-Tuning

Fine-tune the individual teachers before running the distillation stages.
All fine-tuning options live in the teacher configs under `configs/model/teacher/`.
The bundled `SwinTeacher` expects the Swin backbone to call
`features`, `norm`, `permute`, `avgpool` and `flatten` in sequence when
producing its feature map. This mirrors torchvision's official
`SwinTransformer` forward path.
Adjust the parameters in your chosen teacher YAML:

```bash
# configs/model/teacher/resnet152.yaml
finetune_epochs=100   # number of fine-tuning epochs
finetune_lr=0.0005    # learning rate
finetune_cutmix_alpha=0  # set to 0 to disable CutMix
lr_schedule=step   # step or cosine
```

Alternatively edit the YAML file used by `scripts/fine_tuning.py`:

```yaml
# configs/model/teacher/resnet152.yaml
finetune_epochs: 100
finetune_lr: 0.0005
finetune_use_cutmix: false
efficientnet_dropout: 0.3  # dropout probability for EfficientNet teachers
```

Set `efficientnet_dropout` to control the dropout rate used in EfficientNet
teachers. The default value is **0.3**.

#### Fine-tuning a Teacher

Run the fine-tuning script directly to update a single teacher:

```bash
python scripts/fine_tuning.py --config-name base \
  +teacher_type=resnet152 +finetune_epochs=100 +finetune_lr=0.0005 \
  +dropout_p=0.5
```

The script uses **CIFAR-100** by default. Change `dataset.name` via a Hydra override
or edit the dataset YAML (e.g., `dataset=imagenet32`) to switch datasets.

For partial freezing with EfficientNet, a new freeze scope
`features_classifier` unfreezes the feature extractor and classifier modules
along with the MBM:

```yaml
# configs/model/teacher/resnet152.yaml
teacher2_freeze_scope: "features_classifier"
```

After saving the changes, run `python main.py --config-name base` (or your batch
script) to fine-tune the teacher and continue with distillation.

Once the teacher checkpoints are in place you can disable the fine-tuning step
in subsequent runs. Either set `finetune_epochs: 0` in the teacher YAML or
point `finetune_ckpt1` and `finetune_ckpt2` to the existing `.pth` files so
the script skips the fine-tuning loops.

### Teacher Adapter & BN-Head-Only Options

With partial freezing you can further restrict the teachers to small
adapters or only update their batch-norm layers and classifier heads.
Set the following flags in the teacher YAML or via CLI overrides:

```yaml
use_distillation_adapter: true
teacher1_bn_head_only: 1
teacher2_bn_head_only: 0
```

When `use_distillation_adapter` is enabled, each teacher routes its features
through a small MLP adapter before computing synergy. The output dimension of
these adapters is controlled by `distill_out_dim` (default `512`). All
teachers must share the same value so their features can be stacked.

`run_experiments.sh` exports these values so you can toggle them for
sweeps or batch runs without editing every config file.

### Freeze Levels
* **-1 \u2192 no freeze** (new)  
* 0 \u2192 head only  
* 1 \u2192 last block train  
* 2 \u2192 last two blocks train  
  (architecture\u2011specific mapping\ub294 `modules/partial_freeze.py` \ucc38\uace0)

The amount of each model that remains trainable is controlled by three keys in the model configs:

```yaml
teacher1_freeze_level: 0
teacher2_freeze_level: 1
student_freeze_level: 0
```

Lower numbers unfreeze fewer layers. In the example above, only teacher&nbsp;2's
final block and the classifier heads remain trainable. After editing the YAML
file, run `python main.py --config-name base` to apply the new freeze levels.



---
```plaintext
Folder Structure

(Repo Root)
├── main.py                 # Main training script
├── eval.py                 # Evaluation script
├── requirements.txt        # Python dependencies
├── run.sh                  # Example run script
├── run_finetune_clean.sh   # Finetuning helper
├── pyproject.toml          # Build metadata
├── setup.py                # Package setup
├── README.md
├── LICENSE
├── environment.yml         # Conda environment

├── analysis/
│   ├── compare_ablation.py
│   └── plot_results.ipynb

├── configs/
│   ├── base.yaml
│   ├── dataset/
│   ├── experiment/
│   ├── finetune/
│   ├── method/
│   ├── model/
│   │   ├── teacher/
│   │   └── student/
│   └── schedule/

├── data/
│   ├── cifar100.py
│   ├── cifar100_overlap.py
│   ├── imagenet32.py
│   └── __init__.py

├── examples/
│   └── run_cifar100_cl.sh

├── methods/                # KD algorithms
│   ├── asmb.py
│   ├── fitnet.py
│   ├── crd.py
│   ├── dkd.py
│   ├── at.py
│   ├── vanilla_kd.py
│   └── __init__.py

├── models/
│   ├── __init__.py
│   ├── mbm.py
│   ├── common/
│   │   └── adapter.py
│   ├── students/
│   │   └── resnet101_student.py
│   └── teachers/
│       ├── resnet152_teacher.py
│       └── efficientnet_l2_teacher.py

├── modules/
│   ├── trainer_student.py
│   ├── trainer_teacher.py
│   ├── cutmix_finetune_teacher.py
│   ├── disagreement.py
│   ├── partial_freeze.py
│   ├── losses.py
│   └── __init__.py

├── scripts/
│   ├── run_single_teacher.py
│   ├── run_experiments.sh
│   ├── run_overlap_experiments.sh
│   ├── run_sweep.sh
│   ├── fine_tuning.py
│   ├── setup_tests.sh
│   └── train_student_baseline.py

├── sweeps/
│   ├── asmb_grid.yaml
│   ├── asmb_mixed.yaml
│   └── overlap_grid.yaml

├── tests/
│   └── ...

└── utils/
    ├── logger.py
    ├── misc.py
    └── config_utils.py

        • analysis/: Scripts or notebooks for experiment comparisons
        • configs/: Hydra configuration groups
        • data/: Dataset wrappers
        • examples/: Example shell scripts
        • methods/: KD implementations (ASMB, FitNet, etc.)
        • models/: Student and teacher models
        • modules/: Training utilities and losses
        • scripts/: Helper entrypoints
        • sweeps/: YAML sweep definitions
        • tests/: Unit tests
        • utils/: Misc utilities (logging, scheduling)

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

