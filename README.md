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

  | Student model                | Feature dim |
  |------------------------------|-------------|
   | `student_efficientnet_adapter` | 1408        |
   | `student_resnet_adapter`       | 2048        |
   | `student_swin_adapter`         | 768         |
   *The adapters above were part of the original project and are no longer
   shipped in this repository.*
- **Swin Adapter Dim**: `swin_adapter_dim` sets the hidden size of the MLP
  adapter used by `student_swin_adapter` (default `64`)
- **Student Projection Normalization**: set `proj_normalize` (default `true`) to
  apply L2 normalization on features before the student projection head
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
> Student model adapters (`student_resnet_adapter`, etc.) were removed from this
> repository. To run the code you must provide your own student definitions.
> Place custom modules under `models/students/` or copy them from the upstream
> project. Each student should implement a `create_*` factory returning a model
> whose `forward` yields `(feature_dict, logits, extra)` similar to the teacher
> wrappers.

The unified script `run_experiments.sh` automatically tries to activate a
Conda environment named `facil_env`. If you use a different environment name,
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

The base config merged by `generate_config.py` defaults to
`configs/default.yaml`. This file defines universal settings such as
device and paths and enables Automatic Mixed Precision (AMP) by default.
The script can also merge optional fragments such as
`configs/partial_freeze.yaml`. Freeze levels are now defined in
`configs/partial_freeze.yaml`. The variables `teacher1_freeze_level`,
`teacher2_freeze_level` and `student_freeze_level` control how much of each
model is unfrozen during training, so this fragment also sets the freeze levels
in addition to BN freezing and adapter options. Pass one or more
fragment files (or a directory containing them) to assemble a config from
multiple pieces. Override the selection by setting the `BASE_CONFIG`
environment variable:

```bash
BASE_CONFIG=configs/partial_freeze.yaml bash scripts/run_experiments.sh --mode loop
```

You can also merge several fragments directly:

```bash
python scripts/generate_config.py \
  --base configs/default.yaml configs/partial_freeze.yaml \
  --out combined.yaml
```

To combine additional fragments, simply list them all after `--base`:

```bash
python scripts/generate_config.py \
  --base configs/default.yaml configs/partial_freeze.yaml extras.yaml \
  --out full.yaml
```

Or point `--base` to a directory to load every YAML file inside (sorted by
name):

```bash
python scripts/generate_config.py --base configs/fragments/ --out combined.yaml
```

`configs/hparams.yaml` holds the numeric hyperparameters used by the batch
scripts while `configs/*.yaml` describe the model architectures and freeze
settings. `generate_config.py` loads this file with `--hparams` so its values
override or supplement those defined in the fragments. Edit
`configs/hparams.yaml` before running `bash scripts/run_experiments.sh --mode loop`
or `bash scripts/run_experiments.sh --mode sweep` to customize the default hyperparameters.
`N_STAGE_LIST` can contain a space-separated list such as `"2 3 4 5"` to run
multiple stage counts in one batch.

### Batch scripts & hyperparameter overrides

`configs/hparams.yaml` stores the default hyperparameters used by
`run_experiments.sh`. The selected YAML file (via
`BASE_CONFIG`) supplies the base settings such as model types and
partial‑freeze options. When you run a script, the values are merged in
the following order:

1. YAML files passed to `--base` (via `BASE_CONFIG` or manually), merged in order
2. Variables from `configs/hparams.yaml` (unless overridden)
3. Command-line overrides passed to `generate_config.py` or `main.py`

Values defined in `configs/hparams.yaml` **supersede** those in
`configs/default.yaml` or any other fragments included with `--base`.
For example, if `configs/default.yaml` declares
`student_epochs_per_stage: 15` but `configs/hparams.yaml` sets
`student_iters: 40`, the environment variable `STUDENT_ITERS=40` from
`hparams.yaml` overrides the default 15‑epoch value during training.

`run_experiments.sh` exports the values from `configs/hparams.yaml` as
environment variables. These variables are fed back into
`generate_config.py` so they can still override any field defined in the
YAML fragments. The recommended workflow is to edit
`configs/hparams.yaml` whenever you need experiment‑specific values and
then invoke the batch script.

You can override any variable by exporting it before calling the script.
For example, run the batch script with the partial-freeze configuration
(freeze levels come from `configs/partial_freeze.yaml`) and a different teacher
learning rate:

```bash
teacher_lr=0.0002 BASE_CONFIG=configs/partial_freeze.yaml bash scripts/run_experiments.sh --mode loop
```

When launching jobs via `run.sh`, the script writes the merged configuration to
`outputs/asmb_${SLURM_JOB_ID}/config.yaml`. During the batch loop, this file is
copied into each experiment directory so you can recover the exact settings used
for every run.

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
> Predefined student adapters were removed. Add your own modules under
> `models/students/` or copy them from the upstream repository before running
> the examples below.

1. Fine-tune each teacher (optional but recommended).
2. For each stage, perform a teacher adaptive update followed by student knowledge distillation.
3. Repeat for the configured number of stages.

Baseline runs (e.g., `vanilla_kd`) produce their own logs such as `VanillaKD => ...`.

1) Multi-Stage Distillation (main.py)

python main.py --config configs/partial_freeze.yaml --device cuda \
  --mbm_type LA --mbm_r 4 --mbm_n_head 1 --mbm_learnable_q 1
  # Freeze levels (`teacher1_freeze_level`, `teacher2_freeze_level`,
  # `student_freeze_level`) are loaded from `configs/partial_freeze.yaml`
  # mbm_query_dim and mbm_out_dim are automatically set to the student feature dimension
        •       Adjust partial-freeze or architecture settings in `configs/*.yaml`.
        •       Edit `configs/hparams.yaml` to change numeric hyperparameters like learning rates or dropout.
        •       Set `LR_SCHEDULE` to "step" or "cosine" to choose the learning rate scheduler.
	•	Teacher checkpoints load automatically from `checkpoints/{teacher_type}_ft.pth` when available.
        •       Each trainer creates its own optimizer and scheduler at the start of every stage.

2) Single-Teacher Distillation (run_single_teacher.py)

```bash
python scripts/run_single_teacher.py --config configs/default.yaml \
  --method vanilla_kd --teacher_type resnet152 --teacher_ckpt teacher.pth \
  --student_type resnet_adapter --epochs 40 \
  --dataset imagenet100
```

The `--method` flag selects one of `vanilla_kd`, `fitnet`, `dkd`, `at` or `crd`.
Pass `--dataset` to override the dataset specified in the YAML config (either
`cifar100` or `imagenet100`).
Partial freezing is automatically turned off for these methods—`run_single_teacher.py`
sets `use_partial_freeze: false` when the selected `method` is not `asmb`.

3) Student Baseline (train_student_baseline.py)

### Student Baseline

Run the student alone using the same partial-freeze settings to gauge its standalone performance:

```bash
python scripts/train_student_baseline.py --config configs/partial_freeze.yaml \
  --student_type resnet_adapter --epochs 40 --dataset cifar100
# Freeze levels come from `configs/partial_freeze.yaml`
```

The script uses the same optimizer and scheduler configuration as the distillation runs. The resulting accuracy serves as the reference for all distillation experiments and is saved under `results/`.


4) Evaluation (utils.eval.evaluate_acc)

Use `utils.eval.evaluate_acc` to compute accuracy for your model. The old script lives under `legacy_scripts/eval_asmb_old.py` for reference.

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

Fine-tune the individual teachers before running the distillation stages.
All fine-tuning options live in `configs/hparams.yaml`.
The bundled `TeacherSwinWrapper` expects the Swin backbone to call
`features`, `norm`, `permute`, `avgpool` and `flatten` in sequence when
producing its feature map. This mirrors torchvision's official
`SwinTransformer` forward path.
Adjust the parameters in `configs/hparams.yaml`:

```bash
# configs/hparams.yaml
finetune_epochs=100   # number of fine-tuning epochs
finetune_lr=0.0005    # learning rate
finetune_cutmix_alpha=0  # set to 0 to disable CutMix
lr_schedule=step   # step or cosine
```

Alternatively edit the YAML file used by `scripts/fine_tuning.py`:

```yaml
# configs/hparams.yaml
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
python scripts/fine_tuning.py --config configs/hparams.yaml \
  --teacher_type resnet152 --finetune_epochs 100 --finetune_lr 0.0005 \
  --dropout_p 0.5
```

The script uses **CIFAR-100** by default. Change the `dataset_name` key in
`configs/hparams.yaml` (e.g., `dataset_name: imagenet100`) to switch datasets.

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
You can specify several stage counts by setting `n_stage_list="2 3 4 5"`.

```bash
bash scripts/run_experiments.sh --mode loop
```

Once the teacher checkpoints are in place you can disable the fine-tuning step
in subsequent runs. Either set `finetune_epochs: 0` in `configs/hparams.yaml`
or point `finetune_ckpt1` and `finetune_ckpt2` to the existing `.pth` files so
`run_experiments.sh` skips the fine-tuning loops.

### Teacher Adapter & BN-Head-Only Options

With partial freezing you can further restrict the teachers to small
adapters or only update their batch-norm layers and classifier heads.
Set the following flags in `configs/hparams.yaml`:

```yaml
teacher1_use_adapter: 1
teacher1_bn_head_only: 1
teacher2_use_adapter: 1
teacher2_bn_head_only: 0
```

`run_experiments.sh` exports these values so you can toggle them for
sweeps or batch runs without editing every config file.

### Freeze Levels

The amount of each model that remains trainable is controlled by three keys in `configs/partial_freeze.yaml`:

```yaml
teacher1_freeze_level: 0
teacher2_freeze_level: 1
student_freeze_level: 0
```

Lower numbers unfreeze fewer layers. In the example above, only teacher&nbsp;2's
final block and the classifier heads remain trainable. After editing the YAML
file, rerun `bash scripts/run_experiments.sh --mode loop` to generate new
configs with your chosen freeze levels.



---
```plaintext
Folder Structure

(Repo Root)
├── main.py               # Main training script (ASMB, partial freeze)
├── legacy_scripts
│   └── eval_asmb_old.py   # Old evaluation script
├── requirements.txt      # Dependencies
├── README.md             # Project info
├── LICENSE               # MIT License

├── analysis
│   ├── compare_ablation.py
│   └── plot_results.ipynb

├── configs
│   ├── default.yaml
│   ├── hparams.yaml        # default hyperparameters for run_experiments.sh
│   └── partial_freeze.yaml    # BN/adapters and freeze levels

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

├── scripts
│   ├── fine_tuning.py
│   └── run_experiments.sh

└── utils
    ├── logger.py
    ├── eval.py           # evaluation helpers
    └── misc.py

	• analysis/: Scripts or notebooks for comparing experiments (compare_ablation.py, plot_results.ipynb)
	• configs/: YAML config files for partial-freeze settings, hyperparameters
	• methods/: KD implementations (ASMB, FitNet, CRD, DKD, etc.)
	• modules/: Partial freeze utility, trainers, custom losses
        • scripts/: Shell scripts for multiple or batch experiments
            ◦ Edit `configs/hparams.yaml` to change the default hyperparameters consumed by `run_experiments.sh`

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

