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
pip install -r requirements.txt
```

The bash scripts (`run_many.sh` and `run_sweep.sh`) automatically try to
activate a Conda environment named `facil_env`. If you use a different
environment name, set the `CONDA_ENV` variable accordingly. You can also
skip activation entirely by exporting `USE_CONDA=0` before running the
scripts.

---

Usage

1) Multi-Stage Distillation (main.py)

python main.py --config configs/partial_freeze.yaml --device cuda \
--teacher1_ckpt teacher1.pth --teacher2_ckpt teacher2.pth
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
  --head_ckpt synergy_head.pth

	•	Prints Train/Test accuracy, optionally logs to CSV if configured.

### Data Augmentation

Use the `--data_aug` flag to control dataset transforms. When set to `1` (default), the loaders apply `RandomCrop`, `RandomHorizontalFlip` and `RandAugment` for stronger augmentation. Passing `--data_aug 0` disables these operations and only performs normalization/resizing.

```bash
python main.py --config configs/default.yaml --data_aug 0
```

### MixUp, CutMix & Label Smoothing

Control MixUp or CutMix augmentation and label smoothing via CLI flags:

```bash
python main.py --mixup_alpha 0.2 --cutmix_alpha_distill 0.5 --label_smoothing 0.1
```

### Small Input Checkpoints

Models fine-tuned with `--small_input 1` replace their conv stems for small
images. When distilling or evaluating such checkpoints you must pass the same
`--small_input 1` flag to `main.py` or `eval.py` so the architectures match.
The flag now also configures the student Swin adapter to expect 32×32 inputs.

### Teacher Fine-Tuning

Fine-tune the individual teachers before running the distillation stages.
Adjust the parameters in `scripts/hparams.sh`:

```bash
# scripts/hparams.sh
FT_EPOCHS=100   # number of fine-tuning epochs
FT_LR=0.0005    # learning rate
CUTMIX_ALPHA=0  # set to 0 to disable CutMix
```

Alternatively edit the YAML file used by `scripts/fine_tuning.py`:

```yaml
# configs/fine_tune.yaml
finetune_epochs: 100
finetune_lr: 0.0005
use_cutmix: false
efficientnet_dropout: 0.3  # dropout probability for EfficientNet teachers
```

Set `efficientnet_dropout` to control the dropout rate used in EfficientNet
teachers. The default value is **0.3**. You can override it on the command line:

```bash
python scripts/fine_tuning.py --config configs/fine_tune.yaml --dropout_p 0.5
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

```bash
bash scripts/run_many.sh
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
`scripts/hparams.sh`. Both `run_many.sh` and `run_sweep.sh` read the
values from `hparams.sh` so you can toggle them for sweeps or batch
runs without editing every config file.



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
│   ├── continual_asmb.py
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
            ◦ Edit `scripts/hparams.sh` to change common hyperparameters

```
---

Results

Experiment outputs (CSV, logs, etc.) reside in the results/ folder.
You can run analysis/compare_ablation.py or analysis/plot_results.ipynb to analyze or visualize them.

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

