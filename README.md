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
- **MixUp & Label Smoothing**: enable with `--mixup_alpha` and `--label_smoothing`
- **MBM Dropout**: set `mbm_dropout` in configs to add dropout within the
  Manifold Bridging Module
- **CIFAR-friendly ResNet/EfficientNet stem**: use `--small_input 1` when
  fine-tuning to replace the large-stride stem with a 3x3, stride-1 version
  (and remove max-pool for ResNet)

---

## Installation

1. **Clone** this repo:
   ```bash
   git clone https://github.com/YourName/ASMB-KD.git
   cd ASMB-KD
```
2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

---

Usage

1) Multi-Stage Distillation (main.py)

python main.py --config configs/partial_freeze.yaml --device cuda \
--teacher1_ckpt teacher1.pth --teacher2_ckpt teacher2.pth
	•	Adjust hyperparameters in configs/*.yaml (partial freeze, learning rates, etc.).
	•	Optionally load pre-finetuned teacher checkpoints via `--teacher1_ckpt` and `--teacher2_ckpt`.

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

### MixUp & Label Smoothing

Control MixUp augmentation and label smoothing via CLI flags:

```bash
python main.py --mixup_alpha 0.2 --label_smoothing 0.1
```



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

