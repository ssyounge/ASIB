# ASMB Knowledge Distillation Framework

This repository provides an **Adaptive Synergy Manifold Bridging (ASMB)** multi-stage knowledge distillation framework, along with various KD methods (FitNet, CRD, AT, DKD, VanillaKD, etc.) and a partial-freeze mechanism for large models.

---

## Features

- **Multi-Stage Distillation**: Teacher ↔ Student updates in a phased (block-wise) manner  
- **ASMB** (Adaptive Synergy Manifold Bridging): Uses a Manifold Bridging Module (MBM) to fuse two Teacher features into synergy logits  
- **Partial Freeze**: Freeze backbone parameters, adapt BN/Heads/MBM for efficiency  
- **Multiple KD Methods**: FitNet, CRD, AT, DKD, VanillaKD, plus custom `asmb.py`  
- **CIFAR-100 / ImageNet100** dataset support

---

## Installation

1. **Clone** this repo:
   ```bash
   git clone https://github.com/YourName/ASMB-KD.git
   cd ASMB-KD

	2.	Install dependencies:

pip install -r requirements.txt



⸻

Usage

1) Multi-Stage Distillation (main.py)

python main.py --config configs/partial_freeze.yaml --device cuda

	•	Adjust hyperparameters in configs/*.yaml (partial freeze, learning rates, etc.).

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

⸻

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
│   ├── vanilla_kd.py
│   └── __init__.py

├── models
│   ├── teachers/
│   ├── students/
│   ├── mbm.py
│   └── __init__.py

├── modules
│   ├── trainer_student.py
│   ├── trainer_teacher.py
│   ├── partial_freeze.py
│   ├── losses.py
│   └── __init__.py

├── results
│   ├── cifar100_asmb.csv
│   └── summary.csv

├── scripts
│   ├── run_many.sh
│   └── run_sweep.sh

└── utils
    ├── logger.py
    └── misc.py

	•	analysis/: Scripts or notebooks for comparing experiments (compare_ablation.py, plot_results.ipynb)
	•	configs/: YAML config files for partial-freeze settings, hyperparameters
	•	methods/: KD implementations (ASMB, FitNet, CRD, etc.)
	•	modules/: Partial freeze utility, trainers, custom losses
	•	results/: CSV logs, outputs from training/evaluation
	•	scripts/: Shell scripts for multiple or batch experiments

⸻

Results

Experiment outputs (CSV, logs, etc.) reside in the results/ folder.
You can run analysis/compare_ablation.py or analysis/plot_results.ipynb to analyze or visualize them.

⸻

License

This project is distributed under the MIT License.

MIT License

Copyright (c) 2024 Suyoung Yang

Permission is hereby granted, free of charge, to any person obtaining a copy
...


⸻

Citation

If you use this framework, please cite:

@misc{ASMB-KD,
  title   = {ASMB Knowledge Distillation Framework},
  author  = {Suyoung Yang},
  year    = {2024},
  howpublished = {\url{https://github.com/YourName/ASMB-KD}}
}


⸻

Contact

For questions or issues, please open a GitHub issue or email suyoung425@yonsei.ac.kr

