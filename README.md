# ASMB Knowledge Distillation (Minimal)

This repository provides a lightweight implementation of **Adaptive Synergy Manifold Bridging** (ASMB) for multi-stage knowledge distillation on CIFAR‑100. Only the components required for information bottleneck KD are included.

## Installation

Install the minimal dependencies:

```bash
pip install -r requirements.txt
```
The project only requires **torch**, **torchvision**, **PyYAML** and **tqdm**.

## Workflow

1. **Fine-tune teachers** with cross entropy:

```bash
python scripts/train_teacher.py --teacher resnet152 --ckpt checkpoints/resnet152_ft.pth
python scripts/train_teacher.py --teacher efficientnet_b2 --ckpt checkpoints/efficientnet_b2_ft.pth
```

Pass `--overwrite` to `scripts/fine_tuning.py` to remove an existing checkpoint
before starting a new run:

```bash
python scripts/fine_tuning.py --teacher_type resnet152 --overwrite
```

2. **Run IB-KD** using the provided checkpoints:

```bash
bash scripts/run_ibkd.sh
```

Both scripts read default options from `configs/minimal.yaml`.
Set `disable_tqdm: true` in that file to suppress progress bars during training.
Set `grad_scaler_init_scale` to control the initial scale used by the AMP grad
scaler.

## Directory Layout

```
configs/
    minimal.yaml        # experiment settings
checkpoints/            # teacher checkpoints
models/                 # teacher and student architectures
scripts/
    train_teacher.py    # CE fine-tuning
    run_ibkd.sh         # distillation entry point
utils/
main.py                 # training driver
trainer.py              # training loops
```

## Testing

Unit tests require PyTorch. Run:

```bash
pytest -q
```

## License

This project is released under the MIT license.
