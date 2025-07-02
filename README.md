# ASMB Knowledge Distillation (Minimal)

This repository provides a lightweight implementation of **Adaptive Synergy Manifold Bridging** (ASMB) for multi-stage knowledge distillation on CIFAR‑100. Only the components required for information bottleneck KD are included.

## Installation

```bash
pip install -r requirements.txt
```

## Workflow

1. **Fine-tune teachers** with cross entropy:

```bash
python scripts/train_teacher.py --teacher resnet152 --ckpt ckpts/resnet152_ft.pth
python scripts/train_teacher.py --teacher efficientnet_b2 --ckpt ckpts/efficientnet_b2_ft.pth
```

2. **Run IB-KD** using the provided checkpoints:

```bash
bash scripts/run_ibkd.sh
```

Both scripts read default options from `configs/minimal.yaml`.

## Directory Layout

```
configs/
    minimal.yaml        # experiment settings
ckpts/                  # teacher checkpoints
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
