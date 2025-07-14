# ASMB Knowledge Distillation (Minimal)

This repository provides a lightweight implementation of **Adaptive Synergy Manifold Bridging** (ASMB) for multi-stage knowledge distillation on CIFAR‑100. Only the components required for information bottleneck KD are included.

## Installation

Install the minimal dependencies:

```bash
# ①  필수 라이브러리
pip install -r requirements.txt

# ②  (선택) GPU 용 PyTorch\u202f휠 직접 설치
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# ③  테스트 실행
pytest  # 모든 테스트가 PASS 또는 SKIP 없이 실행돼야 합니다.
```
The project only requires **torch**, **torchvision**, **pyyaml**, **tqdm** and **pandas**.

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

3. **(Optional) CE baseline** without distillation:

```bash
python main.py --config configs/base.yaml --method ce
```

Both scripts read default options from `configs/base.yaml`.
Specify comma-separated checkpoint paths under `teacher1_ckpt` or
`teacher2_ckpt` to load a snapshot ensemble. Place any comments on a separate
line rather than at the end of the list.
Set `disable_tqdm: true` in that file to suppress progress bars during training.
Set `grad_scaler_init_scale` to control the initial scale used by the AMP grad
scaler.
Use `grad_clip_norm_init`, `grad_clip_norm_final` and
`grad_clip_warmup_frac` to schedule gradient clipping during student updates.

## Distillation Methods

| `method` | Reference | Implementation |
|---------|-----------|----------------|
| `at` | Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer | `methods/at.py` |
| `fitnet` | FitNets: Hints for Thin Deep Nets | `methods/fitnet.py` |

Use `--method` to override the selected distillation method when calling `main.py`,
or set `method` in `configs/base.yaml`.

## Directory Layout

```
configs/
    base.yaml           # experiment settings
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

Run all tests:

```bash
pytest
```

## Logging

We rely on Weights & Biases for experiment tracking.
If you do not wish to use WandB, run with WANDB_MODE=disabled.

## License

This project is released under the MIT license.
