#!/usr/bin/env bash
# Simple Split-CIFAR run (5 tasks, quick debug)

python main.py \
  --config configs/cl.yaml \
  --cl_mode 1 \
  --num_tasks 5 \
  --epochs 3 \
  --batch_size 64
