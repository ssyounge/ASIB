#!/usr/bin/env bash
# examples/run_cifar100_cl.sh
# Simple Split-CIFAR run (5 tasks, quick debug)

python main.py \
  --config-name base \
  --cl_mode 1 \
  --num_tasks 5 \
  --epochs 3 \
  --batch_size 64
