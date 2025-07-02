#!/usr/bin/env bash
#SBATCH --job-name=ibkd_cifar
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH -o outputs/ibkd_%j.log

set -e
source ~/.bashrc
conda activate facil_env

T1_CKPT="ckpts/resnet152_ft.pth"
T2_CKPT="ckpts/efficientnet_b2_ft.pth"

python main.py --cfg configs/minimal.yaml \
       --teacher1_type resnet152 \
       --teacher2_type efficientnet_b2 \
       --teacher1_ckpt ${T1_CKPT} \
       --teacher2_ckpt ${T2_CKPT}

