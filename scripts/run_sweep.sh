#!/usr/bin/env bash
# scripts/run_sweep.sh

set -e
export PYTHONPATH="$(pwd):${PYTHONPATH}"

source ~/.bashrc
conda activate facil_env

source "$(dirname "$0")/hparams.sh"

# Example sweep over teacher_lr & synergy_ce_alpha
for teacher_lr in 0.0001 0.0002 0.0005; do
  for sc_alpha in 0.2 0.3 0.5; do
    echo "=========================================="
    echo "[RUN] teacher_lr=$teacher_lr synergy_ce_alpha=$sc_alpha"
    echo "=========================================="

    CFG_TMP=$(mktemp)
    python scripts/generate_config.py \
      --base configs/default.yaml \
      --out "$CFG_TMP" \
      teacher_lr=${teacher_lr} \
      student_lr=${S_LR} \
      teacher_weight_decay=${T_WD} \
      student_weight_decay=${S_WD} \
      ce_alpha=${CE_ALPHA} \
      kd_alpha=${KD_ALPHA} \
      temperature=${TEMPERATURE} \
      student_epochs_per_stage=${STUDENT_EPS} \
      teacher_iters=${TEACHER_ITERS} \
      student_iters=${STUDENT_ITERS} \
      mbm_hidden_dim=${MBM_HIDDEN_DIM} \
      mbm_out_dim=${MBM_OUT_DIM} \
      use_partial_freeze=${USE_PARTIAL_FREEZE} \
      batch_size=${BATCH_SIZE}

    python main.py \
      --config "$CFG_TMP" \
      --synergy_ce_alpha ${sc_alpha} \
      --device ${DEVICE} \
      --data_aug ${DATA_AUG}
  done
done
