#!/usr/bin/env bash
# scripts/run_many.sh
set -e
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# Conda setup (optional)
USE_CONDA=${USE_CONDA:-1}
CONDA_ENV=${CONDA_ENV:-facil_env}
if [ "$USE_CONDA" -eq 1 ]; then
  if command -v conda >/dev/null 2>&1; then
    if conda info --envs | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
      source ~/.bashrc
      conda activate "$CONDA_ENV"
    else
      echo "[run_many.sh] Conda env '$CONDA_ENV' not found. Skipping activation." >&2
    fi
  else
    echo "[run_many.sh] Conda executable not found. Skipping activation." >&2
  fi
fi

# ---------------------------------------------------------------------------
# Central hyperparameter config
# ---------------------------------------------------------------------------
source "$(dirname "$0")/hparams.sh"

mkdir -p checkpoints results
RESULT_ROOT="results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RESULT_ROOT}"

T1=resnet101
for T2 in efficientnet_b2 swin_tiny; do
  ###########################################################################
  # 1) Teacher fine-tuning (skip if ckpt already exists)
  ###########################################################################
  for T in "$T1" "$T2"; do
    CKPT="checkpoints/${T}_ft.pth"
    echo ">>> [run_many.sh] fine-tuning teacher=${T}  (epochs=${FT_EPOCHS}, lr=${FT_LR})"
    if [ ! -f "${CKPT}" ]; then
      python scripts/fine_tuning.py \
        --teacher_name "${T}" \
        --device cuda \
        --batch_size ${BATCH_SIZE} \
        --finetune_epochs ${FT_EPOCHS} \
        --finetune_lr ${FT_LR} \
        --finetune_weight_decay ${FT_WD} \
        --cutmix_alpha ${CUTMIX_ALPHA} \
        --finetune_ckpt_path "${CKPT}" \
        --data_aug ${DATA_AUG}
    fi
  done

  ###########################################################################
  # 2) ASMB multi-stage distillation
  ###########################################################################
  for STUDENT in ${STUDENT_LIST}; do
    for SC_ALPHA in ${SC_ALPHA_LIST}; do
      for STAGE in ${N_STAGE_LIST}; do
        OUTDIR="${RESULT_ROOT}/${T2}_${STUDENT}_a${SC_ALPHA}_s${STAGE}"
        mkdir -p "${OUTDIR}"

        CFG_TMP=$(mktemp)
        python scripts/generate_config.py \
          --base configs/default.yaml \
          --out "$CFG_TMP" \
          teacher_lr=${T_LR} \
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
          mbm_reg_lambda=${MBM_REG} \
          use_partial_freeze=${USE_PARTIAL_FREEZE} \
          batch_size=${BATCH_SIZE} \
          mixup_alpha=${MIXUP_ALPHA} \
          label_smoothing=${LABEL_SMOOTHING}

        python main.py \
          --config "$CFG_TMP" \
          --teacher1_type "${T1}" \
          --teacher2_type "${T2}" \
          --teacher1_ckpt checkpoints/${T1}_ft.pth \
          --teacher2_ckpt checkpoints/${T2}_ft.pth \
          --student_type "${STUDENT}" \
          --num_stages ${STAGE} \
          --synergy_ce_alpha ${SC_ALPHA} \
          --teacher_lr ${T_LR} \
          --student_lr ${S_LR} \
          --batch_size ${BATCH_SIZE} \
          --results_dir "${OUTDIR}" \
          --seed 42 \
          --data_aug ${DATA_AUG} \
          --mixup_alpha ${MIXUP_ALPHA} \
          --label_smoothing ${LABEL_SMOOTHING}
      done
    done
  done
done
