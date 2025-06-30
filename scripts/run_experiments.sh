#!/usr/bin/env bash
# scripts/run_experiments.sh
set -e
export PYTHONPATH="$(pwd):${PYTHONPATH}"

LOG_ID=${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR="" # This will be populated by the --output_dir argument

BASE_CONFIG=${BASE_CONFIG:-configs/default.yaml}
USE_CONDA=${USE_CONDA:-1}
CONDA_ENV=${CONDA_ENV:-facil_env}

activate_conda() {
  if [ "$USE_CONDA" -eq 1 ]; then
    if command -v conda >/dev/null 2>&1; then
      if conda info --envs | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
        source ~/.bashrc
        conda activate "$CONDA_ENV"
      else
        echo "[run_experiments.sh] Conda env '$CONDA_ENV' not found. Skipping activation." >&2
      fi
    else
      echo "[run_experiments.sh] Conda executable not found. Skipping activation." >&2
    fi
  fi
}

# Generate config, echo path to a temporary file, and ensure cleanup on exit
generate_config() {
  local cfg_tmp
  cfg_tmp=$(mktemp)
  # Ensure temporary config is cleaned up even if the script exits early
  trap 'rm -f "$cfg_tmp"' EXIT
  python scripts/generate_config.py \
    --base "$BASE_CONFIG" \
    --hparams configs/hparams.yaml \
    --method "$METHOD" \
    --out "$cfg_tmp" \
    teacher_lr=${teacher_lr} \
    student_lr=${student_lr} \
    teacher_weight_decay=${teacher_weight_decay} \
    student_weight_decay=${student_weight_decay} \
    ce_alpha=${ce_alpha} \
    kd_alpha=${kd_alpha} \
    lr_schedule=${lr_schedule} \
    teacher_step_size=${teacher_step_size} \
    teacher_gamma=${teacher_gamma} \
    student_step_size=${student_step_size} \
    student_gamma=${student_gamma} \
    temperature_schedule=${temperature_schedule} \
    tau_start=${tau_start} \
    tau_end=${tau_end} \
    tau_decay_epochs=${tau_decay_epochs} \
    student_epochs_per_stage=${student_epochs_per_stage} \
    teacher_adapt_epochs=${teacher_adapt_epochs} \
    mbm_out_dim=${mbm_out_dim} \
    mbm_reg_lambda=${mbm_reg} \
    reg_lambda=${reg_lambda} \
    mbm_dropout=${mbm_dropout} \
    synergy_head_dropout=${head_dropout} \
    use_partial_freeze=${use_partial_freeze} \
    teacher1_use_adapter=${teacher1_use_adapter} \
    teacher1_bn_head_only=${teacher1_bn_head_only} \
    teacher2_use_adapter=${teacher2_use_adapter} \
    teacher2_bn_head_only=${teacher2_bn_head_only} \
    batch_size=${batch_size} \
    mixup_alpha=${mixup_alpha} \
    cutmix_alpha_distill=${cutmix_alpha_distill} \
    label_smoothing=${label_smoothing}
  echo "$cfg_tmp"
}

run_loop() {
  source <(python scripts/load_hparams.py configs/hparams.yaml)
  source <(python scripts/load_hparams.py configs/partial_freeze.yaml)
  
  METHOD_LIST="${method_list:-$method}"
  T1_LIST="${teacher1_list}"
  T2_LIST="${teacher2_list}"
  mkdir -p "${OUTPUT_DIR}"
  mkdir -p checkpoints

  for T1 in $T1_LIST; do
    for T2 in $T2_LIST; do
      for STUDENT in ${student_list}; do
        for METHOD in $METHOD_LIST; do
          echo ">>> [run_experiments.sh] running METHOD=${METHOD}"
          # 1) Teacher fine-tuning
          # Checkpoints are now saved to a global checkpoints folder
          for T in "$T1" "$T2"; do
        mkdir -p checkpoints # Ensure global checkpoint dir exists
        CKPT="checkpoints/${T}_ft.pth"
        # Only fine-tune when epochs>0 and checkpoint doesn't already exist
        if [ ${finetune_epochs} -gt 0 ] && [ ! -f "${CKPT}" ]; then
        echo ">>> [run_experiments.sh] fine-tuning teacher=${T}  (epochs=${finetune_epochs}, lr=${finetune_lr})"
        python scripts/fine_tuning.py \
          --teacher_type "${T}" \
          --device cuda \
          --batch_size ${batch_size} \
          --finetune_epochs ${finetune_epochs} \
          --finetune_lr ${finetune_lr} \
          --finetune_weight_decay ${finetune_weight_decay} \
          --finetune_cutmix_alpha ${finetune_cutmix_alpha} \
          --finetune_ckpt_path "${CKPT}" \
          --data_aug ${data_aug}
        fi
        done

        # 2) ASMB multi-stage distillation
        for SC_ALPHA in ${sc_alpha_list}; do
          # N_STAGE_LIST may contain space-separated values like "2 3 4 5"
          # Iterate over each item without quoting to allow word splitting.
        for STAGE in $n_stage_list; do
          EXP_ID="${METHOD}_${T2}_vs_${T1}_${STUDENT}_s${STAGE}_a${SC_ALPHA}"
          # Use the directory passed from run.sh as the final output location
          # No more nested directories
          OUTDIR="${OUTPUT_DIR}"
          CKPT_DIR="${OUTDIR}/checkpoints"
          mkdir -p "${CKPT_DIR}"

          CFG_TMP=$(generate_config)
          # Save the one true config file for this job
          cp "$CFG_TMP" "${OUTDIR}/config.yaml"

          if [ "$METHOD" = "asmb" ]; then
          python main.py \
            --config "${CFG_TMP}" \
            --teacher1_type "${T1}" \
            --teacher2_type "${T2}" \
            --finetune_epochs 0 \
            --student_type "${STUDENT}" \
            --num_stages ${STAGE} \
            --synergy_ce_alpha ${SC_ALPHA} \
            --teacher_lr ${teacher_lr} \
            --student_lr ${student_lr} \
            --batch_size ${batch_size} \
            --teacher1_use_adapter ${teacher1_use_adapter} \
            --teacher1_bn_head_only ${teacher1_bn_head_only} \
            --teacher2_use_adapter ${teacher2_use_adapter} \
            --teacher2_bn_head_only ${teacher2_bn_head_only} \
            --student_freeze_level ${student_freeze_level} \
            --results_dir "${OUTDIR}" \
            --ckpt_dir "${CKPT_DIR}" \
            --exp_id "${EXP_ID}" \
            --seed 42 \
            --data_aug ${data_aug} \
            --mixup_alpha ${mixup_alpha} \
            --cutmix_alpha_distill ${cutmix_alpha_distill} \
            --label_smoothing ${label_smoothing} \
            --method ${METHOD}
          else
          python scripts/run_single_teacher.py \
            --config "${CFG_TMP}" \
            --teacher_type "${T2}" \
            --student_type "${STUDENT}" \
            --student_lr ${student_lr} \
            --batch_size ${batch_size} \
            --epochs ${student_epochs_per_stage} \
            --student_freeze_level ${student_freeze_level} \
            --results_dir "${OUTDIR}" \
            --ckpt_dir "${CKPT_DIR}" \
            --seed 42 \
            --data_aug ${data_aug} \
            --mixup_alpha ${mixup_alpha} \
            --cutmix_alpha_distill ${cutmix_alpha_distill} \
            --label_smoothing ${label_smoothing} \
            --method ${METHOD}
          fi
        done
      done
    done
      done  # closes 'for STUDENT' loop
    done    # closes 'for T2' loop
  done      # closes 'for T1' loop
}

run_sweep() {
  source <(python scripts/load_hparams.py configs/hparams.yaml)
  source <(python scripts/load_hparams.py configs/partial_freeze.yaml)
  echo ">>> [run_experiments.sh] running METHOD=${METHOD}"

  # Use the first entry from the teacher lists for sweeps
  local T1="${teacher1_list%% *}"
  local T2="${teacher2_list%% *}"

  for teacher_lr in 0.0001 0.0002 0.0005; do
    for sc_alpha in 0.2 0.3 0.5; do
      echo "=========================================="
      echo "[RUN] teacher_lr=$teacher_lr synergy_ce_alpha=$sc_alpha"
      echo "=========================================="

      T_LR=${teacher_lr}
      CFG_TMP=$(generate_config)

      python main.py \
        --config "${CFG_TMP}" \
        --teacher1_type "${T1}" \
        --teacher2_type "${T2}" \
        --synergy_ce_alpha ${sc_alpha} \
        --device ${device} \
        --finetune_epochs 0 \
        --data_aug ${data_aug} \
        --mixup_alpha ${mixup_alpha} \
        --cutmix_alpha_distill ${cutmix_alpha_distill} \
        --teacher1_use_adapter ${teacher1_use_adapter} \
        --teacher1_bn_head_only ${teacher1_bn_head_only} \
        --teacher2_use_adapter ${teacher2_use_adapter} \
        --teacher2_bn_head_only ${teacher2_bn_head_only} \
        --student_freeze_level ${student_freeze_level} \
        --label_smoothing ${label_smoothing} \
        --method ${METHOD}
    done
  done
}

MODE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [ -z "$MODE" ]; then
  echo "Usage: $0 --mode {loop,sweep} [--output_dir DIR]" >&2
  exit 1
fi

activate_conda

case "$MODE" in
  loop)
    run_loop
    ;;
  sweep)
    run_sweep
    ;;
  *)
    echo "Invalid mode: $MODE" >&2
    exit 1
    ;;
esac
