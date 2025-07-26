#!/usr/bin/env bash
# scripts/run_experiments.sh
set -e
export PYTHONPATH="$(pwd):${PYTHONPATH}"

LOG_ID=${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR="" # This will be populated by the --output_dir argument

USE_CONDA=${USE_CONDA:-1}
CONDA_ENV=${CONDA_ENV:-tlqkf}

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

# Main experiment loop
run_loop() {
  
  METHOD_LIST="${method_list:-$method}"
  T1_LIST="${teacher1_list}"
  T2_LIST="${teacher2_list}"
  STUD_LIST="${student_list}"
  IFS=' ' read -ra T1S <<< "$T1_LIST"
  IFS=' ' read -ra T2S <<< "$T2_LIST"
  IFS=' ' read -ra STUDS <<< "$STUD_LIST"
  mkdir -p "${OUTPUT_DIR}"
  mkdir -p checkpoints

  for TEACH_EP in ${teacher_adapt_epochs_list}; do
    for STUD_EP in ${student_epochs_per_stage_list}; do
      teacher_adapt_epochs=${TEACH_EP}
      student_epochs_per_stage=${STUD_EP}
      for T1 in "${T1S[@]}"; do
        for T2 in "${T2S[@]}"; do
          for STUDENT in "${STUDS[@]}"; do
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
          --config-name base \
          +teacher_type="${T}" \
          device=cuda \
          batch_size="${batch_size}" \
          +finetune_epochs="${finetune_epochs}" \
          +finetune_lr="${finetune_lr}" \
          +finetune_weight_decay="${finetune_weight_decay}" \
          +finetune_cutmix_alpha="${finetune_cutmix_alpha}" \
          +finetune_ckpt_path="${CKPT}" \
          +data_aug="${data_aug}"
        fi
        done

        # 2) ASMB multi-stage distillation
        for SC_ALPHA in ${sc_alpha_list}; do
          for H_BETA in ${hybrid_beta_list}; do
            synergy_ce_alpha=${SC_ALPHA}
            hybrid_beta=${H_BETA}
            # N_STAGE_LIST may contain space-separated values like "2 3 4 5"
            # Iterate over each item without quoting to allow word splitting.
        for STAGE in $n_stage_list; do
          EXP_ID="${METHOD}_${T2}_vs_${T1}_${STUDENT}_s${STAGE}_a${SC_ALPHA}"
          # Use the directory passed from run.sh as the final output location
          # No more nested directories
          OUTDIR="${OUTPUT_DIR}"
          CKPT_DIR="${OUTDIR}/checkpoints"
          mkdir -p "${CKPT_DIR}"

          if [ "$METHOD" = "asmb" ]; then
          python main.py \
            --config-name base \
            +teacher1_type="${T1}" \
            +teacher2_type="${T2}" \
            +finetune_epochs=0 \
            +student_type="${STUDENT}" \
            +num_stages="${STAGE}" \
            +synergy_ce_alpha="${SC_ALPHA}" \
            +hybrid_beta="${H_BETA}" \
            +ib_beta="${ib_beta}" \
            +teacher_lr="${teacher_lr}" \
            +student_lr="${student_lr}" \
            batch_size="${batch_size}" \
            +teacher1_use_adapter="${teacher1_use_adapter}" \
            +teacher1_bn_head_only="${teacher1_bn_head_only}" \
            +teacher2_use_adapter="${teacher2_use_adapter}" \
            +teacher2_bn_head_only="${teacher2_bn_head_only}" \
            +student_freeze_level="${student_freeze_level}" \
            +results_dir="${OUTDIR}" \
            +ckpt_dir="${CKPT_DIR}" \
            +exp_id="${EXP_ID}" \
            seed=42 \
            +data_aug="${data_aug}" \
            +mixup_alpha="${mixup_alpha}" \
            +cutmix_alpha_distill="${cutmix_alpha_distill}" \
            +label_smoothing="${label_smoothing}" \
            method="${METHOD}" \
            "${EXTRA_ARGS[@]}"
          else
          python scripts/run_single_teacher.py \
            --config-name base \
            +teacher_type="${T2}" \
            +student_type="${STUDENT}" \
            +student_lr="${student_lr}" \
            batch_size="${batch_size}" \
            +epochs="${student_epochs_per_stage}" \
            +student_freeze_level="${student_freeze_level}" \
            +results_dir="${OUTDIR}" \
            +ckpt_dir="${CKPT_DIR}" \
            seed=42 \
            +data_aug="${data_aug}" \
            +mixup_alpha="${mixup_alpha}" \
            +cutmix_alpha_distill="${cutmix_alpha_distill}" \
            +label_smoothing="${label_smoothing}" \
            method="${METHOD}" \
            "${EXTRA_ARGS[@]}"
          fi 2>&1 | tee -a "${OUTDIR}/train.log"
                done            # closes STAGE loop
              done              # closes 'for H_BETA' loop
            done                # closes 'for SC_ALPHA' loop
          done                  # closes 'for METHOD' loop
        done                    # closes 'for STUDENT' loop
      done                      # closes 'for T2' loop
    done                        # closes 'for T1' loop
  done                          # closes 'for STUD_EP' loop
done                            # closes 'for TEACH_EP' loop
}

run_sweep() {
  echo ">>> [run_experiments.sh] running METHOD=${METHOD}"

  # Use the first entry from the teacher lists for sweeps
  local T1="${teacher1_list%% *}"
  local T2="${teacher2_list%% *}"

  for teacher_lr in 0.0001 0.0002 0.0005; do
    for sc_alpha in 0.2 0.3 0.5; do
      for h_beta in 0.1 0.5 0.9; do
      echo "=========================================="
      echo "[RUN] teacher_lr=$teacher_lr synergy_ce_alpha=$sc_alpha hybrid_beta=$h_beta"
      echo "=========================================="

      synergy_ce_alpha=${sc_alpha}
      hybrid_beta=${h_beta}
      python main.py \
        --config-name base \
        +teacher1_type="${T1}" \
        +teacher2_type="${T2}" \
        +synergy_ce_alpha="${sc_alpha}" \
        +hybrid_beta="${h_beta}" \
        +ib_beta="${ib_beta}" \
        device="${device}" \
        +finetune_epochs=0 \
        +data_aug="${data_aug}" \
        +mixup_alpha="${mixup_alpha}" \
        +cutmix_alpha_distill="${cutmix_alpha_distill}" \
        +teacher1_use_adapter="${teacher1_use_adapter}" \
        +teacher1_bn_head_only="${teacher1_bn_head_only}" \
        +teacher2_use_adapter="${teacher2_use_adapter}" \
        +teacher2_bn_head_only="${teacher2_bn_head_only}" \
        +student_freeze_level="${student_freeze_level}" \
        +label_smoothing="${label_smoothing}" \
        method="${METHOD}" 2>&1 | tee -a "${OUTPUT_DIR}/train.log"
      done
    done
  done
}

MODE=""
# ---------- 인자 파싱 ----------
#   알 수 없는 옵션은 EXTRA_ARGS 배열에 보관해서
#   나중에 python main.py ... "${EXTRA_ARGS[@]}" 로 그대로 전달한다.
POSITIONAL=()
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
      POSITIONAL+=("$1")      # ← pass-through
      shift
      ;;
  esac
done
# 파이썬 커맨드에 붙일 여분 인자
EXTRA_ARGS=("${POSITIONAL[@]}")

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
