#!/usr/bin/env bash
set -e
export PYTHONPATH="$(pwd):${PYTHONPATH}"

###############################################################################
# ★ 여기 변수만 바꾸면 학습 전부 재조정할 수 있습니다 ★
###############################################################################
# (A) Fine-tune Teacher
FT_EPOCHS=50           # --finetune_epochs
FT_LR=0.001            # --finetune_lr
FT_WD=0.0005           # --finetune_weight_decay
FT_BATCH=128           # --batch_size  (fine-tune & distill 공통 사용)
CUTMIX_ALPHA=1.0       # --cutmix_alpha

# (B) ASMB Distillation
T_LR=2e-4              # --teacher_lr  (adaptive update)
S_LR=1e-2              # --student_lr
N_STAGE_LIST="2 3"     # for STAGE in …
SC_ALPHA_LIST="0.3 0.6"
STUDENT_LIST="resnet_adapter efficientnet_adapter swin_adapter"
###############################################################################

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
        --batch_size ${FT_BATCH} \
        --finetune_epochs ${FT_EPOCHS} \
        --finetune_lr ${FT_LR} \
        --finetune_weight_decay ${FT_WD} \
        --cutmix_alpha ${CUTMIX_ALPHA} \
        --finetune_ckpt_path "${CKPT}"
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

        python main.py \
          --teacher1_type "${T1}" \
          --teacher2_type "${T2}" \
          --teacher1_ckpt checkpoints/${T1}_ft.pth \
          --teacher2_ckpt checkpoints/${T2}_ft.pth \
          --student_type "${STUDENT}" \
          --num_stages ${STAGE} \
          --synergy_ce_alpha ${SC_ALPHA} \
          --teacher_lr ${T_LR} \
          --student_lr ${S_LR} \
          --batch_size ${FT_BATCH} \
          --results_dir "${OUTDIR}" \
          --seed 42
      done
    done
  done
done