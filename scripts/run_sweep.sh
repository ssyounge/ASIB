# scripts/run_sweep.sh

# Example: sweep teacher_lr & synergy_ce_alpha

TEACHER_LR_LIST="0.0001 0.0002 0.0005"
SC_ALPHA_LIST="0.2 0.3 0.5"
DISTILL_OVERRIDE=""    # 예) "student_lr=0.01,student_epochs_per_stage=15"

for teacher_lr in ${TEACHER_LR_LIST}
do
  for sc_alpha in ${SC_ALPHA_LIST}
  do
    echo "=========================================="
    echo "[RUN] teacher_lr=$teacher_lr synergy_ce_alpha=$sc_alpha"
    echo "=========================================="

    python main.py \
      --config configs/default.yaml \
      --teacher_lr $teacher_lr \
      --synergy_ce_alpha $sc_alpha \
      --device cuda \
      ${DISTILL_OVERRIDE:+--override "$DISTILL_OVERRIDE"}
  done
done
