# scripts/run_sweep.sh

# Example: sweep teacher_lr & synergy_ce_alpha

for teacher_lr in 0.0001 0.0002 0.0005
do
  for sc_alpha in 0.2 0.3 0.5
  do
    echo "=========================================="
    echo "[RUN] teacher_lr=$teacher_lr synergy_ce_alpha=$sc_alpha"
    echo "=========================================="

    python main.py \
      --config configs/default.yaml \
      --teacher_lr $teacher_lr \
      --synergy_ce_alpha $sc_alpha \
      --device cuda
  done
done
