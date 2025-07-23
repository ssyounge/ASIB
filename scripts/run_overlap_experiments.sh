# scripts/run_overlap_experiments.sh
#!/usr/bin/env bash
set -e
export PYTHONPATH="$(pwd):$PYTHONPATH"

OVERLAPS="0 10 20 30 40 50 60 70 80 90 100"
KD_LIST="asmb vanilla_kd dkd at crd fitnet"
STUDENT="resnet152_adapter"

for P in $OVERLAPS; do
  ## 1) 두 교사 학습 (ResNet-152 ×2)
  CLS_A=$(python - <<PY
from data.cifar100_overlap import _split_classes
print(",".join(map(str,_split_classes($P)[0])))
PY
)
  CLS_B=$(python - <<PY
from data.cifar100_overlap import _split_classes
print(",".join(map(str,_split_classes($P)[1])))
PY
)

  for ID in A B; do
    CLS_VAR=CLS_${ID}
    CKPT="checkpoints/resnet152_overlap${P}_${ID}.pth"
    if [ ! -f "$CKPT" ]; then
      python scripts/fine_tuning.py \
        --config-name base \
        +teacher_type=resnet152 \
        +finetune_epochs=10 \
        +finetune_lr=3e-4 \
        +class_subset="${!CLS_VAR}" \
        +finetune_ckpt_path="$CKPT"
    fi
  done

  ## 2) Distillation – 모든 KD 방법 루프
  for KD in $KD_LIST; do
    python main.py \
      --config-name base \
      +overlap_pct=$P \
      +method=$KD \
      +teacher1_type=resnet152 \
      +teacher1_ckpt=checkpoints/resnet152_overlap${P}_A.pth \
      +teacher2_type=resnet152 \
      +teacher2_ckpt=checkpoints/resnet152_overlap${P}_B.pth \
      +student_type=$STUDENT \
      +results_dir=outputs/overlap_${P}/${KD} \
      +num_stages=4 +student_epochs_per_stage=15
  done

done
