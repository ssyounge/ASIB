#scripts/run_many.sh

# run_many.sh
# Example script to launch multiple KD experiments (ASMB or others)
# by sweeping over parameters like teacher1, teacher2, alpha, stage, etc.

# If you want to run them in parallel, you can add "&" to each python line,
# but be mindful of GPU resource conflicts.

# Basic usage: 
#   chmod +x run_many.sh
#   ./run_many.sh

METHOD="asmb"      # or "fitnet", "crd" ...
LR=0.0002
EPOCHS=10
BATCH=128

for TEACHER1 in "resnet50" "mobilenet_v2"
do
  for TEACHER2 in "efficientnet_b2" "densenet121"
  do
    for ALPHA in 0.3 0.6
    do
      for STAGE in 2 3
      do
        echo "============================================"
        echo "Running experiment: method=$METHOD, t1=$TEACHER1, t2=$TEACHER2, alpha=$ALPHA, stage=$STAGE"
        echo "============================================"
        
        # Option 1) Run in the foreground
        python main.py \
          --method $METHOD \
          --teacher1 $TEACHER1 \
          --teacher2 $TEACHER2 \
          --alpha $ALPHA \
          --stage $STAGE \
          --lr $LR \
          --epochs $EPOCHS \
          --batch_size $BATCH \
          --results_dir "results" \
          --seed 42

        # Option 2) If you want to run in background parallel, add '&'
        # python main.py ... &

        # If running in parallel, you might want a wait or a GPU scheduling logic 
        # (like CUDA_VISIBLE_DEVICES or tools like slurm).
      done
    done
  done
done

echo "All experiments done."
