# scripts/hparams.sh
# Common hyperparameters for ASMB experiments.
# Modify values below and run the bash scripts.

# General
DEVICE=cuda
BATCH_SIZE=128
SEED=42
DATA_AUG=1

# Teacher fine-tuning
FT_EPOCHS=50
FT_LR=0.001
FT_WD=0.0005
CUTMIX_ALPHA=1.0

# Distillation
T_LR=2e-4
S_LR=1e-2
T_WD=0.0003
S_WD=0.0005
CE_ALPHA=0.5
KD_ALPHA=0.5
CUTMIX_ALPHA_DISTILL=0.0   # CutMix alpha used during distillation
# Temperature scheduling
TEMPERATURE_SCHEDULE="fixed"
TAU_START=4.0
TAU_END=1.0
TAU_DECAY_EPOCHS=40
STUDENT_EPS=10
TEACHER_ITERS=20
STUDENT_ITERS=40
MBM_HIDDEN_DIM=1024
MBM_OUT_DIM=2048
MBM_REG=5e-5          # default sweep value
USE_PARTIAL_FREEZE=true
TEACHER1_USE_ADAPTER=0
TEACHER1_BN_HEAD_ONLY=0
TEACHER2_USE_ADAPTER=0
TEACHER2_BN_HEAD_ONLY=0
MIXUP_ALPHA=0.2
LABEL_SMOOTHING=0.1

# Sweep/loop variables
# N_STAGE_LIST accepts a space-separated list of stage counts,
# e.g. "2 3 4 5" to run multiple values in a batch script.
N_STAGE_LIST=5
SC_ALPHA_LIST="0.3 0.6"
STUDENT_LIST="resnet_adapter efficientnet_adapter swin_adapter"
