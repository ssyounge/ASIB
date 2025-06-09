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
TEMPERATURE=4.0
STUDENT_EPS=15
TEACHER_ITERS=10
STUDENT_ITERS=20
MBM_HIDDEN_DIM=1024
MBM_OUT_DIM=2048
MBM_REG=1e-4          # default sweep value
USE_PARTIAL_FREEZE=true
TEACHER1_USE_ADAPTER=0
TEACHER1_BN_HEAD_ONLY=0
TEACHER2_USE_ADAPTER=0
TEACHER2_BN_HEAD_ONLY=0
MIXUP_ALPHA=0.2
LABEL_SMOOTHING=0.1

# Sweep/loop variables
N_STAGE_LIST="2 3"
SC_ALPHA_LIST="0.3 0.6"
STUDENT_LIST="resnet_adapter efficientnet_adapter swin_adapter"
