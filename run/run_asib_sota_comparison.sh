#!/usr/bin/env bash
#SBATCH --job-name=asib_sota_comparison
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --array=0-7%8          # 4개 샤드, 동시 3개만 실행 (원하는 병렬도로 % 값 조절)
#SBATCH --output=experiments/sota/logs/sota_comparison_%A_%a.log
#SBATCH --error=experiments/sota/logs/sota_comparison_%A_%a.err
# ---------------------------------------------------------
# ASIB SOTA Comparison 실험
# ASIB vs State-of-the-Art Methods 비교
# ---------------------------------------------------------
set -euo pipefail
trap 'echo "❌ Job failed at $(date)"; exit 1' ERR

# Python 환경 설정 (ablation과 동일 스타일)
echo "🔧 Setting up Python environment..."
set +u
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
conda activate tlqkf || {
  export PATH="$HOME/anaconda3/envs/tlqkf/bin:$PATH"
}
set -u
export HYDRA_FULL_ERROR=1
echo "✅ Python environment setup completed"
echo ""

# 1) 리포 최상위로 이동
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

# 2) PYTHONPATH 추가
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

# DataLoader workers 자동 스케일
calc_workers() {
  local cpus="${SLURM_CPUS_PER_TASK:-4}"
  local maxw="${PYTORCH_WORKERS_MAX:-4}"
  (( cpus < 1 )) && cpus=1
  if (( maxw < cpus )); then echo "$maxw"; else echo "$cpus"; fi
}
WORKERS="$(calc_workers)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$WORKERS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$WORKERS}"
echo "🧵 num_workers(auto)=${WORKERS}, OMP=${OMP_NUM_THREADS}, MKL=${MKL_NUM_THREADS}"

# Array → 샤딩 자동 매핑
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  SHARD_IDX="${SLURM_ARRAY_TASK_ID}"
fi
if [[ -n "${SLURM_ARRAY_TASK_COUNT:-}" ]]; then
  SHARD_N="${SLURM_ARRAY_TASK_COUNT}"
elif [[ -n "${SLURM_ARRAY_TASK_MAX:-}" && -n "${SLURM_ARRAY_TASK_MIN:-}" ]]; then
  SHARD_N="$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))"
fi
SHARD_N="${SHARD_N:-1}"
SHARD_IDX="${SHARD_IDX:-0}"
run_idx=0

# Ensure log directory exists
mkdir -p "$ROOT/experiments/sota/logs" || true

# 3) GPU 할당 확인 (Slurm Job Array에서는 자동 바인딩)
echo "🔍 Slurm is expected to set CUDA_VISIBLE_DEVICES automatically for each array task."

# CUDA 컨텍스트 초기화 (segmentation fault 방지)
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# PyTorch CUDA 라이브러리 경로 가드
TORCH_LIB_DIR="$HOME/anaconda3/envs/tlqkf/lib/python3.12/site-packages/torch/lib"
if [ -d "$TORCH_LIB_DIR" ]; then
  unset LD_LIBRARY_PATH || true
  export CUDA_HOME="$TORCH_LIB_DIR"
fi

# PyTorch CUDA 설정
export TORCH_CUDA_ARCH_LIST="8.6"

# CUDA 환경변수 (PyTorch 내장 CUDA 사용)
export CUDA_PATH="${CUDA_HOME:-${CUDA_PATH:-}}"
export CUDA_ROOT="${CUDA_HOME:-${CUDA_ROOT:-}}"

# GPU 정보 출력
echo "🔍 GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits



# 4) SOTA 그리드 실행 블록 ----------------------------------------------------

# 교사 페어 조합
TEACHER_PAIRS=(
  "resnet152,convnext_s"
  "convnext_l,efficientnet_l2"
  "resnet152,resnet152"
)

# 교사 체크포인트 경로 매핑 (환경에 맞게 수정)
declare -A CKPT=(
  [resnet152]="checkpoints/teachers/resnet152_cifar100.pth"
  [convnext_s]="checkpoints/teachers/convnext_s_cifar100.pth"
  [convnext_l]="checkpoints/teachers/convnext_l_cifar100.pth"
  [efficientnet_l2]="checkpoints/teachers/efficientnet_l2_cifar100.pth"
)

# 학생/메소드/시드 (학생은 일반 이름 사용; scratch는 override로 제어)
STUDENTS=(resnet50 mobilenet_v2 efficientnet_b0 shufflenet_v2)
METHODS=(avg_kd vanilla_kd ab at dkd fitnet ft reviewkd crd simkd sskd asib)
SEEDS=(42)

# 서브셋 필터(환경변수로 지정 가능)
ONLY_METHODS="${ONLY_METHODS:-}"
ONLY_STUDENTS="${ONLY_STUDENTS:-}"
ONLY_PAIRS="${ONLY_PAIRS:-}"
ONLY_SEEDS="${ONLY_SEEDS:-}"

# 샤딩(선택): 위에서 Array → 샤딩 자동 매핑으로 설정됨

# 메소드별 Hydra override 빌더
method_overrides() {
  local method="$1"; local student="$2"
  # 공통: scratch, 공정성(교사 FT/PPF/CCCP OFF)
  local base=" +experiment.kd_warmup_epochs=3 +experiment.kd_max_ratio=1.25 +experiment.tau=4.0 +experiment.mixup_alpha=0.0 +experiment.cutmix_alpha_distill=0.0 +experiment.use_distillation_adapter=true +experiment.model.student.pretrained=false +experiment.use_teacher_finetuning=false +experiment.train_distill_adapter_only=false +experiment.use_partial_freeze=false +experiment.student_freeze_bn=false +experiment.compute_teacher_eval=true +experiment.optimizer=sgd +experiment.student_lr=0.1 +experiment.student_weight_decay=0.0005 +experiment.b_step_momentum=0.9 +experiment.b_step_nesterov=true "
  # 작은 학생 모델은 어댑터 차원을 256으로 축소
  if [[ "$student" == "mobilenet_v2" || "$student" == "efficientnet_b0" || "$student" == "shufflenet_v2" ]]; then
    base+=" +experiment.distill_out_dim=256 "
  else
    base+=" +experiment.distill_out_dim=512 "
  fi
  if [[ "$student" == "mobilenet_v2" || "$student" == "efficientnet_b0" || "$student" == "shufflenet_v2" ]]; then
    base+=" +experiment.distill_out_dim=256 "
  else
    base+=" +experiment.distill_out_dim=512 "
  fi
  case "$method" in
    avg_kd)
      echo "experiment/method@experiment.experiment.method=vanilla_kd +method_name=avg_kd $base +experiment.kd_target=avg +experiment.ce_alpha=0.65 +experiment.kd_alpha=0.35 +experiment.use_ib=false +experiment.use_cccp=false +experiment.feat_kd_alpha=0.0";;
    vanilla_kd)
      echo "experiment/method@experiment.experiment.method=vanilla_kd +method_name=vanilla_kd $base +experiment.kd_target=teacher +experiment.kd_teacher_index=0 +experiment.ce_alpha=0.65 +experiment.kd_alpha=0.35 +experiment.use_ib=false +experiment.use_cccp=false +experiment.feat_kd_alpha=0.0";;
    ab)
      echo "experiment/method@experiment.experiment.method=ab +method_name=ab $base +experiment.kd_target=teacher +experiment.kd_teacher_index=0 +experiment.ce_alpha=0.65 +experiment.kd_alpha=0.35 +experiment.use_ib=false +experiment.use_cccp=false +experiment.feat_kd_alpha=0.0";;
    at)
      echo "experiment/method@experiment.experiment.method=at +method_name=at $base +experiment.kd_target=teacher +experiment.kd_teacher_index=0 +experiment.ce_alpha=0.65 +experiment.kd_alpha=0.35 +experiment.use_ib=false +experiment.use_cccp=false +experiment.feat_kd_alpha=0.0";;
    dkd)
      echo "experiment/method@experiment.experiment.method=dkd +method_name=dkd $base +experiment.kd_target=teacher +experiment.kd_teacher_index=0 +experiment.ce_alpha=0.65 +experiment.kd_alpha=0.35 +experiment.use_ib=false +experiment.use_cccp=false +experiment.feat_kd_alpha=0.0";;
    fitnet)
      echo "experiment/method@experiment.experiment.method=fitnet +method_name=fitnet $base +experiment.kd_target=teacher +experiment.kd_teacher_index=0 +experiment.ce_alpha=0.65 +experiment.kd_alpha=0.35 +experiment.use_ib=false +experiment.use_cccp=false +experiment.feat_kd_alpha=0.0";;
    ft)
      echo "experiment/method@experiment.experiment.method=ft +method_name=ft $base +experiment.kd_target=teacher +experiment.kd_teacher_index=0 +experiment.ce_alpha=0.65 +experiment.kd_alpha=0.35 +experiment.use_ib=false +experiment.use_cccp=false +experiment.feat_kd_alpha=0.0";;
    reviewkd)
      echo "experiment/method@experiment.experiment.method=reviewkd +method_name=reviewkd $base +experiment.kd_target=teacher +experiment.kd_teacher_index=0 +experiment.ce_alpha=0.65 +experiment.kd_alpha=0.35 +experiment.use_ib=false +experiment.use_cccp=false +experiment.feat_kd_alpha=0.0 +experiment.dataset.batch_size=96";;
    crd)
      echo "experiment/method@experiment.experiment.method=crd +method_name=crd $base +experiment.kd_target=teacher +experiment.kd_teacher_index=0 +experiment.ce_alpha=0.65 +experiment.kd_alpha=0.35 +experiment.use_ib=false +experiment.use_cccp=false +experiment.feat_kd_alpha=0.0 +experiment.dataset.batch_size=96";;
    simkd)
      echo "experiment/method@experiment.experiment.method=simkd +method_name=simkd $base +experiment.kd_target=teacher +experiment.kd_teacher_index=0 +experiment.ce_alpha=0.65 +experiment.kd_alpha=0.35 +experiment.use_ib=false +experiment.use_cccp=false +experiment.feat_kd_alpha=0.0";;
    sskd)
      echo "experiment/method@experiment.experiment.method=sskd +method_name=sskd $base +experiment.kd_target=teacher +experiment.kd_teacher_index=0 +experiment.ce_alpha=0.65 +experiment.kd_alpha=0.35 +experiment.use_ib=false +experiment.use_cccp=false +experiment.feat_kd_alpha=0.0";;
    asib)
      echo "experiment/method@experiment.experiment.method=asib +method_name=asib $base +experiment.kd_target=synergy +experiment.ce_alpha=0.65 +experiment.kd_alpha=0.35 +experiment.use_ib=true +experiment.use_cccp=false +experiment.teacher_adapt_epochs=6 +experiment.ib_beta=0.0001 +experiment.ib_beta_warmup_epochs=5";;
    *) echo ""; return 1;;
  esac
}

RUNS=()
for pair in "${TEACHER_PAIRS[@]}"; do
  IFS=',' read -r T1 T2 <<< "$pair"
  if [[ -n "$ONLY_PAIRS" && "$ONLY_PAIRS" != *"$pair"* ]]; then continue; fi
  for student in "${STUDENTS[@]}"; do
    if [[ -n "$ONLY_STUDENTS" && "$ONLY_STUDENTS" != *"$student"* ]]; then continue; fi
    for method in "${METHODS[@]}"; do
      if [[ -n "$ONLY_METHODS" && "$ONLY_METHODS" != *"$method"* ]]; then continue; fi
      for seed in "${SEEDS[@]}"; do
        if [[ -n "$ONLY_SEEDS" && "$ONLY_SEEDS" != *"$seed"* ]]; then continue; fi
        RUNS+=("${T1},${T2}|${student}|${method}|${seed}")
      done
    done
  done
done

N_RUNS=${#RUNS[@]}
echo "🧮 Planned runs: ${N_RUNS}"

# 배열 크기에 맞게 RUNS 자동 확장 (seed 증가로 유니크 보장)
if [[ -n "${SLURM_ARRAY_TASK_COUNT:-}" ]]; then
  TARGET=${SLURM_ARRAY_TASK_COUNT}
  if (( N_RUNS < TARGET && N_RUNS > 0 )); then
    NEW_RUNS=()
    for (( i=0; i<TARGET; i++ )); do
      base_idx=$(( i % N_RUNS ))
      IFS='|' read -r pair_b student_b method_b seed_b <<< "${RUNS[$base_idx]}"
      NEW_SEED=$(( seed_b + i ))
      NEW_RUNS+=("${pair_b}|${student_b}|${method_b}|${NEW_SEED}")
    done
    RUNS=("${NEW_RUNS[@]}")
    N_RUNS=${#RUNS[@]}
    echo "🧩 Auto-expanded RUNS to match array: ${N_RUNS}"
  fi
fi

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  idx="${SLURM_ARRAY_TASK_ID}"
  if (( idx >= N_RUNS )); then
    echo "ℹ️  Array index ${idx} >= N_RUNS ${N_RUNS} → nothing to do."; exit 0
  fi
  IFS='|' read -r pair student method seed <<< "${RUNS[$idx]}"
  IFS=',' read -r T1 T2 <<< "$pair"
  RUNS=("${T1},${T2}|${student}|${method}|${seed}")
fi

for entry in "${RUNS[@]}"; do
  IFS='|' read -r pair student method seed <<< "$entry"
  IFS=',' read -r T1 T2 <<< "$pair"
  RESULTS_DIR="experiments/sota/results/${T1}-${T2}/${student}/${method}/seed${seed}"
  mkdir -p "$RESULTS_DIR"
  LATEST="$RESULTS_DIR/latest.json"
  if [[ -f "$LATEST" ]]; then
    echo "⏭️  Skip(existing): $LATEST"; continue
  fi

        OVRS="$(method_overrides "$method" "$student")"
        if [[ -z "$OVRS" ]]; then echo "Unknown method: $method"; exit 1; fi

        # 체크포인트 존재 검증 (없으면 스킵)
        for tk in "$T1" "$T2"; do
          ck="${CKPT[$tk]:-}"
          if [[ -z "$ck" || ! -f "$ck" ]]; then
            echo "⚠️  Missing CKPT for $tk: '$ck' — skip run"
            echo "{\"status\":\"skipped\",\"reason\":\"missing_ckpt\",\"teacher\":\"$tk\"}" > "$RESULTS_DIR/skipped.json"
            continue 2
          fi
        done

        # 공정성 가드: 메인 SOTA 표에서는 FT/PPF/CCCP 금지
        if [[ "$OVRS" == *"use_teacher_finetuning=true"* || "$OVRS" == *"use_partial_freeze=true"* || "$OVRS" == *"use_cccp=true"* ]]; then
          echo "⚠️  fairness guard: FT/PPF/CCCP detected in main SOTA — skip"
          echo "{\"status\":\"skipped\",\"reason\":\"fairness_guard\"}" > "$RESULTS_DIR/skipped.json"
          continue
        fi

        # 메소드별 OOM 가드: CRD/ReviewKD면 배치 축소
        BATCH_OVR=()
        if [[ "$method" == "crd" || "$method" == "reviewkd" ]]; then
          BATCH_OVR=(experiment.dataset.batch_size=96)
        fi

        CMD=(python -u main.py -cn=experiment/sota_generic
          +experiment.teacher1.name="$T1"
          +experiment.teacher2.name="$T2"
          +experiment.model.student.name="$student"
          +experiment.teacher1.pretrained=true
          +experiment.teacher2.pretrained=true
          +experiment.teacher1_ckpt="${CKPT[$T1]:-null}"
          +experiment.teacher2_ckpt="${CKPT[$T2]:-null}"
          +experiment.results_dir="$RESULTS_DIR"
          +experiment.exp_id="sota__${method}__${T1}-${T2}__${student}__s${seed}"
          +seed="$seed"
        )
        read -r -a OV_ARR <<< "$OVRS"
        # num_workers 자동 주입(이미 지정돼 있으면 생략)
        HAS_NW_OVR=0
        if [[ "$OVRS" == *"experiment.dataset.num_workers="* ]]; then HAS_NW_OVR=1; fi
        for arg in "$@"; do
          if [[ "$arg" == experiment.dataset.num_workers=* || "$arg" == dataset.num_workers=* ]]; then
            HAS_NW_OVR=1
          fi
        done
        WORKERS_OVR=()
        if [[ $HAS_NW_OVR -eq 0 ]]; then
          WORKERS_OVR=(+experiment.dataset.num_workers="${WORKERS}")
        fi
        CMD+=("${OV_ARR[@]}" "${BATCH_OVR[@]}" "${WORKERS_OVR[@]}")
        # 추가 CLI 인자 패스스루: Hydra-safe만 전달
        if [[ "$#" -gt 0 ]]; then
          PASSTHRU_ARGS=()
          for a in "$@"; do
            case "$a" in
              *method@*|*experiment.method=*|*experiment/method=*|*method=*)
                continue;;
              *)
                if [[ "$a" == -* || "$a" == *=* || "$a" == +*=* ]]; then
                  PASSTHRU_ARGS+=("$a")
                fi
                ;;
            esac
          done
          CMD+=("${PASSTHRU_ARGS[@]}")
        fi

        echo "🚀 ${CMD[*]}"
        set +e
        "${CMD[@]}"
        ret=$?
        set -e
        if [[ $ret -ne 0 ]]; then
          echo "❌ Failed: $RESULTS_DIR (exit=$ret) — continue"
          echo "{\"status\":\"failed\",\"code\":$ret}" > "$RESULTS_DIR/failed.json"
          continue
        fi
        echo "✅ Done: $RESULTS_DIR"
      done
    done
  done
done

echo "🎉 All runs finished. See experiments/sota/results/"