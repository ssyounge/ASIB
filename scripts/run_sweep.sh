#!/bin/bash
# scripts/run_sweep.sh  ── W&B Sweep + SLURM array

# ---------- SLURM 옵션 (run.sh 과 동일 스타일) ----------
#SBATCH --job-name=asmb_sweep
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-3                    # agent 4개 (필요 시 확장)
#SBATCH --output=outputs/asmb_%A_%a/run.log
#SBATCH --error=outputs/asmb_%A_%a/run.log

# ---------- 공통 경로 & 환경 ----------
JOB_ID=${SLURM_ARRAY_JOB_ID:-manual}
AGENT_ID=${SLURM_ARRAY_TASK_ID:-0}
OUTPUT_DIR="outputs/asmb_${JOB_ID}_${AGENT_ID}"
mkdir -p "${OUTPUT_DIR}"

source ~/.bashrc
conda activate tlqkf

# ---------- W&B 설정 ----------
export WANDB_ENTITY="kakamy0820-yonsei-university"
export WANDB_PROJECT="kd_monitor"
# export WANDB_API_KEY="<원하면_직접_기입>"

# ---------------------------------------------------------------------------
#  Sweep 생성 (array-id 0)  ⟶  jq / --json 없이 파싱
# ---------------------------------------------------------------------------
SWEEP_FILE="sweeps/asmb_grid.yaml"
if [[ "${AGENT_ID}" == "0" ]]; then
    echo "📡  Creating sweep from ${SWEEP_FILE} ..."
    CREATE_LOG=$(wandb sweep "${SWEEP_FILE}" 2>&1)   # 표준·오류 모두 캡처
    echo "${CREATE_LOG}"

    # wandb sweep 출력에서 전체 경로를 그대로 추출
    SWEEP_PATH=$(echo "${CREATE_LOG}" | grep -oE 'wandb agent [^`]*' | awk '{print $3}')

    if [[ -z "${SWEEP_PATH}" ]]; then
        echo "❌  Sweep path 파싱 실패"; exit 1
    fi
    echo "${SWEEP_PATH}" | tee "sweep_id_${JOB_ID}.txt"
fi

# 다른 agent 들은 Sweep 경로 준비될 때까지 대기
while [[ ! -f "sweep_id_${JOB_ID}.txt" ]]; do sleep 5; done
SWEEP_PATH=$(cat "sweep_id_${JOB_ID}.txt")

echo "🚀  Launching W&B agent ${AGENT_ID} for sweep ${SWEEP_PATH}"
# logs/* 의 step‑별 출력은 wandb 내부에 저장, SLURM log 로도 기본 info 출력
wandb agent "${SWEEP_PATH}"

