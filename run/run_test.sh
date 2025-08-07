#!/usr/bin/env bash
#SBATCH --job-name=run_tests
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
# ---------------------------------------------------------
# GPU를 활용한 빠른 테스트 실행
# tests/ 폴더의 모든 테스트 파일을 병렬로 실행
# ---------------------------------------------------------
set -euo pipefail

# Python 환경 설정
echo "🔧 Setting up Python environment..."
export PATH="$HOME/anaconda3/envs/tlqkf/bin:$PATH"
echo "✅ Python environment setup completed"
echo ""

# 1) 리포 최상위로 이동
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

# 2) PYTHONPATH 추가 및 환경 설정
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
if [ -n "${CONDA_PREFIX:-}" ]; then
    export PATH="$CONDA_PREFIX/bin:$PATH"
fi

# GPU 할당 확인 및 설정
echo "🔍 Checking GPU allocation..."
if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
    # GPU 인덱스를 0부터 시작하도록 조정
    if [ "$SLURM_GPUS_ON_NODE" = "1" ]; then
        export CUDA_VISIBLE_DEVICES=0
        echo "✅ CUDA_VISIBLE_DEVICES set to: 0 (mapped from SLURM_GPUS_ON_NODE=1)"
    else
        export CUDA_VISIBLE_DEVICES=0
        echo "✅ CUDA_VISIBLE_DEVICES set to: 0 (default for any GPU allocation)"
    fi
else
    echo "⚠️  SLURM_GPUS_ON_NODE not set, using default GPU 0"
    export CUDA_VISIBLE_DEVICES=0
fi

# CUDA 컨텍스트 초기화 (segmentation fault 방지)
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# PyTorch CUDA 12.4 라이브러리 사용
export LD_LIBRARY_PATH="$HOME/anaconda3/envs/tlqkf/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH"
export CUDA_HOME="$HOME/anaconda3/envs/tlqkf/lib/python3.12/site-packages/torch/lib"

# PyTorch CUDA 설정
export TORCH_CUDA_ARCH_LIST="8.6"

# CUDA 환경변수 (PyTorch 내장 CUDA 12.4 사용)
export CUDA_PATH="$HOME/anaconda3/envs/tlqkf/lib/python3.12/site-packages/torch/lib"
export CUDA_ROOT="$HOME/anaconda3/envs/tlqkf/lib/python3.12/site-packages/torch/lib"

# GPU 정보 출력
echo "🔍 GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits

# 3) 테스트 디렉토리 생성
mkdir -p experiments/test/logs experiments/test/results

# 4) 타임스탬프 함수
timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

# 5) 테스트 결과 파싱 함수
parse_test_results() {
    local log_file="$1"
    local test_name="$2"
    
    echo "📊 $test_name Results:" | tee -a experiments/test/results/summary.log
    
    # 성공/실패/스킵/경고 카운트 추출
    local passed=$(grep -c "PASSED" "$log_file" 2>/dev/null || echo "0")
    local failed=$(grep -c "FAILED" "$log_file" 2>/dev/null || echo "0")
    local skipped=$(grep -c "SKIPPED" "$log_file" 2>/dev/null || echo "0")
    local warnings=$(grep -c "WARNING" "$log_file" 2>/dev/null || echo "0")
    
    echo "   ✅ PASSED: $passed" | tee -a experiments/test/results/summary.log
    echo "   ❌ FAILED: $failed" | tee -a experiments/test/results/summary.log
    echo "   ⏭️  SKIPPED: $skipped" | tee -a experiments/test/results/summary.log
    echo "   ⚠️  WARNINGS: $warnings" | tee -a experiments/test/results/summary.log
    
    # 실패한 테스트 목록 추출
    if [ "$failed" -gt 0 ]; then
        echo "   🔍 Failed tests:" | tee -a experiments/test/results/summary.log
        grep "FAILED" "$log_file" | head -5 | sed 's/^/      /' | tee -a experiments/test/results/summary.log
        if [ "$failed" -gt 5 ]; then
            echo "      ... and $((failed - 5)) more" | tee -a experiments/test/results/summary.log
        fi
    fi
    
    echo "" | tee -a experiments/test/results/summary.log
    
    # 실패 여부 반환
    if [ "$failed" -gt 0 ]; then
        return 1
    else
        return 0
    fi
}

# 6) 메인 실행 시작
echo "🚀 Starting GPU-accelerated tests at $(timestamp)" | tee experiments/test/results/summary.log
echo "==================================================" | tee -a experiments/test/results/summary.log
echo "Python version: $(python --version)" | tee -a experiments/test/results/summary.log
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')" | tee -a experiments/test/results/summary.log
echo "GPU device: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")')" | tee -a experiments/test/results/summary.log
echo "==================================================" | tee -a experiments/test/results/summary.log
echo "" | tee -a experiments/test/results/summary.log

# 7) 모든 테스트 파일 찾기 (conftest.py 제외)
echo "📋 Discovering test files..." | tee -a experiments/test/results/summary.log
TEST_FILES=($(find tests/ -name "test_*.py" -not -name "conftest.py" | sort))
echo "   Found ${#TEST_FILES[@]} test files" | tee -a experiments/test/results/summary.log
echo "" | tee -a experiments/test/results/summary.log

# 8) 테스트 그룹별로 병렬 실행
echo "📋 Starting test groups at $(timestamp)..." | tee -a experiments/test/results/summary.log

# 🔧 Core Functionality Tests (3개)
echo "   🔄 Running core functionality tests..." | tee -a experiments/test/results/summary.log
python -m pytest tests/test_core.py tests/test_core_utils.py tests/test_utils_common.py -v --tb=short -W ignore > experiments/test/logs/core_functionality_test.log 2>&1 &
CORE_FUNCTIONALITY_PID=$!

# 🧠 Model & Module Tests (4개)
echo "   🔄 Running model and module tests..." | tee -a experiments/test/results/summary.log
python -m pytest tests/test_models.py tests/test_models_advanced.py tests/test_modules.py tests/test_modules_partial_freeze.py -v --tb=short -W ignore > experiments/test/logs/model_module_test.log 2>&1 &
MODEL_MODULE_PID=$!

# 📊 Data & Configuration Tests (4개)
echo "   🔄 Running data and configuration tests..." | tee -a experiments/test/results/summary.log
python -m pytest tests/test_data.py tests/test_configs.py tests/test_finetune_configs.py tests/test_experiment_configs.py -v --tb=short -W ignore > experiments/test/logs/data_config_test.log 2>&1 &
DATA_CONFIG_PID=$!

# 🔄 Training & Experiment Tests (7개)
echo "   🔄 Running training and experiment tests..." | tee -a experiments/test/results/summary.log
python -m pytest tests/test_main.py tests/test_main_training.py tests/test_main_step_by_step.py tests/test_asib_step.py tests/test_asib_cl.py tests/test_cl_experiments.py tests/test_training_pipeline.py -v --tb=short -W ignore > experiments/test/logs/training_experiment_test.log 2>&1 &
TRAINING_EXPERIMENT_PID=$!

# 🧪 Integration & Validation Tests (4개)
echo "   🔄 Running integration and validation tests..." | tee -a experiments/test/results/summary.log
python -m pytest tests/test_integration.py tests/test_final_validation.py tests/test_experiment_execution.py tests/test_framework_robustness.py -v --tb=short -W ignore > experiments/test/logs/integration_validation_test.log 2>&1 &
INTEGRATION_VALIDATION_PID=$!

# 🔍 Analysis & Script Tests (3개)
echo "   🔄 Running analysis and script tests..." | tee -a experiments/test/results/summary.log
python -m pytest tests/test_scripts.py tests/test_pycil_integration.py tests/test_pycil_models.py -v --tb=short -W ignore > experiments/test/logs/analysis_script_test.log 2>&1 &
ANALYSIS_SCRIPT_PID=$!

# 🛡️ Error Prevention & Edge Cases (2개)
echo "   🔄 Running error prevention and edge case tests..." | tee -a experiments/test/results/summary.log
python -m pytest tests/test_error_prevention.py tests/test_overlap_dataset.py -v --tb=short -W ignore > experiments/test/logs/error_prevention_test.log 2>&1 &
ERROR_PREVENTION_PID=$!

# 📐 Specialized Component Tests (4개)
echo "   🔄 Running specialized component tests..." | tee -a experiments/test/results/summary.log
python -m pytest tests/test_mbm_tensor_shapes.py tests/test_ib_mbm_shapes.py tests/test_kd_methods.py tests/test_registry_comprehensive.py -v --tb=short -W ignore > experiments/test/logs/specialized_component_test.log 2>&1 &
SPECIALIZED_COMPONENT_PID=$!

# 🧩 Utility Tests (2개)
echo "   🔄 Running utility tests..." | tee -a experiments/test/results/summary.log
python -m pytest tests/test_utils.py tests/test_disagreement.py -v --tb=short -W ignore > experiments/test/logs/utility_test.log 2>&1 &
UTILITY_PID=$!

# 9) 실시간 진행 상황 모니터링
echo "" | tee -a experiments/test/results/summary.log
echo "⏳ Monitoring test progress..." | tee -a experiments/test/results/summary.log

# 각 테스트 완료 대기 및 결과 확인
wait_and_check() {
    local pid="$1"
    local test_name="$2"
    local log_file="$3"
    
    wait $pid
    echo "   ✅ $test_name completed at $(timestamp)" | tee -a experiments/test/results/summary.log
    parse_test_results "$log_file" "$test_name"
    return $?
}

# 모든 테스트 완료 대기
FAILED_COUNT=0

wait_and_check $CORE_FUNCTIONALITY_PID "Core Functionality Tests" "experiments/test/logs/core_functionality_test.log" || ((FAILED_COUNT++))
wait_and_check $MODEL_MODULE_PID "Model & Module Tests" "experiments/test/logs/model_module_test.log" || ((FAILED_COUNT++))
wait_and_check $DATA_CONFIG_PID "Data & Configuration Tests" "experiments/test/logs/data_config_test.log" || ((FAILED_COUNT++))
wait_and_check $TRAINING_EXPERIMENT_PID "Training & Experiment Tests" "experiments/test/logs/training_experiment_test.log" || ((FAILED_COUNT++))
wait_and_check $INTEGRATION_VALIDATION_PID "Integration & Validation Tests" "experiments/test/logs/integration_validation_test.log" || ((FAILED_COUNT++))
wait_and_check $ANALYSIS_SCRIPT_PID "Analysis & Script Tests" "experiments/test/logs/analysis_script_test.log" || ((FAILED_COUNT++))
wait_and_check $ERROR_PREVENTION_PID "Error Prevention & Edge Cases" "experiments/test/logs/error_prevention_test.log" || ((FAILED_COUNT++))
wait_and_check $SPECIALIZED_COMPONENT_PID "Specialized Component Tests" "experiments/test/logs/specialized_component_test.log" || ((FAILED_COUNT++))
wait_and_check $UTILITY_PID "Utility Tests" "experiments/test/logs/utility_test.log" || ((FAILED_COUNT++))

# 10) 전체 결과 요약
echo "==================================================" | tee -a experiments/test/results/summary.log
echo "📊 FINAL TEST SUMMARY at $(timestamp)" | tee -a experiments/test/results/summary.log
echo "==================================================" | tee -a experiments/test/results/summary.log

# 전체 통계 계산 (더 정확한 방법)
TOTAL_PASSED=$(grep "✅ PASSED:" experiments/test/results/summary.log | awk '{sum += $3} END {print sum+0}')
TOTAL_FAILED=$(grep "❌ FAILED:" experiments/test/results/summary.log | awk '{sum += $3} END {print sum+0}')
TOTAL_SKIPPED=$(grep "⏭️  SKIPPED:" experiments/test/results/summary.log | awk '{sum += $3} END {print sum+0}')
TOTAL_WARNINGS=$(grep "⚠️  WARNINGS:" experiments/test/results/summary.log | awk '{sum += $3} END {print sum+0}')

echo "🎯 Overall Statistics:" | tee -a experiments/test/results/summary.log
echo "   ✅ Total PASSED: $TOTAL_PASSED" | tee -a experiments/test/results/summary.log
echo "   ❌ Total FAILED: $TOTAL_FAILED" | tee -a experiments/test/results/summary.log
echo "   ⏭️  Total SKIPPED: $TOTAL_SKIPPED" | tee -a experiments/test/results/summary.log
echo "   ⚠️  Total WARNINGS: $TOTAL_WARNINGS" | tee -a experiments/test/results/summary.log
echo "   📁 Failed test groups: $FAILED_COUNT" | tee -a experiments/test/results/summary.log
echo "   📋 Total test files: ${#TEST_FILES[@]}" | tee -a experiments/test/results/summary.log

echo "" | tee -a experiments/test/results/summary.log
echo "📁 Detailed logs available in:" | tee -a experiments/test/results/summary.log
echo "   Summary: experiments/test/results/summary.log" | tee -a experiments/test/results/summary.log
echo "   Individual: experiments/test/logs/*.log" | tee -a experiments/test/results/summary.log

# 11) 최종 결과
echo "" | tee -a experiments/test/results/summary.log
echo "==================================================" | tee -a experiments/test/results/summary.log
if [ $FAILED_COUNT -eq 0 ] && [ $TOTAL_FAILED -eq 0 ]; then
    echo "🎉 ALL TESTS PASSED SUCCESSFULLY!" | tee -a experiments/test/results/summary.log
    echo "✅ No failures detected" | tee -a experiments/test/results/summary.log
    echo "📊 All ${#TEST_FILES[@]} test files processed" | tee -a experiments/test/results/summary.log
    exit 0
else
    echo "⚠️  SOME TESTS FAILED" | tee -a experiments/test/results/summary.log
    echo "❌ $TOTAL_FAILED individual tests failed" | tee -a experiments/test/results/summary.log
    echo "📋 $FAILED_COUNT test groups have failures" | tee -a experiments/test/results/summary.log
    echo "" | tee -a experiments/test/results/summary.log
    echo "🔍 To debug specific failures:" | tee -a experiments/test/results/summary.log
    echo "   cat experiments/test/results/summary.log" | tee -a experiments/test/results/summary.log
    echo "   python -m pytest tests/test_<name>.py -v" | tee -a experiments/test/results/summary.log
    exit 1
fi 