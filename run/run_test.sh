#!/usr/bin/env bash
#SBATCH --job-name=run_tests
#SBATCH --partition=base_suma_rtx3090
#SBATCH --qos=base_qos
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

# 1) 리포 최상위로 이동
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

# 2) PYTHONPATH 추가 및 환경 설정
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PATH="/home/suyoung425/anaconda3/envs/tlqkf/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0

# 3) 테스트 디렉토리 생성 (experiments/logs는 생성하지 않음)
mkdir -p experiments/test_results

# 4) 타임스탬프 함수
timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

# 5) 테스트 결과 파싱 함수
parse_test_results() {
    local log_file="$1"
    local test_name="$2"
    
    echo "📊 $test_name Results:" | tee -a experiments/test_results/summary.log
    
    # 성공/실패/스킵/경고 카운트 추출
    local passed=$(grep -c "PASSED" "$log_file" 2>/dev/null || echo "0")
    local failed=$(grep -c "FAILED" "$log_file" 2>/dev/null || echo "0")
    local skipped=$(grep -c "SKIPPED" "$log_file" 2>/dev/null || echo "0")
    local warnings=$(grep -c "WARNING" "$log_file" 2>/dev/null || echo "0")
    
    echo "   ✅ PASSED: $passed" | tee -a experiments/test_results/summary.log
    echo "   ❌ FAILED: $failed" | tee -a experiments/test_results/summary.log
    echo "   ⏭️  SKIPPED: $skipped" | tee -a experiments/test_results/summary.log
    echo "   ⚠️  WARNINGS: $warnings" | tee -a experiments/test_results/summary.log
    
    # 실패한 테스트 목록 추출
    if [ "$failed" -gt 0 ]; then
        echo "   🔍 Failed tests:" | tee -a experiments/test_results/summary.log
        grep "FAILED" "$log_file" | head -5 | sed 's/^/      /' | tee -a experiments/test_results/summary.log
        if [ "$failed" -gt 5 ]; then
            echo "      ... and $((failed - 5)) more" | tee -a experiments/test_results/summary.log
        fi
    fi
    
    echo "" | tee -a experiments/test_results/summary.log
    
    # 실패 여부 반환
    if [ "$failed" -gt 0 ]; then
        return 1
    else
        return 0
    fi
}

# 6) 메인 실행 시작
echo "🚀 Starting GPU-accelerated tests at $(timestamp)" | tee experiments/test_results/summary.log
echo "==================================================" | tee -a experiments/test_results/summary.log
echo "Python version: $(python --version)" | tee -a experiments/test_results/summary.log
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')" | tee -a experiments/test_results/summary.log
echo "GPU device: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")')" | tee -a experiments/test_results/summary.log
echo "==================================================" | tee -a experiments/test_results/summary.log
echo "" | tee -a experiments/test_results/summary.log

# 7) 모든 테스트 파일 찾기 (conftest.py 제외)
echo "📋 Discovering test files..." | tee -a experiments/test_results/summary.log
TEST_FILES=($(find tests/ -name "test_*.py" -not -name "conftest.py" | sort))
echo "   Found ${#TEST_FILES[@]} test files" | tee -a experiments/test_results/summary.log
echo "" | tee -a experiments/test_results/summary.log

# 8) 테스트 그룹별로 병렬 실행
echo "📋 Starting test groups at $(timestamp)..." | tee -a experiments/test_results/summary.log

# 핵심 기능 테스트 (ASIB-CL, ASIB Step)
echo "   🔄 Running core ASIB tests..." | tee -a experiments/test_results/summary.log
python -m pytest tests/test_asib_cl.py tests/test_asib_step.py -v --tb=short -W ignore > experiments/test_results/core_asib_test.log 2>&1 &
CORE_ASIB_PID=$!

# PyCIL 통합 테스트
echo "   🔄 Running PyCIL integration tests..." | tee -a experiments/test_results/summary.log
python -m pytest tests/test_pycil_integration.py tests/test_pycil_models.py -v --tb=short -W ignore > experiments/test_results/pycil_test.log 2>&1 &
PYCIL_PID=$!

# 데이터 및 유틸리티 테스트
echo "   🔄 Running data and utils tests..." | tee -a experiments/test_results/summary.log
python -m pytest tests/test_data.py tests/test_utils.py tests/test_core.py -v --tb=short -W ignore > experiments/test_results/data_utils_test.log 2>&1 &
DATA_UTILS_PID=$!

# 모델 테스트
echo "   🔄 Running model tests..." | tee -a experiments/test_results/summary.log
python -m pytest tests/test_models.py tests/test_models_advanced.py tests/test_new_methods.py tests/test_new_students.py -v --tb=short -W ignore > experiments/test_results/models_test.log 2>&1 &
MODELS_PID=$!

# 설정 및 실험 테스트
echo "   🔄 Running config and experiment tests..." | tee -a experiments/test_results/summary.log
python -m pytest tests/test_configs.py tests/test_finetune_configs.py tests/test_cl_experiments.py tests/test_registry_comprehensive.py -v --tb=short -W ignore > experiments/test_results/configs_test.log 2>&1 &
CONFIGS_PID=$!

# 스크립트 및 기타 테스트
echo "   🔄 Running script and misc tests..." | tee -a experiments/test_results/summary.log
python -m pytest tests/test_scripts.py tests/test_integration.py tests/test_modules.py -v --tb=short -W ignore > experiments/test_results/scripts_test.log 2>&1 &
SCRIPTS_PID=$!

# KD 및 특수 테스트
echo "   🔄 Running KD and special tests..." | tee -a experiments/test_results/summary.log
python -m pytest tests/test_kd_methods.py tests/test_disagreement.py tests/test_convnext_s_teacher.py tests/test_ib_mbm_shapes.py tests/test_partial_freeze.py -v --tb=short -W ignore > experiments/test_results/kd_test.log 2>&1 &
KD_PID=$!

# 9) 실시간 진행 상황 모니터링
echo "" | tee -a experiments/test_results/summary.log
echo "⏳ Monitoring test progress..." | tee -a experiments/test_results/summary.log

# 각 테스트 완료 대기 및 결과 확인
wait_and_check() {
    local pid="$1"
    local test_name="$2"
    local log_file="$3"
    
    wait $pid
    echo "   ✅ $test_name completed at $(timestamp)" | tee -a experiments/test_results/summary.log
    parse_test_results "$log_file" "$test_name"
    return $?
}

# 모든 테스트 완료 대기
FAILED_COUNT=0

wait_and_check $CORE_ASIB_PID "Core ASIB Tests" "experiments/test_results/core_asib_test.log" || ((FAILED_COUNT++))
wait_and_check $PYCIL_PID "PyCIL Tests" "experiments/test_results/pycil_test.log" || ((FAILED_COUNT++))
wait_and_check $DATA_UTILS_PID "Data & Utils Tests" "experiments/test_results/data_utils_test.log" || ((FAILED_COUNT++))
wait_and_check $MODELS_PID "Model Tests" "experiments/test_results/models_test.log" || ((FAILED_COUNT++))
wait_and_check $CONFIGS_PID "Config & Experiment Tests" "experiments/test_results/configs_test.log" || ((FAILED_COUNT++))
wait_and_check $SCRIPTS_PID "Script & Integration Tests" "experiments/test_results/scripts_test.log" || ((FAILED_COUNT++))
wait_and_check $KD_PID "KD & Special Tests" "experiments/test_results/kd_test.log" || ((FAILED_COUNT++))

# 10) 전체 결과 요약
echo "==================================================" | tee -a experiments/test_results/summary.log
echo "📊 FINAL TEST SUMMARY at $(timestamp)" | tee -a experiments/test_results/summary.log
echo "==================================================" | tee -a experiments/test_results/summary.log

# 전체 통계 계산
TOTAL_PASSED=$(grep "PASSED:" experiments/test_results/summary.log | awk '{sum += $2} END {print sum}')
TOTAL_FAILED=$(grep "FAILED:" experiments/test_results/summary.log | awk '{sum += $2} END {print sum}')
TOTAL_SKIPPED=$(grep "SKIPPED:" experiments/test_results/summary.log | awk '{sum += $2} END {print sum}')
TOTAL_WARNINGS=$(grep "WARNINGS:" experiments/test_results/summary.log | awk '{sum += $2} END {print sum}')

echo "🎯 Overall Statistics:" | tee -a experiments/test_results/summary.log
echo "   ✅ Total PASSED: $TOTAL_PASSED" | tee -a experiments/test_results/summary.log
echo "   ❌ Total FAILED: $TOTAL_FAILED" | tee -a experiments/test_results/summary.log
echo "   ⏭️  Total SKIPPED: $TOTAL_SKIPPED" | tee -a experiments/test_results/summary.log
echo "   ⚠️  Total WARNINGS: $TOTAL_WARNINGS" | tee -a experiments/test_results/summary.log
echo "   📁 Failed test groups: $FAILED_COUNT" | tee -a experiments/test_results/summary.log
echo "   📋 Total test files: ${#TEST_FILES[@]}" | tee -a experiments/test_results/summary.log

echo "" | tee -a experiments/test_results/summary.log
echo "📁 Detailed logs available in:" | tee -a experiments/test_results/summary.log
echo "   Summary: experiments/test_results/summary.log" | tee -a experiments/test_results/summary.log
echo "   Individual: experiments/test_results/*.log" | tee -a experiments/test_results/summary.log

# 11) 최종 결과
echo "" | tee -a experiments/test_results/summary.log
echo "==================================================" | tee -a experiments/test_results/summary.log
if [ $FAILED_COUNT -eq 0 ] && [ $TOTAL_FAILED -eq 0 ]; then
    echo "🎉 ALL TESTS PASSED SUCCESSFULLY!" | tee -a experiments/test_results/summary.log
    echo "✅ No failures detected" | tee -a experiments/test_results/summary.log
    echo "📊 All ${#TEST_FILES[@]} test files processed" | tee -a experiments/test_results/summary.log
    exit 0
else
    echo "⚠️  SOME TESTS FAILED" | tee -a experiments/test_results/summary.log
    echo "❌ $TOTAL_FAILED individual tests failed" | tee -a experiments/test_results/summary.log
    echo "📋 $FAILED_COUNT test groups have failures" | tee -a experiments/test_results/summary.log
    echo "" | tee -a experiments/test_results/summary.log
    echo "🔍 To debug specific failures:" | tee -a experiments/test_results/summary.log
    echo "   cat experiments/test_results/summary.log" | tee -a experiments/test_results/summary.log
    echo "   python -m pytest tests/test_<name>.py -v" | tee -a experiments/test_results/summary.log
    exit 1
fi 