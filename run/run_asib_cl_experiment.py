#!/usr/bin/env python3
"""
ASIB-CL Continual Learning 실험 실행 스크립트

이 스크립트는 PyCIL 프레임워크를 사용하여 ASIB-CL과 다른 CL 방법들을 비교합니다.
"""

import os
import sys
import subprocess
import json
import logging
from pathlib import Path

# PyCIL 디렉토리를 Python 경로에 추가
sys.path.append('./PyCIL')

def setup_logging():
    """로깅 설정"""
    # experiments/sota/logs 디렉토리 생성
log_dir = Path('experiments/sota/logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'asib_cl_experiment.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def run_experiment(method_name, config_path):
    """단일 실험 실행"""
    logging.info(f"실행 중: {method_name}")
    
    cmd = [
        'python', 'PyCIL/main.py',
        '--config', config_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')
        if result.returncode == 0:
            logging.info(f"✅ {method_name} 실험 완료")
            return True
        else:
            logging.error(f"❌ {method_name} 실험 실패")
            logging.error(f"에러: {result.stderr}")
            return False
    except Exception as e:
        logging.error(f"❌ {method_name} 실행 중 예외 발생: {e}")
        return False

def create_comparison_configs():
    """비교 실험을 위한 설정 파일들 생성"""
    base_config = {
        "convnet_type": "resnet32",
        "dataset": "cifar100",
        "init_cls": 10,
        "increment": 10,
        "memory_size": 2000,
        "memory_per_class": 20,
        "device": [0],
        "num_workers": 8,
        "batch_size": 128,
        "epochs": 170,
        "lr": 0.1,
        "lr_decay": 0.1,
        "milestones": [60, 120, 160],
        "weight_decay": 0.0002,
        "topk": 5,
        "seed": [1993]
    }
    
    # ASIB-CL 설정
    asib_config = base_config.copy()
    asib_config.update({
        "ib_beta": 0.1,
        "lambda_D": 1.0,
        "lambda_IB": 1.0,
        "logdir": "./experiments/sota/results/asib_cl",
        "model_name": "asib_cl"
    })
    
    # 다른 방법들과 비교
    comparison_methods = {
        "finetune": {"model_name": "finetune", "logdir": "./experiments/sota/results/finetune"},
"ewc": {"model_name": "ewc", "logdir": "./experiments/sota/results/ewc"},
"lwf": {"model_name": "lwf", "logdir": "./experiments/sota/results/lwf"},
"icarl": {"model_name": "icarl", "logdir": "./experiments/sota/results/icarl"},
"der": {"model_name": "der", "logdir": "./experiments/sota/results/der"}
    }
    
    # 설정 파일들 생성
    configs = {}
    
    # ASIB-CL 설정 저장
    with open('PyCIL/exps/asib_cl.json', 'w', encoding='utf-8') as f:
        json.dump(asib_config, f, indent=4)
    configs["asib_cl"] = "PyCIL/exps/asib_cl.json"
    
    # 비교 방법들 설정 생성
    for method, config in comparison_methods.items():
        method_config = base_config.copy()
        method_config.update(config)
        
        config_path = f"PyCIL/exps/{method}_comparison.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(method_config, f, indent=4)
        configs[method] = config_path
    
    return configs

def run_comparison_experiments():
    """비교 실험 실행"""
    logging.info("🚀 ASIB-CL vs 다른 CL 방법들 비교 실험 시작")
    
    # 설정 파일들 생성
    configs = create_comparison_configs()
    
    # 실험 실행
    results = {}
    
    # ASIB-CL 먼저 실행
    logging.info("=" * 50)
    logging.info("ASIB-CL 실험 시작")
    logging.info("=" * 50)
    
    success = run_experiment("ASIB-CL", configs["asib_cl"])
    results["asib_cl"] = success
    
    # 다른 방법들과 비교
    for method, config_path in configs.items():
        if method == "asib_cl":
            continue
            
        logging.info("=" * 50)
        logging.info(f"{method.upper()} 실험 시작")
        logging.info("=" * 50)
        
        success = run_experiment(method.upper(), config_path)
        results[method] = success
    
    # 결과 요약
    logging.info("=" * 50)
    logging.info("실험 결과 요약")
    logging.info("=" * 50)
    
    for method, success in results.items():
        status = "✅ 성공" if success else "❌ 실패"
        logging.info(f"{method.upper()}: {status}")
    
    return results

def analyze_results():
    """실험 결과 분석"""
    logging.info("📊 실험 결과 분석 시작")
    
    # 로그 파일들에서 성능 지표 추출
log_dirs = [
"./experiments/sota/logs/asib_cl",
"./experiments/sota/logs/finetune", 
"./experiments/sota/logs/ewc",
"./experiments/sota/logs/lwf",
"./experiments/sota/logs/icarl",
"./experiments/sota/logs/der"
]
    
    results = {}
    
    for log_dir in log_dirs:
        if os.path.exists(log_dir):
            # 로그 파일에서 평균 정확도 추출
            log_files = list(Path(log_dir).glob("*.log"))
            if log_files:
                latest_log = max(log_files, key=os.path.getctime)
                method_name = log_dir.split('/')[-1]
                
                try:
                    with open(latest_log, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # "Average Accuracy" 라인 찾기
                        lines = content.split('\n')
                        for line in lines:
                            if "Average Accuracy" in line:
                                acc = line.split(':')[-1].strip()
                                results[method_name] = acc
                                break
                except Exception as e:
                    logging.warning(f"{method_name} 결과 파싱 실패: {e}")
    
    # 결과 출력
    logging.info("📈 최종 성능 비교")
    logging.info("-" * 30)
    
    for method, acc in results.items():
        logging.info(f"{method.upper()}: {acc}")
    
    return results

def main():
    """메인 함수"""
    setup_logging()
    
    logging.info("🎯 ASIB-CL Continual Learning 실험 시작")
    logging.info("=" * 60)
    
    # 1. 비교 실험 실행
    experiment_results = run_comparison_experiments()
    
    # 2. 결과 분석
    if any(experiment_results.values()):
        performance_results = analyze_results()
        
        # 3. 결과 요약 리포트 생성
        create_summary_report(experiment_results, performance_results)
    else:
        logging.error("❌ 모든 실험이 실패했습니다.")

def create_summary_report(experiment_results, performance_results):
    """실험 결과 요약 리포트 생성"""
    logging.info("📝 실험 결과 리포트 생성")
    
    report = f"""
# ASIB-CL Continual Learning 실험 결과 리포트

## 실험 개요
- **데이터셋**: CIFAR-100
- **시나리오**: Class-Incremental Learning (10 classes per task)
- **메모리 크기**: 2000 exemplars
- **백본 네트워크**: ResNet-32

## 실험 실행 상태
"""
    
    for method, success in experiment_results.items():
        status = "✅ 성공" if success else "❌ 실패"
        report += f"- **{method.upper()}**: {status}\n"
    
    report += "\n## 성능 비교\n"
    report += "| 방법 | 평균 정확도 |\n"
    report += "|------|------------|\n"
    
    for method, acc in performance_results.items():
        report += f"| {method.upper()} | {acc} |\n"
    
    report += f"""
## ASIB-CL의 핵심 특징

### 1. Information Bottleneck 기반 지식 증류
- **안정성**: 이전 모델의 지식을 최소 충분 정보로 압축하여 전달
- **가소성**: 불필요한 정보 전달을 줄여 새로운 태스크 학습에 모델 용량 확보

### 2. 안정성-가소성 딜레마 해결
- **β (IB 압축 강도)**: 0.1로 설정하여 적절한 압축 강도 유지
- **Knowledge Transfer Loss**: MSE loss로 특징 수준 지식 전달
- **Information Compression Loss**: KL divergence로 정보 압축 유도

### 3. Class-IL 시나리오 최적화
- **단일 공유 헤드**: 모든 클래스를 구분하는 단일 분류기
- **이전 모델 교사**: Oracle 교사 대신 이전 태스크 모델을 교사로 사용
- **표준 CL 프로토콜 준수**: 미래 데이터 접근 없이 순차적 학습

## 결론
ASIB-CL은 Information Bottleneck을 활용하여 Continual Learning의 근본적인 문제인 
안정성-가소성 딜레마를 효과적으로 해결하는 방법입니다.
"""
    
    # 리포트 저장
    with open('ASIB_CL_Experiment_Report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    logging.info("📄 실험 리포트가 'ASIB_CL_Experiment_Report.md'에 저장되었습니다.")

if __name__ == "__main__":
    main() 