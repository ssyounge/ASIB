#!/usr/bin/env python3
"""
Overlap Analysis Script

여러 KD 방법들을 다양한 overlap percentage에서 비교 분석합니다.
"""

import subprocess
import os
import json
from datetime import datetime
from typing import Dict, List, Any

# KD 방법들 정의
KD_METHODS = {
    "asib": {
        "method": "asib",
        "kd_alpha": 0.3,
        "kd_ens_alpha": 0.7,
        "use_cccp": True,
        "use_ib": True,
        "use_disagree_weight": True,
        "use_distillation_adapter": True,
        "use_partial_freeze": True
    },
    
    "dkd": {
        "method": "dkd", 
        "kd_alpha": 1.0,
        "kd_ens_alpha": 0.0,
        "use_cccp": False,
        "use_ib": False,
        "use_disagree_weight": False,
        "use_distillation_adapter": False,
        "use_partial_freeze": False,
        "dkd_temp": 4.0,
        "dkd_alpha": 1.0,
        "dkd_beta": 8.0
    },
    
    "crd": {
        "method": "crd",
        "kd_alpha": 0.0,
        "kd_ens_alpha": 0.0,
        "use_cccp": False,
        "use_ib": False,
        "use_disagree_weight": False,
        "use_distillation_adapter": False,
        "use_partial_freeze": False,
        "crd_alpha": 1.0,
        "crd_beta": 0.8
    },
    
    "fitnet": {
        "method": "fitnet",
        "kd_alpha": 0.0,
        "kd_ens_alpha": 0.0,
        "use_cccp": False,
        "use_ib": False,
        "use_disagree_weight": False,
        "use_distillation_adapter": False,
        "use_partial_freeze": False,
        "feat_kd_alpha": 1.0,
        "feat_kd_key": "feat_2d"
    },
    
    "at": {
        "method": "at",
        "kd_alpha": 0.0,
        "kd_ens_alpha": 0.0,
        "use_cccp": False,
        "use_ib": False,
        "use_disagree_weight": False,
        "use_distillation_adapter": False,
        "use_partial_freeze": False,
        "at_alpha": 1.0,
        "at_beta": 1.0
    }
}

# Overlap percentages
OVERLAP_PCTS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

def run_overlap_experiment(method_name: str, overlap_pct: int, method_config: Dict[str, Any]) -> str:
    """단일 overlap 실험을 실행합니다."""
    print(f"\n🚀 Running {method_name.upper()} with {overlap_pct}% overlap")
    
    # 기본 명령어
    cmd = [
        "python", "main.py", 
        "experiment=res152_convnext_effi",  # 기본 설정 사용
        f"overlap_pct={overlap_pct}"
    ]
    
    # Teacher 모델을 ResNet152로 설정
    cmd.extend([
        "teacher1.model.teacher.name=resnet152_teacher",
        "teacher2.model.teacher.name=resnet152_teacher",
        "teacher1_ckpt=checkpoints/resnet152_cifar32.pth",
        "teacher2_ckpt=checkpoints/resnet152_cifar32.pth"
    ])
    
    # KD 방법별 설정 추가
    for key, value in method_config.items():
        if isinstance(value, bool):
            cmd.append(f"{key}={str(value).lower()}")
        else:
            cmd.append(f"{key}={value}")
    
    # 고유한 결과 디렉토리 설정
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"outputs/overlap_{method_name}_{overlap_pct}pct_{timestamp}"
    cmd.extend([f"results_dir={results_dir}", f"exp_id=overlap_{method_name}_{overlap_pct}pct"])
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # 실험 실행
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2시간 타임아웃
        
        if result.returncode == 0:
            print(f"✅ {method_name} {overlap_pct}% completed successfully")
            return results_dir
        else:
            print(f"❌ {method_name} {overlap_pct}% failed:")
            print(result.stderr)
            return None
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {method_name} {overlap_pct}% timed out")
        return None
    except Exception as e:
        print(f"❌ {method_name} {overlap_pct}% error: {e}")
        return None

def extract_results(results_dir: str) -> Dict[str, Any]:
    """실험 결과를 추출합니다."""
    try:
        # JSON 결과 파일 읽기
        json_file = os.path.join(results_dir, "results", "latest.json")
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                results = json.load(f)
            
            # 주요 메트릭 추출
            return {
                "final_student_acc": results.get("final_student_acc", 0.0),
                "total_time_sec": results.get("total_time_sec", 0.0),
                "best_acc": max([
                    results.get(f"stage{i}_student_acc", 0.0) 
                    for i in range(1, 5)
                ])
            }
    except Exception as e:
        print(f"Warning: Could not extract results from {results_dir}: {e}")
    
    return {"final_student_acc": 0.0, "total_time_sec": 0.0, "best_acc": 0.0}

def run_overlap_analysis():
    """전체 overlap analysis를 실행합니다."""
    print("🎯 Starting Overlap Analysis")
    print("=" * 80)
    print(f"Methods: {list(KD_METHODS.keys())}")
    print(f"Overlap percentages: {OVERLAP_PCTS}")
    print("=" * 80)
    
    results = {}
    
    for method_name, method_config in KD_METHODS.items():
        results[method_name] = {}
        
        for overlap_pct in OVERLAP_PCTS:
            results_dir = run_overlap_experiment(method_name, overlap_pct, method_config)
            
            if results_dir:
                metrics = extract_results(results_dir)
                results[method_name][overlap_pct] = {
                    "results_dir": results_dir,
                    "metrics": metrics,
                    "config": method_config
                }
    
    # 결과 요약
    print("\n" + "=" * 80)
    print("📊 OVERLAP ANALYSIS RESULTS")
    print("=" * 80)
    
    # 헤더 출력
    header = f"{'Method':<8} | {'Overlap':<7} | {'Accuracy':<9} | {'Time':<8}"
    print(header)
    print("-" * len(header))
    
    # 결과 출력
    for method_name in KD_METHODS.keys():
        for overlap_pct in OVERLAP_PCTS:
            if overlap_pct in results.get(method_name, {}):
                data = results[method_name][overlap_pct]
                metrics = data["metrics"]
                acc = metrics["final_student_acc"]
                time_min = metrics["total_time_sec"] / 60.0
                
                print(f"{method_name:<8} | {overlap_pct:>6}% | {acc:>7.2f}% | {time_min:>6.1f}m")
    
    # 결과를 JSON으로 저장
    summary_file = f"outputs/overlap_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("=" * 80)
    print(f"📁 Detailed results saved to: {summary_file}")
    print("=" * 80)

if __name__ == "__main__":
    run_overlap_analysis() 