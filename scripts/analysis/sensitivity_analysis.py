#!/usr/bin/env python3
"""
Sensitivity Analysis Script

각 기능을 하나씩 끄면서 성능 변화를 분석합니다.
"""

import subprocess
import os
import json
from datetime import datetime
from typing import Dict, List, Any

# Sensitivity analysis 설정들
SENSITIVITY_CONFIGS = {
    "baseline": {},  # 기본 설정 (모든 기능 ON)
    
    "no_cccp": {
        "use_cccp": False
    },
    
    "no_ib": {
        "use_ib": False
    },
    
    "no_disagree": {
        "use_disagree_weight": False
    },
    
    "no_ensemble": {
        "kd_ens_alpha": 0.0
    },
    
    "no_adapter": {
        "use_distillation_adapter": False
    },
    
    "no_partial_freeze": {
        "use_partial_freeze": False
    }
}

def run_experiment(config_name: str, overrides: Dict[str, Any]) -> str:
    """단일 실험을 실행합니다."""
    print(f"\n🚀 Running experiment: {config_name}")
    print(f"Overrides: {overrides}")
    
    # Hydra 명령어 구성
    cmd = ["python", "main.py", "experiment=res152_convnext_effi"]
    
    # 오버라이드 추가
    for key, value in overrides.items():
        if isinstance(value, bool):
            cmd.append(f"{key}={str(value).lower()}")
        else:
            cmd.append(f"{key}={value}")
    
    # 고유한 결과 디렉토리 설정
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"outputs/sensitivity_{config_name}_{timestamp}"
    cmd.extend([f"results_dir={results_dir}", f"exp_id=sensitivity_{config_name}"])
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # 실험 실행
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1시간 타임아웃
        
        if result.returncode == 0:
            print(f"✅ {config_name} completed successfully")
            return results_dir
        else:
            print(f"❌ {config_name} failed:")
            print(result.stderr)
            return None
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {config_name} timed out")
        return None
    except Exception as e:
        print(f"❌ {config_name} error: {e}")
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

def run_sensitivity_analysis():
    """전체 sensitivity analysis를 실행합니다."""
    print("🎯 Starting Sensitivity Analysis")
    print("=" * 60)
    
    results = {}
    
    for config_name, overrides in SENSITIVITY_CONFIGS.items():
        results_dir = run_experiment(config_name, overrides)
        
        if results_dir:
            metrics = extract_results(results_dir)
            results[config_name] = {
                "results_dir": results_dir,
                "metrics": metrics,
                "overrides": overrides
            }
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📊 SENSITIVITY ANALYSIS RESULTS")
    print("=" * 60)
    
    baseline_acc = results.get("baseline", {}).get("metrics", {}).get("final_student_acc", 0.0)
    
    for config_name, data in results.items():
        metrics = data["metrics"]
        acc = metrics["final_student_acc"]
        time_min = metrics["total_time_sec"] / 60.0
        
        acc_diff = acc - baseline_acc
        
        print(f"{config_name:15} | {acc:6.2f}% | {acc_diff:+6.2f}% | {time_min:6.1f}min")
    
    # 결과를 JSON으로 저장
    summary_file = f"outputs/sensitivity_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Detailed results saved to: {summary_file}")
    print("=" * 60)

if __name__ == "__main__":
    run_sensitivity_analysis() 