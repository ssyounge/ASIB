#!/usr/bin/env python3
"""
Beta Sensitivity Analysis Script (Phase 1.2)

IB β 값의 변화에 따른 성능과 정보 압축 정도를 분석합니다.
VIB 연구에서 필수적인 분석으로, β 값 선택의 정당성을 입증합니다.
"""

import subprocess
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any, Tuple
import pandas as pd

# β 값 범위 (로그 스케일)
BETA_VALUES = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]

# 실험 설정
EXPERIMENT_CONFIG = "ablation_full"  # ASIB Full 설정 사용
NUM_RUNS = 3  # 각 β 값당 3회 반복
TIMEOUT_HOURS = 2  # 각 실험당 2시간 타임아웃

def run_beta_experiment(beta_value: float, run_id: int) -> Dict[str, Any]:
    """단일 β 값 실험을 실행합니다."""
    print(f"\n🚀 Running β={beta_value:.1e} (Run {run_id+1}/{NUM_RUNS})")
    print("=" * 60)
    
    # 고유한 결과 디렉토리 설정
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"outputs/analysis/beta_sensitivity/beta_{beta_value:.1e}_run_{run_id}_{timestamp}"
    exp_id = f"beta_sensitivity_{beta_value:.1e}_run_{run_id}"
    
    # Hydra 명령어 구성
    cmd = [
        "python", "main.py",
        f"--config-name", f"experiment/{EXPERIMENT_CONFIG}",
        f"ib_beta={beta_value}",
        f"results_dir={results_dir}",
        f"exp_id={exp_id}"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # 실험 실행
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=TIMEOUT_HOURS * 3600
        )
        
        if result.returncode == 0:
            print(f"✅ β={beta_value:.1e} (Run {run_id+1}) completed successfully")
            return extract_results(results_dir, beta_value, run_id)
        else:
            print(f"❌ β={beta_value:.1e} (Run {run_id+1}) failed:")
            print(result.stderr)
            return None
            
    except subprocess.TimeoutExpired:
        print(f"⏰ β={beta_value:.1e} (Run {run_id+1}) timed out")
        return None
    except Exception as e:
        print(f"❌ β={beta_value:.1e} (Run {run_id+1}) error: {e}")
        return None

def extract_results(results_dir: str, beta_value: float, run_id: int) -> Dict[str, Any]:
    """실험 결과를 추출합니다."""
    try:
        # JSON 결과 파일 읽기
        json_file = os.path.join(results_dir, "results", "latest.json")
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                results = json.load(f)
            
            # 주요 메트릭 추출
            return {
                "beta": beta_value,
                "run_id": run_id,
                "final_student_acc": results.get("final_student_acc", 0.0),
                "best_student_acc": results.get("best_student_acc", 0.0),
                "total_time_sec": results.get("total_time_sec", 0.0),
                "kl_divergence": results.get("avg_kl_divergence", 0.0),
                "info_compression": results.get("info_compression_ratio", 0.0),
                "results_dir": results_dir
            }
        else:
            print(f"⚠️ Results file not found: {json_file}")
            return None
            
    except Exception as e:
        print(f"❌ Error extracting results: {e}")
        return None

def analyze_beta_results(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """β 값별 결과를 분석합니다."""
    # β 값별로 결과 그룹화
    beta_groups = {}
    for result in all_results:
        if result is None:
            continue
        beta = result["beta"]
        if beta not in beta_groups:
            beta_groups[beta] = []
        beta_groups[beta].append(result)
    
    # 통계 계산
    analysis = {}
    for beta, results in beta_groups.items():
        if len(results) == 0:
            continue
            
        accuracies = [r["final_student_acc"] for r in results]
        kl_divs = [r["kl_divergence"] for r in results]
        times = [r["total_time_sec"] for r in results]
        
        analysis[beta] = {
            "mean_acc": np.mean(accuracies),
            "std_acc": np.std(accuracies),
            "mean_kl": np.mean(kl_divs),
            "std_kl": np.std(kl_divs),
            "mean_time": np.mean(times),
            "num_runs": len(results)
        }
    
    return analysis

def plot_beta_analysis(analysis: Dict[str, Any], save_path: str = "outputs/analysis/beta_sensitivity"):
    """β 분석 결과를 시각화합니다."""
    os.makedirs(save_path, exist_ok=True)
    
    betas = sorted(analysis.keys())
    mean_accs = [analysis[beta]["mean_acc"] for beta in betas]
    std_accs = [analysis[beta]["std_acc"] for beta in betas]
    mean_kls = [analysis[beta]["mean_kl"] for beta in betas]
    std_kls = [analysis[beta]["std_kl"] for beta in betas]
    
    # 1. 정확도 vs β 그래프
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.errorbar(betas, mean_accs, yerr=std_accs, marker='o', capsize=5)
    plt.xscale('log')
    plt.xlabel('β (Information Bottleneck Strength)')
    plt.ylabel('Final Student Accuracy')
    plt.title('Student Accuracy vs β')
    plt.grid(True, alpha=0.3)
    
    # 최적 β 표시
    best_beta = betas[np.argmax(mean_accs)]
    best_acc = max(mean_accs)
    plt.axvline(x=best_beta, color='red', linestyle='--', alpha=0.7)
    plt.text(best_beta, best_acc, f'Best: β={best_beta:.1e}', 
             rotation=90, verticalalignment='bottom')
    
    # 2. KL Divergence vs β 그래프
    plt.subplot(2, 2, 2)
    plt.errorbar(betas, mean_kls, yerr=std_kls, marker='s', capsize=5, color='orange')
    plt.xscale('log')
    plt.xlabel('β (Information Bottleneck Strength)')
    plt.ylabel('Average KL Divergence')
    plt.title('Information Compression vs β')
    plt.grid(True, alpha=0.3)
    
    # 3. 정확도 vs KL Divergence (트레이드오프)
    plt.subplot(2, 2, 3)
    plt.scatter(mean_kls, mean_accs, s=100, alpha=0.7)
    for i, beta in enumerate(betas):
        plt.annotate(f'β={beta:.1e}', (mean_kls[i], mean_accs[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    plt.xlabel('Average KL Divergence')
    plt.ylabel('Final Student Accuracy')
    plt.title('Accuracy vs Information Compression (Trade-off)')
    plt.grid(True, alpha=0.3)
    
    # 4. β 값별 실행 시간
    plt.subplot(2, 2, 4)
    mean_times = [analysis[beta]["mean_time"] / 3600 for beta in betas]  # 시간 단위
    plt.bar(range(len(betas)), mean_times, alpha=0.7)
    plt.xticks(range(len(betas)), [f'{beta:.1e}' for beta in betas], rotation=45)
    plt.xlabel('β Value')
    plt.ylabel('Average Runtime (hours)')
    plt.title('Runtime vs β')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'beta_sensitivity_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return best_beta

def save_analysis_results(analysis: Dict[str, Any], best_beta: float, save_path: str):
    """분석 결과를 CSV로 저장합니다."""
    os.makedirs(save_path, exist_ok=True)
    
    # 결과를 DataFrame으로 변환
    results_data = []
    for beta, stats in analysis.items():
        results_data.append({
            'beta': beta,
            'mean_accuracy': stats['mean_acc'],
            'std_accuracy': stats['std_acc'],
            'mean_kl_divergence': stats['mean_kl'],
            'std_kl_divergence': stats['std_kl'],
            'mean_runtime_hours': stats['mean_time'] / 3600,
            'num_runs': stats['num_runs']
        })
    
    df = pd.DataFrame(results_data)
    df = df.sort_values('beta')
    
    # CSV 저장
    csv_path = os.path.join(save_path, 'beta_sensitivity_results.csv')
    df.to_csv(csv_path, index=False)
    
    # 요약 보고서 생성
    report_path = os.path.join(save_path, 'beta_analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write("Beta Sensitivity Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Best β value: {best_beta:.1e}\n")
        f.write(f"Best accuracy: {analysis[best_beta]['mean_acc']:.4f} ± {analysis[best_beta]['std_acc']:.4f}\n\n")
        
        f.write("Detailed Results:\n")
        f.write("-" * 30 + "\n")
        for beta in sorted(analysis.keys()):
            stats = analysis[beta]
            f.write(f"β={beta:.1e}: {stats['mean_acc']:.4f}±{stats['std_acc']:.4f} "
                   f"(KL: {stats['mean_kl']:.4f}±{stats['std_kl']:.4f})\n")
    
    print(f"📊 Results saved to: {save_path}")
    print(f"📈 Best β value: {best_beta:.1e}")
    print(f"🎯 Best accuracy: {analysis[best_beta]['mean_acc']:.4f} ± {analysis[best_beta]['std_acc']:.4f}")

def run_beta_sensitivity_analysis():
    """전체 β 민감도 분석을 실행합니다."""
    print("🎯 Starting Beta Sensitivity Analysis")
    print("=" * 60)
    print(f"β values: {BETA_VALUES}")
    print(f"Number of runs per β: {NUM_RUNS}")
    print(f"Total experiments: {len(BETA_VALUES) * NUM_RUNS}")
    print(f"Expected duration: ~{len(BETA_VALUES) * NUM_RUNS * TIMEOUT_HOURS} hours")
    print("=" * 60)
    
    all_results = []
    
    # 각 β 값에 대해 여러 번 실행
    for beta in BETA_VALUES:
        for run_id in range(NUM_RUNS):
            result = run_beta_experiment(beta, run_id)
            all_results.append(result)
    
    # 결과 분석
    print("\n📊 Analyzing results...")
    analysis = analyze_beta_results(all_results)
    
    if not analysis:
        print("❌ No valid results to analyze")
        return
    
    # 시각화 및 저장
    save_path = "outputs/analysis/beta_sensitivity"
    best_beta = plot_beta_analysis(analysis, save_path)
    save_analysis_results(analysis, best_beta, save_path)
    
    print("\n🎉 Beta sensitivity analysis completed!")
    print(f"📁 Results saved in: {save_path}")

if __name__ == "__main__":
    run_beta_sensitivity_analysis() 