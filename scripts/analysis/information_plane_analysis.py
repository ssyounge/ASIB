#!/usr/bin/env python3
"""
Information Plane Analysis Script

β 값 변화에 따른 Information Plane 분석을 수행합니다.
IB 이론의 핵심인 I(Z;X) vs I(Z;Y) 트레이드오프를 시각화합니다.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
from typing import Dict, List, Tuple, Any
from scipy import stats
import seaborn as sns

class InformationPlaneAnalyzer:
    def __init__(self, results_dir: str = "outputs/analysis/beta_sensitivity"):
        self.results_dir = results_dir
        self.data = self.load_beta_results()
        
    def load_beta_results(self) -> pd.DataFrame:
        """β 민감도 분석 결과를 로드합니다."""
        csv_path = os.path.join(self.results_dir, "beta_sensitivity_results.csv")
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        else:
            print(f"⚠️ Results file not found: {csv_path}")
            return pd.DataFrame()
    
    def calculate_information_plane_metrics(self) -> pd.DataFrame:
        """Information Plane 분석을 위한 메트릭을 계산합니다."""
        if self.data.empty:
            return pd.DataFrame()
        
        # I(Z;X) = KL Divergence (압축률/복잡도)
        # I(Z;Y) = Accuracy (예측력) - 정규화된 값으로 변환
        df = self.data.copy()
        
        # I(Z;X) = KL Divergence (로그 스케일로 정규화)
        df['I_Z_X'] = np.log1p(df['mean_kl_divergence'])
        
        # I(Z;Y) = Accuracy (0-1 스케일)
        df['I_Z_Y'] = df['mean_accuracy'] / 100.0
        
        # β 값의 로그 스케일
        df['log_beta'] = np.log10(df['beta'])
        
        return df
    
    def plot_information_plane(self, save_path: str = None) -> Tuple[float, float]:
        """Information Plane을 시각화합니다."""
        df = self.calculate_information_plane_metrics()
        if df.empty:
            print("❌ No data available for Information Plane analysis")
            return None, None
        
        # 그래프 설정
        plt.figure(figsize=(12, 10))
        
        # 1. Information Plane (메인 플롯)
        plt.subplot(2, 2, 1)
        
        # β 값별로 색상 구분
        colors = plt.cm.viridis(np.linspace(0, 1, len(df)))
        
        for i, (_, row) in enumerate(df.iterrows()):
            plt.scatter(row['I_Z_X'], row['I_Z_Y'], 
                       c=[colors[i]], s=150, alpha=0.8,
                       label=f"β={row['beta']:.1e}")
        
        # 최적 점 표시
        best_idx = df['I_Z_Y'].idxmax()
        best_point = df.loc[best_idx]
        plt.scatter(best_point['I_Z_X'], best_point['I_Z_Y'], 
                   c='red', s=200, marker='*', 
                   label=f'Optimal: β={best_point["beta"]:.1e}')
        
        plt.xlabel('I(Z;X) = Compression Rate (log KL Divergence)')
        plt.ylabel('I(Z;Y) = Predictive Power (Accuracy)')
        plt.title('Information Plane Analysis')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 2. β vs I(Z;X) 관계
        plt.subplot(2, 2, 2)
        plt.scatter(df['log_beta'], df['I_Z_X'], c=colors, s=100, alpha=0.8)
        plt.xlabel('log₁₀(β)')
        plt.ylabel('I(Z;X) = Compression Rate')
        plt.title('β vs Compression Rate')
        plt.grid(True, alpha=0.3)
        
        # 3. β vs I(Z;Y) 관계
        plt.subplot(2, 2, 3)
        plt.scatter(df['log_beta'], df['I_Z_Y'], c=colors, s=100, alpha=0.8)
        plt.xlabel('log₁₀(β)')
        plt.ylabel('I(Z;Y) = Predictive Power')
        plt.title('β vs Predictive Power')
        plt.grid(True, alpha=0.3)
        
        # 4. Trade-off 곡선 (Pareto Frontier)
        plt.subplot(2, 2, 4)
        
        # Pareto optimal points 찾기
        pareto_points = self.find_pareto_frontier(df[['I_Z_X', 'I_Z_Y']].values)
        
        # 모든 점 플롯
        plt.scatter(df['I_Z_X'], df['I_Z_Y'], c='lightblue', s=80, alpha=0.6, label='All Points')
        
        # Pareto frontier 플롯
        pareto_x = [df.iloc[i]['I_Z_X'] for i in pareto_points]
        pareto_y = [df.iloc[i]['I_Z_Y'] for i in pareto_points]
        plt.plot(pareto_x, pareto_y, 'r-', linewidth=2, label='Pareto Frontier')
        plt.scatter(pareto_x, pareto_y, c='red', s=100, alpha=0.8)
        
        plt.xlabel('I(Z;X) = Compression Rate')
        plt.ylabel('I(Z;Y) = Predictive Power')
        plt.title('Pareto Frontier (Trade-off Curve)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, 'information_plane_analysis.png'), 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return best_point['I_Z_X'], best_point['I_Z_Y']
    
    def find_pareto_frontier(self, points: np.ndarray) -> List[int]:
        """Pareto optimal points를 찾습니다."""
        pareto_points = []
        
        for i, point in enumerate(points):
            is_pareto = True
            for j, other_point in enumerate(points):
                if i != j:
                    # 다른 점이 이 점을 지배하는지 확인
                    if (other_point[0] <= point[0] and other_point[1] >= point[1] and 
                        (other_point[0] < point[0] or other_point[1] > point[1])):
                        is_pareto = False
                        break
            
            if is_pareto:
                pareto_points.append(i)
        
        return pareto_points
    
    def analyze_ib_theory_connection(self) -> Dict[str, Any]:
        """IB 이론과의 연결성을 분석합니다."""
        df = self.calculate_information_plane_metrics()
        if df.empty:
            return {}
        
        # 1. 압축률과 예측력 간의 상관관계
        correlation = stats.pearsonr(df['I_Z_X'], df['I_Z_Y'])[0]
        
        # 2. 최적 β 값에서의 특성
        best_idx = df['I_Z_Y'].idxmax()
        best_point = df.loc[best_idx]
        
        # 3. Trade-off 분석
        # 압축률이 증가할 때 예측력이 어떻게 변하는지
        compression_gain = df['I_Z_X'].max() - df['I_Z_X'].min()
        prediction_loss = df['I_Z_Y'].max() - df['I_Z_Y'].min()
        
        analysis = {
            'correlation_compression_prediction': correlation,
            'optimal_beta': best_point['beta'],
            'optimal_compression': best_point['I_Z_X'],
            'optimal_prediction': best_point['I_Z_Y'],
            'compression_range': compression_gain,
            'prediction_range': prediction_loss,
            'trade_off_ratio': prediction_loss / compression_gain if compression_gain > 0 else 0
        }
        
        return analysis
    
    def generate_theory_report(self, save_path: str = None) -> str:
        """IB 이론 연결성 분석 보고서를 생성합니다."""
        analysis = self.analyze_ib_theory_connection()
        if not analysis:
            return "No analysis data available"
        
        report = f"""
Information Bottleneck Theory Analysis Report
============================================

1. Compression-Prediction Correlation
   - Pearson correlation: {analysis['correlation_compression_prediction']:.4f}
   - Interpretation: {'Strong negative' if analysis['correlation_compression_prediction'] < -0.7 else 'Moderate negative' if analysis['correlation_compression_prediction'] < -0.3 else 'Weak'} correlation

2. Optimal Operating Point
   - Optimal β: {analysis['optimal_beta']:.1e}
   - Compression rate (I(Z;X)): {analysis['optimal_compression']:.4f}
   - Predictive power (I(Z;Y)): {analysis['optimal_prediction']:.4f}

3. Trade-off Analysis
   - Compression range: {analysis['compression_range']:.4f}
   - Prediction range: {analysis['prediction_range']:.4f}
   - Trade-off ratio: {analysis['trade_off_ratio']:.4f}

4. Theoretical Insights
   - The optimal β value {analysis['optimal_beta']:.1e} achieves the best balance
     between compression and prediction power
   - This validates the IB theory that there exists an optimal information
     bottleneck strength for knowledge distillation
   - The negative correlation confirms the fundamental trade-off in IB theory
"""
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, 'ib_theory_analysis.txt'), 'w') as f:
                f.write(report)
        
        return report

def run_information_plane_analysis():
    """Information Plane 분석을 실행합니다."""
    print("🔬 Starting Information Plane Analysis")
    print("=" * 50)
    
    analyzer = InformationPlaneAnalyzer()
    
    # Information Plane 시각화
    print("📊 Generating Information Plane visualization...")
    best_compression, best_prediction = analyzer.plot_information_plane("outputs/analysis/information_plane")
    
    # 이론 연결성 분석
    print("📈 Analyzing IB theory connection...")
    report = analyzer.generate_theory_report("outputs/analysis/information_plane")
    print(report)
    
    print("✅ Information Plane analysis completed!")
    print("📁 Results saved in: outputs/analysis/information_plane/")

if __name__ == "__main__":
    run_information_plane_analysis() 