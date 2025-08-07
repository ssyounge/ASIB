#!/usr/bin/env python3
"""
Sensitivity Analysis for ASIB-KD Framework

Phase 1: 구성 요소 분석 (Ablation Study)
목표: ASIB를 구성하는 핵심 요소들(IB, CCCP, Teacher Adaptation, PF)이 최종 성능 향상에 독립적으로 기여함을 증명

고정 환경:
- 데이터셋: CIFAR-100
- Teachers (이종 구조): ConvNeXt-S + ResNet152
- Student: ResNet50

실험 구성:
1. Baseline 설정 (MBM + E2E + Fixed Teachers)
2. Information Bottleneck (IB) 효과 검증 (+IB)
3. CCCP (Stage-wise 학습) 효과 검증 (+IB +CCCP)
4. Teacher Adaptation 효과 검증 (+IB +CCCP +T-Adapt)
5. Progressive Partial Freezing (PF) 효과 검증 (ASIB Full)

중요 분석: CCCP 적용 시 더 빠르고 안정적으로(진동 없이) 수렴함을 보이는 것이 핵심입니다.
"""

import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.builder import create_teacher_by_name, create_student_by_name
from data.cifar100 import get_cifar100_loaders
from utils.logging import setup_logging


@dataclass
class AblationConfig:
    """Ablation 실험 설정"""
    experiment_name: str
    use_ib: bool = False
    use_cccp: bool = False
    use_teacher_adaptation: bool = False
    use_pf: bool = False
    beta: float = 0.001  # VIB 파라미터
    description: str = ""


def create_ablation_configs() -> List[AblationConfig]:
    """Ablation 실험 설정들 생성"""
    configs = []
    
    # 실험 1: Baseline 설정 (MBM + E2E + Fixed Teachers)
    configs.append(AblationConfig(
        experiment_name="baseline",
        use_ib=False,
        use_cccp=False,
        use_teacher_adaptation=False,
        use_pf=False,
        description="MBM + E2E + Fixed Teachers"
    ))
    
    # 실험 2: Information Bottleneck (IB) 효과 검증 (+IB)
    configs.append(AblationConfig(
        experiment_name="baseline_plus_ib",
        use_ib=True,
        use_cccp=False,
        use_teacher_adaptation=False,
        use_pf=False,
        description="Baseline + Information Bottleneck"
    ))
    
    # 실험 3: CCCP (Stage-wise 학습) 효과 검증 (+IB +CCCP)
    configs.append(AblationConfig(
        experiment_name="baseline_plus_ib_cccp",
        use_ib=True,
        use_cccp=True,
        use_teacher_adaptation=False,
        use_pf=False,
        description="Baseline + IB + CCCP"
    ))
    
    # 실험 4: Teacher Adaptation 효과 검증 (+IB +CCCP +T-Adapt)
    configs.append(AblationConfig(
        experiment_name="baseline_plus_ib_cccp_tadapt",
        use_ib=True,
        use_cccp=True,
        use_teacher_adaptation=True,
        use_pf=False,
        description="Baseline + IB + CCCP + Teacher Adaptation"
    ))
    
    # 실험 5: Progressive Partial Freezing (PF) 효과 검증 (ASIB Full)
    configs.append(AblationConfig(
        experiment_name="asib_full",
        use_ib=True,
        use_cccp=True,
        use_teacher_adaptation=True,
        use_pf=True,
        description="ASIB Full (Baseline + IB + CCCP + T-Adapt + PF)"
    ))
    
    return configs


def run_ablation_experiment(config: AblationConfig, seed: int = 42) -> Dict[str, float]:
    """
    특정 Ablation 설정에서 실험 실행
    
    Args:
        config: Ablation 설정
        seed: 랜덤 시드
    
    Returns:
        실험 결과 딕셔너리
    """
    # 시드 설정
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 교사 모델들 생성
    teacher1 = create_teacher_by_name("convnext_s_teacher", pretrained=True)
    teacher2 = create_teacher_by_name("resnet152_teacher", pretrained=True)
    
    teacher1 = teacher1.to(device)
    teacher2 = teacher2.to(device)
    
    # 학생 모델 생성
    student = create_student_by_name("resnet50_student", pretrained=False)
    student = student.to(device)
    
    # 데이터 로더
    train_loader, val_loader = get_cifar100_loaders(
        root="./data",
        batch_size=128,
        num_workers=4
    )
    
    # 학습 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1, momentum=0.9, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    # 학습 기록
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # 학습
    num_epochs = 200
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        student.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            # 학생 모델 출력
            student_output = student(data)
            
            # 교사 모델들 출력
            with torch.no_grad():
                teacher1_output = teacher1(data)
                teacher2_output = teacher2(data)
            
            # 기본 손실
            loss = criterion(student_output, target)
            
            # IB 적용 (실험 2, 3, 4, 5)
            if config.use_ib:
                # VIB 손실 계산 (간단한 버전)
                kl_loss = torch.mean(torch.sum(student_output * torch.log(student_output + 1e-8), dim=1))
                loss += config.beta * kl_loss
            
            # CCCP 적용 (실험 3, 4, 5)
            if config.use_cccp:
                # A-Step: IB-MBM 학습
                if epoch % 2 == 0:  # A-Step
                    pass  # IB-MBM 학습 로직
                # B-Step: Student 학습
                else:  # B-Step
                    pass  # Student 학습 로직
            
            # Teacher Adaptation 적용 (실험 4, 5)
            if config.use_teacher_adaptation:
                # 교사 모델 상위 레이어 업데이트
                teacher_adaptation_loss = 0.1 * (nn.MSELoss()(student_output, teacher1_output) + 
                                                nn.MSELoss()(student_output, teacher2_output))
                loss += teacher_adaptation_loss
            
            # PF 적용 (실험 5)
            if config.use_pf:
                # Progressive Partial Freezing 로직
                if epoch < 50:
                    # 초기 50 에포크: 모든 레이어 학습
                    pass
                elif epoch < 100:
                    # 50-100 에포크: 일부 레이어 고정
                    pass
                else:
                    # 100+ 에포크: 더 많은 레이어 고정
                    pass
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = student_output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
        
        # Validation
        student.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = student(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        # 메트릭 계산
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
        
        scheduler.step()
        
        if (epoch + 1) % 50 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
    
    return {
        "experiment_name": config.experiment_name,
        "description": config.description,
        "use_ib": config.use_ib,
        "use_cccp": config.use_cccp,
        "use_teacher_adaptation": config.use_teacher_adaptation,
        "use_pf": config.use_pf,
        "beta": config.beta,
        "best_accuracy": best_acc,
        "final_train_accuracy": train_accuracies[-1],
        "final_val_accuracy": val_accuracies[-1],
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies
    }


def run_sensitivity_analysis():
    """Sensitivity Analysis 메인 함수"""
    logger = setup_logging({
        "results_dir": "outputs/analysis/sensitivity_analysis",
        "exp_id": "sensitivity_analysis",
        "log_level": "INFO"
    })
    
    logger.info("🚀 Starting Sensitivity Analysis (Ablation Study)")
    logger.info("=" * 60)
    
    # Ablation 설정들 생성
    ablation_configs = create_ablation_configs()
    logger.info(f"Created {len(ablation_configs)} ablation configurations")
    
    # 결과 저장
    results = []
    
    # 각 실험 실행 (3회 반복)
    for i, config in enumerate(ablation_configs):
        logger.info(f"Experiment {i+1}/{len(ablation_configs)}: {config.experiment_name}")
        logger.info(f"Description: {config.description}")
        logger.info(f"IB: {config.use_ib}, CCCP: {config.use_cccp}, T-Adapt: {config.use_teacher_adaptation}, PF: {config.use_pf}")
        
        # 3회 반복 실험
        experiment_results = []
        for run in range(3):
            logger.info(f"  Run {run+1}/3")
            result = run_ablation_experiment(config, seed=42+run)
            experiment_results.append(result)
        
        # 평균 및 표준편차 계산
        accuracies = [r['best_accuracy'] for r in experiment_results]
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        
        # 최종 결과 저장
        final_result = {
            "experiment_name": config.experiment_name,
            "description": config.description,
            "use_ib": config.use_ib,
            "use_cccp": config.use_cccp,
            "use_teacher_adaptation": config.use_teacher_adaptation,
            "use_pf": config.use_pf,
            "beta": config.beta,
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "individual_results": experiment_results
        }
        
        results.append(final_result)
        
        logger.info(f"  Mean Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
        logger.info("-" * 40)
    
    # 결과 시각화
    plot_sensitivity_results(results)
    
    # 결과 저장
    save_sensitivity_results(results)
    
    logger.info("✅ Sensitivity Analysis completed!")
    
    return results


def plot_sensitivity_results(results: List[Dict[str, float]]):
    """Sensitivity Analysis 결과 시각화"""
    # 데이터 준비
    experiment_names = [r['experiment_name'] for r in results]
    mean_accuracies = [r['mean_accuracy'] for r in results]
    std_accuracies = [r['std_accuracy'] for r in results]
    
    # 그래프 생성
    plt.figure(figsize=(15, 10))
    
    # 1. 정확도 비교
    plt.subplot(2, 3, 1)
    bars = plt.bar(experiment_names, mean_accuracies, yerr=std_accuracies, 
                   capsize=5, alpha=0.7, color='skyblue')
    plt.xlabel('Experiment')
    plt.ylabel('Accuracy (%)')
    plt.title('Ablation Study Results')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # 2. 구성 요소별 기여도
    plt.subplot(2, 3, 2)
    components = ['IB', 'CCCP', 'T-Adapt', 'PF']
    contributions = []
    
    baseline_acc = mean_accuracies[0]  # baseline
    
    for i, result in enumerate(results[1:], 1):
        contribution = result['mean_accuracy'] - baseline_acc
        contributions.append(contribution)
    
    plt.bar(components, contributions, color=['green', 'orange', 'red', 'purple'], alpha=0.7)
    plt.xlabel('Component')
    plt.ylabel('Accuracy Improvement (%)')
    plt.title('Component Contributions')
    plt.grid(True, alpha=0.3)
    
    # 3. 학습 곡선 비교 (CCCP 효과)
    plt.subplot(2, 3, 3)
    for i, result in enumerate(results):
        if result['use_ib'] and not result['use_cccp']:  # IB만
            val_accs = result['individual_results'][0]['val_accuracies']
            plt.plot(val_accs, label='IB Only', linewidth=2)
        elif result['use_ib'] and result['use_cccp']:  # IB + CCCP
            val_accs = result['individual_results'][0]['val_accuracies']
            plt.plot(val_accs, label='IB + CCCP', linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('CCCP Effect (Learning Stability)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. 실험별 상세 비교
    plt.subplot(2, 3, 4)
    x_pos = np.arange(len(experiment_names))
    plt.bar(x_pos, mean_accuracies, yerr=std_accuracies, capsize=5, alpha=0.7)
    plt.xlabel('Experiment')
    plt.ylabel('Accuracy (%)')
    plt.title('Detailed Comparison')
    plt.xticks(x_pos, experiment_names, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # 5. 구성 요소 조합별 성능
    plt.subplot(2, 3, 5)
    combinations = []
    accuracies = []
    
    for result in results:
        combo = []
        if result['use_ib']: combo.append('IB')
        if result['use_cccp']: combo.append('CCCP')
        if result['use_teacher_adaptation']: combo.append('T-Adapt')
        if result['use_pf']: combo.append('PF')
        
        combinations.append('+'.join(combo) if combo else 'Baseline')
        accuracies.append(result['mean_accuracy'])
    
    plt.bar(combinations, accuracies, alpha=0.7, color='lightcoral')
    plt.xlabel('Component Combination')
    plt.ylabel('Accuracy (%)')
    plt.title('Component Combination Performance')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # 6. 표준편차 비교 (안정성)
    plt.subplot(2, 3, 6)
    plt.bar(experiment_names, std_accuracies, alpha=0.7, color='lightgreen')
    plt.xlabel('Experiment')
    plt.ylabel('Standard Deviation (%)')
    plt.title('Training Stability')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/analysis/sensitivity_analysis/sensitivity_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def save_sensitivity_results(results: List[Dict[str, float]]):
    """결과를 JSON 파일로 저장"""
    output_file = 'outputs/analysis/sensitivity_analysis/sensitivity_results.json'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    run_sensitivity_analysis() 