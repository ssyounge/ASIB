# Phase 1: Ablation Study 실험 보고서

## 📋 실험 개요

**목표**: ASIB 프레임워크의 핵심 구성 요소들이 최종 성능 향상에 독립적으로 기여함을 증명

**방법**: 하나의 대표 환경을 고정하고, 요소를 점진적으로 추가하는 방식

**환경 설정**:
- **데이터셋**: CIFAR-100
- **Teachers**: ConvNeXt-S + ResNet152 (이종 구조)
- **Student**: ResNet50
- **총 에포크**: 200

---

## 🧪 실험 1: Baseline (MBM + E2E + Fixed Teachers)

### 📊 하이퍼파라미터 설정

| 카테고리 | 파라미터 | 값 | 설명 |
|---------|---------|-----|------|
| **모델 설정** | Teacher1 | ConvNeXt-S | 첫 번째 교사 모델 |
| | Teacher2 | ResNet152 | 두 번째 교사 모델 |
| | Student | ResNet50 | 학생 모델 |
| **최적화** | Teacher LR | 0.0 | 교사 모델 고정 |
| | Student LR | 2.0e-4 | 학생 모델 학습률 |
| | Student WD | 3.0e-4 | 학생 모델 가중치 감쇠 |
| **학습 방식** | Partial Freeze | False | 교사 완전 고정 |
| | Num Stages | 1 | 단일 스테이지 |
| | Teacher Adapt | 0 | 교사 적응 없음 |
| **MBM/Adapter** | Distillation Adapter | True | 어댑터 사용 |
| | Distill Out Dim | 512 | 어댑터 출력 차원 |
| | MBM Query Dim | 2048 | MBM 쿼리 차원 |
| | MBM Out Dim | 2048 | MBM 출력 차원 |
| | MBM N Head | 8 | 멀티헤드 어텐션 |
| **Information Bottleneck** | Use IB | False | IB 비활성화 |
| | IB Beta | 0.0 | 정보 압축 강도 |
| **Knowledge Distillation** | KD Alpha | 0.5 | 기본 KD 가중치 |
| | KD Ens Alpha | 0.5 | 앙상블 KD 가중치 |
| | Disagree Weight | False | Disagreement 가중치 없음 |
| **CCCP** | Use CCCP | False | E2E 학습 |
| **데이터** | Batch Size | 64 | 배치 크기 |
| | CutMix Alpha | 0.0 | CutMix 비활성화 |
| | MixUp Alpha | 0.0 | MixUp 비활성화 |

### 🎯 실험 목적
가장 기본적인 형태로, 교사 모델들은 완전히 고정하고 VIB는 비활성화한 상태에서의 성능을 측정합니다.

### 📈 예상 결과
- **기준 성능**: 다른 실험들과 비교할 기준선 제공
- **MBM 효과**: 기본적인 특징 융합 모듈의 효과 측정

---

## 🧪 실험 2: +IB (Information Bottleneck)

### 📊 하이퍼파라미터 설정

| 카테고리 | 파라미터 | 값 | 설명 |
|---------|---------|-----|------|
| **모델 설정** | Teacher1 | ConvNeXt-S | 첫 번째 교사 모델 |
| | Teacher2 | ResNet152 | 두 번째 교사 모델 |
| | Student | ResNet50 | 학생 모델 |
| **최적화** | Teacher LR | 0.0 | 교사 모델 고정 |
| | Student LR | 2.0e-4 | 학생 모델 학습률 |
| | Student WD | 3.0e-4 | 학생 모델 가중치 감쇠 |
| **학습 방식** | Partial Freeze | False | 교사 완전 고정 |
| | Num Stages | 1 | 단일 스테이지 |
| | Teacher Adapt | 0 | 교사 적응 없음 |
| **MBM/Adapter** | Distillation Adapter | True | 어댑터 사용 |
| | Distill Out Dim | 512 | 어댑터 출력 차원 |
| | MBM Query Dim | 2048 | MBM 쿼리 차원 |
| | MBM Out Dim | 2048 | MBM 출력 차원 |
| | MBM N Head | 8 | 멀티헤드 어텐션 |
| **Information Bottleneck** | Use IB | **True** | **IB 활성화** |
| | IB Beta | **0.001** | **정보 압축 강도** |
| | IB Beta Warmup | **5** | **IB warmup 에포크** |
| **Knowledge Distillation** | KD Alpha | 0.5 | 기본 KD 가중치 |
| | KD Ens Alpha | 0.5 | 앙상블 KD 가중치 |
| | Disagree Weight | False | Disagreement 가중치 없음 |
| **CCCP** | Use CCCP | False | E2E 학습 |
| **데이터** | Batch Size | 64 | 배치 크기 |
| | CutMix Alpha | 0.0 | CutMix 비활성화 |
| | MixUp Alpha | 0.0 | MixUp 비활성화 |

### 🎯 실험 목적
Baseline 설정에 VIB 모듈을 활성화하여 노이즈와 중복 정보를 효과적으로 제거하는지 검증합니다.

### 📈 예상 결과
- **성능 향상**: Baseline 대비 정확도 향상
- **정보 압축**: KL Divergence를 통한 정보량 측정
- **노이즈 제거**: 더 깔끔한 특징 전달

---

## 🧪 실험 3: +IB +CCCP (Stage-wise 학습)

### 📊 하이퍼파라미터 설정

| 카테고리 | 파라미터 | 값 | 설명 |
|---------|---------|-----|------|
| **모델 설정** | Teacher1 | ConvNeXt-S | 첫 번째 교사 모델 |
| | Teacher2 | ResNet152 | 두 번째 교사 모델 |
| | Student | ResNet50 | 학생 모델 |
| **최적화** | Teacher LR | 0.0 | 교사 모델 고정 |
| | Student LR | 2.0e-4 | 학생 모델 학습률 |
| | Student WD | 3.0e-4 | 학생 모델 가중치 감쇠 |
| **학습 방식** | Partial Freeze | False | 교사 완전 고정 |
| | Num Stages | 1 | 단일 스테이지 |
| | Teacher Adapt | 0 | 교사 적응 없음 |
| **MBM/Adapter** | Distillation Adapter | True | 어댑터 사용 |
| | Distill Out Dim | 512 | 어댑터 출력 차원 |
| | MBM Query Dim | 2048 | MBM 쿼리 차원 |
| | MBM Out Dim | 2048 | MBM 출력 차원 |
| | MBM N Head | 8 | 멀티헤드 어텐션 |
| **Information Bottleneck** | Use IB | True | IB 활성화 |
| | IB Beta | 0.001 | 정보 압축 강도 |
| | IB Beta Warmup | 5 | IB warmup 에포크 |
| **Knowledge Distillation** | KD Alpha | 0.5 | 기본 KD 가중치 |
| | KD Ens Alpha | 0.5 | 앙상블 KD 가중치 |
| | Disagree Weight | False | Disagreement 가중치 없음 |
| **CCCP** | Use CCCP | **True** | **A-Step/B-Step 교대 최적화** |
| | Tau | 4.0 | CCCP 온도 파라미터 |
| **데이터** | Batch Size | 64 | 배치 크기 |
| | CutMix Alpha | 0.0 | CutMix 비활성화 |
| | MixUp Alpha | 0.0 | MixUp 비활성화 |

### 🎯 실험 목적
IB 설정에서 학습 방식을 E2E에서 A-Step/B-Step 교대 최적화 방식으로 변경하여 더 빠르고 안정적인 수렴을 달성하는지 검증합니다.

### 📈 예상 결과
- **수렴 속도**: 실험 2 대비 더 빠른 수렴
- **안정성**: 학습 곡선의 진동 감소
- **최종 성능**: 실험 2 대비 추가 성능 향상

---

## 🔄 실험 간 차이점 요약

| 구성 요소 | 실험 1 (Baseline) | 실험 2 (+IB) | 실험 3 (+CCCP) |
|-----------|------------------|--------------|----------------|
| **Information Bottleneck** | ❌ (β=0) | ✅ (β=0.001) | ✅ (β=0.001) |
| **CCCP** | ❌ (E2E) | ❌ (E2E) | ✅ (A-Step/B-Step) |
| **예상 성능** | 기준선 | Baseline + IB 효과 | Baseline + IB + CCCP 효과 |

---

## 🚀 실행 방법

### 전체 Ablation Study 실행
```bash
sbatch run/run_ablation_study.sh
```

### 개별 실험 실행
```bash
# 실험 1: Baseline
python main.py --config-name experiment/ablation_baseline

# 실험 2: +IB
python main.py --config-name experiment/ablation_ib

# 실험 3: +IB +CCCP
python main.py --config-name experiment/ablation_cccp
```

---

## 📊 결과 분석 방법

### 1. 성능 비교
- **Top-1 Accuracy**: 각 실험의 최종 정확도 비교
- **학습 곡선**: 에포크별 정확도 변화 추이
- **수렴 속도**: 목표 정확도 도달까지의 에포크 수

### 2. 정보 압축 분석 (실험 2, 3)
- **KL Divergence**: 정보 압축 정도 측정
- **β 값 영향**: 정보량과 성능 간의 트레이드오프

### 3. 학습 안정성 분석 (실험 3)
- **학습 곡선 비교**: 실험 2와 3의 진동 정도 비교
- **수렴 안정성**: 표준편차를 통한 안정성 측정

---

## 🎯 기대 효과

1. **실험 1 → 실험 2**: IB의 노이즈 제거 효과 입증
2. **실험 2 → 실험 3**: CCCP의 학습 안정성 향상 효과 입증
3. **전체**: 각 구성 요소의 독립적 기여도 정량화

이를 통해 ASIB 프레임워크의 각 구성 요소가 성능 향상에 실제로 기여함을 체계적으로 입증할 수 있습니다. 