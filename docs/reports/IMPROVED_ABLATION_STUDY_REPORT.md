# Phase 1: Complete Ablation Study 실험 보고서 (개선판)

## 📋 실험 개요

**목표**: ASIB 프레임워크의 모든 핵심 구성 요소들이 최종 성능 향상에 독립적으로 기여함을 증명

**방법**: 하나의 대표 환경을 고정하고, 요소를 점진적으로 추가하는 방식

**환경 설정**:
- **데이터셋**: CIFAR-100
- **Teachers**: ConvNeXt-S + ResNet152 (이종 구조)
- **Student**: ResNet50
- **총 에포크**: 200
- **반복 실험**: 각 실험 3회 (통계적 유의성 확보)

---

## 🧪 완전한 Ablation Study 설계

### **수정된 실험 테이블**

| 실험 | IB‑MBM | VIB (IB) | CCCP | T-Adapt | PF | 분석 목표 |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 (Baseline) | ✅ | ❌ | ❌ (E2E) | ❌ | ❌ | 기본 성능 확인 |
| 2 (+IB) | ✅ | **✅** | ❌ (E2E) | ❌ | ❌ | IB 효과 (정보 압축) |
| 3 (+CCCP) | ✅ | ✅ | **✅** | ❌ | ❌ | CCCP 효과 (안정성) |
| **4 (+T-Adapt)** | ✅ | ✅ | ✅ | **✅** | ❌ | **교사 적응 시너지 효과** |
| **5 (ASIB Full)** | ✅ | ✅ | ✅ | ✅ | **✅** | **PF 효과 (효율성, 성능)** |

---

## 🧪 실험 1: Baseline (IB‑MBM + E2E + Fixed Teachers)

### 📊 개선된 하이퍼파라미터 설정

| 카테고리 | 파라미터 | 값 | 설명 |
|---------|---------|-----|------|
| **모델 설정** | Teacher1 | ConvNeXt-S | 첫 번째 교사 모델 |
| | Teacher2 | ResNet152 | 두 번째 교사 모델 |
| | Student | ResNet50 | 학생 모델 |
| **최적화** | Optimizer | **SGD** | **명시적 optimizer 설정** |
| | Teacher LR | 0.0 | 교사 모델 고정 |
| | Student LR | **0.1** | **CIFAR-100 ResNet50에 적합한 학습률** |
| | Student WD | 3.0e-4 | 학생 모델 가중치 감쇠 |
| | Momentum | **0.9** | **SGD momentum** |
| | Nesterov | **True** | **Nesterov momentum** |
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
| | CCCP NT | **1** | **A-Step 반복 횟수 명시** |
| | CCCP NS | **1** | **B-Step 반복 횟수 명시** |
| **데이터** | Batch Size | 64 | 배치 크기 |
| | CutMix Alpha | 0.0 | CutMix 비활성화 |
| | MixUp Alpha | 0.0 | MixUp 비활성화 |

### 🎯 실험 목적
가장 기본적인 형태로, 교사 모델들은 완전히 고정하고 VIB는 비활성화한 상태에서의 성능을 측정합니다.

---

## 🧪 실험 2: +IB (Information Bottleneck)

### 📊 핵심 변경사항

| 구성 요소 | 변경사항 | 설명 |
|-----------|----------|------|
| **Information Bottleneck** | Use IB: False → **True** | IB 활성화 |
| | IB Beta: 0.0 → **0.001** | 정보 압축 강도 설정 |
| | IB Beta Warmup: 0 → **5** | IB warmup 에포크 |

### 🎯 실험 목적
Baseline 설정에 VIB 모듈을 활성화하여 노이즈와 중복 정보를 효과적으로 제거하는지 검증합니다.

---

## 🧪 실험 3: +IB +CCCP (Stage-wise 학습)

### 📊 핵심 변경사항

| 구성 요소 | 변경사항 | 설명 |
|-----------|----------|------|
| **CCCP** | Use CCCP: False → **True** | A-Step/B-Step 교대 최적화 |
| | CCCP NT: 1 | A-Step 반복 횟수 (1 iteration) |
| | CCCP NS: 1 | B-Step 반복 횟수 (1 iteration) |

### 🎯 실험 목적
IB 설정에서 학습 방식을 E2E에서 A-Step/B-Step 교대 최적화 방식으로 변경하여 더 빠르고 안정적인 수렴을 달성하는지 검증합니다.

---

## 🧪 실험 4: +IB +CCCP +T-Adapt (Teacher Adaptation)

### 📊 핵심 변경사항

| 구성 요소 | 변경사항 | 설명 |
|-----------|----------|------|
| **Teacher Adaptation** | Teacher LR: 0.0 → **1.0e-5** | 교사 적응 학습률 |
| | Teacher Freeze Level: -1 → **0** | 교사 상위 레이어 학습 |
| | Teacher Freeze BN: True → **False** | 교사 BN도 학습 |
| | MBM Reg Lambda: 0.0 → **0.01** | 교사 정규화 (사전 지식 보호) |

### 🎯 실험 목적
IB+CCCP 설정에서 A-Step 시 교사 모델의 상위 레이어를 업데이트하여 시너지 효과를 검증합니다.

---

## 🧪 실험 5: ASIB Full (Progressive Partial Freezing)

### 📊 핵심 변경사항

| 구성 요소 | 변경사항 | 설명 |
|-----------|----------|------|
| **Progressive Freezing** | Use Partial Freeze: False → **True** | Progressive Freezing 활성화 |
| | Num Stages: 1 → **4** | 4단계 Progressive Freezing |
| | Teacher Freeze Level: 0 → **3** | Progressive: 3→2→1→0 |
| | Student Epochs: [200] → **[50,50,50,50]** | 각 스테이지 50 에포크 |
| **Disagreement Weight** | Use Disagree Weight: False → **True** | Disagreement 가중치 활성화 |
| **Data Augmentation** | CutMix Alpha: 0.0 → **0.3** | CutMix 활성화 |

### 🎯 실험 목적
모든 ASIB 구성 요소를 포함한 완전한 프레임워크의 성능과 효율성을 검증합니다.

---

## 📊 Phase 1.2: β 민감도 분석

### **분석 목적**
VIB 연구에서 필수적인 β 값 선택의 정당성을 입증합니다.

### **실험 설정**
- **β 값 범위**: [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1] (로그 스케일)
- **반복 실험**: 각 β 값당 3회
- **총 실험 수**: 27개 (9개 β × 3회 반복)
- **기준 설정**: ASIB Full config 사용

### **분석 지표**
1. **정확도 vs β**: 최적 β 값 도출
2. **KL Divergence vs β**: 정보 압축 정도 측정
3. **정확도 vs KL Divergence**: 트레이드오프 분석
4. **실행 시간 vs β**: 계산 효율성 분석

### **시각화**
- 4개 서브플롯으로 구성된 종합 분석 그래프
- 최적 β 값 표시 및 정당화
- 통계적 신뢰구간 포함

---

## 🔄 실험 간 차이점 요약

| 구성 요소 | 실험 1 | 실험 2 | 실험 3 | 실험 4 | 실험 5 |
|-----------|--------|--------|--------|--------|--------|
| **Information Bottleneck** | ❌ | ✅ | ✅ | ✅ | ✅ |
| **CCCP** | ❌ | ❌ | ✅ | ✅ | ✅ |
| **Teacher Adaptation** | ❌ | ❌ | ❌ | ✅ | ✅ |
| **Progressive Freezing** | ❌ | ❌ | ❌ | ❌ | ✅ |
| **예상 성능** | 기준선 | +IB | +CCCP | +T-Adapt | +PF |

---

## 🚀 실행 방법

### 전체 Ablation Study 실행
```bash
sbatch run/run_ablation_study.sh
```

### β 민감도 분석 실행
```bash
python scripts/analysis/beta_sensitivity.py
```

### 개별 실험 실행
```bash
# 실험 1-5: 개별 실행
python main.py --config-name experiment/ablation_baseline
python main.py --config-name experiment/ablation_ib
python main.py --config-name experiment/ablation_cccp
python main.py --config-name experiment/ablation_tadapt
python main.py --config-name experiment/ablation_full
```

---

## 📊 결과 분석 방법

### 1. 성능 비교 (정량적)
- **Top-1 Accuracy**: 각 실험의 최종 정확도 비교
- **학습 곡선**: 에포크별 정확도 변화 추이
- **수렴 속도**: 목표 정확도 도달까지의 에포크 수
- **통계적 유의성**: 평균 ± 표준편차 (3회 반복)

### 2. 정보 압축 분석 (실험 2-5)
- **KL Divergence**: 정보 압축 정도 측정
- **β 값 영향**: 정보량과 성능 간의 트레이드오프

### 3. 학습 안정성 분석 (실험 3-5)
- **표준편차**: 학습 후반부 20 에포크 검증 정확도의 표준편차
- **수렴 안정성**: CCCP 적용 시 안정성 향상 정량화

### 4. 효율성 분석 (실험 5)
- **메모리 효율성**: Peak VRAM 사용량 측정
- **학습 속도**: Epoch당 평균 학습 시간
- **PF 효과**: 실험 4 대비 효율성 개선 수치

### 5. β 민감도 분석 (Phase 1.2)
- **최적 β 도출**: 정확도 기준 최적값
- **트레이드오프 곡선**: 정확도 vs 정보 압축
- **정당화**: 0.001 선택 이유 입증

---

## 🎯 기대 효과 및 검증 포인트

### **1. 실험 1 → 실험 2**
- **기대**: IB 활성화로 인한 성능 향상
- **검증**: 노이즈 제거 효과의 정량화

### **2. 실험 2 → 실험 3**
- **기대**: CCCP 적용으로 인한 학습 안정성 향상
- **검증**: 표준편차 감소로 안정성 정량화

### **3. 실험 3 → 실험 4**
- **기대**: Teacher Adaptation으로 인한 시너지 효과
- **검증**: 교사 적응의 추가 성능 향상

### **4. 실험 4 → 실험 5**
- **기대**: PF 적용으로 인한 효율성 개선
- **검증**: 메모리 사용량 30% 절감, 학습 시간 단축

### **5. β 민감도 분석**
- **기대**: 최적 β 값 도출 및 정당화
- **검증**: 0.001 선택의 과학적 근거 제시

---

## 📈 논문 작성 시 강조 포인트

1. **완전한 Ablation Study**: 모든 핵심 구성 요소 검증
2. **정량적 분석**: 통계적 유의성과 정량적 지표
3. **β 민감도**: VIB 연구의 필수 요소 완비
4. **효율성 검증**: 메모리 및 시간 효율성 정량화
5. **재현성**: 명확한 하이퍼파라미터 설정과 3회 반복

이를 통해 ASIB 프레임워크의 각 구성 요소가 성능 향상에 실제로 기여함을 체계적이고 과학적으로 입증할 수 있습니다. 