# ASIB 실험 계획

## 개요
ASIB (Adaptive Synergy Information-Bottleneck) 프레임워크의 우수성과 기여도를 체계적으로 검증하기 위한 3단계 실험 계획입니다.

## 실험 환경 설정

### 공통 환경
- **데이터셋**: CIFAR-100 (메인), ImageNet/Tiny ImageNet (확장)
- **교사 모델**: ConvNeXt-S/L, EfficientNet-L2, ResNet152 (사전 파인튜닝 완료)
- **학생 모델**: ResNet50, MobileNetV2, ShuffleNetV2, EfficientNet-B0
- **최적화**: AdamW, Cosine Annealing, 200-240 Epochs
- **재현성**: 최소 3회 반복 실행

## Phase 1: 구성 요소 분석 (Ablation Study)

### 목표
ASIB 구성 요소들(IB, CCCP, Teacher Adaptation, PF)의 독립적 기여도 증명

### 실험 설정
- **고정 환경**: CIFAR-100, Teachers (ConvNeXt-S + ResNet152), Student (ResNet50)

### 실험 구성
1. **Baseline** (`ablation_baseline.yaml`)
   - IB_MBM + E2E + Fixed Teachers
   - 교사 완전 고정, VIB 비활성화 (β=0)

2. **+IB** (`ablation_ib.yaml`)
   - Baseline + Information Bottleneck
   - VIB 활성화 (β=0.001)

3. **+CCCP** (`ablation_cccp.yaml`)
   - +IB + Stage-wise 학습
   - A-Step/B-Step 교대 최적화

4. **ASIB Full** (`ablation_full.yaml`)
   - 모든 구성 요소 활성화
   - Progressive Partial Freezing 포함

### 추가 분석
- **β 민감도 분석**: β = [0.0001, 0.001, 0.01, 0.1]
- **학습 곡선 비교**: CCCP 효과 시각화

## Phase 2: 최신 성능 비교 (SOTA Comparison)

### 목표
다양한 교사-학생 조합에서 기존 방법론 대비 우수성 입증

### 비교 대상
- **Logit 기반**: Vanilla KD, DKD
- **Feature 기반**: FitNet, AT, CRD, ReviewKD, SimKD
- **Multi-teacher Baseline**: Average Logits

### 시나리오

#### 시나리오 A: Heavy to Light (`sota_scenario_a.yaml`)
- **Teachers**: ConvNeXt-L, EfficientNet-L2
- **Student**: MobileNetV2, ShuffleNetV2
- **목표**: 현실적인 경량화 시나리오

#### 시나리오 B: Standard
- **Teachers**: ConvNeXt-S, ResNet152
- **Student**: ResNet50, EfficientNet-B0
- **목표**: 일반적인 환경에서의 성능

#### 시나리오 C: Homogeneous
- **Teachers**: ResNet152 (Seed 1) + ResNet152 (Seed 2)
- **Student**: ResNet50
- **목표**: 중복 정보 처리 능력 검증

## Phase 3: 강건성 분석 (Class Overlap 실험)

### 목표
상충 및 중복 정보 처리 능력을 극한 상황에서 검증

### 실험 환경
- **데이터셋**: CIFAR-100 (100개 클래스)
- **Teachers**: ResNet152 두 개
- **Student**: ResNet50

### Overlap 설정
1. **100% Overlap** (`overlap_100.yaml`)
   - T1, T2 모두 0-99 클래스 전체 학습
   - 완전 중복 환경

2. **50% Overlap** (`overlap_50.yaml`)
   - T1: 0-74 클래스, T2: 25-99 클래스
   - 부분 보완 환경

3. **0% Overlap** (`overlap_0.yaml`)
   - T1: 0-49 클래스, T2: 50-99 클래스
   - 완전 보완/상충 환경

### 예상 결과
- **단순 평균**: Overlap 감소 시 성능 급락
- **ASIB**: IB_MBM을 통한 동적 선택으로 안정적 성능 유지

## 실행 방법

### Phase 1: Ablation Study
```bash
sbatch run/run_ablation_study.sh
```

### Phase 2: SOTA Comparison
```bash
# 시나리오 A
python main.py --config-name experiment/sota_scenario_a

# 시나리오 B
python main.py --config-name experiment/sota_scenario_b

# 시나리오 C
python main.py --config-name experiment/sota_scenario_c
```

### Phase 3: Overlap Analysis
```bash
# 100% Overlap
python main.py --config-name experiment/overlap_100

# 50% Overlap
python main.py --config-name experiment/overlap_50

# 0% Overlap
python main.py --config-name experiment/overlap_0
```

### β 민감도 분석
```bash
python scripts/analysis/beta_sensitivity.py
```

## 결과 분석

### 주요 메트릭
- **정확도**: Top-1 Accuracy
- **효율성**: GPU 메모리 사용량, 학습 시간
- **강건성**: Overlap 비율에 따른 성능 변화

### 시각화
- 학습 곡선 비교
- β 값에 따른 성능 변화
- Overlap 비율에 따른 성능 변화

## 예상 기여도

1. **이론적 기여**: ASIB 구성 요소들의 독립적 효과 증명
2. **실용적 기여**: 다양한 실용 시나리오에서의 우수성 입증
3. **방법론적 기여**: 상충/중복 정보 처리 능력의 체계적 검증 