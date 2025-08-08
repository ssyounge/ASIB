# ASIB (Adaptive Synergy Information-Bottleneck) Method Guide

## 📋 목차
1. [개요](#개요)
2. [핵심 아이디어](#핵심-아이디어)
3. [아키텍처](#아키텍처)
4. [하이퍼파라미터](#하이퍼파라미터)
5. [구현 세부사항](#구현-세부사항)
6. [실험 설정](#실험-설정)
7. [사용법](#사용법)
8. [성능 최적화](#성능-최적화)

---

## 🎯 개요

**ASIB (Adaptive Synergy Information-Bottleneck)**는 지식 증류(Knowledge Distillation)와 정보 병목(Information Bottleneck)을 결합한 혁신적인 방법론입니다.

### 주요 특징
- **2명의 Teacher 모델**과 **1명의 Student 모델** 구조
- **IB-MBM (Information-Bottleneck Manifold Bridging Module)**을 통한 정보 압축
- **Multi-stage 학습** 과정 (Teacher Adaptive → Student Distillation)
- **안정성-가소성 균형** 최적화

---

## 🧠 핵심 아이디어

### 1. 정보 병목 (Information Bottleneck)
```
Input → Encoder → Compressed Representation → Decoder → Output
                ↑
            IB Module
```

- **목적**: 불필요한 정보를 제거하고 핵심 정보만 전달
- **장점**: 모델 용량 절약, 과적합 방지, 일반화 성능 향상

### 2. 시너지 효과 (Synergy)
- **Teacher1**과 **Teacher2**의 지식을 **MBM**을 통해 결합
- **Student**는 압축된 시너지 지식을 학습

### 3. 적응적 학습 (Adaptive Learning)
- **Stage A**: Teacher 모델들 적응적 업데이트
- **Stage B**: Student 지식 증류
- 반복을 통한 점진적 성능 향상

---

## 🏗️ 아키텍처

### 전체 구조 및 데이터 플로우

```
Input Image
    │
    ├── Teacher1 ──┐
    │              │
    ├── Teacher2 ──┼── MBM ── Synergy Head ── Student ── Output
    │              │
    └── Student ───┘
```

### 상세 아키텍처 구성

#### 1. ASIBDistiller (메인 디스틸러)
```python
class ASIBDistiller(nn.Module):
    """
    Adaptive Synergy Information-Bottleneck Distiller
    
    핵심 구성요소:
    - teacher1, teacher2: 두 개의 교사 모델
    - student: 학습할 학생 모델
    - mbm: Manifold Bridging Module (정보 병목 기반)
    - synergy_head: 시너지 효과를 생성하는 헤드
    """
    
    def __init__(
        self,
        teacher1,                    # 첫 번째 교사 모델 (예: ConvNeXt-L)
        teacher2,                    # 두 번째 교사 모델 (예: ResNet-152)
        student,                     # 학생 모델 (예: ResNet-50)
        mbm,                         # Manifold Bridging Module
        synergy_head,                # 시너지 헤드
        alpha=0.5,                   # CE vs KL 비율 (0.5 = 50:50)
        synergy_ce_alpha=0.3,        # 시너지 CE 비중
        temperature=4.0,             # 지식 증류 온도 (높을수록 부드러운 확률 분포)
        reg_lambda=1e-4,             # 정규화 가중치
        mbm_reg_lambda=1e-4,         # MBM 정규화 가중치
        num_stages=2,                # 학습 스테이지 수
        device="cuda",               # 디바이스
        config=None                  # 추가 설정
    ):
        super().__init__()
        
        # 모델들
        self.teacher1 = teacher1
        self.teacher2 = teacher2
        self.student = student
        self.mbm = mbm
        self.synergy_head = synergy_head
        
        # 하이퍼파라미터
        cfg = config or {}
        self.alpha = cfg.get("ce_alpha", alpha)
        self.synergy_ce_alpha = cfg.get("synergy_ce_alpha", synergy_ce_alpha)
        self.kd_warmup_stage = cfg.get("teacher_adapt_kd_warmup", 2)
        self.T = cfg.get("tau_start", temperature)
        self.reg_lambda = cfg.get("reg_lambda", reg_lambda)
        self.mbm_reg_lambda = cfg.get("mbm_reg_lambda", mbm_reg_lambda)
        self.num_stages = cfg.get("num_stages", num_stages)
        self.device = device
        self.config = config if config is not None else {}
        
        # 손실 함수
        self.ce_loss_fn = nn.CrossEntropyLoss()
```

#### 2. IB-MBM (Information-Bottleneck Manifold Bridging Module)
```python
class IB_MBM(nn.Module):
    """
    Information-Bottleneck Manifold Bridging Module
    
    기능:
    1. Teacher들의 특징을 정보 병목을 통해 압축
    2. 압축된 정보를 Student에게 효율적으로 전달
    3. 불필요한 정보 제거로 모델 용량 절약
    """
    
    def __init__(
        self,
        query_dim,                   # 쿼리 차원 (Student 특징 차원)
        key_dim,                     # 키 차원 (Teacher 특징 차원)
        out_dim,                     # 출력 차원 (압축된 특징 차원)
        n_head=8,                    # 멀티헤드 어텐션 헤드 수
        dropout=0.0,                 # 드롭아웃 비율
        learnable_q=False,           # 학습 가능한 쿼리 여부
        mbm_reg_lambda=0.0           # 정규화 가중치
    ):
        super().__init__()
        
        # 차원 설정
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.out_dim = out_dim
        self.n_head = n_head
        self.dropout = dropout
        
        # 쿼리 생성 (Student 특징에서)
        if learnable_q:
            self.query = nn.Parameter(torch.randn(1, out_dim, query_dim))
        else:
            self.query = None
        
        # 키/값 변환 (Teacher 특징에서)
        self.key_proj = nn.Linear(key_dim, out_dim)
        self.value_proj = nn.Linear(key_dim, out_dim)
        
        # 멀티헤드 어텐션
        self.attention = nn.MultiheadAttention(
            embed_dim=out_dim,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True
        )
        
        # 출력 변환
        self.output_proj = nn.Linear(out_dim, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)
        
        # 정규화
        self.mbm_reg_lambda = mbm_reg_lambda
    
    def forward(self, query_feat, key_feat):
        """
        Args:
            query_feat: Student 특징 (B, query_dim)
            key_feat: Teacher 특징들 (B, num_teachers, key_dim)
        
        Returns:
            compressed_feat: 압축된 특징 (B, out_dim)
        """
        batch_size = query_feat.size(0)
        
        # 쿼리 생성
        if self.query is not None:
            q = self.query.expand(batch_size, -1, -1)  # (B, out_dim, query_dim)
        else:
            q = query_feat.unsqueeze(1)  # (B, 1, query_dim)
        
        # 키/값 변환
        k = self.key_proj(key_feat)  # (B, num_teachers, out_dim)
        v = self.value_proj(key_feat)  # (B, num_teachers, out_dim)
        
        # 어텐션 계산
        attn_output, attn_weights = self.attention(q, k, v)
        
        # 출력 변환
        output = self.output_proj(attn_output.squeeze(1))
        output = self.layer_norm(output)
        
        return output, attn_weights
```

#### 3. Synergy Head (시너지 헤드)
```python
class SynergyHead(nn.Module):
    """
    시너지 효과를 생성하는 헤드
    
    기능:
    1. MBM에서 나온 압축된 특징을 분류 가능한 형태로 변환
    2. Teacher들의 시너지 효과를 Student에게 전달
    """
    
    def __init__(
        self,
        input_dim,                   # 입력 차원 (MBM 출력 차원)
        num_classes,                 # 클래스 수
        dropout=0.0,                 # 드롭아웃 비율
        hidden_dim=None              # 숨겨진 차원
    ):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim // 2
        
        self.synergy_classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, compressed_feat):
        """
        Args:
            compressed_feat: MBM에서 압축된 특징 (B, input_dim)
        
        Returns:
            synergy_logit: 시너지 로짓 (B, num_classes)
        """
        return self.synergy_classifier(compressed_feat)
```

#### 4. ASIB-CL (Continual Learning 버전)
```python
class ASIB_CL(BaseLearner):
    """
    ASIB-CL: Information Bottleneck 기반 Class-Incremental Learning
    
    핵심 아이디어:
    1. 이전 모델(M_{T-1})을 교사로 사용
    2. IB 기반 지식 증류로 안정성-가소성 최적화
    3. 최소 충분 정보만 전달하여 모델 용량 확보
    """
    
    def __init__(self, args):
        super().__init__(args)
        
        # 네트워크 초기화
        self._network = IncrementalNet(args, False)
        self._class_means = None
        self._old_network = None
        
        # IB 관련 하이퍼파라미터
        self._ib_beta = args.get('ib_beta', 0.1)      # IB 압축 강도
        self.lambda_D = args.get('lambda_D', 1.0)     # 증류 손실 가중치
        self.lambda_IB = args.get('lambda_IB', 1.0)   # IB 모듈 손실 가중치
        
        # IB 모듈들
        self._ib_encoder = None  # VIB 인코더
        self._ib_decoder = None  # VIB 디코더
        
        # 데이터 메모리 (Continual Learning용)
        self._data_memory = np.array([])
        self._targets_memory = np.array([])
        
        # 네트워크 초기화
        self._network.update_fc(10)  # 기본 클래스 수로 초기화
```

### 데이터 플로우 상세 설명

#### 1. Forward Pass (순전파)
```
1. Input Image → Teacher1, Teacher2, Student
2. Teacher 특징들 → MBM (정보 압축)
3. MBM 출력 → Synergy Head (시너지 생성)
4. Synergy + Student → 최종 출력
```

#### 2. Backward Pass (역전파)
```
1. 손실 계산 (CE + KL + IB)
2. Student 업데이트
3. MBM 업데이트 (선택적)
4. Synergy Head 업데이트
```

#### 3. Multi-Stage 학습
```
Stage 1: Teacher Adaptive Update
├── Teacher1, Teacher2 특징 추출
├── MBM을 통한 정보 압축
├── Synergy Head 학습
└── CCCP 손실로 안정성 확보

Stage 2: Student Distillation
├── Student 특징 추출
├── MBM을 통한 시너지 생성
├── KL Divergence 계산
└── Student 업데이트
```

---

## ⚙️ 하이퍼파라미터

### 📊 하이퍼파라미터 분류 및 상세 설명

#### 1. 기본 지식 증류 하이퍼파라미터

| 파라미터 | 기본값 | 범위 | 설명 | 영향 |
|---------|--------|------|------|------|
| `ce_alpha` | 0.3 | [0.0, 1.0] | Student CE vs KL 비율 | 높을수록 원본 태스크에 집중 |
| `kd_alpha` | 0.0 | [0.0, 1.0] | 기본 KD 가중치 | 기본 지식 증류 강도 |
| `kd_ens_alpha` | 0.7 | [0.0, 1.0] | 앙상블 KD 가중치 | Teacher 앙상블 효과 강도 |
| `temperature` | 4.0 | [1.0, 10.0] | 지식 증류 온도 | 높을수록 부드러운 확률 분포 |
| `num_stages` | 2 | [1, 5] | 학습 스테이지 수 | Teacher-Student 반복 횟수 |

**상세 설명:**
- **`ce_alpha`**: Cross-Entropy와 KL Divergence의 균형을 조절
  - `0.0`: 순수 지식 증류 (Teacher만 학습)
  - `1.0`: 순수 분류 학습 (Student만 학습)
  - `0.3`: 30% CE + 70% KD (권장값)

- **`temperature`**: 확률 분포의 부드러움 조절
  - 낮은 값: 확실한 예측 (하드 타겟)
  - 높은 값: 불확실한 예측 (소프트 타겟)
  - `4.0`: 일반적으로 좋은 성능

#### 2. 정보 병목 (IB) 관련 하이퍼파라미터

| 파라미터 | 기본값 | 범위 | 설명 | 영향 |
|---------|--------|------|------|------|
| `use_ib` | false | {true, false} | IB 사용 여부 | IB 모듈 활성화 |
| `ib_beta` | 0.01 | [0.001, 1.0] | IB 압축 강도 | 높을수록 강한 압축 |
| `ib_beta_warmup_epochs` | 0 | [0, 10] | IB 베타 워밍업 에포크 | 점진적 압축 강화 |
| `mbm_out_dim` | 512 | [256, 2048] | MBM 출력 차원 | 압축된 특징 차원 |
| `mbm_n_head` | 1 | [1, 16] | MBM 어텐션 헤드 수 | 멀티헤드 어텐션 |
| `mbm_dropout` | 0.0 | [0.0, 0.5] | MBM 드롭아웃 비율 | 과적합 방지 |
| `mbm_learnable_q` | false | {true, false} | 학습 가능한 쿼리 | 쿼리 최적화 |
| `mbm_reg_lambda` | 0.0 | [0.0, 1.0] | MBM 정규화 가중치 | MBM 정규화 |

**상세 설명:**
- **`ib_beta`**: 정보 병목의 압축 강도
  - `0.001`: 매우 약한 압축 (거의 원본 유지)
  - `0.01`: 약한 압축 (권장 시작값)
  - `0.1`: 중간 압축 (균형점)
  - `1.0`: 강한 압축 (많은 정보 손실)

- **`mbm_out_dim`**: 압축된 특징의 차원
  - Teacher 특징 차원보다 작아야 함
  - 너무 작으면 정보 손실
  - 너무 크면 압축 효과 없음

#### 3. CCCP (Concave-Convex Procedure) 관련 하이퍼파라미터

| 파라미터 | 기본값 | 범위 | 설명 | 영향 |
|---------|--------|------|------|------|
| `use_cccp` | true | {true, false} | CCCP 사용 여부 | Teacher 안정성 |
| `tau` | 4.0 | [1.0, 10.0] | CCCP 온도 파라미터 | Teacher 학습 안정성 |

**상세 설명:**
- **`use_cccp`**: Teacher 모델의 안정성을 위한 CCCP 사용
- **`tau`**: CCCP의 온도 파라미터로 Teacher 학습의 부드러움 조절

#### 4. 학습 관련 하이퍼파라미터

| 파라미터 | 기본값 | 범위 | 설명 | 영향 |
|---------|--------|------|------|------|
| `teacher_lr` | 0.0002 | [1e-5, 1e-2] | Teacher 학습률 | Teacher 업데이트 속도 |
| `student_lr` | 0.001 | [1e-4, 1e-1] | Student 학습률 | Student 업데이트 속도 |
| `student_epochs_per_stage` | 15 | [5, 50] | 스테이지당 Student 에포크 | Student 학습 시간 |
| `batch_size` | 128 | [32, 512] | 배치 크기 | 메모리와 성능 균형 |
| `use_amp` | true | {true, false} | Mixed Precision 사용 | 학습 속도 향상 |

#### 5. 정규화 및 최적화 하이퍼파라미터

| 파라미터 | 기본값 | 범위 | 설명 | 영향 |
|---------|--------|------|------|------|
| `reg_lambda` | 0.0 | [0.0, 1e-2] | 일반 정규화 가중치 | 모델 복잡도 제어 |
| `weight_decay` | 1e-4 | [1e-5, 1e-3] | 가중치 감쇠 | 과적합 방지 |
| `grad_clip_norm` | 0.0 | [0.0, 10.0] | 그래디언트 클리핑 | 학습 안정성 |
| `adam_beta1` | 0.9 | [0.8, 0.99] | Adam β1 | 모멘텀 조절 |
| `adam_beta2` | 0.999 | [0.9, 0.9999] | Adam β2 | 적응적 학습률 |

#### 6. 데이터 증강 관련 하이퍼파라미터

| 파라미터 | 기본값 | 범위 | 설명 | 영향 |
|---------|--------|------|------|------|
| `mixup_alpha` | 0.0 | [0.0, 1.0] | Mixup 강도 | 데이터 증강 |
| `cutmix_alpha_distill` | 0.0 | [0.0, 1.0] | CutMix 강도 | 지식 증류용 증강 |
| `data_aug` | true | {true, false} | 데이터 증강 사용 | 일반화 성능 |

#### 7. 불일치 가중치 (Disagreement Weighting) 하이퍼파라미터

| 파라미터 | 기본값 | 범위 | 설명 | 영향 |
|---------|--------|------|------|------|
| `use_disagree_weight` | false | {true, false} | 불일치 가중치 사용 | Teacher 불일치 활용 |
| `disagree_mode` | none | {pred, both_wrong, any_wrong, none} | 불일치 모드 | 가중치 적용 방식 |
| `disagree_lambda_high` | 1.0 | [0.5, 2.0] | 높은 불일치 가중치 | 불일치 시 강화 |
| `disagree_lambda_low` | 1.0 | [0.5, 2.0] | 낮은 불일치 가중치 | 일치 시 유지 |

### 🎯 하이퍼파라미터 튜닝 가이드

#### 1. 우선순위별 튜닝 순서
```
1순위: ib_beta (0.01 → 0.1 → 0.5)
2순위: ce_alpha (0.3 → 0.5 → 0.7)
3순위: temperature (4.0 → 6.0 → 8.0)
4순위: mbm_out_dim (512 → 1024 → 2048)
5순위: learning_rate (student_lr, teacher_lr)
```

#### 2. 데이터셋별 권장값

**CIFAR-100:**
```yaml
ib_beta: 0.01
ce_alpha: 0.3
temperature: 4.0
mbm_out_dim: 512
student_lr: 0.001
teacher_lr: 0.0002
```

**ImageNet:**
```yaml
ib_beta: 0.05
ce_alpha: 0.4
temperature: 6.0
mbm_out_dim: 1024
student_lr: 0.0005
teacher_lr: 0.0001
```

**Continual Learning:**
```yaml
ib_beta: 0.1
ce_alpha: 0.3
temperature: 4.0
mbm_out_dim: 512
lambda_D: 1.0
lambda_IB: 1.0
```

#### 3. 성능별 최적화 전략

**정확도 최적화:**
```yaml
ib_beta: 0.01  # 약한 압축으로 정보 보존
ce_alpha: 0.3  # 균형잡힌 학습
temperature: 4.0  # 적당한 부드러움
```

**속도 최적화:**
```yaml
use_amp: true  # Mixed Precision
batch_size: 256  # 큰 배치
mbm_out_dim: 256  # 작은 차원
```

**메모리 최적화:**
```yaml
batch_size: 64  # 작은 배치
mbm_out_dim: 256  # 작은 차원
use_amp: true  # Mixed Precision
```

---

## 🔧 구현 세부사항

### 1. 손실 함수

#### 기본 손실 함수
```python
def forward(self, x, y=None):
    # Teacher 특징 추출
    t1_out = self.teacher1(x)
    t2_out = self.teacher2(x)
    
    # Student 특징 추출
    feat_dict, s_logit, _ = self.student(x)
    s_feat = feat_dict["feat_2d"]
    
    # MBM을 통한 시너지 특징 생성
    syn_feat, *_ = self.mbm(s_feat, feats_2d)
    
    # 손실 계산
    ce_loss = self.ce_loss_fn(s_logit, y)
    kl_loss = F.kl_div(
        F.log_softmax(s_logit / self.T, dim=1),
        F.softmax(syn_logit / self.T, dim=1),
        reduction='batchmean'
    ) * (self.T ** 2)
    
    total_loss = self.alpha * ce_loss + (1 - self.alpha) * kl_loss
    return total_loss, s_logit
```

#### IB 손실 함수 (ASIB-CL)
```python
def _compute_separated_losses(self, inputs, targets, features):
    # IB 인코더
    mu, logvar = self._ib_encoder(features).chunk(2, dim=-1)
    z = self._reparameterize(mu, logvar)
    
    # IB 디코더
    reconstructed = self._ib_decoder(z)
    
    # IB 손실
    recon_loss = F.mse_loss(reconstructed, features)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # 분리된 손실
    ib_loss = recon_loss + self._ib_beta * kl_loss
    student_loss = self.lambda_D * ce_loss + self.lambda_IB * ib_loss
    
    return {
        'ib_loss': ib_loss,
        'student_loss': student_loss,
        'total_loss': student_loss
    }
```

### 2. Multi-Stage 학습

#### Stage A: Teacher Adaptive Update
```python
def _teacher_adaptive_update(self, train_loader, optimizer, epochs, stage=1):
    """Teacher 모델들 적응적 업데이트"""
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            # Teacher 특징 추출
            t1_feat = self.teacher1(inputs)
            t2_feat = self.teacher2(inputs)
            
            # MBM 업데이트
            syn_feat = self.mbm(t1_feat, t2_feat)
            
            # CCCP 손실 계산
            if self.config.get("use_cccp", True):
                tau = self.config.get("tau", 4.0)
                loss = self._compute_cccp_loss(syn_feat, targets, tau)
            else:
                loss = self.ce_loss_fn(syn_feat, targets)
            
            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

#### Stage B: Student Distillation
```python
def _student_distill_update(self, train_loader, test_loader, optimizer, scheduler, epochs, stage=1):
    """Student 지식 증류"""
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            # Forward pass
            total_loss, student_logit = self.forward(inputs, targets)
            
            # 역전파
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        # 스케줄러 업데이트
        if scheduler is not None:
            scheduler.step()
```

### 3. 정보 병목 모듈

#### IB 인코더/디코더
```python
def _init_ib_modules(self, feature_dim):
    """Information Bottleneck 모듈 초기화"""
    latent_dim = feature_dim // 4  # 압축된 표현 차원
    
    self._ib_encoder = nn.Sequential(
        nn.Linear(feature_dim, feature_dim // 2),
        nn.ReLU(),
        nn.Linear(feature_dim // 2, latent_dim * 2)  # mu, logvar
    )
    
    self._ib_decoder = nn.Sequential(
        nn.Linear(latent_dim, feature_dim // 2),
        nn.ReLU(),
        nn.Linear(feature_dim // 2, feature_dim)
    )
```

---

## 🧪 실험 설정

### 📋 실험 환경 구성

#### 1. 하드웨어 요구사항
```yaml
# 최소 요구사항
GPU: NVIDIA RTX 3080 (12GB VRAM)
CPU: Intel i7-10700K 또는 AMD Ryzen 7 3700X
RAM: 32GB DDR4
Storage: 500GB SSD

# 권장 사양
GPU: NVIDIA RTX 4090 (24GB VRAM) 또는 A100 (40GB VRAM)
CPU: Intel i9-12900K 또는 AMD Ryzen 9 5950X
RAM: 64GB DDR4
Storage: 1TB NVMe SSD
```

#### 2. 소프트웨어 환경
```bash
# Python 환경
Python: 3.8-3.11
PyTorch: 2.0.0+
CUDA: 11.8+
cuDNN: 8.6+

# 필수 패키지
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
scipy>=1.7.0
tqdm>=4.62.0
matplotlib>=3.5.0
tensorboard>=2.8.0
hydra-core>=1.3.0
omegaconf>=2.2.0
```

### 🎯 실험 시나리오별 설정

#### 1. ASIB-KD (Knowledge Distillation) 실험

##### 1.1 CIFAR-100 실험 설정
```yaml
# configs/experiment/asib_cifar100.yaml
defaults:
  - base
  - method: asib
  - dataset: cifar100
  - _self_

# 모델 설정
model:
  teacher1:
    name: convnext_l
    pretrained: true
    num_classes: 100
  teacher2:
    name: resnet152
    pretrained: true
    num_classes: 100
  student:
    name: resnet50_scratch
    pretrained: false
    num_classes: 100

# ASIB 하이퍼파라미터
method:
  name: asib
  ce_alpha: 0.3
  kd_ens_alpha: 0.7
  use_ib: true
  ib_beta: 0.001
  ib_beta_warmup_epochs: 3
  use_cccp: true
  tau: 4.0

# MBM 설정
mbm_query_dim: 1024
mbm_out_dim: 1024
mbm_n_head: 8
mbm_dropout: 0.0
mbm_learnable_q: false
mbm_reg_lambda: 0.0

# 학습 설정
num_stages: 4
teacher_lr: 0.0002
student_lr: 0.001
teacher_weight_decay: 0.0001
student_weight_decay: 0.0003
student_epochs_per_stage: 15

# 데이터 설정
batch_size: 128
num_workers: 8
data_aug: true
mixup_alpha: 0.0
cutmix_alpha_distill: 0.3

# 최적화 설정
use_amp: true
amp_dtype: float16
grad_clip_norm: 1.0
adam_beta1: 0.9
adam_beta2: 0.999

# 정규화 설정
reg_lambda: 0.0
weight_decay: 0.0001

# 디바이스 설정
device: cuda
seed: 42
```

##### 1.2 ImageNet 실험 설정
```yaml
# configs/experiment/asib_imagenet.yaml
defaults:
  - base
  - method: asib
  - dataset: imagenet
  - _self_

# 모델 설정
model:
  teacher1:
    name: convnext_l
    pretrained: true
    num_classes: 1000
  teacher2:
    name: efficientnet_l2
    pretrained: true
    num_classes: 1000
  student:
    name: resnet50_scratch
    pretrained: false
    num_classes: 1000

# ASIB 하이퍼파라미터 (ImageNet에 최적화)
method:
  name: asib
  ce_alpha: 0.4
  kd_ens_alpha: 0.6
  use_ib: true
  ib_beta: 0.005
  ib_beta_warmup_epochs: 5
  use_cccp: true
  tau: 6.0

# MBM 설정 (더 큰 모델)
mbm_query_dim: 2048
mbm_out_dim: 2048
mbm_n_head: 16
mbm_dropout: 0.1
mbm_learnable_q: true
mbm_reg_lambda: 0.001

# 학습 설정 (ImageNet에 맞춤)
num_stages: 3
teacher_lr: 0.0001
student_lr: 0.0005
teacher_weight_decay: 0.0001
student_weight_decay: 0.0001
student_epochs_per_stage: 30

# 데이터 설정
batch_size: 256
num_workers: 16
data_aug: true
mixup_alpha: 0.2
cutmix_alpha_distill: 0.5

# 최적화 설정
use_amp: true
amp_dtype: bfloat16  # ImageNet에서는 bfloat16이 더 안정적
grad_clip_norm: 1.0
adam_beta1: 0.9
adam_beta2: 0.999

# 정규화 설정
reg_lambda: 0.001
weight_decay: 0.0001

# 디바이스 설정
device: cuda
seed: 42
```

#### 2. ASIB-CL (Continual Learning) 실험

##### 2.1 CIFAR-100 Class-IL 설정
```json
{
    "prefix": "asib_cl_cifar100",
    "dataset": "cifar100",
    "init_cls": 10,
    "increment": 10,
    "memory_size": 2000,
    "memory_per_class": 20,
    "fixed_memory": false,
    "shuffle": true,
    "convnet_type": "resnet32",
    "model_name": "asib_cl",
    "device": ["0"],
    "seed": [1993],
    
    "ib_beta": 0.1,
    "lambda_D": 1.0,
    "lambda_IB": 1.0,
    
    "batch_size": 64,
    "epochs": 200,
    "learning_rate": 0.1,
    "weight_decay": 5e-4,
    "milestones": [60, 120, 160],
    "gamma": 0.2,
    
    "num_workers": 8,
    "topk": 5,
    
    "logdir": "./experiments/sota/logs/asib_cl",
    "save_path": "./checkpoints/students/asib_cl"
}
```

##### 2.2 ImageNet-100 Class-IL 설정
```json
{
    "prefix": "asib_cl_imagenet100",
    "dataset": "imagenet100",
    "init_cls": 10,
    "increment": 10,
    "memory_size": 5000,
    "memory_per_class": 50,
    "fixed_memory": false,
    "shuffle": true,
    "convnet_type": "resnet18",
    "model_name": "asib_cl",
    "device": ["0"],
    "seed": [1993],
    
    "ib_beta": 0.05,
    "lambda_D": 1.0,
    "lambda_IB": 1.0,
    
    "batch_size": 128,
    "epochs": 100,
    "learning_rate": 0.1,
    "weight_decay": 1e-4,
    "milestones": [30, 60, 80],
    "gamma": 0.1,
    
    "num_workers": 16,
    "topk": 5,
    
    "logdir": "./experiments/sota/logs/asib_cl_imagenet100",
    "save_path": "./checkpoints/students/asib_cl_imagenet100"
}
```

### 🔬 하이퍼파라미터 튜닝 실험

#### 1. IB 베타 (β) 튜닝 실험
```python
# IB 베타 튜닝 스크립트
def ib_beta_tuning_experiment():
    """IB 베타 값에 따른 성능 비교 실험"""
    
    beta_values = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    results = {}
    
    for beta in beta_values:
        print(f"Testing ib_beta = {beta}")
        
        # 설정 파일 업데이트
        config = load_config('configs/experiment/asib_cifar100.yaml')
        config['method']['ib_beta'] = beta
        
        # 실험 실행
        result = run_experiment(config)
        
        results[beta] = {
            'student_acc': result['student_accuracy'],
            'teacher_agreement': result['teacher_agreement'],
            'knowledge_transfer': result['knowledge_transfer_efficiency']
        }
    
    # 결과 분석 및 시각화
    plot_ib_beta_results(results)
    return results

# 결과 시각화
def plot_ib_beta_results(results):
    """IB 베타 튜닝 결과 시각화"""
    import matplotlib.pyplot as plt
    
    betas = list(results.keys())
    student_accs = [results[beta]['student_acc'] for beta in betas]
    teacher_agreements = [results[beta]['teacher_agreement'] for beta in betas]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.semilogx(betas, student_accs, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('IB Beta (β)')
    plt.ylabel('Student Accuracy (%)')
    plt.title('Student Accuracy vs IB Beta')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.semilogx(betas, teacher_agreements, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('IB Beta (β)')
    plt.ylabel('Teacher Agreement (%)')
    plt.title('Teacher Agreement vs IB Beta')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('ib_beta_tuning_results.png', dpi=300, bbox_inches='tight')
    plt.show()
```

#### 2. MBM 차원 튜닝 실험
```python
# MBM 출력 차원 튜닝
def mbm_dimension_tuning_experiment():
    """MBM 출력 차원에 따른 성능 비교 실험"""
    
    mbm_dims = [256, 512, 1024, 2048, 4096]
    results = {}
    
    for dim in mbm_dims:
        print(f"Testing mbm_out_dim = {dim}")
        
        # 설정 파일 업데이트
        config = load_config('configs/experiment/asib_cifar100.yaml')
        config['mbm_out_dim'] = dim
        config['mbm_query_dim'] = dim
        
        # 실험 실행
        result = run_experiment(config)
        
        results[dim] = {
            'student_acc': result['student_accuracy'],
            'memory_usage': result['memory_usage'],
            'training_time': result['training_time']
        }
    
    # 결과 분석
    plot_mbm_dimension_results(results)
    return results

def plot_mbm_dimension_results(results):
    """MBM 차원 튜닝 결과 시각화"""
    import matplotlib.pyplot as plt
    
    dims = list(results.keys())
    student_accs = [results[dim]['student_acc'] for dim in dims]
    memory_usage = [results[dim]['memory_usage'] for dim in dims]
    training_time = [results[dim]['training_time'] for dim in dims]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.plot(dims, student_accs, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('MBM Output Dimension')
    ax1.set_ylabel('Student Accuracy (%)')
    ax1.set_title('Accuracy vs MBM Dimension')
    ax1.grid(True)
    
    ax2.plot(dims, memory_usage, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('MBM Output Dimension')
    ax2.set_ylabel('Memory Usage (GB)')
    ax2.set_title('Memory Usage vs MBM Dimension')
    ax2.grid(True)
    
    ax3.plot(dims, training_time, 'go-', linewidth=2, markersize=8)
    ax3.set_xlabel('MBM Output Dimension')
    ax3.set_ylabel('Training Time (hours)')
    ax3.set_title('Training Time vs MBM Dimension')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('mbm_dimension_tuning_results.png', dpi=300, bbox_inches='tight')
    plt.show()
```

#### 3. 학습률 튜닝 실험
```python
# 학습률 튜닝
def learning_rate_tuning_experiment():
    """학습률 튜닝 실험"""
    
    lr_combinations = [
        {'teacher_lr': 0.0001, 'student_lr': 0.0005},
        {'teacher_lr': 0.0002, 'student_lr': 0.001},
        {'teacher_lr': 0.0005, 'student_lr': 0.002},
        {'teacher_lr': 0.001, 'student_lr': 0.005},
        {'teacher_lr': 0.002, 'student_lr': 0.01}
    ]
    
    results = {}
    
    for i, lr_combo in enumerate(lr_combinations):
        print(f"Testing combination {i+1}: {lr_combo}")
        
        # 설정 파일 업데이트
        config = load_config('configs/experiment/asib_cifar100.yaml')
        config['teacher_lr'] = lr_combo['teacher_lr']
        config['student_lr'] = lr_combo['student_lr']
        
        # 실험 실행
        result = run_experiment(config)
        
        results[f"combo_{i+1}"] = {
            'teacher_lr': lr_combo['teacher_lr'],
            'student_lr': lr_combo['student_lr'],
            'student_acc': result['student_accuracy'],
            'convergence_epoch': result['convergence_epoch']
        }
    
    return results
```

### 📊 실험 결과 분석

#### 1. 성능 지표 정의
```python
# 성능 지표 계산 함수들
def calculate_student_accuracy(model, test_loader):
    """학생 모델 정확도 계산"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(model.device), targets.to(model.device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return 100. * correct / total

def calculate_teacher_agreement(teacher1, teacher2, test_loader):
    """교사 모델들 간의 일치도 계산"""
    teacher1.eval()
    teacher2.eval()
    agreement = 0
    total = 0
    
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(teacher1.device)
            outputs1 = teacher1(inputs)
            outputs2 = teacher2(inputs)
            
            _, pred1 = outputs1.max(1)
            _, pred2 = outputs2.max(1)
            
            agreement += (pred1 == pred2).sum().item()
            total += inputs.size(0)
    
    return 100. * agreement / total

def calculate_knowledge_transfer_efficiency(student, teacher1, teacher2, test_loader):
    """지식 전달 효율성 계산"""
    # KL Divergence 기반 효율성 계산
    student.eval()
    teacher1.eval()
    teacher2.eval()
    
    total_kl = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(student.device)
            
            # 확률 분포 계산
            student_probs = F.softmax(student(inputs), dim=1)
            teacher1_probs = F.softmax(teacher1(inputs), dim=1)
            teacher2_probs = F.softmax(teacher2(inputs), dim=1)
            
            # 앙상블 교사 확률
            ensemble_probs = (teacher1_probs + teacher2_probs) / 2
            
            # KL Divergence 계산
            kl_div = F.kl_div(
                student_probs.log(), ensemble_probs, reduction='batchmean'
            )
            
            total_kl += kl_div.item()
            total_samples += 1
    
    avg_kl = total_kl / total_samples
    # 효율성은 KL Divergence의 역수 (낮을수록 효율적)
    efficiency = 1.0 / (1.0 + avg_kl)
    
    return efficiency
```

#### 2. 실험 결과 시각화
```python
# 실험 결과 시각화
def visualize_experiment_results(results, save_path='experiment_results.png'):
    """실험 결과 종합 시각화"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 스타일 설정
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 정확도 비교
    methods = list(results.keys())
    accuracies = [results[method]['student_acc'] for method in methods]
    
    axes[0, 0].bar(methods, accuracies, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Student Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. 학습 곡선
    for method in methods:
        train_loss = results[method]['train_loss']
        val_loss = results[method]['val_loss']
        epochs = range(1, len(train_loss) + 1)
        
        axes[0, 1].plot(epochs, train_loss, label=f'{method} (Train)', alpha=0.7)
        axes[0, 1].plot(epochs, val_loss, label=f'{method} (Val)', linestyle='--', alpha=0.7)
    
    axes[0, 1].set_title('Training Curves')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 3. 지식 전달 효율성
    efficiencies = [results[method]['knowledge_transfer'] for method in methods]
    
    axes[0, 2].bar(methods, efficiencies, color='lightgreen', alpha=0.7)
    axes[0, 2].set_title('Knowledge Transfer Efficiency')
    axes[0, 2].set_ylabel('Efficiency')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # 4. 메모리 사용량
    memory_usage = [results[method]['memory_usage'] for method in methods]
    
    axes[1, 0].bar(methods, memory_usage, color='orange', alpha=0.7)
    axes[1, 0].set_title('Memory Usage')
    axes[1, 0].set_ylabel('Memory (GB)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 5. 학습 시간
    training_time = [results[method]['training_time'] for method in methods]
    
    axes[1, 1].bar(methods, training_time, color='red', alpha=0.7)
    axes[1, 1].set_title('Training Time')
    axes[1, 1].set_ylabel('Time (hours)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 6. 하이퍼파라미터 민감도
    param_sensitivity = results.get('param_sensitivity', {})
    if param_sensitivity:
        params = list(param_sensitivity.keys())
        sensitivities = list(param_sensitivity.values())
        
        axes[1, 2].bar(params, sensitivities, color='purple', alpha=0.7)
        axes[1, 2].set_title('Parameter Sensitivity')
        axes[1, 2].set_ylabel('Sensitivity Score')
        axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
```

### 🚀 실험 실행 스크립트

#### 1. 단일 실험 실행
```bash
#!/bin/bash
# run_single_experiment.sh

# 환경 설정
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 실험 설정
EXPERIMENT_NAME="asib_cifar100_baseline"
CONFIG_PATH="configs/experiment/asib_cifar100.yaml"
LOG_DIR="experiments/logs/${EXPERIMENT_NAME}"
SAVE_DIR="experiments/results/${EXPERIMENT_NAME}"

# 디렉토리 생성
mkdir -p ${LOG_DIR}
mkdir -p ${SAVE_DIR}

# 실험 실행
python main.py \
    --config-name=${CONFIG_PATH} \
    hydra.run.dir=${SAVE_DIR} \
    hydra.sweep.dir=${SAVE_DIR} \
    hydra.sweep.subdir=${EXPERIMENT_NAME} \
    2>&1 | tee ${LOG_DIR}/experiment.log

echo "Experiment completed: ${EXPERIMENT_NAME}"
```

#### 2. 하이퍼파라미터 스윕 실험
```bash
#!/bin/bash
# run_hyperparameter_sweep.sh

# 환경 설정
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 스윕 설정
SWEEP_NAME="asib_ib_beta_sweep"
BASE_CONFIG="configs/experiment/asib_cifar100.yaml"
LOG_DIR="experiments/logs/${SWEEP_NAME}"
SAVE_DIR="experiments/results/${SWEEP_NAME}"

# 디렉토리 생성
mkdir -p ${LOG_DIR}
mkdir -p ${SAVE_DIR}

# Hydra 멀티런으로 스윕 실행
python main.py \
    --multirun \
    --config-name=${BASE_CONFIG} \
    method.ib_beta=0.001,0.01,0.05,0.1,0.2,0.5 \
    hydra.sweep.dir=${SAVE_DIR} \
    hydra.sweep.subdir=${SWEEP_NAME} \
    2>&1 | tee ${LOG_DIR}/sweep.log

echo "Hyperparameter sweep completed: ${SWEEP_NAME}"
```

#### 3. 비교 실험 실행
```bash
#!/bin/bash
# run_comparison_experiments.sh

# 환경 설정
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 비교할 방법들
METHODS=("asib" "vanilla_kd" "fitnet" "attention" "hint" "crd")

# 각 방법별 실험 실행
for method in "${METHODS[@]}"; do
    echo "Running experiment for method: ${method}"
    
    EXPERIMENT_NAME="comparison_${method}"
    CONFIG_PATH="configs/experiment/${method}_cifar100.yaml"
    LOG_DIR="experiments/logs/${EXPERIMENT_NAME}"
    SAVE_DIR="experiments/results/${EXPERIMENT_NAME}"
    
    # 디렉토리 생성
    mkdir -p ${LOG_DIR}
    mkdir -p ${SAVE_DIR}
    
    # 실험 실행
    python main.py \
        --config-name=${CONFIG_PATH} \
        hydra.run.dir=${SAVE_DIR} \
        hydra.sweep.dir=${SAVE_DIR} \
        hydra.sweep.subdir=${EXPERIMENT_NAME} \
        2>&1 | tee ${LOG_DIR}/experiment.log
    
    echo "Completed experiment for method: ${method}"
done

echo "All comparison experiments completed"
```

---

## 🚀 사용법

### 1. 기본 ASIB-KD 실행

#### 설정 파일 생성
```yaml
# configs/experiment/asib_experiment.yaml
defaults:
  - base
  - _self_

# 모델 설정
model:
  teacher1:
    name: convnext_l
    pretrained: true
  teacher2:
    name: resnet152
    pretrained: true
  student:
    name: resnet50_scratch
    pretrained: false

# ASIB 설정
method:
  name: asib
  ce_alpha: 0.3
  kd_ens_alpha: 0.7
  use_ib: true
  ib_beta: 0.001
  use_cccp: true
  tau: 4.0

# 학습 설정
num_stages: 4
teacher_lr: 0.0002
student_lr: 0.001
student_epochs_per_stage: 15
```

#### 실행 명령
```bash
# Hydra를 사용한 실행
python main.py --config-name=asib_experiment

# 직접 실행
python main.py \
  --method=asib \
  --teacher1=convnext_l \
  --teacher2=resnet152 \
  --student=resnet50_scratch \
  --ib_beta=0.001 \
  --num_stages=4
```

### 2. ASIB-CL 실행

#### 설정 파일
```json
{
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
    "ib_beta": 0.1,
    "topk": 5,
    "seed": 1993,
    "logdir": "./experiments/sota/logs/asib_cl",
    "model_name": "asib_cl"
}
```

#### 실행 명령
```bash
# ASIB-CL 실험 실행
python PyCIL/main.py --config=PyCIL/exps/asib_cl.json

# 비교 실험 실행
python run_asib_cl_experiment.py
```

### 3. 하이퍼파라미터 튜닝

#### 자동 튜닝 스크립트
```python
def hyperparameter_tuning():
    """하이퍼파라미터 튜닝"""
    
    # 튜닝할 파라미터 조합
    param_combinations = [
        {'ib_beta': 0.01, 'lambda_D': 1.0, 'lambda_IB': 1.0},
        {'ib_beta': 0.1, 'lambda_D': 1.0, 'lambda_IB': 1.0},
        {'ib_beta': 0.5, 'lambda_D': 1.0, 'lambda_IB': 1.0},
        {'ib_beta': 1.0, 'lambda_D': 1.0, 'lambda_IB': 1.0},
        {'ib_beta': 0.1, 'lambda_D': 0.5, 'lambda_IB': 1.0},
        {'ib_beta': 0.1, 'lambda_D': 2.0, 'lambda_IB': 1.0},
        {'ib_beta': 0.1, 'lambda_D': 1.0, 'lambda_IB': 0.5},
        {'ib_beta': 0.1, 'lambda_D': 1.0, 'lambda_IB': 2.0},
    ]
    
    results = {}
    
    for i, params in enumerate(param_combinations):
        print(f"Testing combination {i+1}/{len(param_combinations)}: {params}")
        
        # 설정 파일 업데이트
        config = load_config('PyCIL/exps/asib_cl.json')
        config.update(params)
        save_config(config, f'PyCIL/exps/asib_cl_tune_{i}.json')
        
        # 실험 실행
        result = run_experiment(f'PyCIL/exps/asib_cl_tune_{i}.json')
        results[f'combo_{i}'] = {
            'params': params,
            'aia': result['aia'],
            'af': result['af']
        }
    
    # 결과 분석
    analyze_tuning_results(results)
```

---

## ⚡ 성능 최적화

### 1. 학습 속도 최적화

#### Mixed Precision Training
```python
from torch.cuda.amp import autocast, GradScaler

def _train_with_amp(self, train_loader, optimizer):
    """Mixed Precision Training"""
    scaler = GradScaler()
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        
        # Mixed precision forward pass
        with autocast():
            features = self._network.extract_vector(inputs)
            losses = self._compute_separated_losses(inputs, targets, features)
        
        # Scaled backward pass
        scaler.scale(losses['total_loss']).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

#### 데이터 로딩 최적화
```python
# DataLoader 최적화
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,  # 멀티프로세싱
    pin_memory=True,  # GPU 메모리 핀
    persistent_workers=True  # 워커 재사용
)
```

### 2. 메모리 최적화

#### 그래디언트 체크포인팅
```python
# 메모리 효율적인 학습
from torch.utils.checkpoint import checkpoint

def forward_with_checkpoint(self, x):
    return checkpoint(self._forward_pass, x)
```

#### 모델 압축
```python
# 모델 양자화
from torch.quantization import quantize_dynamic

def quantize_model(self, model):
    return quantize_dynamic(
        model, 
        {nn.Linear, nn.Conv2d}, 
        dtype=torch.qint8
    )
```

### 3. 하이퍼파라미터 최적화

#### 베이지안 최적화
```python
from skopt import gp_minimize
from skopt.space import Real

def bayesian_optimization():
    """베이지안 최적화를 통한 하이퍼파라미터 튜닝"""
    
    # 탐색 공간 정의
    space = [
        Real(0.01, 1.0, name='ib_beta'),
        Real(0.1, 10.0, name='lambda_D'),
        Real(0.1, 10.0, name='lambda_IB')
    ]
    
    def objective(params):
        ib_beta, lambda_D, lambda_IB = params
        
        # 설정 업데이트
        config = load_config('PyCIL/exps/asib_cl.json')
        config.update({
            'ib_beta': ib_beta,
            'lambda_D': lambda_D,
            'lambda_IB': lambda_IB
        })
        
        # 실험 실행
        result = run_experiment(config)
        
        # 목표: AIA 최대화, AF 최소화
        objective_value = result['af'] - result['aia']  # 최소화 문제
        return objective_value
    
    # 최적화 실행
    result = gp_minimize(
        objective,
        space,
        n_calls=50,
        random_state=42
    )
    
    print(f"Best parameters: {result.x}")
    print(f"Best objective value: {result.fun}")
```
