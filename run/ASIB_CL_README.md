# ASIB-CL: Information Bottleneck 기반 Continual Learning

## 📖 개요

ASIB-CL은 ASIB(Adaptive Sampling Information Bottleneck)를 Continual Learning 환경에 적용한 방법입니다. Information Bottleneck을 활용하여 안정성-가소성 딜레마(Stability-Plasticity Dilemma)를 해결하는 Class-Incremental Learning 방법입니다.

## 🎯 핵심 아이디어

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

## 🏗️ 아키텍처

```
이전 모델 (M_{T-1}) → IB 인코더 → 압축된 표현 (Z) → IB 디코더 → 복원된 특징
                                                      ↓
현재 모델 (M_T) ← Knowledge Transfer Loss ← MSE Loss
```

### 핵심 구성 요소

1. **IB 인코더**: 교사 특징을 압축된 표현으로 변환
2. **IB 디코더**: 압축된 표현을 원본 특징 차원으로 복원
3. **Knowledge Transfer Loss**: 학생 특징이 복원된 특징을 모방하도록 유도
4. **Information Compression Loss**: KL divergence로 정보 압축 유도

## 🚀 설치 및 설정

### 1. PyCIL 프레임워크 설치
```bash
# PyCIL 클론 (이미 완료됨)
git clone https://github.com/LAMDA-CL/PyCIL.git

# 의존성 설치
pip install torch torchvision tqdm numpy scipy quadprog POT
```

### 2. ASIB-CL 모듈 확인
```bash
# ASIB-CL 모델 파일 확인
ls PyCIL/models/asib_cl.py

# 설정 파일 확인
ls PyCIL/exps/asib_cl.json

# Factory 등록 확인
grep "asib_cl" PyCIL/utils/factory.py
```

## 📊 실험 설정

### 기본 실험 설정 (CIFAR-100)
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

### 주요 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `ib_beta` | 0.1 | IB 압축 강도 (높을수록 더 강한 압축) |
| `memory_size` | 2000 | 총 exemplar 개수 |
| `memory_per_class` | 20 | 클래스당 exemplar 개수 |
| `init_cls` | 10 | 첫 번째 태스크의 클래스 수 |
| `increment` | 10 | 각 태스크당 추가되는 클래스 수 |

## 🧪 실험 실행

### 1. 단일 실험 실행
```bash
# ASIB-CL 실험 실행
python PyCIL/main.py --config=PyCIL/exps/asib_cl.json
```

### 2. 비교 실험 실행
```bash
# 전체 비교 실험 실행
python run_asib_cl_experiment.py
```

### 3. 개별 방법 비교
```bash
# Fine-tuning
python PyCIL/main.py --config=PyCIL/exps/finetune.json

# EWC
python PyCIL/main.py --config=PyCIL/exps/ewc.json

# LwF
python PyCIL/main.py --config=PyCIL/exps/lwf.json

# iCaRL
python PyCIL/main.py --config=PyCIL/exps/icarl.json

# DER
python PyCIL/main.py --config=PyCIL/exps/der.json
```

## 📈 결과 분석

### 성능 지표
- **Average Incremental Accuracy (AIA)**: 모든 태스크 완료 후 평균 정확도
- **Average Forgetting (AF)**: 이전 태스크에 대한 성능 망각 정도
- **Forward Transfer**: 새로운 태스크 학습 시 이전 지식의 활용도

### 결과 확인
```bash
# 로그 파일 확인
tail -f experiments/sota/logs/asib_cl/*.log

# 결과 리포트 생성
python run_asib_cl_experiment.py
```

## 🔬 하이퍼파라미터 튜닝

### IB 압축 강도 (β) 실험
```python
# 다양한 β 값으로 실험
beta_values = [0.01, 0.05, 0.1, 0.2, 0.5]

for beta in beta_values:
    # 설정 파일 수정
    config["ib_beta"] = beta
    # 실험 실행
```

### 메모리 크기 실험
```python
# 다양한 메모리 크기로 실험
memory_sizes = [1000, 2000, 3000, 5000]

for memory_size in memory_sizes:
    # 설정 파일 수정
    config["memory_size"] = memory_size
    # 실험 실행
```

## 📝 코드 구조

```
PyCIL/
├── models/
│   └── asib_cl.py          # ASIB-CL 모델 구현
├── exps/
│   └── asib_cl.json        # ASIB-CL 실험 설정
└── utils/
    └── factory.py          # 모델 팩토리 (ASIB-CL 등록됨)

run_asib_cl_experiment.py   # 실험 실행 스크립트
ASIB_CL_README.md          # 이 파일
```

## 🎯 핵심 구현 세부사항

### 1. IB 모듈 초기화
```python
def _init_ib_modules(self, feature_dim):
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

### 2. IB 기반 지식 증류 손실
```python
def _ib_distillation_loss(self, student_features, teacher_features):
    # 교사 특징을 IB 인코더로 압축
    ib_output = self._ib_encoder(teacher_features)
    mu, logvar = ib_output.chunk(2, dim=1)
    
    # Reparameterization
    z = self._reparameterize(mu, logvar)
    
    # 압축된 표현을 디코더로 복원
    reconstructed_features = self._ib_decoder(z)
    
    # Knowledge Transfer Loss (안정성)
    knowledge_transfer_loss = F.mse_loss(student_features, reconstructed_features)
    
    # Information Compression Loss (가소성)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # 전체 IB 손실
    ib_loss = knowledge_transfer_loss + self._ib_beta * kl_loss
    
    return ib_loss
```

## 🔍 문제 해결

### 일반적인 문제들

1. **CUDA 메모리 부족**
   ```bash
   # 배치 크기 줄이기
   "batch_size": 64  # 128에서 64로 변경
   ```

2. **학습이 수렴하지 않음**
   ```bash
   # 학습률 조정
   "lr": 0.05  # 0.1에서 0.05로 변경
   ```

3. **IB 압축이 너무 강함**
   ```bash
   # β 값 줄이기
   "ib_beta": 0.05  # 0.1에서 0.05로 변경
   ```

## 📚 참고 문헌

1. **PyCIL**: Zhou, D. W., et al. "PyCIL: A Python Toolbox for Class-Incremental Learning." Science China Information Sciences, 2023.
2. **Information Bottleneck**: Tishby, N., et al. "The information bottleneck method." Allerton Conference, 1999.
3. **Variational Information Bottleneck**: Alemi, A. A., et al. "Deep variational information bottleneck." ICLR, 2017.

## 🤝 기여

ASIB-CL 구현에 대한 질문이나 개선 제안이 있으시면 이슈를 생성해 주세요.

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 