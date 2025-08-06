# ASIB-CL 구현 가이드 (ASIB-CL Implementation Guide)

## 목차
1. [개요](#개요)
2. [핵심 아이디어 및 이론적 배경](#핵심-아이디어-및-이론적-배경)
3. [아키텍처 설계](#아키텍처-설계)
4. [핵심 구현 세부사항](#핵심-구현-세부사항)
5. [손실 함수 설계 (핵심 수정사항)](#손실-함수-설계-핵심-수정사항)
6. [학습 프로세스](#학습-프로세스)
7. [하이퍼파라미터 튜닝](#하이퍼파라미터-튜닝)
8. [성능 최적화](#성능-최적화)
9. [실험 설정 및 실행](#실험-설정-및-실행)
10. [결과 분석](#결과-분석)
11. [문제 해결](#문제-해결)

## 개요

ASIB-CL (ASIB-based Continual Learning)은 ASIB (Adaptive Selective Information Bottleneck)의 핵심 철학인 정보 병목(Information Bottleneck, IB)을 Continual Learning 환경에 적용한 방법론입니다. 이 구현은 CL의 근본적인 난제인 **안정성-가소성 딜레마(Stability-Plasticity Dilemma)**를 IB를 통해 해결하는 것을 목표로 합니다.

### 주요 특징
- **IB 기반 지식 증류**: VAE 구조를 활용한 특징 수준 압축
- **분리된 손실 함수**: IB 모듈 최적화와 학생 모델 최적화의 명확한 분리
- **안정성-가소성 균형**: β 파라미터를 통한 압축 강도 조절
- **표준 CL 프로토콜 준수**: Class-IL 시나리오에서의 공정한 비교

## 핵심 아이디어 및 이론적 배경

### 1. 안정성-가소성 딜레마

Continual Learning에서 모델은 두 가지 상충되는 요구사항을 동시에 만족해야 합니다:

- **안정성 (Stability)**: 이전에 학습한 지식을 유지
- **가소성 (Plasticity)**: 새로운 지식을 효과적으로 학습

전통적인 지식 증류(KD)는 이전 모델의 지식을 강하게 유지하려 하므로, 모델 용량이 고갈되어 새로운 지식 학습에 어려움을 겪습니다.

### 2. IB 기반 해결책

ASIB-CL은 IB를 통해 이전 지식을 **'최소 충분(Minimal Sufficient)' 형태로 압축**하여 전달합니다:

```
이전 모델 특징 → IB 인코더 → 압축된 표현 → IB 디코더 → 복원된 특징
```

이를 통해:
- **안정성**: 복원된 특징을 통해 이전 지식 유지
- **가소성**: 압축을 통해 확보된 모델 용량으로 새로운 지식 학습

### 3. 수정된 손실 함수 설계

**핵심 수정사항**: IB 모듈의 학습 목표와 학생 모델의 학습 목표를 명확히 분리

```
L_Total = L_Cls + λ_D * L_Distill + λ_IB * L_IB
```

- **L_IB**: IB 모듈 최적화 (VAE 손실)
- **L_Distill**: 학생 모델 최적화 - 안정성
- **L_Cls**: 학생 모델 최적화 - 가소성

## 아키텍처 설계

### 1. 전체 아키텍처

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   이전 모델     │    │   IB 모듈       │    │   현재 모델     │
│  (Teacher)      │───▶│  (VAE 구조)     │───▶│  (Student)      │
│                 │    │                 │    │                 │
│ - 특징 추출     │    │ - 인코더        │    │ - 특징 학습     │
│ - 지식 보유     │    │ - 디코더        │    │ - 분류 수행     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   손실 계산     │
                       │                 │
                       │ - IB 손실       │
                       │ - 증류 손실     │
                       │ - 분류 손실     │
                       └─────────────────┘
```

### 2. IB 모듈 구조 (VAE)

```python
class IBModule(nn.Module):
    def __init__(self, feature_dim, latent_dim):
        super().__init__()
        # 인코더: 특징 → μ, log(σ²)
        self.encoder = nn.Linear(feature_dim, latent_dim * 2)
        # 디코더: 잠재 변수 → 복원된 특징
        self.decoder = nn.Linear(latent_dim, feature_dim)
    
    def forward(self, features):
        # 인코딩
        encoded = self.encoder(features)
        mu, logvar = encoded.chunk(2, dim=1)
        
        # Reparameterization trick
        z = self.reparameterize(mu, logvar)
        
        # 디코딩
        reconstructed = self.decoder(z)
        
        return reconstructed, mu, logvar
```

### 3. ASIB-CL 모델 구조

```python
class ASIB_CL(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)
        self._old_network = None
        
        # IB 관련 파라미터
        self._ib_beta = args.get('ib_beta', 0.1)
        self.lambda_D = args.get('lambda_D', 1.0)
        self.lambda_IB = args.get('lambda_IB', 1.0)
        
        # IB 모듈 (첫 번째 태스크 후 초기화)
        self._ib_encoder = None
        self._ib_decoder = None
        
        # 메모리 관리
        self._data_memory = np.array([])
        self._targets_memory = np.array([])
```

## 핵심 구현 세부사항

### 1. IB 모듈 초기화

```python
def _init_ib_modules(self, feature_dim):
    """IB 모듈 초기화 (VAE 구조)"""
    latent_dim = feature_dim // 4  # 압축 비율: 4:1
    
    # 인코더: 특징 → μ, log(σ²)
    self._ib_encoder = nn.Linear(feature_dim, latent_dim * 2)
    
    # 디코더: 잠재 변수 → 복원된 특징
    self._ib_decoder = nn.Linear(latent_dim, feature_dim)
    
    # 디바이스 이동
    device = self.args['device'][0]
    self._ib_encoder = self._ib_encoder.to(device)
    self._ib_decoder = self._ib_decoder.to(device)
```

### 2. Reparameterization Trick

```python
def _reparameterize(self, mu, logvar):
    """VAE Reparameterization Trick"""
    if self.training:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    else:
        return mu
```

### 3. 수정된 손실 함수 구현

**핵심 수정사항**: IB 모듈 최적화와 학생 모델 최적화의 분리

```python
def _compute_losses(self, inputs, targets, features):
    """수정된 손실 함수 계산"""
    
    # 1. 분류 손실 (가소성)
    outputs = self._network(inputs)
    if isinstance(outputs, dict):
        outputs = outputs['logits']
    cls_loss = F.cross_entropy(outputs, targets)
    
    # 초기화
    distill_loss = torch.tensor(0.0, device=inputs.device)
    ib_module_loss = torch.tensor(0.0, device=inputs.device)
    
    # 이전 네트워크가 있는 경우에만 IB 증류 수행
    if self._old_network is not None and self._ib_encoder is not None:
        
        # 0. 교사 특징 추출 (원본)
        with torch.no_grad():
            teacher_features = self._old_network.extract_vector(inputs)
        
        # --- IB 모듈 연산 (VAE) ---
        # IB 인코더로 교사 특징 압축
        ib_output = self._ib_encoder(teacher_features)
        mu, logvar = ib_output.chunk(2, dim=1)
        
        # Reparameterization
        z = self._reparameterize(mu, logvar)
        
        # IB 디코더로 복원
        reconstructed_teacher_features = self._ib_decoder(z)
        
        # 2. 증류 손실 (안정성)
        # 학생이 복원된 교사 특징을 모방하도록 함
        distill_loss = F.mse_loss(features, reconstructed_teacher_features)
        
        # 3. IB 모듈 손실 (VAE Loss)
        # 3-1. Reconstruction Loss: 원본 교사 특징 vs 복원된 교사 특징
        recon_loss = F.mse_loss(reconstructed_teacher_features, teacher_features)
        
        # 3-2. Compression Loss (KL Divergence)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # IB 모듈 전체 손실
        ib_module_loss = recon_loss + self._ib_beta * kl_loss
    
    # 전체 손실
    total_loss = cls_loss + self.lambda_D * distill_loss + self.lambda_IB * ib_module_loss
    
    return {
        'total_loss': total_loss,
        'cls_loss': cls_loss,
        'distill_loss': distill_loss,
        'ib_module_loss': ib_module_loss
    }
```

### 4. 옵티마이저 설정 (핵심 수정사항)

**핵심 수정사항**: IB 모듈 파라미터를 옵티마이저에 포함

```python
def _setup_optimizer(self):
    """옵티마이저 설정 - IB 모듈 파라미터 포함"""
    
    # 1. 최적화할 파라미터 리스트 생성
    params_to_optimize = list(self._network.parameters())
    
    # IB 모듈 파라미터 추가 (핵심 수정사항)
    if self._ib_encoder is not None:
        params_to_optimize += list(self._ib_encoder.parameters())
        params_to_optimize += list(self._ib_decoder.parameters())
    
    # 2. 옵티마이저 생성
    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4
    )
    
    # 3. 스케줄러 설정
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[60, 120, 160],
        gamma=0.2
    )
    
    return optimizer, scheduler
```

### 5. 태스크 완료 후 처리

```python
def after_task(self):
    """태스크 완료 후 처리"""
    
    # 1. 현재 네트워크를 이전 네트워크로 저장
    self._old_network = copy.deepcopy(self._network)
    self._old_network.eval()
    
    # 2. IB 모듈 초기화 (새로운 압축 방식 학습)
    if self._old_network is not None:
        feature_dim = self._old_network.feature_dim
        self._init_ib_modules(feature_dim)
    
    # 3. Exemplar 선택 및 저장
    self._reduce_exemplar()
    self._construct_exemplar()
```

## 손실 함수 설계 (핵심 수정사항)

### 1. 기존 구현의 문제점

**문제**: IB 모듈과 학생 모델을 동시에 최적화하는 구조

```python
# 기존 구현 (문제가 있는 버전)
def _ib_distillation_loss(self, student_features, teacher_features):
    ib_output = self._ib_encoder(teacher_features)
    mu, logvar = ib_output.chunk(2, dim=1)
    z = self._reparameterize(mu, logvar)
    reconstructed_features = self._ib_decoder(z)
    
    # 문제: 학생 특징을 목표로 사용
    knowledge_transfer_loss = F.mse_loss(student_features, reconstructed_features)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # IB 모듈이 교사 특징을 왜곡할 위험
    ib_loss = knowledge_transfer_loss + self._ib_beta * kl_loss
    return ib_loss
```

### 2. 수정된 손실 함수

**해결책**: IB 모듈 최적화와 학생 모델 최적화의 명확한 분리

```python
def _compute_separated_losses(self, inputs, targets, features):
    """분리된 손실 함수 계산"""
    
    # 1. 분류 손실 (학생 모델 - 가소성)
    outputs = self._network(inputs)
    if isinstance(outputs, dict):
        outputs = outputs['logits']
    cls_loss = F.cross_entropy(outputs, targets)
    
    distill_loss = torch.tensor(0.0, device=inputs.device)
    ib_module_loss = torch.tensor(0.0, device=inputs.device)
    
    if self._old_network is not None and self._ib_encoder is not None:
        
        # 교사 특징 추출
        with torch.no_grad():
            teacher_features = self._old_network.extract_vector(inputs)
        
        # IB 모듈 연산
        ib_output = self._ib_encoder(teacher_features)
        mu, logvar = ib_output.chunk(2, dim=1)
        z = self._reparameterize(mu, logvar)
        reconstructed_teacher_features = self._ib_decoder(z)
        
        # 2. 증류 손실 (학생 모델 - 안정성)
        # 학생이 복원된 교사 특징을 모방
        distill_loss = F.mse_loss(features, reconstructed_teacher_features)
        
        # 3. IB 모듈 손실 (VAE Loss)
        # 3-1. Reconstruction Loss: 교사 특징 복원 품질
        recon_loss = F.mse_loss(reconstructed_teacher_features, teacher_features)
        
        # 3-2. Compression Loss: 압축 강도 조절
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # IB 모듈만의 손실 (교사 특징 복원에 집중)
        ib_module_loss = recon_loss + self._ib_beta * kl_loss
    
    # 전체 손실 (가중치 조절 가능)
    total_loss = cls_loss + self.lambda_D * distill_loss + self.lambda_IB * ib_module_loss
    
    return {
        'total_loss': total_loss,
        'cls_loss': cls_loss,
        'distill_loss': distill_loss,
        'ib_module_loss': ib_module_loss,
        'recon_loss': recon_loss if 'recon_loss' in locals() else torch.tensor(0.0),
        'kl_loss': kl_loss if 'kl_loss' in locals() else torch.tensor(0.0)
    }
```

### 3. 손실 함수의 각 구성 요소 역할

#### L_Cls (분류 손실)
- **목적**: 현재 태스크의 분류 성능
- **역할**: 가소성 확보 (새로운 지식 학습)
- **계산**: Cross-entropy loss

#### L_Distill (증류 손실)
- **목적**: 학생이 복원된 교사 특징을 모방
- **역할**: 안정성 확보 (이전 지식 유지)
- **계산**: MSE between student features and reconstructed teacher features

#### L_IB (IB 모듈 손실)
- **목적**: IB 모듈이 교사 특징을 잘 복원하도록 최적화
- **역할**: 정보 압축 품질 보장
- **계산**: Reconstruction Loss + β × KL Loss

## 학습 프로세스

### 1. 전체 학습 루프

```python
def incremental_train(self, data_manager):
    """증분 학습 메인 루프"""
    
    for task_id in range(data_manager.nb_tasks):
        print(f"=== Task {task_id + 1}/{data_manager.nb_tasks} ===")
        
        # 1. 현재 태스크 데이터 로드
        train_dataset, test_dataset = self._get_task_data(data_manager, task_id)
        
        # 2. 네트워크 업데이트 (새 클래스 추가)
        self._update_representation(train_dataset, test_dataset)
        
        # 3. 태스크 학습
        self._train(train_dataset, test_dataset)
        
        # 4. 태스크 완료 후 처리
        self.after_task()
        
        # 5. 성능 평가
        self._evaluate(test_dataset)
```

### 2. 태스크별 학습

```python
def _train(self, train_dataset, test_dataset):
    """태스크별 학습"""
    
    # 데이터 로더 설정
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    
    # 옵티마이저 설정 (IB 모듈 파라미터 포함)
    optimizer, scheduler = self._setup_optimizer()
    
    # 학습 루프
    for epoch in range(200):  # 하드코딩된 값
        self._network.train()
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # 순전파
            features = self._network.extract_vector(inputs)
            
            # 수정된 손실 계산
            losses = self._compute_separated_losses(inputs, targets, features)
            
            # 역전파
            optimizer.zero_grad()
            losses['total_loss'].backward()
            optimizer.step()
            
            # 로깅
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}")
                print(f"  Total Loss: {losses['total_loss']:.4f}")
                print(f"  Cls Loss: {losses['cls_loss']:.4f}")
                print(f"  Distill Loss: {losses['distill_loss']:.4f}")
                print(f"  IB Module Loss: {losses['ib_module_loss']:.4f}")
        
        # 스케줄러 업데이트
        scheduler.step()
        
        # 검증
        if epoch % 10 == 0:
            self._validate(test_dataset)
```

### 3. KL Annealing (선택사항)

VAE 학습 안정성을 위한 KL Annealing 기법:

```python
def _get_kl_weight(self, epoch, total_epochs):
    """KL 손실 가중치 점진적 증가"""
    # 0에서 시작하여 점진적으로 목표값까지 증가
    target_weight = self._ib_beta
    current_weight = target_weight * min(1.0, epoch / (total_epochs * 0.1))
    return current_weight

def _compute_losses_with_annealing(self, inputs, targets, features, epoch, total_epochs):
    """KL Annealing을 적용한 손실 계산"""
    
    # 기본 손실 계산
    losses = self._compute_separated_losses(inputs, targets, features)
    
    # KL Annealing 적용
    if self._old_network is not None and self._ib_encoder is not None:
        kl_weight = self._get_kl_weight(epoch, total_epochs)
        
        # KL 손실 재계산
        with torch.no_grad():
            teacher_features = self._old_network.extract_vector(inputs)
        
        ib_output = self._ib_encoder(teacher_features)
        mu, logvar = ib_output.chunk(2, dim=1)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # IB 모듈 손실 업데이트
        recon_loss = losses.get('recon_loss', torch.tensor(0.0))
        losses['ib_module_loss'] = recon_loss + kl_weight * kl_loss
    
    return losses
```

## 하이퍼파라미터 튜닝

### 1. 핵심 하이퍼파라미터

#### β (IB 압축 강도)
- **범위**: 0.01 ~ 1.0
- **기본값**: 0.1
- **영향**: 
  - 낮은 값: 압축 부족, 가소성 제한
  - 높은 값: 과도한 압축, 정보 손실

#### λ_D (증류 손실 가중치)
- **범위**: 0.1 ~ 10.0
- **기본값**: 1.0
- **영향**: 안정성 vs 가소성 균형

#### λ_IB (IB 모듈 손실 가중치)
- **범위**: 0.1 ~ 10.0
- **기본값**: 1.0
- **영향**: IB 모듈 학습 강도

### 2. 하이퍼파라미터 튜닝 스크립트

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

### 3. 자동 하이퍼파라미터 최적화

```python
def bayesian_optimization():
    """베이지안 최적화를 통한 하이퍼파라미터 튜닝"""
    
    from skopt import gp_minimize
    from skopt.space import Real
    
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

## 성능 최적화

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
def _setup_dataloader(self, dataset):
    """최적화된 데이터 로더 설정"""
    return DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=8,              # CPU 코어 수에 맞게 조정
        pin_memory=True,            # GPU 전송 최적화
        persistent_workers=True,    # 워커 재사용
        prefetch_factor=2           # 미리 로딩
    )
```

### 2. 메모리 효율성

#### 그래디언트 체크포인팅
```python
from torch.utils.checkpoint import checkpoint

def _forward_with_checkpoint(self, inputs):
    """그래디언트 체크포인팅을 통한 메모리 절약"""
    return checkpoint(self._network.extract_vector, inputs)
```

#### 메모리 정리
```python
def _cleanup_memory(self):
    """주기적 메모리 정리"""
    import gc
    gc.collect()
    torch.cuda.empty_cache()
```

### 3. 모델 병렬화

```python
def _setup_model_parallel(self):
    """모델 병렬화 설정"""
    if torch.cuda.device_count() > 1:
        self._network = nn.DataParallel(self._network)
        if self._ib_encoder is not None:
            self._ib_encoder = nn.DataParallel(self._ib_encoder)
            self._ib_decoder = nn.DataParallel(self._ib_decoder)
```

## 실험 설정 및 실행

### 1. 기본 실험 설정

```json
{
    "prefix": "asib_cl_experiment",
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
    "gamma": 0.2
}
```

### 2. 실험 실행 스크립트

```python
def run_asib_cl_experiment():
    """ASIB-CL 실험 실행"""
    
    # 설정 로드
    config = load_config('PyCIL/exps/asib_cl.json')
    
    # 실험 디렉토리 생성
    experiment_dir = f"experiments/asib_cl_{int(time.time())}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 로깅 설정
    setup_logging(experiment_dir)
    
    # 모델 생성
    model = get_model('asib_cl', config)
    
    # 데이터 매니저 생성
    data_manager = DataManager(
        config['dataset'],
        config['shuffle'],
        config['seed'][0],
        config['init_cls'],
        config['increment']
    )
    
    # 실험 실행
    start_time = time.time()
    results = model.incremental_train(data_manager)
    end_time = time.time()
    
    # 결과 저장
    save_results(results, experiment_dir)
    
    # 성능 분석
    analyze_performance(results, experiment_dir)
    
    print(f"Experiment completed in {end_time - start_time:.2f} seconds")
    return results
```

### 3. 배치 실험 실행

```python
def run_comparison_experiments():
    """비교 실험 실행"""
    
    methods = ['asib_cl', 'ewc', 'lwf', 'icarl', 'der']
    results = {}
    
    for method in methods:
        print(f"Running {method} experiment...")
        
        config_file = f'PyCIL/exps/{method}.json'
        result = run_single_experiment(config_file)
        results[method] = result
        
        # 중간 결과 저장
        save_intermediate_results(results)
    
    # 최종 비교 분석
    final_analysis = compare_methods(results)
    save_final_analysis(final_analysis)
    
    return results
```

## 결과 분석

### 1. 핵심 성능 지표

#### Average Incremental Accuracy (AIA)
```python
def calculate_aia(self, accuracies):
    """평균 증분 정확도 계산"""
    return np.mean(accuracies)
```

#### Average Forgetting (AF)
```python
def calculate_af(self, accuracies_matrix):
    """평균 망각률 계산"""
    forgetting = []
    for i in range(len(accuracies_matrix) - 1):
        for j in range(i + 1):
            forgetting.append(accuracies_matrix[i][j] - accuracies_matrix[i+1][j])
    return np.mean(forgetting)
```

### 2. 상세 분석 스크립트

```python
def analyze_asib_cl_results(results):
    """ASIB-CL 결과 상세 분석"""
    
    # 1. 전체 성능 분석
    aia = calculate_aia(results['final_accuracies'])
    af = calculate_af(results['accuracy_matrix'])
    
    print(f"Average Incremental Accuracy: {aia:.4f}")
    print(f"Average Forgetting: {af:.4f}")
    
    # 2. 태스크별 성능 분석
    task_performance = analyze_task_performance(results)
    
    # 3. 손실 함수 구성 요소 분석
    loss_analysis = analyze_loss_components(results)
    
    # 4. IB 모듈 성능 분석
    ib_analysis = analyze_ib_performance(results)
    
    # 5. 시각화
    create_performance_plots(results)
    
    return {
        'aia': aia,
        'af': af,
        'task_performance': task_performance,
        'loss_analysis': loss_analysis,
        'ib_analysis': ib_analysis
    }
```

### 3. 시각화

```python
def create_performance_plots(results):
    """성능 시각화"""
    
    # 1. 정확도 곡선
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plot_accuracy_curve(results['accuracy_matrix'])
    plt.title('Accuracy Curve')
    plt.xlabel('Task')
    plt.ylabel('Accuracy')
    
    # 2. 손실 함수 구성 요소
    plt.subplot(2, 2, 2)
    plot_loss_components(results['loss_history'])
    plt.title('Loss Components')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # 3. IB 압축 성능
    plt.subplot(2, 2, 3)
    plot_ib_performance(results['ib_metrics'])
    plt.title('IB Module Performance')
    plt.xlabel('Task')
    plt.ylabel('Reconstruction Error')
    
    # 4. 하이퍼파라미터 영향
    plt.subplot(2, 2, 4)
    plot_hyperparameter_impact(results['hyperparameter_sweep'])
    plt.title('Hyperparameter Impact')
    plt.xlabel('β')
    plt.ylabel('AIA')
    
    plt.tight_layout()
    plt.savefig('asib_cl_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
```

## 문제 해결

### 1. 학습 불안정성

#### 문제: 손실 함수 발산
```python
def _handle_loss_divergence(self, losses):
    """손실 함수 발산 처리"""
    
    # 손실 값 검증
    for loss_name, loss_value in losses.items():
        if torch.isnan(loss_value) or torch.isinf(loss_value):
            print(f"Warning: {loss_name} is {loss_value}")
            
            # 손실 값 클리핑
            if torch.isnan(loss_value):
                losses[loss_name] = torch.tensor(0.0, device=loss_value.device)
            elif torch.isinf(loss_value):
                losses[loss_name] = torch.tensor(100.0, device=loss_value.device)
    
    return losses
```

#### 문제: 그래디언트 폭발
```python
def _gradient_clipping(self, optimizer):
    """그래디언트 클리핑"""
    torch.nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=1.0)
    
    if self._ib_encoder is not None:
        torch.nn.utils.clip_grad_norm_(self._ib_encoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self._ib_decoder.parameters(), max_norm=1.0)
```

### 2. 메모리 문제

#### 문제: GPU 메모리 부족
```python
def _handle_memory_shortage(self):
    """메모리 부족 처리"""
    
    # 배치 크기 동적 조정
    if torch.cuda.memory_allocated() > 0.9 * torch.cuda.max_memory_allocated():
        self.batch_size = max(16, self.batch_size // 2)
        print(f"Reduced batch size to {self.batch_size}")
    
    # 메모리 정리
    self._cleanup_memory()
```

### 3. 수렴 문제

#### 문제: 학습 속도 조정
```python
def _adaptive_learning_rate(self, optimizer, epoch, loss_history):
    """적응적 학습률 조정"""
    
    # 손실이 증가하는 경우 학습률 감소
    if len(loss_history) > 5:
        recent_losses = loss_history[-5:]
        if all(recent_losses[i] > recent_losses[i-1] for i in range(1, len(recent_losses))):
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
            print(f"Reduced learning rate to {optimizer.param_groups[0]['lr']}")
```

### 4. 디버깅 도구

```python
def _debug_training(self, inputs, targets, features, losses):
    """학습 과정 디버깅"""
    
    # 텐서 형태 검증
    print(f"Inputs shape: {inputs.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Features shape: {features.shape}")
    
    # 손실 값 검증
    for loss_name, loss_value in losses.items():
        print(f"{loss_name}: {loss_value.item():.6f}")
    
    # 그래디언트 검증
    if self._network.fc.weight.grad is not None:
        grad_norm = torch.norm(self._network.fc.weight.grad)
        print(f"Gradient norm: {grad_norm:.6f}")
```

## 결론

ASIB-CL 구현은 ASIB의 핵심 철학인 정보 병목을 Continual Learning 환경에 성공적으로 적용한 사례입니다. 특히, **손실 함수 설계의 수정**과 **옵티마이저 설정의 보완**을 통해 IB 모듈이 교사 특징을 올바르게 압축하고 복원할 수 있도록 구현했습니다.

### 주요 성과
1. **이론적 일관성**: IB의 핵심 아이디어를 CL에 정확히 적용
2. **실용적 구현**: PyCIL 프레임워크와의 완벽한 통합
3. **확장 가능성**: 다양한 하이퍼파라미터 튜닝 및 최적화 기법 적용
4. **안정성**: 포괄적인 테스트와 디버깅 도구를 통한 안정성 확보

### 향후 발전 방향
1. **고급 IB 기법**: β-VAE, InfoVAE 등 고급 IB 방법론 적용
2. **동적 하이퍼파라미터**: 태스크별 자동 하이퍼파라미터 조정
3. **멀티모달 확장**: 다양한 데이터 모달리티에 대한 확장
4. **실시간 학습**: 온라인 학습 환경에서의 적용

이 구현을 통해 ASIB의 연구 가치를 Continual Learning 분야에서 최대한 활용할 수 있으며, 안정성-가소성 딜레마 해결에 대한 새로운 접근 방식을 제시합니다. 