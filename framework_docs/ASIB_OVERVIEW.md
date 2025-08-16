# ASIB Framework Overview

ASIB 프레임워크의 전반 구조, 메소드(IB, CCCP, PPF/FFP), 핵심 모듈, 주요 설정 키, 학습 플로우, ablation 규칙을 한 문서로 요약합니다. 다른 프롬프트에 넣을 때 본문 전체를 복사해도 됩니다.

## 최신 Ablation 업데이트 요약 (2025-08)
- 공통 하이퍼(모든 L0~L4):
  - Distillation: `ce_alpha: 0.65`, `kd_alpha: 0.35`, `kd_max_ratio: 2.0`, `tau_schedule: [2.0, 6.0]`
  - A‑Step: `a_step_lr: 0.0001`, `synergy_only_epochs: 12`, `enable_kd_after_syn_acc: 0.01`
  - IB‑MBM: `ib_mbm_out_dim: 512`, `ib_mbm_logvar_clip: 6`, `ib_mbm_min_std: 0.001`, `ib_mbm_lr_factor: 10`, `ib_beta: 0.00005`
  - Adapter/Feature: `use_distillation_adapter: true`, `distill_out_dim: 512`, `feat_kd_alpha: 0.0`, `feat_kd_key: distill_feat`
- A‑Step 안정화 로직(코드 반영): 초기 `synergy_only_epochs` 동안 IB‑KL off, certainty 가중 무시(CE 균등), logvar 클리핑, 에폭 말에 `cfg["last_synergy_acc"]` 저장.
- 실행 스크립트: `-cn="experiment/<CFG>"` + 루트 오버라이드(`+seed=`), `CUDA_VISIBLE_DEVICES` 강제 해제.

## 1) 아키텍처 개요
- Teachers(교사 2개): `teacher1`, `teacher2` 고정 백본에서 특징 추출
- Student(학생): 증류 대상 모델
- IB_MBM + SynergyHead: 교사 특징(KV) + 학생 질의(Q) → 잠재표현 z → 로짓. IB의 KL 항은 β 가중
- 손실: CE + (선택) KD + (선택) Feature KD + (선택) CCCP surrogate + 정규항
- 정책: PPF/FFP(Partial Freeze/Finetuning)로 일부 블록/정규화를 동결하거나 교사를 소규모 파인튜닝

참고 문서: `framework_docs/configs.md`, `framework_docs/root_files.md`, `framework_docs/utils.md`, `framework_docs/modules.md`, `framework_docs/run.md`, `framework_docs/tests.md`

## 2) 핵심 메소드 요약
### IB (Information Bottleneck)
- 목적: 교사 정보 과잉전달을 억제해 일반화 향상. 학생 질의(Q)와 교사 특징(KV)로 잠재 z를 생성, KL(β)로 정보량 제어
- 주요 키:
  - `use_ib`(bool)
  - `ib_beta`(float), `ib_beta_warmup_epochs`(int, optional), `ib_epochs_per_stage`(int)
  - `ib_mbm_query_dim`(auto), `ib_mbm_out_dim`(int), `ib_mbm_n_head`(int)
  - `ib_mbm_feature_norm`(str), `ib_mbm_reg_lambda`(float)
- 유틸: `utils.get_beta(cfg, epoch)`로 β 선형 워밍업/스케줄 지원

### CCCP (Contrastive Consistency Proxy)
- 목적: 대비적·온도 기반 KD 대체항을 총 손실에 더해 학생의 예측 일관성을 강화
- 주요 키:
  - `use_cccp`(bool), `cccp_alpha`(float)
  - `tau`(float), `cccp_nt`(int), `cccp_ns`(int)
- 비고: ablation에서는 CCCP 유무 외 나머지 하이퍼는 베이스라인과 동일 유지 권장

### PPF/FFP (Partial Freeze / Finetuning Policy)
- 목적: 학습 안정화와 속도/일반화 균형, 필요 시 교사 미세튜닝
- 주요 키:
  - 부분동결: `use_partial_freeze`(bool)
  - 동결 레벨: `student_freeze_level`, `teacher1_freeze_level`, `teacher2_freeze_level` (−1: 없음, 0: 헤드만, 1: 마지막 블록, 2: 마지막 두 블록)
  - 정규화 정책: `student_freeze_bn`, `teacher1_freeze_bn`, `teacher2_freeze_bn`
  - 교사 파인튜닝: `use_teacher_finetuning`(bool), `train_distill_adapter_only`(bool)
- 유틸: `apply_partial_freeze(model, level, freeze_bn)`로 간단 적용

## 3) 주요 설정 키 레퍼런스(요약)
- 데이터/모델
  - `dataset.name`, `small_input`, `data_aug`, `batch_size`, `num_workers`
  - `model.student.name`, `pretrained`, `use_adapter`
  - `teacher1/2.name`, `pretrained`
- 손실/KD
  - `kd_target`('synergy'), `ce_alpha`, `kd_alpha`, `kd_ens_alpha`, `feat_kd_alpha`, `feat_kd_key`('distill_feat')
  - `use_loss_clamp`, `loss_clamp_max`
- IB
  - `use_ib`, `ib_beta`, `ib_beta_warmup_epochs`, `ib_epochs_per_stage`, `ib_mbm_query_dim`(auto), `ib_mbm_out_dim`, `ib_mbm_n_head`, `ib_mbm_feature_norm`, `ib_mbm_reg_lambda`
- CCCP
  - `use_cccp`, `cccp_alpha`, `tau`, `cccp_nt`, `cccp_ns`
- PPF/Finetuning
  - `use_partial_freeze`, `student_freeze_level`, `teacher*_freeze_level`, `*freeze_bn`, `use_teacher_finetuning`, `train_distill_adapter_only`
- 최적화/스케줄/AMP
  - `optimizer`, `student_lr`, `student_weight_decay`, `schedule.type`, `lr_warmup_epochs`, `min_lr`, `use_amp`, `amp_dtype`('float16'|'bfloat16')

## 4) 최소 설정 템플릿(복붙용)
```yaml
# Baseline(예)
kd_target: synergy
ce_alpha: 1.0
kd_alpha: 0.0
kd_ens_alpha: 0.0
feat_kd_alpha: 0.0
use_loss_clamp: false
teacher_adapt_epochs: 0
teacher_adapt_kd_warmup: 0

# IB
use_ib: false
ib_beta: 0.001
ib_beta_warmup_epochs: 0
ib_epochs_per_stage: 12
ib_mbm_out_dim: 512
ib_mbm_n_head: 4

# CCCP
use_cccp: false
cccp_alpha: 0.5
tau: 4.0
cccp_nt: 1
cccp_ns: 1

# PPF/FT
use_partial_freeze: false
student_freeze_level: -1
teacher1_freeze_level: -1
teacher2_freeze_level: -1
student_freeze_bn: false
teacher1_freeze_bn: true
teacher2_freeze_bn: true
use_teacher_finetuning: false
train_distill_adapter_only: false
```

## 5) 학습 플로우(Stages)
- 전처리: `auto_set_ib_mbm_query_dim_with_model`(Q 차원 자동 설정), `renorm_ce_kd`(가중 재정규화), AMP/로깅 초기화
- Stage 루프:
  1) A‑Step(선택): `teacher_adapt_epochs>0`이면 교사 어댑트/IB/Head 학습(교사 백본 동결)
  2) B‑Step: 학생 증류(CE/KD/FeatKD + CCCP + 정규항)
  3) 로깅/저장: `ExperimentLogger` 메타/지표, CSV/JSON 요약
- 평가: teachers + IB_MBM + SynergyHead 조합 가능

## 6) Ablation 권장 규칙
- 단일 요인만 변경: 대상(IB/CCCP/PPF) 외 모든 하이퍼는 베이스라인과 동일 유지
- 로그 검증: `KD`, `IB/CCCP`, `PPF` 라인이 베이스라인과 동일한지 확인

## 7) 트러블슈팅
- CCCP 약함: `cccp_alpha` 0.25/0.5 시도, 필요 시 `cccp_nt/ns=2`
- IB 튜닝: `ib_mbm_out_dim ~= distill_out_dim`, β 워밍업 확인
- PPF 영향: 동결 레벨/BN 정책 변경 여부 확인
- 환경: `run/test_gpu_allocation.sh`를 sbatch로 제출해 노드/GPU/torch CUDA 확인

---

## 8) 중요 코드 스니펫

### 8.1 IB β 스케줄 함수(`utils.get_beta`)
```python
def get_beta(cfg: dict, epoch: int = 0) -> float:
    """Return β for the IB KL term at epoch (fixed/warmup/scheduled)."""
    if "beta_schedule" in cfg:
        start, end, total = cfg["beta_schedule"]
        t = min(epoch / max(1, total), 1.0)
        beta = start + t * (end - start)
    else:
        beta = float(cfg.get("ib_beta", 1e-3))
        warmup = int(cfg.get("ib_beta_warmup_epochs", 0))
        if warmup > 0:
            scale = min(float(epoch) / warmup, 1.0)
            beta *= scale
    return beta
```

### 8.2 부분 동결 정책(`modules.apply_partial_freeze`)
```python
def apply_partial_freeze(model, level: int, freeze_bn: bool = False):
    if level < 0:
        for p in model.parameters():
            p.requires_grad = True
        return
    if level == 0:
        for p in model.parameters():
            p.requires_grad = True
        apply_bn_ln_policy(model, train_bn=not freeze_bn)
        return
    freeze_all(model)
    patterns = [
        r"\.layer4\.", r"features\.7\.", r"features\.8\.", r"\.layers\.3\.",
        r"(?:^|\.)fc\.", r"(?:^|\.)classifier\.", r"(?:^|\.)head\.",
    ] if level == 1 else [
        r"\.layer3\.", r"\.layer4\.", r"features\.6\.", r"features\.7\.", r"features\.8\.",
        r"\.layers\.2\.", r"\.layers\.3\.", r"(?:^|\.)fc\.", r"(?:^|\.)classifier\.", r"(?:^|\.)head\.",
    ]
    unfreeze_by_regex(model, patterns)
    apply_bn_ln_policy(model, train_bn=not freeze_bn)
```

### 8.3 CCCP 통합(총 손실 구성부)
```python
kd_cccp = 0.0
if cfg.get("use_cccp", True):
    with torch.no_grad():
        s_out = student_model(x)[1] if isinstance(student_model(x), tuple) else student_model(x)
    tau  = float(cfg.get("tau", 4.0))
    kd_w = float(cfg.get("kd_alpha", 0.0))
    kd_cccp = (
        kd_w * tau * tau *
        F.kl_div(
            F.log_softmax(zsyn / tau, dim=1),
            F.softmax(s_out.detach() / tau, dim=1),
            reduction="batchmean",
        )
    )

total_loss_step = (
    kd_weight * loss_kd
    + synergy_ce_loss
    + cfg.get("feat_kd_alpha", 0) * feat_kd_loss
    + ib_loss_val
    + float(cfg.get("reg_lambda", 0.0)) * reg_loss
    + float(cfg.get("ib_mbm_reg_lambda", 0.0)) * ib_mbm_reg_loss
    + kd_cccp
)
```

### 8.4 Synergy Ensemble 전방패스
```python
class SynergyEnsemble(nn.Module):
    def __init__(self, teacher1, teacher2, ib_mbm, synergy_head, student=None, cfg=None):
        super().__init__()
        self.teacher1, self.teacher2 = teacher1, teacher2
        self.ib_mbm, self.synergy_head = ib_mbm, synergy_head
        self.student, self.cfg = student, (cfg or {})
        self.feat_key = (self.cfg.get("feat_kd_key") or
                         ("distill_feat" if self.cfg.get("use_distillation_adapter", False) else "feat_2d"))
    def forward(self, x):
        autocast_ctx, _ = get_amp_components(self.cfg or {})
        with autocast_ctx:
            s_out = self.student(x)
            s_dict = s_out[0] if isinstance(s_out, tuple) else s_out
            q = s_dict[self.feat_key]
            with torch.no_grad():
                t1d = (self.teacher1(x)[0] if isinstance(self.teacher1(x), tuple) else self.teacher1(x))
                t2d = (self.teacher2(x)[0] if isinstance(self.teacher2(x), tuple) else self.teacher2(x))
                k1, k2 = t1d[self.feat_key], t2d[self.feat_key]
                kv = torch.stack([k1, k2], dim=1)
            z, mu, logvar = self.ib_mbm(q, kv)
            logits = self.synergy_head(z)
            return logits
```

### 8.5 Student B‑Step 트레이너(요지)
```python
def student_distillation_update(teacher_wrappers, ib_mbm, synergy_head, student_model,
                                trainloader, testloader, cfg, logger,
                                optimizer, scheduler, global_ep=0):
    # freeze teachers + ib_mbm + head
    # decide student_epochs from schedule; set AMP autocast+scaler
    for ep in range(student_epochs):
        cur_tau = get_tau(cfg, epoch=global_ep + ep, total_epochs=scheduler.T_max if scheduler else cfg.get("total_epochs", 1))
        for step, (x, y) in enumerate(smart_tqdm(trainloader, desc=f"[StudentDistill ep={ep+1}]")):
            # student forward → s_logit, s_feat
            # teacher forward (조건부) → t1_dict, t2_dict, f1_2d, f2_2d
            # IB_MBM (조건부) → zsyn_ng, mu_ng, logvar_ng
            # CE, KD, (옵션)Feat-KD 계산 후 가중 합산 + (옵션)loss clamp
            # grad clip, scaler.step/update
        # validate, log, scheduler.step, best snapshot
    return best_acc
```

### 8.6 Teacher A‑Step 트레이너(요지)
```python
def teacher_adaptive_update(teacher_wrappers, ib_mbm, synergy_head, student_model,
                            trainloader, testloader, cfg, logger,
                            optimizer, scheduler, global_ep=0):
    # 교사 백본 freeze or finetune 여부 결정
    # student eval 고정, ib/synergy 학습 모드
    for ep in range(teacher_epochs):
        cur_tau = get_tau(cfg, epoch=global_ep + ep, total_epochs=scheduler.T_max if scheduler else cfg.get("total_epochs", 1))
        for step, (x, y) in enumerate(smart_tqdm(trainloader, desc=f"[TeacherAdaptive ep={ep+1}]")):
            # student feat 추출(s_feat)
            # teachers feat 추출(KV), IB_MBM → zsyn, loss 구성(CE/KD/IB/정규화)
            # kd_cccp surrogate(옵션) 추가 가능
            # scaler/optimizer 스텝
        # per-epoch synergy 평가, best snapshot
    return best_synergy
```

### 8.7 Optimizer/Scheduler 생성 요약
```python
(
  teacher_optimizer,
  teacher_scheduler,
  student_optimizer,
  student_scheduler,
) = create_optimizers_and_schedulers(
  teacher_wrappers, ib_mbm, synergy_head, student_model, cfg, num_stages
)
# CosineAnnealing 기반, teacher_adapt_epochs=0이면 T_max=1 안전 처리
```

### 8.8 안전/품질 유틸
```python
def renorm_ce_kd(cfg):
    if "ce_alpha" in cfg and "kd_alpha" in cfg:
        ce, kd = float(cfg["ce_alpha"]), float(cfg["kd_alpha"])
        if abs(ce + kd - 1) > 1e-5:
            tot = ce + kd
            cfg["ce_alpha"], cfg["kd_alpha"] = (0.5, 0.5) if tot == 0 else (ce/tot, kd/tot)

def auto_set_ib_mbm_query_dim_with_model(student_model, cfg):
    if cfg.get("feat_kd_key", "feat_2d") == "distill_feat" and cfg.get("use_distillation_adapter", False):
        qdim = int(cfg.get("distill_out_dim", 0))
        if qdim > 0:
            cfg["ib_mbm_query_dim"] = qdim; return
    if cfg.get("ib_mbm_query_dim", 0) in (0, None):
        with torch.no_grad():
            dummy = torch.randn(1, 3, 32, 32, device=cfg["device"])
            feat_dict, _, _ = student_model(dummy)
            qdim = feat_dict.get("distill_feat", feat_dict.get("feat_2d")).shape[-1]
            cfg["ib_mbm_query_dim"] = int(qdim)
```

---

## 9) 로깅과 실험 기록

### 9.1 ExperimentLogger 핵심 API
```python
class ExperimentLogger:
    """
    Handles experiment configuration + result metrics in a single dict (self.config).
    1) Create a unique exp_id for each run
    2) update_metric(key, value) 로 임의의 메트릭/하이퍼를 누적 기록
    3) finalize() 시 JSON/CSV로 저장
    """
    def update_metric(self, key, value):
        """Save any metric (accuracy, loss, hyperparams, etc.)."""
        self.config[key] = value

    def finalize(self):
        """
        1) total_time_sec 계산
        2) self.config를 JSON으로 저장
        3) 요약 CSV 업데이트
        """
        ...
```

### 9.2 Structured logging 설정
```python
def setup_structured_logging(log_dir: str, experiment_name: str, level: str = "INFO"):
    import os, logging
    os.makedirs(log_dir, exist_ok=True)
    # File + Console 핸들러 구성, 포맷터 지정
    ...
```

## 10) AMP/Autocast/GradScaler

### 10.1 AMP 컴포넌트 팩토리
```python
def get_amp_components(cfg):
    """Return autocast context and GradScaler based on config."""
    use_amp = bool(cfg.get("use_amp", False))
    device = cfg.get("device", "cuda")
    if not use_amp:
        from contextlib import nullcontext
        return nullcontext(), None
    amp_dtype = cfg.get("amp_dtype", "float16")
    dtype = torch.float16 if amp_dtype == "float16" else torch.bfloat16
    autocast = torch.autocast if hasattr(torch, "autocast") else torch.cuda.amp.autocast
    autocast_ctx = autocast("cuda", dtype=dtype)
    GradScaler = torch.amp.GradScaler if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler
    try:
        scaler = GradScaler(device="cuda", init_scale=int(cfg.get("grad_scaler_init_scale", 1024)))
    except TypeError:
        scaler = GradScaler(init_scale=int(cfg.get("grad_scaler_init_scale", 1024)))
    return autocast_ctx, scaler
```

## 11) 데이터 로더

### 11.1 CIFAR‑100 / ImageNet‑32 로더 요약
```python
from data.cifar100 import get_cifar100_loaders
from data.imagenet32 import get_imagenet32_loaders

# 예시
tr, te = get_cifar100_loaders(root="./data", batch_size=128, num_workers=2, augment=True)
tr32, te32 = get_imagenet32_loaders(root="./data", batch_size=128, num_workers=2, augment=True)
```

### 11.2 태스크 분할 로더 (CL)
```python
from utils.data import get_split_cifar100_loaders

task_loaders = get_split_cifar100_loaders(num_tasks=5, batch_size=128, augment=True)
```

## 12) 모델 빌드/구성

### 12.1 Student/Teacher 팩토리
```python
from core import create_student_by_name, create_teacher_by_name

student = create_student_by_name(student_name="resnet101_scratch", num_classes=100, pretrained=False, cfg=cfg)
teacher1 = create_teacher_by_name(teacher_name="convnext_l", num_classes=100, pretrained=True, cfg=cfg)
teacher2 = create_teacher_by_name(teacher_name="resnet152", num_classes=100, pretrained=True, cfg=cfg)
```

### 12.2 IB_MBM/SynergyHead 빌더
```python
from models import build_ib_mbm_from_teachers as build_from_teachers

ib_mbm, synergy_head = build_from_teachers([teacher1, teacher2], cfg, query_dim=cfg.get("ib_mbm_query_dim"))
```

## 13) 손실/헬퍼 함수 (안전 계산)

### 13.1 CE/KL 세이프 계산
```python
def ce_safe(logits, target, ls_eps: float = 0.0):
    # FP16 underflow 방지: float32로 변환 후 softmax clamp
    with torch.autocast('cuda', enabled=False):
        logits = logits.float()
        if ls_eps > 0:
            # label smoothing
            num_classes = logits.size(-1)
            smooth = torch.full_like(logits, ls_eps / (num_classes - 1))
            smooth.scatter_(1, target.unsqueeze(1), 1 - ls_eps)
            return -(smooth * torch.log_softmax(logits, dim=1).clamp(1e-8, 1.0)).sum(dim=1).mean()
        else:
            return F.cross_entropy(logits, target)


def kl_safe(p_logits, q_logits, tau: float = 1.0):
    with torch.autocast('cuda', enabled=False):
        p = torch.softmax(p_logits.float() / tau, dim=1).clamp(1e-8, 1.0)
        q = torch.softmax(q_logits.float() / tau, dim=1).clamp(1e-8, 1.0)
        return F.kl_div(torch.log(p), q, reduction="batchmean") * (tau * tau)
```

### 13.2 IB 손실, Certainty Weights
```python
def ib_loss(mu, logvar, beta: float = 1e-3):
    # β · KL( N(μ,σ²) || N(0,1) ), float32 안전계산 + clipping
    with torch.autocast('cuda', enabled=False):
        mu, logvar = mu.float(), logvar.float()
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        return beta * kld


def certainty_weights(logvar: torch.Tensor) -> torch.Tensor:
    # Higher variance ⇒ lower certainty ⇒ lower weight
    return torch.exp(-logvar)
```

## 14) 트레이너 의사코드 (전체 흐름)

### 14.1 run_training_stages
```python
def run_training_stages(teachers, ib_mbm, synergy_head, student, train_loader, test_loader, cfg, logger, num_stages):
    # 0) Optim/Sched 생성, 메타 기록
    topt, tsched, sopt, ssched = create_optimizers_and_schedulers(teachers, ib_mbm, synergy_head, student, cfg, num_stages)
    global_ep = 0
    final_acc = 0.0
    for stage in range(1, num_stages+1):
        cfg["cur_stage"] = stage
        # A-Step (선택): teacher/IB/Head 적응
        if cfg.get("teacher_adapt_epochs", 0) > 0 or cfg.get("use_partial_freeze", False) or cfg.get("use_cccp", False) or cfg.get("use_teacher_finetuning", False):
            _ = teacher_adaptive_update(teachers, ib_mbm, synergy_head, student, train_loader, test_loader, cfg, logger, topt, tsched, global_ep)
        # B-Step: 학생 증류 학습
        final_acc = student_distillation_update(teachers, ib_mbm, synergy_head, student, train_loader, test_loader, cfg, logger, sopt, ssched, global_ep)
        # global epoch 증가 (A+B 합산)
        global_ep += int(cfg.get("teacher_adapt_epochs", 0)) + int(cfg.get("student_epochs_schedule", [0])[stage-1])
    return final_acc
```

### 14.2 create_optimizers_and_schedulers 요지
```python
def create_optimizers_and_schedulers(teachers, ib_mbm, synergy_head, student, cfg, num_stages):
    # teacher optimizer: Adam (혹은 cfg에 따라), trainable 파라미터만
    # teacher scheduler: CosineAnnealing, teacher_adapt_epochs=0이면 T_max=1
    # student optimizer: AdamW, scheduler: CosineAnnealing
    return teacher_optimizer, teacher_scheduler, student_optimizer, student_scheduler
```

## 15) 실행/엔드투엔드 체크리스트
- 데이터 로더 생성 → 모델 생성(teacher1/2, student) → `auto_set_ib_mbm_query_dim_with_model` → IB_MBM/Head 생성
- 로깅/실험ID 초기화, AMP/GradScaler 준비
- `renorm_ce_kd(cfg)`로 가중치 정규화(필요 시)
- `run_training_stages(...)` 실행 → 결과/메타 `ExperimentLogger` 저장(finalize)
- ablation: 대상 메소드 외 설정은 베이스라인과 동일 유지

## 16) 추가 참고(테스트/스크립트 근거)
- 데이터 유닛테스트: CIFAR‑100/ImageNet‑32 로더 정상 동작 케이스/엣지 케이스 포함
- 모델 빌드 테스트: `create_student_by_name`, `create_teacher_by_name`, `build_ib_mbm_from_teachers` 파이프라인 유효성 확인
- AMP 유닛테스트: `get_amp_components` on/off 분기 검증

---

## 17) ExperimentLogger 구현 상세
```python
import os, json, csv, time, datetime
import torch, torch.nn as nn, torch.nn.functional as F
from utils.utils import get_amp_components, get_tau, smart_tqdm, freeze_all, unfreeze_by_regex, apply_bn_ln_policy
from models.models import build_ib_mbm_from_teachers

def save_json(exp_dict, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(exp_dict, f, indent=4, default=lambda o: str(o))

def save_csv_row(exp_dict, csv_path, fieldnames, write_header_if_new=True):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header_if_new and not file_exists:
            writer.writeheader()
        row_data = {}
        for fn in fieldnames:
            val = exp_dict.get(fn, "")
            row_data[fn] = f"{val:.2f}" if isinstance(val, float) else str(val)
        writer.writerow(row_data)

class ExperimentLogger:
    def __init__(self, args, exp_name="exp"):
        if hasattr(args, "__dict__"):
            self.config = vars(args)
        elif isinstance(args, dict):
            self.config = args
        else:
            raise ValueError("args must be Namespace or dict")
        self.exp_name = exp_name
        self.exp_id = self.config.get("exp_id") or self._generate_exp_id(exp_name)
        self.config["exp_id"] = self.exp_id
        self.results_dir = self.config.get("results_dir", "experiments/test/results")
        self.config.setdefault("results_dir", self.results_dir)
        self.start_time = time.time()
    
    def _generate_exp_id(self, exp_name="exp"):
        eval_mode = self.config.get("eval_mode", "noeval")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{exp_name}_{eval_mode}_{ts}"

    def update_metric(self, key, value):
        self.config[key] = value

    def get_metric(self, key, default=None):
        return self.config.get(key, default)

    def finalize(self):
        self.config["total_time_sec"] = time.time() - self.start_time
        self.config.pop("logger", None)
        os.makedirs(self.results_dir, exist_ok=True)
        json_path = os.path.join(self.results_dir, f"{self.exp_id}.json")
        save_json(self.config, json_path)
        latest_path = os.path.join(self.results_dir, "latest.json")
        try:
            if os.path.islink(latest_path) or os.path.exists(latest_path):
                os.remove(latest_path)
            if os.name != "nt":
                os.symlink(os.path.basename(json_path), latest_path)
            else:
                import shutil; shutil.copy2(json_path, latest_path)
        except OSError:
            import shutil; shutil.copy2(json_path, latest_path)
        base_cols = [
            "exp_id","csv_filename","total_time_sec","final_student_acc","num_classes","batch_size",
            "ce_alpha","kd_alpha","use_ib","ib_beta","ib_epochs_per_stage","use_cccp",
            "optimizer","student_lr","student_weight_decay",
        ]
        epoch_cols = [k for k in self.config.keys() if k.startswith(("student_ep","teacher_ep"))]
        fieldnames = base_cols + sorted(epoch_cols)
        self.config["csv_filename"] = "summary.csv"
        csv_path = os.path.join(self.results_dir, self.config["csv_filename"]) 
        save_csv_row(self.config, csv_path, fieldnames, write_header_if_new=True)
```

## 18) 모델 빌더 / IB_MBM 빌더 (발췌)
```python
from models import build_ib_mbm_from_teachers as build_from_teachers

ib_mbm, synergy_head = build_from_teachers([teacher1, teacher2], cfg, query_dim=cfg.get("ib_mbm_query_dim"))
# 내부 검사: use_distillation_adapter가 꺼져 있으면 teacher feature dims 동일해야 함
# q_dim 미지정 시 오류. out_dim/n_head/clip/min_std 등 cfg로 제어 가능
```

## 19) 설정 예시 모음(발췌)
```yaml
# L2_cccp.yaml 중요 키만 발췌
experiment:
  results_dir: experiments/ablation/cccp/results
  exp_id: L2_cccp
  num_stages: 2
  student_epochs_per_stage: [20, 20]
  teacher_adapt_epochs: 12
  use_amp: true
  amp_dtype: bfloat16
  # IB
  use_ib: true
  ib_epochs_per_stage: 12
  ib_beta: 0.00005
  ib_beta_warmup_epochs: 8
  # Adapter/IB_MBM
  use_distillation_adapter: true
  distill_out_dim: 512
  ib_mbm_query_dim: 512
  ib_mbm_out_dim: 512
  ib_mbm_n_head: 4
  # CCCP
  use_cccp: true
  cccp_alpha: 0.25
  tau: 4.0
  cccp_nt: 1
  cccp_ns: 1
  # Optim/Sched
  optimizer: adamw
  student_lr: 0.001
  student_weight_decay: 0.0003
  grad_clip_norm: 0.5
  a_step_lr: 0.0001
  # KD
  ce_alpha: 0.65
  kd_alpha: 0.35
  kd_ens_alpha: 0.0
  kd_max_ratio: 2.0
  tau_schedule: [2.0, 6.0]
  teacher_adapt_kd_warmup: 0
  use_loss_clamp: false
```

## 20) 실행 스크립트 아웃라인(비교 실험)
```python
# run/run_asib_cl_experiment.py 발췌
configs = create_comparison_configs()
results = {}
results["asib_cl"] = run_experiment("ASIB-CL", configs["asib_cl"])  # PyCIL 기반
for method, cfg_path in configs.items():
    if method == "asib_cl":
        continue
    results[method] = run_experiment(method.upper(), cfg_path)
# 결과 요약 로그 출력
```

## 21) 재현성/체크포인트/프로파일링 팁
- 재현성
  - 시드 고정: 데이터/모델 생성 직후 `set_random_seed(cfg["seed"], deterministic=True)` 호출
  - CUDA 결정론 설정(필요 시): `torch.backends.cudnn.deterministic=True`, `benchmark=False`
- 체크포인트
  - 베스트 스냅샷: B‑Step에서 `best_state = deepcopy(student.state_dict())` 저장 후 마지막에 `load_state_dict`
  - 메타 저장: `ExperimentLogger.save_meta(meta_dict)`로 주요 하이퍼·모델명·성능 한 번 더 기록
- 프로파일링
  - 채널‑라스트: 입력 4D이면 `x = x.to(memory_format=torch.channels_last)` 적용으로 Conv 가속
  - AMP dtype 전환: 불안정 시 bfloat16 ↔ float16 비교, 필요 시 AMP off
  - DataLoader 튜닝: `pin_memory=True`, `persistent_workers=True`, `num_workers` 조절
  - nvidia-smi/torch.cuda.memory_summary로 메모리 피크 확인, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

## 22) CL 모드(참고)
- `run_continual_learning(...)`에서 `get_split_cifar100_loaders`로 태스크 분할, `ReplayBuffer/EWC` 통합 가능
- 태스크별 학습 후 `eval_student`로 평가, `ExperimentLogger.update_metric("task{i}_acc")`

## 23) 자주 나는 이슈와 즉시 점검
- dims mismatch
  - 증상: IB_MBM KV 스택에서 shape 오류
  - 조치: `use_distillation_adapter=true`, `distill_out_dim`을 교사간 동일하게, `feat_kd_key` 체크
- q_dim mismatch
  - 증상: `q_proj.in_features != s_feat.shape[1]`
  - 조치: `auto_set_ib_mbm_query_dim_with_model` 호출 또는 `ib_mbm_query_dim` 명시
- CCCP 영향 과대/과소
  - 조치: ablation에서는 `cccp_alpha`만 조절, 0.25/0.5 그리드. 필요 시 `cccp_nt/ns=2`
- IB 과규제
  - 조치: `ib_beta` 워밍업(`ib_beta_warmup_epochs`) 사용, `ib_mbm_out_dim`을 `distill_out_dim`과 정합

(끝)

---

## 24) Full StudentDistillationUpdate (B‑Step) – Full Source Excerpt
```python
def student_distillation_update(
    teacher_wrappers,
    ib_mbm, synergy_head,
    student_model,
    trainloader,
    testloader,
    cfg,
    logger,
    optimizer,
    scheduler,
    global_ep: int = 0
):
    """Train the student model via knowledge distillation.

    The teachers and IB_MBM are frozen to generate synergy logits while the
    student is optimized using a combination of cross-entropy and KD losses.
    If the IB_MBM operates in query mode, the optional feature-level KD term
    aligns the student query with the teacher attention output in the IB_MBM
    latent space.
    """
    # 1) freeze teacher + ib_mbm
    teacher_reqgrad_states = []
    teacher_train_states = []
    for tw in teacher_wrappers:
        teacher_train_states.append(tw.training)
        states = []
        for p in tw.parameters():
            states.append(p.requires_grad)
            p.requires_grad = False
        teacher_reqgrad_states.append(states)
        tw.eval()

    ib_mbm_reqgrad_states = []
    ib_mbm_train_state = ib_mbm.training
    for p in ib_mbm.parameters():
        ib_mbm_reqgrad_states.append(p.requires_grad)
        p.requires_grad = False
    ib_mbm.eval()

    syn_reqgrad_states = []
    syn_train_state = synergy_head.training
    for p in synergy_head.parameters():
        syn_reqgrad_states.append(p.requires_grad)
        p.requires_grad = False
    synergy_head.eval()

    # ------------------------------------------------------------
    # 학생 Epoch 결정 로직 – schedule 우선, 없으면 student_iters 만 사용
    # ------------------------------------------------------------
    if "student_epochs_schedule" in cfg:
        cur_stage_idx = int(cfg.get("cur_stage", 1)) - 1   # 0-base
        try:
            student_epochs = int(cfg["student_epochs_schedule"][cur_stage_idx])
        except (IndexError, ValueError, TypeError):
            raise ValueError(
                "[trainer_student] student_epochs_schedule가 "
                f"stage {cur_stage_idx+1} 에 대해 정의돼 있지 않습니다."
            )
    else:
        student_epochs = int(cfg.get("student_iters", 1))   # 최후 fallback

    # ──────────────────────────────────────────────────────────
    stage_meter = StageMeter(cfg.get("cur_stage", 1), logger, cfg, student_model)
    best_acc = 0.0
    best_state = copy.deepcopy(student_model.state_dict())

    logger.info(f"[StudentDistill] Using student_epochs={student_epochs}")

    # B-Step 시작 전 안전 복원 (A-Step에서 freeze된 경우 대비)
    for p in student_model.parameters():
        p.requires_grad_(True)
    student_model.train()
    
    # B-Step 시작 전 trainable 파라미터 수 기록
    s_train = sum(p.requires_grad for p in student_model.parameters())
    s_total = sum(1 for p in student_model.parameters())
    logger.info(f"[PPF][B-Step] student trainable: {s_train}/{s_total} ({100*s_train/s_total:.1f}%)")

    autocast_ctx, scaler = get_amp_components(cfg)
    # ---------------------------------------------------------
    # IB_MBM forward
    # ---------------------------------------------------------
    # no attention weights returned in simplified IB_MBM
    for ep in range(student_epochs):
        if scheduler is not None and hasattr(scheduler, "T_max"):
            total_epochs = scheduler.T_max
        else:
            total_epochs = cfg.get("total_epochs", 1)
        cur_tau = get_tau(
            cfg,
            epoch=global_ep + ep,
            total_epochs=total_epochs,
        )
        distill_loss_sum = 0.0
        cnt = 0
        student_model.train()

        mix_mode = (
            "cutmix"
            if cfg.get("cutmix_alpha_distill", 0.0) > 0.0
            else "mixup" if cfg.get("mixup_alpha", 0.0) > 0.0
            else "none"
        )

        for step, (x, y) in enumerate(
            smart_tqdm(trainloader, desc=f"[StudentDistill ep={ep+1}]")
        ):
            x, y = x.to(cfg["device"], non_blocking=True), y.to(cfg["device"], non_blocking=True)
            
            # 채널-라스트 포맷 적용 (Conv 연산 가속)
            if cfg.get("use_channels_last", True) and x.dim() == 4:
                x = x.to(memory_format=torch.channels_last)

            if mix_mode == "cutmix":
                x_mixed, y_a, y_b, lam = cutmix_data(
                    x, y, alpha=cfg["cutmix_alpha_distill"]
                )
            elif mix_mode == "mixup":
                x_mixed, y_a, y_b, lam = mixup_data(x, y, alpha=cfg["mixup_alpha"])
            else:
                x_mixed, y_a, y_b, lam = x, y, y, 1.0

            with autocast_ctx:
                # (A) Student forward (query)
                feat_dict, s_logit, _ = student_model(x_mixed)

                # 교사 모델이 필요한지 확인 (KD, Feature KD, IB 중 하나라도 사용하는 경우)
                need_teachers = (cfg.get("kd_alpha", 0.0) > 0.0) or \
                               (cfg.get("feat_kd_alpha", 0.0) > 0.0) or \
                               bool(cfg.get("use_ib", False))
                
                if need_teachers:
                    with torch.no_grad():
                        t1_out = teacher_wrappers[0](x_mixed)
                        t2_out = teacher_wrappers[1](x_mixed)
                        t1_dict = t1_out[0] if isinstance(t1_out, tuple) else t1_out
                        t2_dict = t2_out[0] if isinstance(t2_out, tuple) else t2_out

                        feat_key = "distill_feat" if cfg.get("use_distillation_adapter", False) \
                                   else "feat_2d"
                        f1_2d = t1_dict[feat_key]
                        f2_2d = t2_dict[feat_key]
                else:
                    # 교사 모델 forward를 스킵하는 경우 None으로 설정
                    t1_dict = t2_dict = None
                    f1_2d = f2_2d = None

                feat_kd_key = cfg.get("feat_kd_key", "feat_2d")
                s_feat = feat_dict[feat_kd_key]
                
                # 첫 배치에서 feat_kd_key와 차원 확인
                if ep == 0 and step == 0:
                    logging.info(f"[B-Step] feat_kd_key={feat_kd_key}, s_feat.shape={s_feat.shape}")
                
                # IB/KD 타깃이 필요할 때만 IB_MBM 실행 (need_teachers와 동일한 조건)
                need_ibm = need_teachers
                if need_ibm:
                    # 스택 전 shape 검증 (차원 불일치 조기 발견)
                    if f1_2d.shape[1] != f2_2d.shape[1]:
                        logging.error(f"[IB-MBM] KV dim mismatch: f1={f1_2d.shape}, f2={f2_2d.shape}. Check distill_out_dim and adapters.")
                        raise RuntimeError("KV dim mismatch")
                    
                    with torch.no_grad():
                        syn_feat_ng, mu_ng, logvar_ng = ib_mbm(
                            s_feat, torch.stack([f1_2d, f2_2d], dim=1)
                        )
                        zsyn_ng = synergy_head(syn_feat_ng)
                else:
                    mu_ng = logvar_ng = zsyn_ng = None

                # stable CE/KL calculations in float32
                ls = cfg.get("ce_label_smoothing", cfg.get("label_smoothing", 0.0))
                loss_ce = ce_safe(
                    s_logit,
                    y,
                    ls_eps=ls,
                )
                # KD는 no-grad 타깃 사용 (필요할 때만)
                kd_loss_val = 0.0
                if cfg.get("kd_alpha", 0.0) > 0 and need_teachers:
                    # KD 타겟 선택
                    kd_target = cfg.get("kd_target", "synergy")
                    if kd_target == "synergy" and zsyn_ng is not None:
                        kd_tgt = zsyn_ng
                    elif kd_target == "avg" and t1_dict is not None and t2_dict is not None:
                        # Teacher 평균
                        t1_logit = t1_dict["logit"]
                        t2_logit = t2_dict["logit"]
                        kd_tgt = (t1_logit + t2_logit) / 2.0
                    elif kd_target == "weighted_conf" and t1_dict is not None and t2_dict is not None:
                        # Teacher confidence 기반 가중 평균
                        t1_logit = t1_dict["logit"]
                        t2_logit = t2_dict["logit"]
                        p1 = torch.softmax(t1_logit / cur_tau, dim=1).amax(dim=1).values
                        p2 = torch.softmax(t2_logit / cur_tau, dim=1).amax(dim=1).values
                        w1 = (p1 / (p1 + p2 + 1e-8)).unsqueeze(1)
                        w2 = 1.0 - w1
                        kd_tgt = w1 * t1_logit + w2 * t2_logit
                    else:
                        # 기본값: synergy
                        kd_tgt = zsyn_ng if zsyn_ng is not None else torch.zeros_like(s_logit)

                    # Warmup 동안 시너지/앙상블 혼합 (ens_alpha 비율)
                    warmup_epochs = int(cfg.get("teacher_adapt_kd_warmup", 0))
                    ens_alpha = float(cfg.get("kd_ens_alpha", 0.0))
                    if (
                        warmup_epochs > 0 and ep < warmup_epochs and ens_alpha > 0.0
                        and zsyn_ng is not None and t1_dict is not None and t2_dict is not None
                    ):
                        t1_logit = t1_dict["logit"]
                        t2_logit = t2_dict["logit"]
                        avg_t = (t1_logit + t2_logit) / 2.0
                        kd_tgt = (1.0 - ens_alpha) * zsyn_ng + ens_alpha * avg_t
                    
                    if kd_tgt is not None:
                        loss_kd = kl_safe(
                            s_logit,
                            kd_tgt.detach(),  # gradient 끊김
                            tau=cur_tau,
                        )
                        kd_loss_val = loss_kd
                    else:
                        loss_kd = torch.tensor(0.0, device=s_logit.device)
                        kd_loss_val = loss_kd
                else:
                    loss_kd = torch.tensor(0.0, device=s_logit.device)
                    kd_loss_val = loss_kd

                # ---- DEBUG: 첫 batch 모양 확인 ----
                if ep == 0 and step == 0 and cfg.get("debug_verbose", False):
                    logging.debug(
                        "[DBG/student] x %s s_logit %s zsyn %s",
                        tuple(x.shape),
                        tuple(s_logit.shape),
                        tuple(zsyn_ng.shape) if 'zsyn_ng' in locals() and zsyn_ng is not None else None,
                    )

            # ── (B1) sample-weights (always same dtype as losses) ───────────
            if cfg.get("use_disagree_weight", False) and need_teachers and t1_dict is not None and t2_dict is not None:
                weights = sample_weights_from_disagreement(
                    t1_dict["logit"],
                    t2_dict["logit"],
                    y,
                    mode=cfg.get("disagree_mode", "pred"),
                    lambda_high=cfg.get("disagree_lambda_high", 1.0),
                    lambda_low=cfg.get("disagree_lambda_low", 1.0),
                )
            else:
                weights = torch.ones_like(y, dtype=s_logit.dtype, device=y.device)

            # AMP / float16 환경에서도 안전
            weights = weights.to(s_logit.dtype)

            # cw 기본값 설정 (안전 가드)
            cw = torch.ones(y.size(0), device=s_logit.device, dtype=s_logit.dtype)

            if cfg.get("use_ib", False) and logvar_ng is not None:
                # 큰 logvar_ng(불확실) → 작은 weight
                cw = torch.exp(-logvar_ng).mean(dim=1).clamp(
                    cfg.get("min_cw", 0.1), 1.0
                ).to(s_logit.dtype)
                weights = weights * cw

            # apply sample weights to CE and KD losses computed above
            ce_loss_val = loss_ce
            kd_loss_val = loss_kd

            # --- μ‑MSE with certainty weight ---------------------------------
            feat_kd_val = None
            if cfg.get("feat_kd_alpha", 0.0) > 0 and need_teachers and mu_ng is not None:
                # 차원 불일치 안전장치
                if s_feat.shape[1] == mu_ng.shape[1]:
                    diff = (s_feat - mu_ng).pow(2).sum(dim=1)  # s_feat에서 grad 흐름
                    cw = certainty_weights(logvar).mean(dim=1).to(s_feat.dtype)
                    feat_kd_val = (cw * diff).mean()
                else:
                    # 첫 배치에서만 경고 출력 (로그 과다 방지)
                    if ep == 0 and step == 0:
                        logging.warning(
                            "[B-Step] Feat-KD skipped (dim mismatch): s=%s, mu=%s",
                            tuple(s_feat.shape),
                            tuple(mu.shape),
                        )

            # IB KL은 B-Step에서 제외 (A-Step에서만 최적화)
            ib_loss_val = None
            # if cfg.get("use_ib_on_student", False):
            #     ib_loss_val = ib_loss(mu_ng, logvar_ng, beta=get_beta(cfg, global_ep + ep))
            
            # 최종 loss는 조건부 합 (상수 0 텐서 더하지 않기)
            loss = _get_cfg_val(cfg, "ce_alpha", 1.0) * loss_ce
            if cfg.get("kd_alpha", 0.0) > 0 and zsyn_ng is not None:
                loss = loss + cfg["kd_alpha"] * kd_loss_val
            if feat_kd_val is not None:
                loss = loss + cfg.get("feat_kd_alpha", 0.0) * feat_kd_val
            
            # 추가 안전장치: loss가 너무 크면 clipping (클리핑 완화)
            if cfg.get("use_loss_clamp", False):
                loss = torch.clamp(loss, 0.0, cfg.get("loss_clamp_max", 1000.0))
            else:
                # 기본적으로는 클리핑 해제 (안정화를 위해)
                pass
            
            # loss.requires_grad 보증 (디버깅용)
            if not loss.requires_grad:
                logging.error("loss grad off: ce=%s kd=%s feat=%s", 
                            loss_ce.requires_grad, loss_kd.requires_grad, 
                            feat_kd_val.requires_grad if feat_kd_val is not None else None)

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                if cfg.get("grad_clip_norm", 0) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        student_model.parameters(), cfg["grad_clip_norm"]
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg.get("debug_verbose", False):
                    logging.debug(
                        "[StudentDistill] batch loss=%.4f", loss.item()
                    )
                if cfg.get("grad_clip_norm", 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        student_model.parameters(), cfg["grad_clip_norm"]
                    )
                optimizer.step()

            bs = x.size(0)
            distill_loss_sum += loss.item() * bs
            cnt += bs
            stage_meter.step(bs)

        ep_loss = distill_loss_sum / cnt

        # (C) validate
        test_acc = eval_student(student_model, testloader, cfg["device"], cfg)

        logging.info(
            "[StudentDistill ep=%d] loss=%.4f testAcc=%.2f best=%.2f",
            ep + 1,
            ep_loss,
            test_acc,
            best_acc,
        )
        if wandb and wandb.run:
            wandb.log({
                "student/loss": ep_loss,
                "student/acc": test_acc,
                "student/epoch": global_ep + ep + 1,
            })

        # ── NEW: per-epoch logging ───────────────────────────────
        logger.update_metric(f"student_ep{ep+1}_acc", test_acc)
        logger.update_metric(f"student_ep{ep+1}_loss", ep_loss)
        logger.update_metric(f"ep{ep+1}_mix_mode", mix_mode)
        logger.update_metric(f"epoch{global_ep+ep+1}_tau", cur_tau)

        if scheduler is not None:
            scheduler.step()

        # (E) best snapshot
        if test_acc > best_acc:
            best_acc = test_acc
            best_state = copy.deepcopy(student_model.state_dict())

    student_model.load_state_dict(best_state)

    # restore original requires_grad and training states
    for tw, states, train_flag in zip(teacher_wrappers, teacher_reqgrad_states, teacher_train_states):
        for p, rg in zip(tw.parameters(), states):
            p.requires_grad = rg
        tw.train(train_flag)
    for p, rg in zip(ib_mbm.parameters(), ib_mbm_reqgrad_states):
        p.requires_grad = rg
    ib_mbm.train(ib_mbm_train_state)
    for p, rg in zip(synergy_head.parameters(), syn_reqgrad_states):
        p.requires_grad = rg
    synergy_head.train(syn_train_state)

    stage_meter.finish(best_acc)
    logger.info(f"[StudentDistill] bestAcc={best_acc:.2f}")
    return best_acc
```

## 25) Full TeacherAdaptiveUpdate (A‑Step) – Full Source Excerpt
```python
def teacher_adaptive_update(
    teacher_wrappers,
    ib_mbm,
    synergy_head,
    student_model,
    trainloader,
    testloader,
    cfg,
    logger,
    optimizer,
    scheduler,
    global_ep: int = 0,
):
    """
    - ``teacher_wrappers``: list containing ``teacher1`` and ``teacher2``.
    - ``ib_mbm`` and ``synergy_head``: assume partial freezing has been applied.
    - ``student_model``: kept fixed for knowledge distillation.
    - ``testloader``: optional loader used to evaluate synergy accuracy.
    """
    # 교사 백본 freeze 보장 (미세조정 OFF일 때)
    use_tf = bool(cfg.get("use_teacher_finetuning", False))
    only_da = cfg.get("train_distill_adapter_only", False)
    
    teacher_params = []
    for tw in teacher_wrappers:
        if not use_tf:
            # 교사 백본 고정 (기본값)
            for p in tw.parameters():
                p.requires_grad = False
            tw.eval()  # 교사를 eval 모드로 설정
        
        # optimizer에 추가할 파라미터 선택 (이미 optimizer에서 처리하지만 여기서도 확인)
        if use_tf:
            # 교사 전체 미세조정 허용
            for p in tw.parameters():
                if p.requires_grad:
                    teacher_params.append(p)
        elif only_da and hasattr(tw, "distillation_adapter"):
            # adapter만 학습
            for p in tw.distillation_adapter.parameters():
                p.requires_grad = True  # adapter는 학습 가능하게
                teacher_params.append(p)
        # else: 교사 백본은 학습하지 않음
    ib_mbm_params = [p for p in ib_mbm.parameters() if p.requires_grad]
    syn_params = [p for p in synergy_head.parameters() if p.requires_grad]

    teacher_epochs = int(cfg.get("teacher_iters", cfg.get("teacher_adapt_epochs", 5)))

    best_synergy = -1
    best_state = {
        "teacher_wraps": [_cpu_state_dict(tw) for tw in teacher_wrappers],
        "ib_mbm": _cpu_state_dict(ib_mbm),
        "syn_head": _cpu_state_dict(synergy_head),
    }

    # 추가 검증 로직: learning rate 조정을 위한 상태
    prev_obj = float("inf")
    backup_state = {
        "teacher_wraps": [_cpu_state_dict(tw) for tw in teacher_wrappers],
        "ib_mbm": _cpu_state_dict(ib_mbm),
        "syn_head": _cpu_state_dict(synergy_head),
    }

    # A-Step 시작 로깅
    stage_info = f"[Stage {cfg.get('cur_stage', '?')}]" if 'cur_stage' in cfg else ""
    logger.info(f"{stage_info} A-Step (Teacher/IB) start - teacher_epochs={teacher_epochs}")
    
    # teacher_epochs=0인 경우 early return
    if teacher_epochs == 0:
        logger.info(f"{stage_info} A-Step (Teacher/IB) skipped - teacher_epochs=0")
        return 0.0  # 기본값 반환
    
    # student를 eval 모드로 고정하고 requires_grad=False
    if student_model is not None:
        student_model.eval()
        for p in student_model.parameters():
            p.requires_grad_(False)
    
    # ib_mbm, synergy_head만 train 모드로 (교사는 use_tf에 따라)
    if use_tf:
        # 교사 미세조정 모드에서만 train 모드로
        for tw in teacher_wrappers:
            tw.train()
    else:
        # 기본값: 교사는 eval 모드 유지
        for tw in teacher_wrappers:
            tw.eval()
    ib_mbm.train()
    synergy_head.train()

    autocast_ctx, scaler = get_amp_components(cfg)
    
    # teacher_epochs=0인 경우를 위한 기본값 설정
    teacher_loss_sum = 0.0
    count = 0
    
    for ep in range(teacher_epochs):
        # 교사 모델 모드 설정
        if use_tf:
            for tw in teacher_wrappers:
                tw.train()
        else:
            for tw in teacher_wrappers:
                tw.eval()
        ib_mbm.train()
        synergy_head.train()
        if student_model is not None:
            student_model.eval()
        if scheduler is not None and hasattr(scheduler, "T_max"):
            total_epochs = scheduler.T_max
        else:
            total_epochs = cfg.get("total_epochs", 1)
        cur_tau = get_tau(
            cfg,
            epoch=global_ep + ep,
            total_epochs=total_epochs,
        )
        teacher_loss_sum = 0.0
        count = 0

        for step, batch in enumerate(
            smart_tqdm(trainloader, desc=f"[TeacherAdaptive ep={ep+1}]")
        ):
            x, y = batch
            x, y = x.to(cfg["device"], non_blocking=True), y.to(cfg["device"], non_blocking=True)
            
            # 채널-라스트 포맷 적용 (Conv 연산 가속)
            if cfg.get("use_channels_last", True) and x.dim() == 4:
                x = x.to(memory_format=torch.channels_last)

            with autocast_ctx:
                # (A) Student features and logits (kept fixed)
                with torch.no_grad():
                    feat_dict, s_logit, _ = student_model(x)
                    key = cfg.get("feat_kd_key", "feat_2d")
                    s_feat = feat_dict[key]

                # (B) Teacher features
                feats_2d = []
                feat_key = (
                    "distill_feat"
                    if cfg.get("use_distillation_adapter", False)
                    else "feat_2d"
                )
                t1_dict = None
                for i, tw in enumerate(teacher_wrappers):
                    out = tw(x)
                    t_dict = out[0] if isinstance(out, tuple) else out
                    if i == 0:
                        t1_dict = t_dict
                    feat = t_dict[feat_key]
                    feats_2d.append(feat)
                    
                    # 차원 확인 로그 (첫 배치에서만)
                    if ep == 0 and step == 0:
                        logging.info(f"Teacher {i} {feat_key} shape: {feat.shape}")

                # (C) IB_MBM + synergy_head (IB_MBM only)
                # 차원 불일치 안전장치
                if len(feats_2d) > 0:
                    target_dim = feats_2d[0].size(1)
                    for i, feat in enumerate(feats_2d):
                        if feat.size(1) != target_dim:
                            logging.warning(f"Teacher {i} feature dim mismatch: {feat.size(1)} vs {target_dim}")
                            # 차원을 맞춰주기 위해 projection 추가 (필요 시)
                            if not hasattr(ib_mbm, f'feat_proj_{i}'):
                                setattr(ib_mbm, f'feat_proj_{i}', 
                                       nn.Linear(feat.size(1), target_dim).to(feat.device))
                            proj_layer = getattr(ib_mbm, f'feat_proj_{i}')
                            feats_2d[i] = proj_layer(feat)
                
                # 스택/검증 직전
                if len(feats_2d) < 2:
                    logging.error("[A-Step] need 2 teacher feats, got %d", len(feats_2d))
                    raise RuntimeError("Not enough teacher features")

                t1, t2 = feats_2d[0], feats_2d[1]
                if t1.shape[1] != t2.shape[1]:
                    logging.error("[A-Step] KV dim mismatch: t1=%s, t2=%s. Check use_distillation_adapter/distill_out_dim.",
                                  tuple(t1.shape), tuple(t2.shape))
                    raise RuntimeError("KV dim mismatch")

                # (선택) 쿼리 q_dim도 점검
                if getattr(ib_mbm.q_proj, "in_features", None) != s_feat.shape[1]:
                    logging.error("[A-Step] q_dim mismatch: q_in=%s, s_feat=%s",
                                  getattr(ib_mbm.q_proj, "in_features", None), s_feat.shape[1])
                    raise RuntimeError("Q dim mismatch")

                # A-Step에서 shape 체크 (첫 배치에서만)
                if ep == 0 and step == 0:
                    logging.info("[A-Step] t1=%s, t2=%s, s_feat=%s, q_in=%s",
                                 tuple(t1.shape), tuple(t2.shape), tuple(s_feat.shape), 
                                 getattr(ib_mbm.q_proj, "in_features", None))

                kv = torch.stack([t1, t2], dim=1)
                syn_feat, mu, logvar = ib_mbm(s_feat, kv)
                ib_loss_val = 0.0
                if cfg.get("use_ib", False):
                    mu, logvar = mu.float(), logvar.float()
                    ib_beta = get_beta(cfg, global_ep + ep)
                    ib_loss_val = ib_loss(mu, logvar, beta=ib_beta)
                fsyn = syn_feat
                zsyn = synergy_head(fsyn)

                # (D) compute loss (KL + synergyCE)
                if cfg.get("use_ib", False) and isinstance(ib_mbm, IB_MBM):
                    ce_vec = ce_loss_fn(
                        zsyn,
                        y,
                        label_smoothing=cfg.get("label_smoothing", 0.0),
                        reduction="none",
                    )
                    kd_vec = kd_loss_fn(zsyn, s_logit, T=cur_tau, reduction="none").sum(
                        dim=1
                    )

                    # ---- DEBUG: 첫 batch 모양 확인 ----
                    if ep == 0 and step == 0 and cfg.get("debug_verbose", False):
                        logging.debug(
                            "[DBG/teacher] t1_logit %s s_logit %s zsyn %s",
                            tuple(t1_dict["logit"].shape),
                            tuple(s_logit.shape),
                            tuple(zsyn.shape),
                        )
                    cw = certainty_weights(logvar).mean(dim=1).to(zsyn.dtype)
                    loss_ce = (cw * ce_vec).mean()
                    loss_kd = (cw * kd_vec).mean()
                else:
                    loss_kd = kd_loss_fn(zsyn, s_logit, T=cur_tau)
                    loss_ce = ce_loss_fn(
                        zsyn,
                        y,
                        label_smoothing=cfg.get("label_smoothing", 0.0),
                    )

                # ① 누락 시 기본값 0.6
                synergy_weight = cfg.get("synergy_ce_alpha", 0.6)
                synergy_ce_loss = synergy_weight * loss_ce

                # Concave‑Convex surrogate (CCCP)
                kd_cccp = 0.0
                if cfg.get("use_cccp", True):
                    with torch.no_grad():
                        s_out = student_model(x)[1] if isinstance(student_model(x), tuple) else student_model(x)
                    tau  = float(cfg.get("tau", 4.0))
                    kd_w = float(cfg.get("kd_alpha", 0.0))
                    kd_cccp = (
                        kd_w * tau * tau *
                        F.kl_div(
                            F.log_softmax(zsyn / tau, dim=1),
                            F.softmax(s_out.detach() / tau, dim=1),
                            reduction="batchmean",
                        )
                    )

                feat_kd_loss = torch.tensor(0.0, device=cfg["device"])
                if cfg.get("feat_kd_alpha", 0) > 0:
                    dims_match = s_feat.shape[1] == mu.shape[1]
                    first_batch = (ep == 0 and step == 0)
                    if dims_match:
                        diff = (s_feat - mu).pow(2).sum(dim=1)
                        cw = certainty_weights(logvar).mean(dim=1).to(s_feat.dtype)
                        feat_kd_loss = (cw * diff).mean()
                    else:
                        if first_batch:
                            logging.warning(
                                "[A-Step] Feat-KD skipped (dim mismatch): s=%s, mu=%s",
                                tuple(s_feat.shape),
                                tuple(mu.shape),
                            )

                # ---- (1) 전체 손실 구성 ----
                kd_weight = cfg.get("teacher_adapt_alpha_kd", cfg.get("kd_alpha", 1.0))

                reg_loss = (
                    torch.stack([(p ** 2).mean() for p in teacher_params]).mean()
                    if teacher_params
                    else torch.tensor(0.0, device=cfg["device"])
                )
                ib_mbm_reg_params = ib_mbm_params + syn_params
                ib_mbm_reg_loss = (
                    torch.stack([(p ** 2).mean() for p in ib_mbm_reg_params]).mean()
                    if ib_mbm_reg_params
                    else torch.tensor(0.0, device=cfg["device"])
                )

                total_loss_step = (
                    kd_weight * loss_kd
                    + synergy_ce_loss
                    + cfg.get("feat_kd_alpha", 0) * feat_kd_loss
                    + ib_loss_val
                    + float(cfg.get("reg_lambda", 0.0)) * reg_loss
                    + float(cfg.get("ib_mbm_reg_lambda", 0.0)) * ib_mbm_reg_loss
                    + kd_cccp
                )
            # loss clamp (옵션)
            if cfg.get("use_loss_clamp", False):
                total_loss_step = torch.clamp(total_loss_step, 0.0, cfg.get("loss_clamp_max", 1000.0))

            # ---- (2) per-batch Optim ----
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(total_loss_step).backward()
                if cfg.get("grad_clip_norm", 0) > 0:
                    scaler.unscale_(optimizer)
                    grad_params = teacher_params + ib_mbm_params + syn_params
                    if grad_params:
                        torch.nn.utils.clip_grad_norm_(
                            grad_params,
                            cfg["grad_clip_norm"],
                        )
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss_step.backward()
                if cfg.get("grad_clip_norm", 0) > 0:
                    grad_params = teacher_params + ib_mbm_params + syn_params
                    if grad_params:
                        torch.nn.utils.clip_grad_norm_(
                            grad_params,
                            cfg["grad_clip_norm"],
                        )
                optimizer.step()

            count += x.size(0)
            teacher_loss_sum += float(total_loss_step.detach())

        # (E) evaluate synergy (optional)
        if testloader is not None:
            try:
                syn_acc = eval_synergy(teacher_wrappers, ib_mbm, synergy_head, testloader, device=cfg["device"], cfg=cfg, student_model=student_model)
                best_synergy = max(best_synergy, syn_acc)
                logger.update_metric(f"teacher_ep{ep+1}_syn_acc", syn_acc)
                if syn_acc > best_synergy:
                    best_state = {
                        "teacher_wraps": [_cpu_state_dict(tw) for tw in teacher_wrappers],
                        "ib_mbm": _cpu_state_dict(ib_mbm),
                        "syn_head": _cpu_state_dict(synergy_head),
                    }
            except Exception as e:
                logging.warning(f"[A-Step] eval_synergy failed: {e}")

        if scheduler is not None:
            scheduler.step()

    # restore if improved
    # (실제 코드에서는 필요 시 best_state를 로드하거나 외부에서 관리)
    logger.info(f"[TeacherAdaptive] best_synergy={best_synergy:.2f}")
    return float(best_synergy)
```

## 26) Data Loader Excerpts
```python
# data.cifar100.get_cifar100_loaders (발췌)
def get_cifar100_loaders(root="./data", batch_size=128, num_workers=2, augment=True):
    """
    CIFAR-100 size = (32x32) - NPZ 파일 사용 (ImageNet-32와 동일한 형식)
    Returns: train_loader, test_loader
    """
    ...

# data.imagenet32.get_imagenet32_loaders (발췌)
def get_imagenet32_loaders(root, batch_size=128, num_workers=2, augment=True):
    """
    ImageNet-32 size = (32x32) - 강화된 증강 적용
    Returns: train_loader, test_loader
    """
    ...
```

## 27) AMP Helper (Full)
```python
def get_amp_components(cfg):
    use_amp = bool(cfg.get("use_amp", False))
    device = cfg.get("device", "cuda")
    if not use_amp:
        from contextlib import nullcontext
        return nullcontext(), None
    amp_dtype = cfg.get("amp_dtype", "float16")
    dtype = torch.float16 if amp_dtype == "float16" else torch.bfloat16
    autocast = torch.autocast if hasattr(torch, "autocast") else torch.cuda.amp.autocast
    autocast_ctx = autocast("cuda", dtype=dtype)
    GradScaler = torch.amp.GradScaler if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler
    try:
        scaler = GradScaler(device="cuda", init_scale=int(cfg.get("grad_scaler_init_scale", 1024)))
    except TypeError:
        scaler = GradScaler(init_scale=int(cfg.get("grad_scaler_init_scale", 1024)))
    return autocast_ctx, scaler
```

## 28) IB_MBM Builder (Full)
```python
def build_ib_mbm_from_teachers(teachers: List[nn.Module], cfg: dict, query_dim: Optional[int] = None) -> Tuple[IB_MBM, SynergyHead]:
    use_da = bool(cfg.get("use_distillation_adapter", False))
    feat_dims = [ (t.distill_dim if use_da and hasattr(t, "distill_dim") else t.get_feat_dim()) for t in teachers ]
    if not use_da:
        unique_dims = set(int(d) for d in feat_dims)
        if len(unique_dims) > 1:
            raise ValueError("Teacher feature dims differ. Enable use_distillation_adapter to align dimensions.")
    qdim = cfg.get("ib_mbm_query_dim") or query_dim
    if not qdim:
        raise ValueError("`ib_mbm_query_dim` must be specified for IB_MBM.")
    ib_mbm = IB_MBM(
        q_dim=qdim,
        kv_dim=max(feat_dims),
        d_emb=cfg.get("ib_mbm_out_dim", 512),
        beta=cfg.get("ib_beta", 1e-2),
        n_head=cfg.get("ib_mbm_n_head", 1),
        logvar_clip=cfg.get("ib_mbm_logvar_clip", 10),
        min_std=cfg.get("ib_mbm_min_std", 1e-4),
    )
    head = SynergyHead(
        in_dim=cfg.get("ib_mbm_out_dim", 512),
        num_classes=cfg.get("num_classes", 100),
        p=cfg.get("synergy_head_dropout", cfg.get("ib_mbm_dropout", 0.0)),
    )
    return ib_mbm, head
```

## 29) Full Config Examples (Extra)
```yaml
# L1_ib.yaml (예시)
experiment:
  results_dir: experiments/ablation/ib/results
  exp_id: L1_ib
  dataset: { batch_size: 64, num_workers: 4, data_aug: 1 }
  num_stages: 2
  student_epochs_per_stage: [20, 20]
  teacher_adapt_epochs: 0
  use_amp: true
  amp_dtype: bfloat16
  use_ib: true
  ib_epochs_per_stage: 12
  ib_beta: 0.001
  ib_beta_warmup_epochs: 5
  use_distillation_adapter: true
  distill_out_dim: 512
  ib_mbm_query_dim: 512
  ib_mbm_out_dim: 512
  ib_mbm_n_head: 4
  use_cccp: false
  optimizer: adamw
  student_lr: 0.001
  student_weight_decay: 0.0003
  grad_clip_norm: 0.5
  ce_alpha: 1.0
  kd_alpha: 0.0
  kd_ens_alpha: 0.0
  feat_kd_alpha: 0.0
  kd_target: synergy
  schedule: { type: cosine, lr_warmup_epochs: 5, min_lr: 1e-5 }

# L0_baseline.yaml (예시)
experiment:
  results_dir: experiments/ablation/baseline/results
  exp_id: L0_baseline
  dataset: { batch_size: 64, num_workers: 4, data_aug: 1 }
  num_stages: 2
  student_epochs_per_stage: [20, 20]
  teacher_adapt_epochs: 0
  use_amp: true
  amp_dtype: bfloat16
  use_ib: false
  use_cccp: false
  optimizer: adamw
  student_lr: 0.001
  student_weight_decay: 0.0003
  grad_clip_norm: 0.5
  ce_alpha: 1.0
  kd_alpha: 0.0
  kd_ens_alpha: 0.0
  feat_kd_alpha: 0.0
  kd_target: synergy
  schedule: { type: cosine, lr_warmup_epochs: 5, min_lr: 1e-5 }
```

(끝)
