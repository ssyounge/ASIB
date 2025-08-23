# ASIB Framework Overview

ASIB 프레임워크의 전반 구조, 메소드(IB, CCCP, PPF/FFP), 핵심 모듈, 주요 설정 키, 학습 플로우, ablation 규칙을 한 문서로 요약합니다. 다른 프롬프트에 넣을 때 본문 전체를 복사해도 됩니다.

## 최신 Ablation 업데이트 요약 (2025-08)
- 공통 베이스(모든 L0~L4/side):
  - KD: `kd_target: avg`, `ce_alpha: 0.65`, `kd_alpha: 0.35`, `kd_max_ratio: 1.25`, `tau_schedule: [3.5, 5.0]`, `kd_warmup_epochs: 3`, `ce_label_smoothing: 0.0`
  - 스테이지/스케줄: `num_stages: 4`, `student_epochs_per_stage: [20, 20, 20, 20]`, `schedule: { type: cosine, lr_warmup_epochs: 5, min_lr: 1e-6 }`
  - 증강: `mixup_alpha: 0.2`, `cutmix_alpha_distill: 1.0`
  - Adapter/Feature: `use_distillation_adapter: true`, `distill_out_dim: 512`, `feat_kd_alpha: 0.0`, `feat_kd_key: distill_feat`
- IB 가드(IB 사용하는 설정에만 적용): `ib_mbm_out_dim: 512`, `ib_mbm_logvar_clip: 4`, `ib_mbm_min_std: 0.01`, `ib_mbm_lr_factor: 2`, `ib_beta: 0.0001(또는 0.00005)`, `ib_beta_warmup_epochs: 4~6`, `synergy_only_epochs: 2`, `enable_kd_after_syn_acc: 0.8`, `disable_loss_clamp_in_a: true`
- PPF/BN: `L4_full`/`side_cccp_ppf`에서 `student_freeze_bn: true` 권장
- A‑Step 안정화: asib_stage.yaml에 `label_smoothing=0.05`, `synergy_head_dropout=0.05`, `teacher_eval_every=2`, `synergy_ema_alpha=0.8` 반영. 초기 `synergy_only_epochs` 동안 CE‑only(IB‑KL=0, KD=0, cw=1.0)로 안정화. 에폭 종료 시 `last_synergy_acc`를 EMA로 갱신.
- B‑Step 시너지 게이트(코드 반영): `_synergy_gate_ok`로 `last_synergy_acc ≥ enable_kd_after_syn_acc`일 때만 IB_MBM/zsyn/μ‑KD 활성. 임계 미만이면 avg‑KD만 사용. KD 벡터는 `nan_to_num`으로 안전 처리, μ‑KD는 Huber/clip 옵션(`feat_kd_clip`, `feat_kd_huber_beta`).
- 실행/구성: `-cn="experiment/<CFG>"` + 루트 오버라이드(`+seed=`). normalize 이후 `method.*` 서브트리 제거, `[CFG] kd_target/ce/kd/ib_beta` 한 줄 로그 출력.

### 2025-08 추가 반영 (코드→문서 동기화)
- IB_MBM 내부 안정화: q/kv에 `LayerNorm`(pre‑norm) 적용, MHA 출력에 q residual 후 `LayerNorm`(`out_norm`). SynergyHead는 `LayerNorm+GELU+Dropout+Linear`로 교체(로짓 안정화), 선택적으로 learnable temperature(`synergy_temp_learnable`, `synergy_temp_init`) 지원.
- μ‑KD 기본화: B‑Step에서 `use_mu_for_kd: true`일 때 `synergy_head(mu)`를 KD 타깃으로 사용(노이즈 억제). 기본값 on.
- KD 클램프 스케줄: `kd_max_ratio`는 `kd_warmup_epochs` 이후에만 적용(초기 과도 제약 방지).
- 시너지 평가/게이팅 안정화: `teacher_eval_every`(기본 2ep 간격)로 평가 빈도 조절, `synergy_ema_alpha`(기본 0.8)로 `last_synergy_acc` EMA 반영.
  - EMA 업데이트 가드: 평가를 건너뛴 에폭에서는 EMA를 업데이트하지 않음(음수/무효 값 유입 방지)
  - eval_synergy 진입 시 teachers/IB_MBM/SynergyHead를 eval()로 강제 후 기존 모드 복원
- 작은 학생 자동 차원 정렬: `mobilenet_v2`/`efficientnet_b0`/`shufflenet_v2`는 `distill_out_dim=256` 권장, `ib_mbm_out_dim`을 동일 값으로 자동 정렬. MobileNetV2는 분류 경로 1280ch 유지(어댑터 분기 분리), CIFAR stem(stride=1) 적용.
- 교사 quick_eval 가속: GPU+AMP+bfloat16로 빠르게 평가(`teacher_eval_on_gpu`, `teacher_eval_amp`, `teacher_eval_batch_size`, `teacher_eval_max_batches`), safe‑mode에서도 평가 구간에 한해 `teacher_eval_force_cudnn=true`로 cuDNN/TF32 일시 활성화.
  - zero_grad 최적화: A‑Step에서 `optimizer.zero_grad(set_to_none=True)` 사용
  - 스냅샷 메모리 최적화: A‑Step `best_state`/`backup_state`를 CPU(state_dict)로 저장해 VRAM 파편화 방지
  - 첫 스텝 요약 로그: ep=1/step=1에 KD 게이트/가중치, cw 통계, raw KLD, ib_beta를 한 줄로 출력해 튜닝 가속

주의: CLI 최소 오버라이드 규칙(Strict 모드 호환)
- 허용(권장): `+experiment/method=<name>`, `+results_dir=...`, `+exp_id=...`, `+seed=...`
- 대안: `experiment/method@experiment.experiment.method=<name>` (defaults 항목 교체)
- 금지(CLI로 덮지 말 것): `optimizer`, `dataset.*`, `kd_target`, `kd_alpha/ce_alpha`, `teacher*_ckpt`, `compute_teacher_eval`, `method_name` 등 모든 leaf 키. 이 값들은 반드시 설정 파일(YAML)에서 정의하세요.

메소드 우선 규칙: `normalize_exp`가 `experiment.method` 값을 최상위로 승격. 메소드별 가드는 코드에서 강제하지 않고 YAML/CLI를 신뢰(asib_stage는 YAML에서 구성).

### 구성 락/해시 정책
- 실행 전 효과적 구성은 해시로 잠금되며 `before_run`/`before_safe_retry`/`after_run` 시점에 검증합니다.
- 해시에서 제외되는 런타임 변동 키(허용된 변이): `config_sha256`, `locked`, `use_amp`, `teacher1_ckpt`, `teacher2_ckpt`, `ib_mbm_out_dim`, `ib_mbm_query_dim`, `auto_align_ib_out_dim`, `_locked_config`, `csv_filename`, `total_time_sec`, `final_student_acc`, `last_synergy_acc`, `last_synergy_acc_pct`, `kd_gate_on`, `optimizer`, `hydra_method`, `cur_stage`, `effective_teacher_lr`, `effective_teacher_wd`, `num_classes`, 그리고 접두어 `student_ep*`/`teacher_ep*`/`epoch*`/`csv_*` 로 시작하는 모든 메트릭.
- SAFE‑RETRY 중 `use_amp`가 `false`로 강제될 수 있으나, 이는 해시 제외 키이므로 락 위반이 아닙니다.
- 작은 학생 자동 정렬로 `distill_out_dim` 기준 `ib_mbm_out_dim`/`ib_mbm_query_dim`이 조정될 수 있으며, 이 역시 해시 제외 대상입니다.
 - finalize_config에서 `teacher_lr`/`teacher_weight_decay`를 `a_step_*`에서 사전 확정하여 락 이후 변이를 방지합니다.

### KD 타깃 정책(요약)
- weighted_conf: 교사별 최고 확률(confidence)을 per‑sample 가중치로 사용해 로짓을 가중합.
- auto: 게이트 통과 시 synergy 사용, 실패 시 weighted_conf로 폴백. 내부적으로 `min(KL_syn, KL_avg)`를 사용하고 승률(`student_epN_auto_syn_win_ratio`)을 로깅해 분석합니다.
- 단일교사: `kd_teacher_index`로 선택.

### 시너지 평가/안정화
- eval 시 IB_MBM은 z 대신 μ를 사용(sample=False)해 노이즈를 줄입니다.
- A‑Step에서만 `last_synergy_acc` EMA를 갱신(update_logger=True); B‑Step은 모니터링 전용.

### W&B 로깅(옵션)
- 에폭별 `gate_ratio`, `kd_syn_ratio`, `kd_clamp_ratio`, `kd_scale_mean`, `kd_tgt_mode`, `kd_alpha_eff` 등을 함께 로깅해 튜닝 피드백을 가속합니다.

## 런타임 안정화 업데이트 (2025-08)
- safe‑mode에서 cuDNN 완전 비활성: `torch.backends.cudnn.enabled = False` (TF32/benchmark도 OFF)
- 멀티프로세싱 start_method 강제 해제: `set_start_method("spawn", ...)` 제거
- DataLoader 정책 통일: `pin_memory = (num_workers > 0)`, `persistent_workers = (num_workers > 0)`, `prefetch_factor = (... if num_workers > 0 else None)` — `imagenet32.py`, `cifar100_overlap.py` 반영
- 러너 환경 정리: `unset LD_LIBRARY_PATH`; NCCL 완전 비활성 `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN`
- 스모크 권장(10~20ep): CE‑only, workers=0, AMP off, channels_last off

```bash
env -i PATH="$PATH" HOME="$HOME" USER="$USER" PYTHONPATH="$PYTHONPATH" \
CUDA_LAUNCH_BLOCKING=1 NVIDIA_TF32_OVERRIDE=0 PYTORCH_NVFUSER_DISABLE=1 \
TORCH_USE_CUDA_DSA=1 CUDA_MODULE_LOADING=LAZY NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN \
$HOME/anaconda3/envs/tlqkf/bin/python -u main.py -cn=experiment/sota_generic \
+results_dir=experiments/smoke/ce_only_gpu +exp_id=ce_only_gpu_smoke +seed=42 \
+experiment.student_lr=0.05 +experiment.compute_teacher_eval=false \
+experiment.dataset.num_workers=0 +experiment.use_amp=false \
+experiment.use_safe_mode=true +experiment.use_channels_last=false \
experiment/method@experiment.experiment.method=ce_only
```

## 1) 아키텍처 개요
- Teachers(교사 2개): `teacher1`, `teacher2` 고정 백본에서 특징 추출
- Student(학생): 증류 대상 모델
- IB_MBM + SynergyHead: 교사 특징(KV) + 학생 질의(Q) → 잠재표현 z → 로짓. IB의 KL 항은 β 가중
- 손실: CE + (선택) KD + (선택) Feature KD + (선택) CCCP surrogate + 정규항
- 정책: PPF/FFP(Partial Freeze/Finetuning)로 일부 블록/정규화를 동결하거나 교사를 소규모 파인튜닝

참고 문서: `framework_docs/root_files.md`, `framework_docs/utils.md`, `framework_docs/modules.md`, `framework_docs/run.md`, `framework_docs/tests.md`

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
  - `kd_target` ('teacher' | 'avg' | 'synergy' | 'auto'), `ce_alpha`, `kd_alpha`, `kd_ens_alpha`, `feat_kd_alpha`, `feat_kd_key`('distill_feat')
  - `synergy_logit_scale`(float), `tau_syn`(float)
  - SynergyHead 옵션: `synergy_head_dropout`(float), `synergy_temp_learnable`(bool), `synergy_temp_init`(float)
  - `use_loss_clamp`, `loss_clamp_max`, `kd_max_ratio`(워밍업 이후 적용)
  - `use_mu_for_kd`(bool; B‑Step KD 타깃으로 μ 사용)
- IB
  - `use_ib`, `ib_beta`, `ib_beta_warmup_epochs`, `ib_epochs_per_stage`, `ib_mbm_query_dim`(auto), `ib_mbm_out_dim`, `ib_mbm_n_head`, `ib_mbm_feature_norm`, `ib_mbm_reg_lambda`
- CCCP
  - `use_cccp`, `cccp_alpha`, `tau`, `cccp_nt`, `cccp_ns`
- PPF/Finetuning
  - `use_partial_freeze`, `student_freeze_level`, `teacher*_freeze_level`, `*freeze_bn`, `use_teacher_finetuning`, `train_distill_adapter_only`
  
- 시너지 평가/게이팅 안정화
  - `teacher_eval_every`(int): A‑Step 시너지 평가 주기(기본 2)
  - `synergy_ema_alpha`(float): `last_synergy_acc` EMA 계수(기본 0.8)

- 교사 quick_eval(시작 지연 단축)
  - `compute_teacher_eval`, `teacher_eval_on_gpu`, `teacher_eval_amp`, `teacher_eval_batch_size`, `teacher_eval_max_batches`, `teacher_eval_force_cudnn`
- 최적화/스케줄/AMP
  - `optimizer`, `student_lr`, `student_weight_decay`, `schedule.type`, `lr_warmup_epochs`, `min_lr`, `use_amp`, `amp_dtype`('float16'|'bfloat16')

주의(라벨 스무딩 키 구분)
- A-Step(교사/IB 적응): `label_smoothing` 키 사용(teacher CE에 적용)
- B-Step(학생 증류): `ce_label_smoothing` 키 사용(학생 CE에 적용)
- 혼선을 피하려면 하나의 키로 통일 가능하지만, 현재 코드는 위처럼 단계별로 분리되어 있으니 설정 시 구분하여 사용하세요.

## 4) 최소 설정 템플릿(복붙용)
```yaml
# Baseline (avg-KD, 공정 비교용)
kd_target: avg
ce_alpha: 0.65
kd_alpha: 0.35
kd_ens_alpha: 0.0
feat_kd_alpha: 0.0

# KD 안정 가드
kd_max_ratio: 1.25
tau_schedule: [3.5, 5.0]
kd_warmup_epochs: 3

# Loss clamp (soft)
use_loss_clamp: true
loss_clamp_mode: soft
loss_clamp_max: 20.0
loss_clamp_warmup_epochs: 8

# IB (기본 OFF; 전환 시 안전 값)
use_ib: false
ib_mbm_out_dim: 512
ib_mbm_n_head: 4
ib_mbm_logvar_clip: 4
ib_mbm_min_std: 0.01
ib_mbm_lr_factor: 2

# CCCP (옵션)
use_cccp: false
cccp_alpha: 0.20
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
# synergy-only 구간/시너지 정확도 임계치(thr) 기반 게이팅 적용
kd_cccp = 0.0
use_cccp_a = cfg.get("use_cccp", False) and cfg.get("use_cccp_in_a", False)
if use_cccp_a:
    syn_warm = int(cfg.get("synergy_only_epochs", 0))
    if ep >= syn_warm:
        thr = float(cfg.get("enable_kd_after_syn_acc", 0.0))
        last_syn = float(logger.get_metric("last_synergy_acc", cfg.get("last_synergy_acc", 0.0))) if hasattr(logger, "get_metric") else float(cfg.get("last_synergy_acc", 0.0))
        if thr <= 0.0 or last_syn >= thr:
            with torch.no_grad():
                s_out = student_model(x)[1] if isinstance(student_model(x), tuple) else student_model(x)
            tau  = float(cfg.get("tau", 4.0))
            cccp_w = float(cfg.get("cccp_alpha", 0.0))
            kd_cccp = cccp_w * (tau**2) * F.kl_div(
                F.log_softmax(zsyn / tau, dim=1),
                F.softmax(s_out.detach() / tau, dim=1),
                reduction="batchmean",
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
            # teacher forward (조건부, K개 일반화) → t_feats(list), t_logits(list)
            # KD 타깃 모드: teacher(idx) | avg(가중) | synergy/auto(게이트)
            # IB_MBM (조건부: use_ib ∧ gate ∧ K>1) → zsyn_ng, mu_ng, logvar_ng
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

### 10.2 Teacher quick_eval 가속 옵션
```yaml
# 빠른 시작을 위한 권장(예)
experiment:
  compute_teacher_eval: true
  teacher_eval_on_gpu: true
  teacher_eval_amp: true
  teacher_eval_batch_size: 128     # OOM 시 64
  teacher_eval_max_batches: 10     # 부분 평가
  teacher_eval_force_cudnn: true   # safe‑mode에서도 평가만 빠르게
```
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

### 13.1 CE/KL 안전 계산(현행)
```python
def ce_safe(logits, target, ls_eps: float = 0.0):
    with torch.autocast('cuda', enabled=False):
        return F.cross_entropy(logits.float(), target, label_smoothing=float(ls_eps))

def kl_safe(p_logits, q_logits, tau: float = 1.0):
    with torch.autocast('cuda', enabled=False):
        s_log = F.log_softmax(p_logits.float() / tau, dim=1)
        t_prob = F.softmax(q_logits.float() / tau, dim=1)
        return F.kl_div(s_log, t_prob, reduction="batchmean") * (tau * tau)
```

추가: B‑Step에서 mixup/cutmix 사용 시 CE를 라벨 두 개(y_a, y_b)와 `lam`으로 직접 혼합합니다.
```python
ls = float(cfg.get("ce_label_smoothing", 0.0))
if mix_mode in ("mixup", "cutmix"):
    ce_loss_val = (
        F.cross_entropy(s_logit.float(), y_a, label_smoothing=ls) * lam
        + F.cross_entropy(s_logit.float(), y_b, label_smoothing=ls) * (1.0 - lam)
    )
else:
    ce_vec = ce_safe_vec(s_logit, y, ls_eps=ls)  # [B]
    ce_loss_val = (weights * ce_vec).mean()
# KD에도 동일 weights 적용 후 kd_max_ratio로 클램프
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
- 실행 예(Strict 모드, 최소 오버라이드):
  - Vanilla KD
    - `python -u main.py -cn=experiment/sota_generic \\
       +experiment/method=vanilla_kd \\
       +results_dir=experiments/sota/r152_mbv2/vanilla_kd/results \\
       +exp_id=r152_mbv2_vanilla \\
       +seed=42`
  - DKD
    - `python -u main.py -cn=experiment/sota_generic \\
       +experiment/method=dkd \\
       +results_dir=experiments/sota/cnexts_eb0/dkd/results \\
       +exp_id=cnexts_eb0_dkd \\
       +seed=42`

참고: 교사/학생 선택, 체크포인트 경로, 데이터/옵티마/하이퍼(예: `kd_target`, `student_lr`) 등 모든 leaf 키는 YAML 설정(`experiment/sota_generic.yaml` 등)에서 정의하세요. CLI에서는 그룹 선택과 실행 메타(`results_dir`, `exp_id`, `seed`)만 추가합니다.
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

ib_mbm, synergy_head = build_from_teachers(
    [teacher1, teacher2], cfg, query_dim=cfg.get("ib_mbm_query_dim")
)
# 내부 검사: use_distillation_adapter=false면 teacher feature dims 동일해야 함
# SynergyHead: dropout(`synergy_head_dropout`), temperature(`synergy_temp_*`) 옵션 지원
```

## 19) 설정 예시 모음(발췌)
```yaml
# SOTA generic (단일 파일 + CLI 오버라이드)
defaults:
  - /base
  - /model/teacher@experiment.teacher1: resnet152
  - /model/teacher@experiment.teacher2: convnext_s
  - /model/student@experiment.model.student: mobilenet_v2_scratch
  - _self_

experiment:
  exp_id: sota_generic
  results_dir: experiments/sota/generic/results
  dataset: { name: cifar100, batch_size: 128, num_workers: 4, data_aug: 1 }
  num_stages: 1
  student_epochs_per_stage: [240]
  compute_teacher_eval: false
  use_amp: true
  amp_dtype: bfloat16
  teacher1_ckpt: null
  teacher2_ckpt: null
  # kd_target / use_ib / teacher_adapt_epochs / use_partial_freeze 는 메소드 파일이 결정
  kd_teacher_index: 0
  ce_alpha: 0.65
  kd_alpha: 0.35
  kd_warmup_epochs: 3
  kd_max_ratio: 1.25
  tau: 4.0
  ce_label_smoothing: 0.0
  use_distillation_adapter: true
  distill_out_dim: 256
  feat_kd_key: distill_feat
  optimizer: sgd
  student_lr: 0.1
  student_weight_decay: 0.0005
  b_step_momentum: 0.9
  b_step_nesterov: true
  schedule: { type: cosine, lr_warmup_epochs: 5, min_lr: 1e-5 }
  mixup_alpha: 0.0
  cutmix_alpha_distill: 0.0
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

## 30) ASIB 전환 가이드 (Migration)

### 30.1 기존 KD 파이프라인 → ASIB 최소 전환 절차
- 1) 모델/데이터 정렬: `teacher1/2`, `student`를 기존 설정으로 생성. 어댑터가 없다면 `use_distillation_adapter: true`, `distill_out_dim: 512` 권장
- 2) KD 베이스라인 통일: `kd_target: avg`, `ce/kd=0.65/0.35`, `tau_schedule: [3.5, 5.0]`, `kd_warmup_epochs: 3`, `kd_max_ratio: 1.25`
- 3) 안정 가드: `use_loss_clamp: true`, `loss_clamp_mode: soft`, `loss_clamp_max: 20.0`, `loss_clamp_warmup_epochs: 8`
- 4) 스테이지 구성: `num_stages: 4`, `student_epochs_per_stage: [20,20,20,20]`, `schedule.cosine + warmup=5`
- 5) AMP: `use_amp: true`, `amp_dtype: bfloat16`
- 6) 실행: `python main.py -cn=experiment/L0_baseline +seed=42`

### 30.2 IB/시너지 경로 활성화 절차
- 1) IB 켜기: `use_ib: true`, `ib_epochs_per_stage: 6`, `ib_beta: 5e-05`, `ib_beta_warmup_epochs: 4`
- 2) IB_MBM 용량: `ib_mbm_out_dim: 512`, `ib_mbm_n_head: 4`, `ib_mbm_logvar_clip: 4`, `ib_mbm_min_std: 0.01`, `ib_mbm_lr_factor: 2`
- 3) 시너지 게이트: `synergy_only_epochs: 8`, `enable_kd_after_syn_acc: 0.8`, `kd_ens_alpha: 0.5`
- 4) 권장 실행: `-cn=experiment/L4_full` 또는 `-cn=experiment/L1_ib`로 시작, 성능/안정 확인 후 하이퍼 튜닝

### 30.3 CCCP 결합
- 1) `use_cccp: true`, `use_cccp_in_a: true`, `cccp_alpha: 0.20`
- 2) KD와 별개로 surrogate 항이 A/B 단계에서 안전하게 합산됨. 과대 시 `cccp_alpha`를 0.10~0.25로 조절

### 30.4 체크리스트
- 로그에 `[KD]`, `[IB/CCCP]`, `[PPF]` 블록이 베이스라인과 비교해 의도한 변경만 포함되는지 확인
- `auto_set_ib_mbm_query_dim_with_model` 호출로 `ib_mbm_query_dim` 자동 세팅 여부 확인
- 첫 배치에서 KV/Q dim mismatch 경고가 없는지 확인

## 31) PPF 스케줄 키 설명 상세

### 31.1 단일 레벨 키
- `use_partial_freeze`(bool): PPF 정책 사용 여부
- `student_freeze_level`, `teacher1_freeze_level`, `teacher2_freeze_level`(int): −1 없음, 0 헤드만, 1 마지막 블록, 2 마지막 두 블록
- `student_freeze_bn`, `teacher1_freeze_bn`, `teacher2_freeze_bn`(bool): BN 고정 여부. BN 고정보다 LN은 기본적으로 학습 유지

### 31.2 스테이지별 스케줄 키
- `student_freeze_level_schedule`, `teacher1_freeze_level_schedule`, `teacher2_freeze_level_schedule`(list[int]): 스테이지 1..N에 대응하는 레벨 값
  - 예) `[-1, -1, 1, 1]`이면 1~2스테이지는 동결 없음, 3~4스테이지는 마지막 블록 동결
- 스케줄 키가 없고 단일 레벨만 주어진 경우, 내부 로직은 스케줄을 유추하거나 그대로 단일 레벨을 사용

### 31.3 적용 타이밍과 동작
- 스테이지 진입 시점에 각 모델(학생/교사)에 대해 `apply_partial_freeze(model, level, freeze_bn)` 호출
- 레벨 < 0: `requires_grad=True`로 전체 해제
- 레벨 = 0: 헤드만 학습. BN은 `freeze_bn`에 따라 동결
- 레벨 ≥ 1: 백본 전체를 먼저 동결 후 레벨 규칙에 맞는 블록/헤드만 해제해 학습

### 31.4 실전 가이드
- 성능 안정 우선: `L4_full.yaml`처럼 후반 스테이지에만 레벨 1을 적용하고 `student_freeze_bn: true` 권장
- 빠른 수렴/VRAM 절감: `side_cccp_ppf.yaml`처럼 전 스테이지에서 `level: 1` 고정도 가능
- 주의: freeze ≥ 0인데 `student_pretrained=false`이면 랜덤 초기화된 동결층이 생길 수 있음. 사전학습 사용 권장 또는 level=-1 유지

### 31.5 예시 스니펫
```yaml
# 안정형 (후반만 동결)
use_partial_freeze: true
student_freeze_level_schedule: [-1, -1, 1, 1]
teacher1_freeze_level_schedule: [-1, -1, 1, 1]
teacher2_freeze_level_schedule: [-1, -1, 1, 1]
student_freeze_bn: true
teacher1_freeze_bn: true
teacher2_freeze_bn: true
```

```yaml
# 경량/고속형 (전 스테이지 동결)
use_partial_freeze: true
student_freeze_level: 1
teacher1_freeze_level: 1
teacher2_freeze_level: 1
student_freeze_bn: true
teacher1_freeze_bn: true
teacher2_freeze_bn: true
```


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
def build_ib_mbm_from_teachers(
    teachers: List[nn.Module], cfg: dict, query_dim: Optional[int] = None
) -> Tuple[IB_MBM, SynergyHead]:
    use_da = bool(cfg.get("use_distillation_adapter", False))
    feat_dims = [
        (t.distill_dim if use_da and hasattr(t, "distill_dim") else t.get_feat_dim())
        for t in teachers
    ]
    if not use_da:
        unique_dims = set(int(d) for d in feat_dims)
        if len(unique_dims) > 1:
            raise ValueError(
                "Teacher feature dims differ. Enable use_distillation_adapter to align dimensions."
            )
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
        learnable_temp=bool(cfg.get("synergy_temp_learnable", False)),
        temp_init=float(cfg.get("synergy_temp_init", 0.0)),
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
