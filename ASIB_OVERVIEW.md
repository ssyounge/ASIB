# ASIB Framework Overview

ASIB 프레임워크의 전반 구조, 메소드(IB, CCCP, PPF/FFP), 핵심 모듈, 주요 설정 키, 학습 플로우, ablation 규칙을 한 문서로 요약합니다. 다른 프롬프트에 넣을 때 본문 전체를 복사해도 됩니다.

## 최신 Ablation 업데이트 요약 (2025-08)
- 공통 베이스(모든 L0~L4/side):
  - KD: `ce_alpha: 0.70`, `kd_alpha: 0.30`, `kd_max_ratio: 1.25`, `kd_warmup_epochs: 10`, `ce_label_smoothing: 0.05`
  - 스테이지/스케줄: `num_stages: 4`, `student_epochs_per_stage: [20, 20, 20, 20]`, `schedule: { type: cosine, lr_warmup_epochs: 5, min_lr: 1e-6 }`
  - 증강: `mixup_alpha: 0.2`, `cutmix_alpha_distill: 1.0`
  - Adapter/Feature: `use_distillation_adapter: true`, `distill_out_dim: 512`, `feat_kd_key: distill_feat`
  - Clamp(A‑Step): `use_loss_clamp=true`, `loss_clamp_mode=soft`, `disable_loss_clamp_in_a=false`, `loss_clamp_warmup_epochs=0`
  - Dataloader: `dataset.num_workers=8`
  - Optim/Schedule: `optimizer=adamw`, `student_lr=0.001`, `student_weight_decay=0.0003`
- A‑Step 예열(시너지 품질 향상): `synergy_only_epochs: 8` (CE‑only)
- Rung 설정(L1~L4): `teacher_adapt_epochs: 8` (A‑Step 길이)
- IB 가드(IB 사용하는 설정에만 적용): `ib_mbm_out_dim: 512`, `ib_mbm_logvar_clip: 4`, `ib_mbm_min_std: 0.01`
- PPF/BN: `L4_full`에서 `student_freeze_bn: true` 권장(사이드 PPF OFF)
- A‑Step 안정화: asib_stage.yaml에 `label_smoothing=0.05`, `synergy_head_dropout=0.05`, `teacher_eval_every=2`, `synergy_ema_alpha=0.8` 반영. 초기 `synergy_only_epochs` 동안 CE‑only로 안정화. 에폭 종료 시 `last_synergy_acc`를 EMA로 갱신.
  - CE‑only 권장치: `synergy_only_epochs` 구간은 `synergy_ce_alpha: 1.0` 권장(코드에서 미달 시 경고 로그 출력).
- B‑Step 시너지 게이트(코드 반영): `_synergy_gate_ok`로 `last_synergy_acc ≥ enable_kd_after_syn_acc`일 때만 IB_MBM/zsyn/μ‑KD 활성. 임계 미만이면 avg‑KD만 사용. KD 벡터는 `nan_to_num`으로 안전 처리, μ‑KD는 Huber/clip 옵션(`feat_kd_clip`, `feat_kd_huber_beta`).
- 실행/구성: `-cn="experiment/<CFG>"` + 루트 오버라이드(`+seed=`). normalize 이후 `method.*` 서브트리 제거, `[CFG] kd_target/ce/kd/ib_beta` 한 줄 로그 출력.

### 2025-08 추가 반영 (코드→문서 동기화)
- IB_MBM 내부 안정화: q/kv에 `LayerNorm`(pre‑norm) 적용, MHA 출력에 q residual 후 `LayerNorm`(`out_norm`). SynergyHead는 `LayerNorm+GELU+Dropout+Linear`로 교체(로짓 안정화), 선택적으로 learnable temperature(`synergy_temp_learnable`, `synergy_temp_init`) 지원.
- μ‑KD 기본화: B‑Step에서 `use_mu_for_kd: true`일 때 `synergy_head(mu)`를 KD 타깃으로 사용(노이즈 억제). 기본값 on.
- KD 클램프 스케줄: `kd_max_ratio`는 `kd_warmup_epochs` 이후에만 적용(초기 과도 제약 방지).
- 시너지 평가/게이팅 안정화: `teacher_eval_every`(기본 2ep 간격)로 평가 빈도 조절, `synergy_ema_alpha`(기본 0.8)로 `last_synergy_acc` EMA 반영.
  - EMA 업데이트 가드: 평가를 건너뛴 에폭에서는 EMA를 업데이트하지 않음(음수/무효 값 유입 방지)
  - eval_synergy 진입 시 teachers/IB_MBM/SynergyHead를 eval()로 강제 후 기존 모드 복원
  - 초기값 보강: A‑Step 직후에도 `last_synergy_acc<0`이면 B‑Step 전 `eval_synergy` 1회로 초기화해 게이트 기준을 확보.
- 작은 학생 자동 차원 정렬: `mobilenet_v2`/`efficientnet_b0`/`shufflenet_v2`는 `distill_out_dim=256` 권장, `ib_mbm_out_dim`을 동일 값으로 자동 정렬. MobileNetV2는 분류 경로 1280ch 유지(어댑터 분기 분리), CIFAR stem(stride=1) 적용.
- B‑Step 최적화(A–D): kd/syn clean‑view 재사용, two_view B‑view EMA 제외 기본, 쿨다운 시 KD/IB 전체 스킵, `kd_two_view_stop_epoch`로 후반 clean 전환.
- auto_min(label_ce) 라우팅/라벨 가드/uncertainty/DKD 하이브리드 바인딩: 학습 신호 품질과 안정성 개선(코드 반영 완료).
  - 샘플 게이팅(T7): `kd_sample_gate`로 KD 대상 subset만 교사/IB 수행. 두‑뷰/시너지/불확실도/μ‑MSE/Disagree 모두 subset→full 확장 처리. 선택 시 `kd_sample_reweight`로 KD 스케일 보존.
  - 운영 키(요약): `kd_cooldown_epochs`, `kd_two_view_stop_epoch`, `kd_sample_gate`, `kd_sample_thr`, `kd_sample_max_ratio`, `kd_sample_min_ratio`, `kd_sample_reweight`, `ema_update_every`.
  - dtype/인덱스 정합: subset scatter는 float32 기준(`torch.zeros(..., dtype=kd_vec_sel.dtype)`), `kd_uncertainty_weight` subset 분기에서 `auto_min` 선택 로짓으로 `q_sel` 구성.
  - SAFE‑RETRY: AMP만 off. `use_channels_last`는 락 이후 변경하지 않음(해시 충돌 방지).
  - 게이트 안전화: `_synergy_gate_ok`는 `last_synergy_acc <= 0`이면 게이트를 닫음.
  - auto_min 권장: `kd_ens_alpha: 0.0` 유지(혼합은 synergy 모드에서만 적용 가치 높음).
- 두‑뷰 지연(권장): 초기에는 clean으로 단순화하기 위해 `kd_two_view_start_epoch=20`을 권장합니다. 종료 시점은 `kd_two_view_stop_epoch=80`으로 통일합니다.

### asib_stage 최신 기본값(요약)
- `kd_target: auto`, `tau_syn: 4.0`, `synergy_logit_scale: 0.80`
- `kd_cooldown_epochs: 60`, `kd_uncertainty_weight: 0.5`
- `kd_sample_gate: true`, `kd_sample_thr: 0.85`, `kd_sample_max_ratio: 0.50`
- `teacher_weights: [0.7, 0.3]`
- `synergy_only_epochs: 2`, `synergy_ce_alpha: 1.0`
- Optim/Schedule: `optimizer=adamw`, `student_lr=0.001`, `student_weight_decay=0.0003`, `schedule.min_lr=1e-6`
- Adapter dims: `distill_out_dim=512`, `ib_mbm_query_dim=512`, `ib_mbm_out_dim=512`, `ib_mbm_lr_factor=1`, `synergy_temp_learnable=false`

### 러너 사용 가이드(갱신)
- 메소드 선택: `+method@experiment.method=<name>`
- 모델/학생 그룹: `model/teacher@experiment.teacher{1,2}`, `model/student@experiment.model.student`
- DRY_RUN=1로 인덱스 매핑 확인 후 Slurm 배열 제출 권장

### ENV 단축키(런타임 오버라이드)
- `TEACHER_PAIR="t1,t2"`: `experiment.teacher1/2.name` 주입 및 `pretrained=true` 보존
- `STUDENT="<key>"`: `experiment.model.student.name`을 `<key>_scratch`로 설정하고 `pretrained=false`
- `METHOD="<name>"`: 루트 `method_name`에 기록(최종 선택은 Hydra 그룹을 신뢰)
- `SEED=<int>`: `seed` 주입

주의: 메소드 선택은 `experiment.method` Hydra 그룹으로 결정됩니다. 루트 `method_name`은 로깅용입니다.

### 구성 락/해시 정책
- SAFE‑RETRY 중 `use_amp`가 `false`로 강제될 수 있으나, 이는 해시 제외 키이므로 락 위반이 아닙니다.
- 작은 학생 자동 정렬로 `distill_out_dim` 기준 `ib_mbm_out_dim`/`ib_mbm_query_dim`이 조정될 수 있으며, 이 역시 해시 제외 대상입니다.
 - finalize_config에서 `teacher_lr`/`teacher_weight_decay`를 `a_step_*`에서 사전 확정하여 락 이후 변이를 방지합니다.
- 오토/프로필 메타 키 해시 제외(추가): `_profile_applied`, `kd_two_view_start_epoch`, `kd_sample_thr`, `auto_tune_target_ratio`, `lr_log_every`.

### 시너지 평가/안정화
- eval 시 IB_MBM은 z 대신 μ를 사용(sample=False)해 노이즈를 줄입니다.
- A‑Step에서만 `last_synergy_acc` EMA를 갱신(update_logger=True); B‑Step은 모니터링 전용.
- K>2 교사: 기본은 상위 2개(인덱스 0,1)를 사용합니다. 모든 교사를 쓰려면 `synergy_eval_use_all_teachers: true`, 특정 교사 지정은 `synergy_eval_teacher_indices: [0,1,...]`를 사용하세요.
- 시너지 로짓 스케일(권장): `synergy_logit_scale=0.8`을 기본 가이드로 사용하세요. 스케일이 클수록 초기 KD가 과도해질 수 있습니다.