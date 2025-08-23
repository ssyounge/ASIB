# Configs Index

All YAML configs under `configs/` with their paths and full contents.

### Note: Runtime Config Lock/Hash
- 실행 전 효과적 구성은 해시로 잠금/검증됩니다(`before_run`/`before_safe_retry`/`after_run`).
- 해시 산출은 sanitize를 거친 복사본으로 계산합니다. 아래 항목/접두사는 무시됩니다(허용된 런타임 변이):
  - 단일 키: `config_sha256`, `locked`, `use_amp`, `teacher1_ckpt`, `teacher2_ckpt`, `ib_mbm_out_dim`, `ib_mbm_query_dim`, `auto_align_ib_out_dim`, `_locked_config`, `csv_filename`, `total_time_sec`, `final_student_acc`, `last_synergy_acc`, `last_synergy_acc_pct`, `kd_gate_on`, `optimizer`, `hydra_method`, `cur_stage`, `effective_teacher_lr`, `effective_teacher_wd`, `num_classes`
  - 접두사: `student_ep*`, `teacher_ep*`, `epoch*`, `csv_*`
- finalize_config: 락 전에 파생/자동 키를 확정합니다(스케줄/PPF/세이프티, 작은 학생 `distill_out_dim=256` 및 `ib_mbm_out_dim` 정렬, 교사 ckpt 주입, 숫자 캐스팅, `teacher_lr`/`teacher_weight_decay`를 `a_step_*`에서 사전 확정).
- AMP: `get_amp_components(cfg)`는 cfg를 수정하지 않으며, 기본은 `use_amp=false`를 존중합니다(안정화 후 bfloat16 권장).
- KD 타깃 가이드: `weighted_conf`는 교사별 최대 확률을 per‑sample 가중치로 사용, `auto`는 게이트 통과 시 synergy, 실패 시 weighted_conf로 폴백합니다.

## Files
- configs/base.yaml
- configs/dataset/cifar100.yaml
- configs/dataset/imagenet32.yaml
- configs/experiment/L0_baseline.yaml
- configs/experiment/L1_ib.yaml
- configs/experiment/L2_cccp.yaml
- configs/experiment/L3_ib_cccp_tadapt.yaml
- configs/experiment/L4_full.yaml
- configs/experiment/method/ab.yaml
- configs/experiment/method/asib.yaml
- configs/experiment/method/at.yaml
- configs/experiment/method/crd.yaml
- configs/experiment/method/dkd.yaml
- configs/experiment/method/fitnet.yaml
- configs/experiment/method/ft.yaml
- configs/experiment/method/reviewkd.yaml
- configs/experiment/method/simkd.yaml
- configs/experiment/method/sskd.yaml
- configs/experiment/method/vanilla_kd.yaml
- configs/experiment/overlap_100.yaml
- configs/experiment/side_cccp_ppf.yaml
- configs/experiment/sota_generic.yaml
- configs/finetune/convnext_l_cifar100.yaml
- configs/finetune/convnext_l_imagenet32.yaml
- configs/finetune/convnext_s_cifar100.yaml
- configs/finetune/convnext_s_imagenet32.yaml
- configs/finetune/efficientnet_l2_cifar100.yaml
- configs/finetune/efficientnet_l2_imagenet32.yaml
- configs/finetune/resnet152_cifar100.yaml
- configs/finetune/resnet152_imagenet32.yaml
- configs/hydra.yaml
- configs/method/ab.yaml
- configs/method/asib.yaml
- configs/method/at.yaml
- configs/method/crd.yaml
- configs/method/dkd.yaml
- configs/method/fitnet.yaml
- configs/method/ft.yaml
- configs/method/reviewkd.yaml
- configs/method/simkd.yaml
- configs/method/sskd.yaml
- configs/method/vanilla_kd.yaml
- configs/model/student/efficientnet_b0_scratch.yaml
- configs/model/student/mobilenet_v2_scratch.yaml
- configs/model/student/resnet101_pretrain.yaml
- configs/model/student/resnet101_scratch.yaml
- configs/model/student/resnet152_pretrain.yaml
- configs/model/student/resnet152_scratch.yaml
- configs/model/student/resnet50_scratch.yaml
- configs/model/student/shufflenet_v2_scratch.yaml
- configs/model/teacher/convnext_l.yaml
- configs/model/teacher/convnext_s.yaml
- configs/model/teacher/efficientnet_l2.yaml
- configs/model/teacher/resnet152.yaml
- configs/registry_key.yaml
- configs/registry_map.yaml
- configs/schedule/cosine.yaml
- configs/schedule/step.yaml
- configs/anchor/fair_baseline.yaml
- configs/experiment/method/asib_stage.yaml

---

## Full Contents

### configs/base.yaml
# configs/base.yaml defaults: - dataset@experiment.dataset: cifar100 - schedule@experiment.schedule: cosine - experiment/method@experiment.method: asib - _self_ experiment: device: cuda seed: 42 small_input: true results_dir: experiments/default/results exp_id: default dataset: batch_size: 64 num_workers: 8 # 더 많은 워커로 데이터 로딩 속도 향상 num_stages: 1 student_epochs_per_stage: [5] teacher_adapt_epochs: 0 use_partial_freeze: false use_amp: true amp_dtype: bfloat16 # bfloat16이 더 안정적이고 빠름 (A100/H100에서) use_ib: false ib_epochs_per_stage: 0 ib_beta: 0.0 ib_beta_warmup_epochs: 0 use_vib_synergy_head: false ib_mbm_query_dim: 2048 ib_mbm_out_dim: 512 ib_mbm_n_head: 1 ib_mbm_feature_norm: l2 ib_mbm_reg_lambda: 0.0 a_step_lr: 1.0e-4 a_step_weight_decay: 1.0e-4 b_step_lr: 1.0e-1 b_step_weight_decay: 3.0e-4 b_step_momentum: 0.9 b_step_nesterov: true grad_clip_norm: 1.0 ce_alpha: 0.3 kd_ens_alpha: 0.0 hybrid_beta: 0.05 use_cccp: false tau: 4.0 cccp_nt: 1 cccp_ns: 1 # Teacher evaluation (초반 평가 스킵으로 속도 향상) compute_teacher_eval: false # Teacher fine-tuning control (교사 백본 학습 제어) use_teacher_finetuning: false # false: 교사 백본 고정 (기본값) train_distill_adapter_only: false # true: distillation adapter만 학습 # Disagreement calculation optimization disagreement_max_samples: 2000 # 전체 대신 샘플링으로 속도 향상 (None: 전체 사용) disagreement_max_batches: 10 # 배치 수 제한 (수 분 → 수 초) # Legacy trainer compatibility keys (do not remove) teacher_lr: 0.0 teacher_weight_decay: 0.0 student_lr: 0.05 student_weight_decay: 0.0003 # KD warmup and loss stabilization teacher_adapt_kd_warmup: 0 use_loss_clamp: false loss_clamp_max: 100.0 # Single-teacher SOTA default index (used only when kd_target: teacher) kd_teacher_index: 0

### configs/dataset/cifar100.yaml
name: cifar100 root: ./data small_input: true data_aug: 1

### configs/dataset/imagenet32.yaml
name: imagenet32 root: ./data small_input: true data_aug: 1

### configs/experiment/L0_baseline.yaml
defaults: - /base - /model/teacher@experiment.teacher1: convnext_l - /model/teacher@experiment.teacher2: resnet152 - /model/student@experiment.model.student: resnet101_scratch - _self_ experiment: exp_id: L0_baseline results_dir: experiments/ablation/L0/results teacher1_ckpt: checkpoints/teachers/convnext_l_cifar100.pth teacher2_ckpt: checkpoints/teachers/resnet152_cifar100.pth dataset: { batch_size: 64, num_workers: 8, data_aug: 1, pin_memory: true, persistent_workers: true, prefetch_factor: 2 } # 4-stage(총 40ep) num_stages: 4 student_epochs_per_stage: [20, 20, 20, 20] teacher_adapt_epochs: 0 compute_teacher_eval: true use_amp: true amp_dtype: bfloat16 # KD (teacher avg) kd_target: avg ce_alpha: 0.65 kd_alpha: 0.35 tau_schedule: [3.5, 5.0] kd_warmup_epochs: 3 kd_ens_alpha: 0.0 kd_max_ratio: 1.25 ce_label_smoothing: 0.0 min_cw: 0.1 feat_kd_alpha: 0.0 # 어댑터로 피처 정렬 use_distillation_adapter: true distill_out_dim: 512 feat_kd_key: distill_feat # IB config (uniform keys; baseline에서는 OFF) use_ib: false ib_epochs_per_stage: 0 ib_beta: 0.0001 ib_beta_warmup_epochs: 0 ib_mbm_query_dim: 512 ib_mbm_out_dim: 512 ib_mbm_n_head: 4 ib_mbm_feature_norm: l2 ib_mbm_logvar_clip: 4 ib_mbm_min_std: 0.01 ib_mbm_reg_lambda: 0.0 ib_mbm_lr_factor: 10 # CCCP (baseline에서는 OFF; 키만 보유) use_cccp: false use_cccp_in_a: false cccp_alpha: 0.20 tau: 4.0 # Optim/Schedule optimizer: adamw student_lr: 0.001 student_weight_decay: 0.0003 grad_clip_norm: 0.5 use_loss_clamp: true loss_clamp_mode: soft loss_clamp_max: 20.0 loss_clamp_warmup_epochs: 8 schedule: { type: cosine, lr_warmup_epochs: 5, min_lr: 1e-6 } mixup_alpha: 0.2 cutmix_alpha_distill: 1.0

### configs/experiment/L1_ib.yaml
defaults: - /base - /model/teacher@experiment.teacher1: convnext_l - /model/teacher@experiment.teacher2: resnet152 - /model/student@experiment.model.student: resnet101_scratch - _self_ experiment: exp_id: L1_ib results_dir: experiments/ablation/L1/results teacher1_ckpt: checkpoints/teachers/convnext_l_cifar100.pth teacher2_ckpt: checkpoints/teachers/resnet152_cifar100.pth dataset: { batch_size: 64, num_workers: 8, data_aug: 1, pin_memory: true, persistent_workers: true, prefetch_factor: 2 } num_stages: 4 student_epochs_per_stage: [20, 20, 20, 20] teacher_adapt_epochs: 6 # IB_MBM 학습용 A‑Step compute_teacher_eval: true use_amp: true amp_dtype: bfloat16 # KD (synergy target with gating/mix) kd_target: synergy teacher_adapt_kd_warmup: 6 kd_ens_alpha: 0.5 enable_kd_after_syn_acc: 0.8 ce_alpha: 0.65 kd_alpha: 0.35 tau_schedule: [3.5, 5.0] kd_warmup_epochs: 3 kd_max_ratio: 1.25 ce_label_smoothing: 0.0 feat_kd_alpha: 0.25 feat_kd_clip: 3.0 feat_kd_huber_beta: 1.0 min_cw: 0.5 max_cw: 1.5 # IB ON (A‑Step에서 시너지CE+KL로 학습 → B‑Step에서 logvar 기반 가중) use_ib: true ib_epochs_per_stage: 6 ib_beta: 5e-05 ib_beta_warmup_epochs: 4 ib_mbm_query_dim: 512 ib_mbm_out_dim: 512 ib_mbm_n_head: 4 ib_mbm_feature_norm: l2 ib_mbm_logvar_clip: 4 ib_mbm_min_std: 0.01 ib_mbm_lr_factor: 2 # A‑Step 안정화 synergy_only_epochs: 2 synergy_ce_alpha: 1.0 use_cccp_in_a: false # 어댑터 use_distillation_adapter: true distill_out_dim: 512 feat_kd_key: distill_feat optimizer: adamw student_lr: 0.001 student_weight_decay: 0.0003 teacher_lr: 0.0001 teacher_weight_decay: 0.0001 a_step_lr: 0.0001 a_step_weight_decay: 0.0001 grad_clip_norm: 0.5 use_loss_clamp: true loss_clamp_mode: soft loss_clamp_max: 20.0 disable_loss_clamp_in_a: true loss_clamp_warmup_epochs: 0 schedule: { type: cosine, lr_warmup_epochs: 5, min_lr: 1e-6 } mixup_alpha: 0.2 cutmix_alpha_distill: 1.0

### configs/experiment/L2_cccp.yaml
defaults: - /base - /model/teacher@experiment.teacher1: convnext_l - /model/teacher@experiment.teacher2: resnet152 - /model/student@experiment.model.student: resnet101_scratch - _self_ experiment: exp_id: L2_cccp results_dir: experiments/ablation/L2/results teacher1_ckpt: checkpoints/teachers/convnext_l_cifar100.pth teacher2_ckpt: checkpoints/teachers/resnet152_cifar100.pth dataset: { batch_size: 64, num_workers: 8, data_aug: 1, pin_memory: true, persistent_workers: true, prefetch_factor: 2 } num_stages: 4 student_epochs_per_stage: [20, 20, 20, 20] teacher_adapt_epochs: 6 compute_teacher_eval: true use_amp: true amp_dtype: bfloat16 # KD (teacher avg) kd_target: avg ce_alpha: 0.65 kd_alpha: 0.35 tau_schedule: [3.5, 5.0] kd_warmup_epochs: 3 kd_ens_alpha: 0.0 kd_max_ratio: 1.25 ce_label_smoothing: 0.0 feat_kd_alpha: 0.0 min_cw: 0.1 # IB use_ib: false ib_epochs_per_stage: 6 ib_beta: 0.0001 ib_beta_warmup_epochs: 4 ib_mbm_query_dim: 512 ib_mbm_out_dim: 512 ib_mbm_n_head: 4 ib_mbm_feature_norm: l2 ib_mbm_logvar_clip: 4 ib_mbm_min_std: 0.01 ib_mbm_lr_factor: 10 # CCCP ON (A‑Step에서만 적용되는 코드 경로) use_cccp: true use_cccp_in_a: true cccp_alpha: 0.20 tau: 4.0 # A‑Step 안정화 synergy_only_epochs: 6 synergy_ce_alpha: 1.0 # enable_kd_after_syn_acc: 0.6 # avg KD에서는 B-Step 게이팅에 무의미 (A-Step CCCP 의도 아니면 비활성 권장) use_distillation_adapter: true distill_out_dim: 512 feat_kd_key: distill_feat optimizer: adamw student_lr: 0.001 student_weight_decay: 0.0003 a_step_lr: 0.0001 a_step_weight_decay: 0.0001 grad_clip_norm: 0.5 use_loss_clamp: true loss_clamp_mode: soft loss_clamp_max: 20.0 loss_clamp_warmup_epochs: 8 schedule: { type: cosine, lr_warmup_epochs: 5, min_lr: 1e-6 } mixup_alpha: 0.2 cutmix_alpha_distill: 1.0

### configs/experiment/L3_ib_cccp_tadapt.yaml
defaults: - /base - /model/teacher@experiment.teacher1: convnext_l - /model/teacher@experiment.teacher2: resnet152 - /model/student@experiment.model.student: resnet101_scratch - _self_ experiment: exp_id: L3_ib_cccp_tadapt results_dir: experiments/ablation/L3/results teacher1_ckpt: checkpoints/teachers/convnext_l_cifar100.pth teacher2_ckpt: checkpoints/teachers/resnet152_cifar100.pth dataset: { batch_size: 64, num_workers: 8, data_aug: 1, pin_memory: true, persistent_workers: true, prefetch_factor: 2 } num_stages: 4 student_epochs_per_stage: [20, 20, 20, 20] teacher_adapt_epochs: 6 compute_teacher_eval: true use_amp: true amp_dtype: bfloat16 # KD (synergy target with gating/mix) kd_target: synergy teacher_adapt_kd_warmup: 6 kd_ens_alpha: 0.5 enable_kd_after_syn_acc: 0.8 ce_alpha: 0.65 kd_alpha: 0.35 tau_schedule: [3.5, 5.0] kd_warmup_epochs: 3 kd_max_ratio: 1.25 ce_label_smoothing: 0.0 feat_kd_alpha: 0.25 max_cw: 1.5 min_cw: 0.5 # IB(동일) use_ib: true ib_epochs_per_stage: 6 ib_beta: 5e-05 ib_beta_warmup_epochs: 4 ib_mbm_query_dim: 512 ib_mbm_out_dim: 512 ib_mbm_n_head: 4 ib_mbm_feature_norm: l2 ib_mbm_logvar_clip: 4 ib_mbm_min_std: 0.01 ib_mbm_lr_factor: 2 # CCCP (A‑Step) use_cccp: true use_cccp_in_a: true cccp_alpha: 0.20 tau: 4.0 # Teacher Adapt(어댑터만) use_teacher_finetuning: false train_distill_adapter_only: true teacher_lr: 3e-06 teacher_weight_decay: 1e-4 # A‑Step 안정화 synergy_only_epochs: 2 disable_loss_clamp_in_a: true synergy_ce_alpha: 1.0 use_distillation_adapter: true distill_out_dim: 512 feat_kd_key: distill_feat optimizer: adamw student_lr: 0.001 student_weight_decay: 0.0003 a_step_lr: 0.0001 a_step_weight_decay: 0.0001 grad_clip_norm: 0.5 use_loss_clamp: true loss_clamp_mode: soft loss_clamp_max: 20.0 loss_clamp_warmup_epochs: 8 schedule: { type: cosine, lr_warmup_epochs: 5, min_lr: 1e-6 } mixup_alpha: 0.2 cutmix_alpha_distill: 1.0

### configs/experiment/L4_full.yaml
defaults: - /base - /model/teacher@experiment.teacher1: convnext_l - /model/teacher@experiment.teacher2: resnet152 - /model/student@experiment.model.student: resnet101_scratch - _self_ experiment: exp_id: L4_full results_dir: experiments/ablation/L4/results teacher1_ckpt: checkpoints/teachers/convnext_l_cifar100.pth teacher2_ckpt: checkpoints/teachers/resnet152_cifar100.pth dataset: { batch_size: 64, num_workers: 8, data_aug: 1, pin_memory: true, persistent_workers: true, prefetch_factor: 2 } num_stages: 4 student_epochs_per_stage: [20, 20, 20, 20] teacher_adapt_epochs: 6 compute_teacher_eval: true use_amp: true amp_dtype: bfloat16 # KD (synergy target with gating/mix) kd_target: synergy teacher_adapt_kd_warmup: 6 kd_ens_alpha: 0.5 enable_kd_after_syn_acc: 0.8 ce_alpha: 0.65 kd_alpha: 0.35 tau_schedule: [3.5, 5.0] kd_warmup_epochs: 3 kd_max_ratio: 1.25 ce_label_smoothing: 0.0 feat_kd_alpha: 0.25 min_cw: 0.5 max_cw: 1.5 # IB use_ib: true ib_epochs_per_stage: 6 ib_beta: 5e-05 ib_beta_warmup_epochs: 4 ib_mbm_query_dim: 512 ib_mbm_out_dim: 512 ib_mbm_n_head: 4 ib_mbm_feature_norm: l2 ib_mbm_logvar_clip: 4 ib_mbm_min_std: 0.01 ib_mbm_lr_factor: 2 # CCCP use_cccp: true use_cccp_in_a: true cccp_alpha: 0.20 tau: 4.0 # Teacher Adapt(어댑터만) use_teacher_finetuning: false train_distill_adapter_only: true teacher_lr: 3e-06 teacher_weight_decay: 1e-4 # PPF(부분 동결) – 안정/속도/VRAM 절감 use_partial_freeze: true student_freeze_level_schedule: [-1, -1, 1, 1] teacher1_freeze_level_schedule: [-1, -1, 1, 1] teacher2_freeze_level_schedule: [-1, -1, 1, 1] student_freeze_bn: true teacher1_freeze_bn: true teacher2_freeze_bn: true # A‑Step 안정화 synergy_only_epochs: 2 synergy_ce_alpha: 1.0 disable_loss_clamp_in_a: true use_distillation_adapter: true distill_out_dim: 512 feat_kd_key: distill_feat optimizer: adamw student_lr: 0.001 student_weight_decay: 0.0003 a_step_lr: 0.0001 a_step_weight_decay: 0.0001 grad_clip_norm: 0.5 use_loss_clamp: true loss_clamp_mode: soft loss_clamp_max: 20.0 loss_clamp_warmup_epochs: 8 schedule: { type: cosine, lr_warmup_epochs: 5, min_lr: 1e-6 } mixup_alpha: 0.2 cutmix_alpha_distill: 1.0

### configs/experiment/method/ab.yaml
name: ab ce_alpha: 0.65 kd_alpha: 0.35 tau_start: 4.0 tau_end: 1.5 attention_weight: 1.0 feature_weight: 0.5

### configs/experiment/method/asib.yaml
name: asib # KD losses ce_alpha: 0.70 kd_alpha: 0.40 kd_ens_alpha: 0.25 kd_ens_alpha_max: 0.8 tau: 4.0 # ASIB core use_ib: true ib_beta: 5.0e-05 ib_beta_warmup_epochs: 5 # KD target policy kd_target: auto synergy_logit_scale: 0.7 tau_syn: 6.0 # A-Step (teacher adaptation) and gating teacher_adapt_epochs: 6 synergy_only_epochs: 2 enable_kd_after_syn_acc: 0.80 teacher_adapt_kd_warmup: 4 kd_warmup_epochs: 5 # Partial freeze policy (light stabilization) use_partial_freeze: true student_freeze_level: 1 student_freeze_bn: true # Optimizer suggestion for CIFAR-100 scale optimizer: sgd student_lr: 0.1 student_weight_decay: 0.0005 # Misc method params feat_kd_alpha: 0.05 teacher_weights: [0.7, 0.3] use_distillation_adapter: true min_cw: 0.5 max_cw: 1.5 use_mu_for_kd: true kd_max_ratio: 1.75

### configs/experiment/method/at.yaml
name: at ce_alpha: 0.5 kd_alpha: 0.5 at_beta: 1e-1

### configs/experiment/method/crd.yaml
name: crd ce_alpha: 0.65 kd_alpha: 0.35 crd_feat_dim: 512 crd_nce_k: 16384 crd_nce_t: 0.07 crd_momentum: 0.5

### configs/experiment/method/dkd.yaml
name: dkd ce_alpha: 0.5 kd_alpha: 0.5 tau: 4.0 kd_target: teacher feat_kd_alpha: 0.0 dkd_alpha: 1.0 dkd_beta: 2.0

### configs/experiment/method/fitnet.yaml
name: fitnet ce_alpha: 0.65 kd_alpha: 0.35 hint_layer: feat_4d_layer2 hint_beta: 100.0

### configs/experiment/method/ft.yaml
name: ft ce_alpha: 0.65 kd_alpha: 0.35 tau_start: 4.0 tau_end: 1.5 factor_weight: 1.0 feature_weight: 0.5

### configs/experiment/method/reviewkd.yaml
name: reviewkd ce_alpha: 0.65 kd_alpha: 0.35 tau_start: 4.0 tau_end: 1.5 feature_weight: 1.0 attention_weight: 0.5

### configs/experiment/method/simkd.yaml
name: simkd ce_alpha: 0.3 kd_alpha: 0.7 tau_start: 4.0 tau_end: 1.5 feature_weight: 1.0

### configs/experiment/method/sskd.yaml
name: sskd ce_alpha: 0.3 kd_alpha: 0.7 tau_start: 4.0 tau_end: 1.5 contrastive_weight: 1.0 feature_weight: 0.5

### configs/experiment/method/vanilla_kd.yaml
name: vanilla_kd ce_alpha: 0.5 kd_alpha: 0.5 tau_start: 4.0 tau_end: 1.5

### configs/experiment/overlap_100.yaml
# configs/experiment/overlap_100.yaml # Phase 3: 100% Overlap (완전 중복) # T1과 T2 모두 0-99 클래스 전체를 학습 defaults: - /base - /model/teacher@experiment.teacher1: convnext_s - /model/teacher@experiment.teacher2: resnet152 - /model/student@experiment.model.student: resnet101_scratch - _self_ experiment: results_dir: experiments/overlap/100/results exp_id: overlap_100 teacher1_ckpt: checkpoints/teachers/convnext_s_cifar100.pth teacher2_ckpt: checkpoints/teachers/resnet152_cifar100.pth dataset: batch_size: 128 num_workers: 4 num_stages: 2 student_epochs_per_stage: [15, 15] teacher_adapt_epochs: 0 use_partial_freeze: false compute_teacher_eval: true use_amp: true amp_dtype: bfloat16 use_ib: true ib_epochs_per_stage: 5 ib_beta: 0.005 ib_beta_warmup_epochs: 3 use_vib_synergy_head: false # Align heterogeneous teacher feature dims via adapter use_distillation_adapter: true distill_out_dim: 512 ib_mbm_query_dim: 2048 ib_mbm_out_dim: 512 ib_mbm_n_head: 4 ib_mbm_feature_norm: l2 overlap_pct: 100 use_overlap_sampling: true a_step_lr: 0.001 a_step_weight_decay: 0.0001 b_step_lr: 0.05 b_step_weight_decay: 0.0003 b_step_momentum: 0.9 b_step_nesterov: true grad_clip_norm: 1.0 ce_alpha: 0.3 kd_ens_alpha: 0.0 hybrid_beta: 0.0

### configs/experiment/side_cccp_ppf.yaml
defaults: - /base - /model/teacher@experiment.teacher1: convnext_l - /model/teacher@experiment.teacher2: resnet152 - /model/student@experiment.model.student: resnet101_scratch - _self_ experiment: results_dir: experiments/ablation/cccp_ppf/results exp_id: side_cccp_ppf teacher1_ckpt: checkpoints/teachers/convnext_l_cifar100.pth teacher2_ckpt: checkpoints/teachers/resnet152_cifar100.pth dataset: batch_size: 64 num_workers: 8 data_aug: 1 pin_memory: true persistent_workers: true prefetch_factor: 2 # 4-stage(총 40ep) - 안정 수렴 num_stages: 4 student_epochs_per_stage: [20, 20, 20, 20] teacher_adapt_epochs: 6 ib_epochs_per_stage: 6 # KD(안정 세팅: teacher avg) kd_target: avg ce_alpha: 0.65 kd_alpha: 0.35 tau_schedule: [3.5, 5.0] kd_warmup_epochs: 3 kd_max_ratio: 1.25 ce_label_smoothing: 0.0 min_cw: 0.1 feat_kd_alpha: 0.0 use_ib: false ib_beta: 0.00005 ib_beta_warmup_epochs: 4 synergy_only_epochs: 6 synergy_ce_alpha: 1.0 use_vib_synergy_head: false use_cccp: true use_cccp_in_a: true cccp_alpha: 0.20 cccp_nt: 1 cccp_ns: 1 tau: 4.0 # IB‑MBM 용량(시너지 표현력↑) ib_mbm_query_dim: 512 ib_mbm_out_dim: 512 ib_mbm_n_head: 4 ib_mbm_feature_norm: l2 ib_mbm_lr_factor: 10 ib_mbm_min_std: 0.01 ib_mbm_logvar_clip: 4 # Optimizers optimizer: adamw student_lr: 0.001 student_weight_decay: 0.0003 a_step_lr: 0.0001 a_step_weight_decay: 0.0001 grad_clip_norm: 0.5 use_loss_clamp: true loss_clamp_mode: soft loss_clamp_max: 20.0 loss_clamp_warmup_epochs: 8 # Schedule schedule: type: cosine lr_warmup_epochs: 5 min_lr: 1e-6 mixup_alpha: 0.2 cutmix_alpha_distill: 1.0 # AMP use_amp: true amp_dtype: bfloat16 # Distillation/Adapter use_distillation_adapter: true distill_out_dim: 512 feat_kd_key: distill_feat # PPF ON (T‑Adapt 없음) use_partial_freeze: true student_freeze_level: 1 teacher1_freeze_level: 1 teacher2_freeze_level: 1 student_freeze_bn: true teacher1_freeze_bn: true teacher2_freeze_bn: true # 교사 백본 고정 use_teacher_finetuning: false train_distill_adapter_only: false teacher_lr: 0.0 teacher_weight_decay: 0.0 # 안전장치 compute_teacher_eval: true

### configs/experiment/sota_generic.yaml
defaults: - /base - /model/teacher@experiment.teacher1: resnet152 - /model/teacher@experiment.teacher2: convnext_s - /model/student@experiment.model.student: mobilenet_v2_scratch - _self_ # strict-safe: ensure method_name exists at root for runtime sync method_name: null experiment: exp_id: sota_generic results_dir: experiments/sota/generic/results dataset: { name: cifar100, batch_size: 128, num_workers: 4, data_aug: 1 } num_stages: 1 student_epochs_per_stage: [240] # Main SOTA: 교사 파인튜닝/A-Step은 메소드 파일 설정 사용 (루트 고정값 제거) compute_teacher_eval: true use_amp: false amp_dtype: bfloat16 teacher1_ckpt: null teacher2_ckpt: null # 학생 scratch 고정 (공정성) model: student: pretrained: false # 메소드 기본값은 base의 defaults(method@experiment.method: asib)로만 구성 # 교사 파인튜닝/PPF/BN 정책은 메소드 파일이 결정 (루트 강제 OFF 제거) use_teacher_finetuning: false train_distill_adapter_only: false teacher1_freeze_bn: true teacher2_freeze_bn: true # (교사는 항상 고정; BN은 eval) # 공정성: SOTA 비교 시 PPF OFF 기본값 force_ppf_off: true # KD (single-teacher 기본) – kd_target은 메소드 파일이 결정 # kd_target: teacher # teacher | avg | synergy kd_teacher_index: 0 # teacher 모드일 때 0=teacher1, 1=teacher2 ce_alpha: 0.65 kd_alpha: 0.35 kd_warmup_epochs: 3 kd_max_ratio: 1.25 # Global tau schedule as default; methods may override via tau_schedule tau_schedule: [4.0, 4.0] mixup_alpha: 0.2 cutmix_alpha_distill: 1.0 ce_label_smoothing: 0.1 # Optimizer defaults (CIFAR-100 recommended) optimizer: sgd student_lr: 0.1 student_weight_decay: 0.0005 b_step_momentum: 0.9 b_step_nesterov: true # IB/CCCP 설정은 메소드 파일이 결정 (루트 OFF 제거) # 안전 기본치: 강한 압축을 피하기 위해 ib_beta 낮게 설정 (메소드에서 덮어씀) ib_beta: 1.0e-4 # Adapter (작은 학생이면 256 권장; finalize_config에서 정렬) use_distillation_adapter: true distill_out_dim: 256 feat_kd_key: distill_feat

### configs/finetune/convnext_l_cifar100.yaml
# configs/finetune/convnext_l_cifar100.yaml # ConvNeXt‑L 파인튜닝 (CIFAR‑100, 32×32 입력) # @package _global_ defaults: - /dataset: cifar100 - _self_ teacher_type: convnext_l small_input: true teacher_pretrained: true teacher_ckpt_init: null # 이어서 학습할 ckpt 경로가 있으면 지정 # ─── 옵티마이저 & 학습 하이퍼파라미터 ────────────────────────── finetune_epochs: 60 # 큰 모델이므로 충분한 학습 시간 finetune_lr: 8e-5 # 큰 모델이므로 더 낮은 학습률 finetune_weight_decay: 8e-3 # 큰 모델이므로 강한 정규화 warmup_epochs: 5 # 큰 모델이므로 충분한 warmup min_lr: 1e-6 batch_size: 64 # ConvNeXt-L은 메모리 제약으로 작은 배치 label_smoothing: 0.5 # 큰 모델이므로 강한 label smoothing finetune_use_cutmix: true finetune_cutmix_alpha: 1.0 # ─── 고급 스케줄링 설정 ────────────────────────────────────── scheduler_type: onecycle # 가장 효과적인 스케줄링 early_stopping_patience: 15 # 큰 모델이므로 더 긴 patience early_stopping_min_delta: 0.05 # 더 작은 개선도 허용 # ─── 출력 경로 ──────────────────────────────────────────────── results_dir: experiments/finetune/convnext_l_cifar100_ft/results exp_id: convnext_l_cifar100 # ─── 공통 옵션 (AMP 등) ─────────────────────────────────────── use_amp: true device: cuda seed: 42 log_level: INFO deterministic: true # 추후 save 할 체크포인트 경로 finetune_ckpt_path: checkpoints/teachers/convnext_l_cifar100.pth # Hydra 의 편의를 위해 명시(로더에서 사용) dataset_name: cifar100 data_root: ./data

### configs/finetune/convnext_l_imagenet32.yaml
# configs/finetune/convnext_l_imagenet32.yaml # ConvNeXt-L 파인튜닝 (ImageNet‑32, 32×32 입력, 1000 클래스) - 성능 개선 버전 # @package _global_ defaults: - /dataset: imagenet32 - _self_ teacher_type: convnext_l small_input: true teacher_pretrained: true teacher_ckpt_init: null # ─── 성능 개선을 위한 하이퍼파라미터 ────────────────────────── finetune_epochs: 120 # 더 긴 학습 시간 finetune_lr: 4e-5 # 더 낮은 학습률 (안정성) finetune_weight_decay: 1.2e-2 # 더 강한 정규화 finetune_use_cutmix: true finetune_cutmix_alpha: 1.0 label_smoothing: 0.7 # 더 강한 label smoothing warmup_epochs: 10 # 더 긴 warmup min_lr: 4e-7 # ─── 고급 스케줄링 설정 ────────────────────────────────────── scheduler_type: cosine_warm_restarts # 더 효과적인 스케줄러 restart_period: 30 # 30 에포크마다 재시작 restart_multiplier: 0.8 # 재시작 시 LR을 0.8배로 감소 early_stopping_patience: 15 # 더 긴 patience early_stopping_min_delta: 0.05 # 더 작은 개선도 요구 seed: 42 device: cuda log_level: INFO deterministic: true # ─── 출력 경로 ──────────────────────────────────────────────── results_dir: experiments/finetune/convnext_l_imagenet32_ft_improved/results exp_id: convnext_l_imagenet32_improved # 추후 save 할 체크포인트 경로 finetune_ckpt_path: checkpoints/teachers/convnext_l_imagenet32_improved.pth # Hydra 의 편의를 위해 명시(로더에서 사용) dataset_name: imagenet32 data_root: ./data batch_size: 64 # A6000 GPU에서 ConvNeXt-L용 use_amp: true

### configs/finetune/convnext_s_cifar100.yaml
# configs/finetune/convnext_s_cifar100.yaml # ConvNeXt‑S 파인튜닝 (CIFAR‑100, 32×32 입력) # @package _global_ defaults: - /dataset: cifar100 - _self_ teacher_type: convnext_s small_input: true teacher_pretrained: true teacher_ckpt_init: null # ─── 옵티마이저 & 학습 하이퍼파라미터 ────────────────────────── finetune_epochs: 80 finetune_lr: 1.5e-4 finetune_weight_decay: 8e-3 warmup_epochs: 4 min_lr: 1e-6 batch_size: 128 label_smoothing: 0.5 finetune_use_cutmix: true finetune_cutmix_alpha: 1.0 # ─── 고급 스케줄링 설정 ────────────────────────────────────── scheduler_type: reduce_on_plateau early_stopping_patience: 10 early_stopping_min_delta: 0.1 # ─── 출력 경로 ──────────────────────────────────────────────── results_dir: experiments/finetune/convnext_s_cifar100_ft/results exp_id: convnext_s_cifar100 # ─── 공통 옵션 (AMP 등) ─────────────────────────────────────── use_amp: true device: cuda seed: 42 log_level: INFO deterministic: true # 추후 save 할 체크포인트 경로 finetune_ckpt_path: checkpoints/teachers/convnext_s_cifar100.pth # Hydra 의 편의를 위해 명시(로더에서 사용) dataset_name: cifar100 data_root: ./data

### configs/finetune/convnext_s_imagenet32.yaml
# configs/finetune/convnext_s_imagenet32.yaml # ConvNeXt‑S 파인튜닝 (ImageNet‑32, 32×32 입력, 1000 클래스) # @package _global_ defaults: - /dataset: imagenet32 - _self_ teacher_type: convnext_s small_input: true teacher_pretrained: true teacher_ckpt_init: null # 이어서 학습할 ckpt 경로가 있으면 지정 # ─── 옵티마이저 & 학습 하이퍼파라미터 ────────────────────────── finetune_epochs: 120 # 작은 모델이 1000 클래스이므로 매우 긴 학습 시간 finetune_lr: 3e-5 # 작은 모델이 1000 클래스이므로 매우 낮은 학습률 finetune_weight_decay: 1.5e-2 # 작은 모델이 1000 클래스이므로 매우 강한 정규화 warmup_epochs: 10 # 작은 모델이 1000 클래스이므로 매우 긴 warmup min_lr: 3e-7 batch_size: 256 # 작은 모델이므로 큰 배치 가능 label_smoothing: 0.7 # 작은 모델이 1000 클래스이므로 매우 강한 label smoothing finetune_use_cutmix: true finetune_cutmix_alpha: 1.0 # ─── 고급 스케줄링 설정 ────────────────────────────────────── scheduler_type: multistep # 작은 모델에 적합한 단계적 감소 lr_milestones: [40, 80, 100] # 40, 80, 100 에포크에서 LR 감소 lr_gamma: 0.5 # LR을 절반씩 감소 early_stopping_patience: 25 # 작은 모델이 1000 클래스이므로 매우 긴 patience early_stopping_min_delta: 0.03 # 작은 모델이 1000 클래스이므로 매우 작은 개선도 요구 # ─── 출력 경로 ──────────────────────────────────────────────── results_dir: experiments/finetune/convnext_s_imagenet32_ft/results exp_id: convnext_s_imagenet32 # ─── 공통 옵션 (AMP 등) ─────────────────────────────────────── use_amp: true device: cuda seed: 42 log_level: INFO deterministic: true # 추후 save 할 체크포인트 경로 finetune_ckpt_path: checkpoints/teachers/convnext_s_imagenet32.pth # Hydra 의 편의를 위해 명시(로더에서 사용) dataset_name: imagenet32 data_root: ./data

### configs/finetune/efficientnet_l2_cifar100.yaml
# configs/finetune/efficientnet_l2_cifar100.yaml # @package _global_ defaults: - /dataset: cifar100 - _self_ teacher_type: efficientnet_l2 small_input: true # Recommended batch sizes for EfficientNet-L2 are noted in README.md teacher_pretrained: true teacher_ckpt_init: null teacher_use_checkpointing: true finetune_epochs: 65 # 효율적 모델이므로 적당한 학습 시간 finetune_lr: 1.8e-4 # 효율적 모델이므로 약간 높은 학습률 finetune_weight_decay: 3e-3 # 과적합 방지를 위해 증가 finetune_use_cutmix: true finetune_cutmix_alpha: 1.0 label_smoothing: 0.4 # 과적합 방지를 위해 증가 warmup_epochs: 3 min_lr: 1e-6 # ─── 고급 스케줄링 설정 ────────────────────────────────────── scheduler_type: multistep # 명시적이고 예측 가능한 스케줄링 lr_milestones: [20, 40, 55] # 20, 40, 55 epoch에서 LR 감소 lr_gamma: 0.3 # LR을 0.3배씩 감소 early_stopping_patience: 6 # 과적합 방지를 위해 감소 early_stopping_min_delta: 0.15 # 과적합 방지를 위해 증가 seed: 42 device: cuda log_level: INFO deterministic: true # ─── 출력 경로 ──────────────────────────────────────────────── results_dir: experiments/finetune/efficientnet_l2_cifar100_ft/results exp_id: efficientnet_l2_cifar100 # 추후 save 할 체크포인트 경로 finetune_ckpt_path: checkpoints/teachers/efficientnet_l2_cifar100.pth # Hydra 의 편의를 위해 명시(로더에서 사용) dataset_name: cifar100 data_root: ./data batch_size: 32 # A6000 GPU에서 EfficientNet-L2용 use_amp: true

### configs/finetune/efficientnet_l2_imagenet32.yaml
# configs/finetune/efficientnet_l2_imagenet32.yaml # EfficientNet-L2 파인튜닝 (ImageNet‑32, 32×32 입력, 1000 클래스) - 성능 개선 버전 # @package _global_ defaults: - /dataset: imagenet32 - _self_ teacher_type: efficientnet_l2 small_input: true teacher_pretrained: true teacher_ckpt_init: null teacher_use_checkpointing: true # ─── 성능 개선을 위한 하이퍼파라미터 ────────────────────────── finetune_epochs: 100 # 더 긴 학습 시간 finetune_lr: 5e-5 # 더 낮은 학습률 (안정성) finetune_weight_decay: 8e-3 # 더 강한 정규화 finetune_use_cutmix: true finetune_cutmix_alpha: 1.0 label_smoothing: 0.7 # 더 강한 label smoothing warmup_epochs: 10 # 더 긴 warmup min_lr: 5e-7 # ─── 고급 스케줄링 설정 ────────────────────────────────────── scheduler_type: multistep # 1000 클래스에 적합한 단계적 감소 lr_milestones: [40, 70, 90] # 40, 70, 90 에포크에서 LR 감소 lr_gamma: 0.3 # LR을 0.3배씩 감소 early_stopping_patience: 15 # 더 긴 patience early_stopping_min_delta: 0.05 # 더 작은 개선도 요구 seed: 42 device: cuda log_level: INFO deterministic: true # ─── 출력 경로 ──────────────────────────────────────────────── results_dir: experiments/finetune/efficientnet_l2_imagenet32_ft_improved/results exp_id: efficientnet_l2_imagenet32_improved # 추후 save 할 체크포인트 경로 finetune_ckpt_path: checkpoints/teachers/efficientnet_l2_imagenet32_improved.pth # Hydra 의 편의를 위해 명시(로더에서 사용) dataset_name: imagenet32 data_root: ./data batch_size: 32 # A6000 GPU에서 EfficientNet-L2용 use_amp: true

### configs/finetune/resnet152_cifar100.yaml
# configs/finetune/resnet152_cifar100.yaml # 32×32 CIFAR-100 – ResNet-152 fine-tune # @package _global_ defaults: - /dataset: cifar100 # 작은 입력용 dataloader - _self_ # ---------- 모델 ---------- teacher_type: resnet152 # registry key small_input: true # 3×3 conv , stride 1 teacher_pretrained: true teacher_ckpt_init: null # 이어서 학습할 ckpt 경로가 있으면 지정 # ---------- 학습 ---------- finetune_epochs: 70 # 전통적 모델이므로 충분한 학습 시간 finetune_lr: 1.2e-4 # 전통적 모델이므로 적당한 학습률 finetune_weight_decay: 2e-3 # 전통적 모델이므로 적당한 정규화 warmup_epochs: 3 min_lr: 1e-6 batch_size: 64 label_smoothing: 0.3 # 전통적 모델이므로 적당한 label smoothing finetune_use_cutmix: true finetune_cutmix_alpha: 1.0 # ─── 고급 스케줄링 설정 ────────────────────────────────────── scheduler_type: cosine_warm_restarts # 주기적 재시작으로 local minima 탈출 restart_period: 20 # 20 epoch마다 재시작 restart_multiplier: 2 # 재시작 주기 2배씩 증가 early_stopping_patience: 10 # 전통적 모델이므로 적당한 patience early_stopping_min_delta: 0.1 # 적당한 개선도 요구 # ---------- 출력 ---------- results_dir: experiments/finetune/resnet152_cifar100_ft/results exp_id: resnet152_cifar100 finetune_ckpt_path: checkpoints/teachers/resnet152_cifar100.pth # ---------- 기타 ---------- use_amp: true device: cuda seed: 42 log_level: INFO deterministic: true # dataloader 편의용 명시 dataset_name: cifar100 data_root: ./data

### configs/finetune/resnet152_imagenet32.yaml
# configs/finetune/resnet152_imagenet32.yaml # ResNet152 파인튜닝 (ImageNet‑32, 32×32 입력, 1000 클래스) # @package _global_ defaults: - /dataset: imagenet32 - _self_ teacher_type: resnet152 small_input: true teacher_pretrained: true teacher_ckpt_init: null # 이어서 학습할 ckpt 경로가 있으면 지정 # ─── 옵티마이저 & 학습 하이퍼파라미터 ────────────────────────── finetune_epochs: 80 # 1000 클래스이므로 충분한 학습 시간 finetune_lr: 8e-5 # 1000 클래스이므로 낮은 학습률 finetune_weight_decay: 8e-3 # 1000 클래스이므로 강한 정규화 warmup_epochs: 6 # 1000 클래스이므로 긴 warmup min_lr: 8e-7 batch_size: 128 # ResNet152는 메모리 효율적 label_smoothing: 0.5 # 1000 클래스이므로 강한 label smoothing finetune_use_cutmix: true finetune_cutmix_alpha: 1.0 # ─── 고급 스케줄링 설정 ────────────────────────────────────── scheduler_type: reduce_on_plateau # 안정적인 스케줄러 early_stopping_patience: 15 # 1000 클래스이므로 긴 patience early_stopping_min_delta: 0.08 # 1000 클래스이므로 작은 개선도 요구 # ─── 출력 경로 ──────────────────────────────────────────────── results_dir: experiments/finetune/resnet152_imagenet32_ft/results exp_id: resnet152_imagenet32 # ─── 공통 옵션 (AMP 등) ─────────────────────────────────────── use_amp: true device: cuda seed: 42 log_level: INFO deterministic: true # 추후 save 할 체크포인트 경로 finetune_ckpt_path: checkpoints/teachers/resnet152_imagenet32.pth # Hydra 의 편의를 위해 명시(로더에서 사용) dataset_name: imagenet32 data_root: ./data

### configs/hydra.yaml
hydra: run: dir: ${experiment.results_dir}/${now:%Y%m%d_%H%M%S} job: chdir: false

### configs/method/ab.yaml
# configs/method/ab.yaml method: name: ab ce_alpha: 0.3 kd_alpha: 0.7 tau_start: 4.0 tau_end: 1.5 attention_weight: 1.0 feature_weight: 0.5

### configs/method/asib.yaml
method: name: asib ce_alpha: 0.65 kd_alpha: 0.35 # KD 손실 활성화 kd_ens_alpha: 0.0 # IB 설정 use_ib: true # IB 손실 활성화 ib_beta: 1.0e-4 # IB 손실 가중치 feat_kd_alpha: 0.1 # Feature KD 활성화 # KD 타겟 설정 kd_target: "synergy" # "synergy", "avg", "weighted_conf" teacher_weights: [0.4, 0.6] # [R152, ConvNeXt-S] 성능 기반 가중치 # Distillation Adapter use_distillation_adapter: true # Distillation Adapter 사용 # Uncertainty weighting min_cw: 0.1 # 최소 certainty weight

### configs/method/at.yaml
# configs/method/at.yaml method: name: at # Attention Transfer ce_alpha: 0.5 kd_alpha: 0.5 at_beta: 1e-1

### configs/method/crd.yaml
# configs/method/crd.yaml method: name: crd ce_alpha: 0.3 kd_alpha: 0.7 crd_feat_dim: 512 crd_nce_k: 16384 crd_nce_t: 0.07 crd_momentum: 0.5

### configs/method/dkd.yaml
# configs/method/dkd.yaml method: name: dkd ce_alpha: 0.5 kd_alpha: 0.5 tau_start: 6.0 tau_end: 2.0 dkd_alpha: 1.0 dkd_beta: 2.0

### configs/method/fitnet.yaml
# configs/method/fitnet.yaml method: name: fitnet ce_alpha: 0.4 kd_alpha: 0.6 hint_layer: "feat_4d_layer2" # student, teacher 동일 키 hint_beta: 100.0

### configs/method/ft.yaml
# configs/method/ft.yaml method: name: ft ce_alpha: 0.3 kd_alpha: 0.7 tau_start: 4.0 tau_end: 1.5 factor_weight: 1.0 feature_weight: 0.5

### configs/method/reviewkd.yaml
# configs/method/reviewkd.yaml method: name: reviewkd ce_alpha: 0.3 kd_alpha: 0.7 tau_start: 4.0 tau_end: 1.5 feature_weight: 1.0 attention_weight: 0.5

### configs/method/simkd.yaml
# configs/method/simkd.yaml method: name: simkd ce_alpha: 0.3 kd_alpha: 0.7 tau_start: 4.0 tau_end: 1.5 feature_weight: 1.0

### configs/method/sskd.yaml
# configs/method/sskd.yaml method: name: sskd ce_alpha: 0.3 kd_alpha: 0.7 tau_start: 4.0 tau_end: 1.5 contrastive_weight: 1.0 feature_weight: 0.5

### configs/method/vanilla_kd.yaml
# configs/method/vanilla_kd.yaml method: name: vanilla_kd ce_alpha: 0.5 kd_alpha: 0.5 tau_start: 4.0 tau_end: 1.5

### configs/model/student/efficientnet_b0_scratch.yaml
# configs/model/student/efficientnet_b0_scratch.yaml name: efficientnet_b0_scratch pretrained: false use_adapter: true

### configs/model/student/mobilenet_v2_scratch.yaml
# configs/model/student/mobilenet_v2_scratch.yaml name: mobilenet_v2_scratch pretrained: false use_adapter: true

### configs/model/student/resnet101_pretrain.yaml
# configs/model/student/resnet101_pretrain.yaml name: resnet101_pretrain pretrained: true use_adapter: true

### configs/model/student/resnet101_scratch.yaml
# configs/model/student/resnet101_scratch.yaml name: resnet101_scratch pretrained: false use_adapter: true

### configs/model/student/resnet152_pretrain.yaml
# configs/model/student/resnet152_pretrain.yaml name: resnet152_pretrain pretrained: true use_adapter: true

### configs/model/student/resnet152_scratch.yaml
# configs/model/student/resnet152_scratch.yaml name: resnet152_scratch pretrained: false use_adapter: true

### configs/model/student/resnet50_scratch.yaml
# configs/model/student/resnet50_scratch.yaml name: resnet50_scratch pretrained: false use_adapter: true

### configs/model/student/shufflenet_v2_scratch.yaml
# configs/model/student/shufflenet_v2_scratch.yaml name: shufflenet_v2_scratch pretrained: false use_adapter: true

### configs/model/teacher/convnext_l.yaml
# configs/model/teacher/convnext_l.yaml name: convnext_l pretrained: true

### configs/model/teacher/convnext_s.yaml
# configs/model/teacher/convnext_s.yaml name: convnext_s pretrained: true

### configs/model/teacher/efficientnet_l2.yaml
# configs/model/teacher/efficientnet_l2.yaml name: efficientnet_l2 pretrained: true

### configs/model/teacher/resnet152.yaml
# configs/model/teacher/resnet152.yaml name: resnet152 pretrained: true

### configs/registry_key.yaml
# configs/registry_key.yaml student_keys: - resnet50_scratch - resnet101_scratch - resnet101_pretrain - resnet152_pretrain - shufflenet_v2_scratch - mobilenet_v2_scratch - efficientnet_b0_scratch teacher_keys: - convnext_s - resnet152 - efficientnet_l2 - convnext_l

### configs/registry_map.yaml
# configs/registry_map.yaml teachers: convnext_s: models.teachers.convnext_s_teacher.create_convnext_s convnext_l: models.teachers.convnext_l_teacher.create_convnext_l efficientnet_l2: models.teachers.efficientnet_l2_teacher.create_efficientnet_l2 resnet152: models.teachers.resnet152_teacher.create_resnet152 students: resnet152_pretrain: models.students.resnet152_student.create_resnet152_student resnet101_pretrain: models.students.resnet101_student.create_resnet101_student resnet101_scratch: models.students.resnet101_student.create_resnet101_scratch_student resnet50_scratch: models.students.resnet50_student.create_resnet50_student shufflenet_v2_scratch: models.students.shufflenet_v2_student.create_shufflenet_v2_scratch_student mobilenet_v2_scratch: models.students.mobilenet_v2_student.create_mobilenet_v2_scratch_student efficientnet_b0_scratch: models.students.efficientnet_b0_student.create_efficientnet_b0_scratch_student

### configs/schedule/cosine.yaml
# configs/schedule/cosine.yaml type: cosine lr_warmup_epochs: 5 min_lr: 1e-5

### configs/schedule/step.yaml
# configs/schedule/step.yaml type: step lr_warmup_epochs: 5 min_lr: 1e-5 step_size: 30 gamma: 0.1

### configs/anchor/fair_baseline.yaml (핵심 필드)
anchor:
  ce_alpha: 0.65
  kd_alpha: 0.35
  kd_target: avg
  tau: 4.0
  ce_label_smoothing: 0.05
  optimizer: sgd
  student_lr: 0.1
  student_weight_decay: 0.0005
  dataset:
    batch_size: 128
  mixup_alpha: 0.2
  cutmix_alpha_distill: 1.0
  use_partial_freeze: false
  student_freeze_bn: false
  strict_mode: true
  auto_renorm: false

### configs/experiment/method/asib_stage.yaml (핵심 필드)
name: asib_stage
use_ib: true
teacher_adapt_epochs: 10
synergy_only_epochs: 6
enable_kd_after_syn_acc: 0.65
kd_target: auto
kd_alpha: 0.35
kd_warmup_epochs: 5
tau: 4.0
tau_syn: 5.0
use_mu_for_kd: true
mixup_alpha: 0.2
cutmix_alpha_distill: 1.0
use_channels_last: true
dataset:
  backend: torchvision
ib_mbm_lr_factor: 2
label_smoothing: 0.05
synergy_head_dropout: 0.05
