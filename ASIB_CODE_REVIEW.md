## 0) Runtime invariants — Config Lock/Hash

- 효과적 구성은 정규화/정책 적용 후 해시로 잠금됩니다. 검증 시점: `before_run`, `before_safe_retry`, `after_run`.
- 해시 계산 전 sanitize 단계에서 숫자/문자 타입을 실제 런타임과 동일하게 `cast_numeric_configs`로 정규화한 다음, 아래 키/패턴을 제외합니다(허용된 런타임 변이):
  - 단일 키: `config_sha256`, `locked`, `use_amp`, `teacher1_ckpt`, `teacher2_ckpt`, `ib_mbm_out_dim`, `ib_mbm_query_dim`, `auto_align_ib_out_dim`, `_locked_config`, `csv_filename`, `total_time_sec`, `final_student_acc`, `last_synergy_acc`, `last_synergy_acc_pct`, `kd_gate_on`, `optimizer`, `hydra_method`, `cur_stage`, `effective_teacher_lr`, `effective_teacher_wd`, `num_classes`
  - 접두어 패턴: `student_ep*`, `teacher_ep*`, `epoch*`, `csv_*`
- 따라서 SAFE‑RETRY에서 `use_amp` 토글, 교사 ckpt 경로 주입, IB/쿼리/아웃 차원 자동 정렬, 에폭별 로깅/메트릭 추가는 락 위반이 아닙니다.
- 디버그를 위해 잠금 시점의 sanitize된 구성을 `_locked_config`로 보관합니다(해시 제외).

### 2025-08-25 업데이트(핵심 수정 요약)
- SAFE-RETRY: 락 이후 `use_channels_last`를 변경하지 않도록 고정(해시 충돌 방지). SAFE-RETRY에서는 AMP만 off.
- IB_MBM 생성 조건: `kd_target in {"synergy","auto","auto_min"}`일 때 생성하도록 확장(main.py).
- KD 샘플 게이팅 dtype/인덱싱 고정(modules/trainer_student.py):
  - subset scatter에서 `s_logit.new_zeros(...)` 대신 `torch.zeros(..., dtype=kd_vec_sel.dtype)` 사용으로 bf16/float 충돌 제거.
  - `kd_uncertainty_weight` subset 분기에서 `auto_min(choose_syn)` 경로 시 선택된 타깃 로짓으로 `q_sel` 구성(인덱스/shape 정합).
- 러너 개선(run/run_asib_sota_comparison.sh):
  - EPOCHS 환경훅 추가(예: EPOCHS=1 시 +student_epochs=1, ASIB 계열은 +teacher_adapt_epochs=1).
  - Hydra 오버라이드 정규화: `+method@experiment.method=<name>` 사용, 모델 그룹은 `model/teacher@experiment.teacher{1,2}` 및 `model/student@experiment.model.student`(선행 슬래시 금지).
  - DRY_RUN 인덱스 매핑/Slurm 배열 가드/로그 파일 경로 안전화 반영.
  - `dkd`의 ce/kd를 0.65/0.35로 통일.
  - reviewkd/crd의 배치 축소 오버라이드 제거(앵커 락 충돌 방지).
- A‑Step 가드 갱신: `auto_min` 포함, `use_cccp_in_a=true`면 IB 여부와 무관하게 A‑Step 실행.
- eval_synergy EMA 초기화: `last_synergy_acc`가 미측정(prev<0)이면 현재 측정값으로 즉시 초기화 후 EMA 진행.
- B‑Step 시너지 경로 활성: `need_ibm` 판정이 `allow_syn`(게이트) 결과를 그대로 재사용하여 `auto_min`에서도 IB 경로가 실제 실행.
- kd_ens_alpha 혼합: 시너지 게이트 통과 후(`allow_syn=true`)에만 워밍업 혼합 허용.
- A‑Step 직후 로깅 안전 가드: 단일 교사 구성에서도 파라미터 카운트 로깅이 안전하게 동작.
- 메소드 YAML 반영:
  - `asib_stage.yaml`(최신): `kd_target: auto`, `tau_syn: 4.0`, `synergy_logit_scale: 0.80`, `kd_cooldown_epochs: 60`, `kd_uncertainty_weight: 0.5`, `kd_two_view_start_epoch: 20`, `kd_two_view_stop_epoch: 80`, `kd_sample_gate: true`, `kd_sample_thr: 0.85`, `kd_sample_max_ratio: 0.50`, `teacher_weights: [0.7, 0.3]`, `synergy_only_epochs: 8`, `synergy_ce_alpha: 1.0`, `optimizer: adamw`, `student_lr: 0.001`, `student_weight_decay: 0.0003`, `schedule.min_lr: 1e-6`, `cutmix_alpha_distill: 1.0`, `distill_out_dim/ib_mbm_query_dim/ib_mbm_out_dim: 512`, `ib_mbm_lr_factor: 1`, `synergy_temp_learnable: false`.
  - `asib_fair.yaml`: `use_amp: true`, `use_channels_last: true`, `kd_two_view_stop_epoch: 80`, `kd_cooldown_epochs: 60`, `kd_sample_thr: 0.90`, `kd_sample_max_ratio: 0.50`.

### 2025-08-25 추가 업데이트(프레임워크 기본 강화)
- IB_MBM/Synergy 빌드 정책: `need_synergy = (kd_target ∈ {synergy, auto, auto_min}) or use_ib`로 일원화. IB‑KL은 `use_ib`로만 on/off.
- 교사 생성 가드: `kd_target ∈ {synergy, auto, auto_min}`이면 `need_teachers=True`로 승격(설령 `kd_alpha=0,use_ib=false,compute_teacher_eval=false`여도 안전).
- 프로필 기본치(권장):
  - stable: `kd_two_view_start_epoch=20`, `kd_uncertainty_weight=0.3`, `kd_ens_alpha=0.0`, `tau_syn=tau`, `auto_tune_target_ratio=0.35`.
  - balanced/aggressive: 동일 키를 10/1, 0.4/0.5, 0.40/0.45로 조정. 실험 의도에 따라 선택.
  - ASIB 계열(asib/asib_stage) 공통 보수 기본(권장): `synergy_logit_scale=0.8`, `kd_two_view_start_epoch=20`, `kd_uncertainty_weight=0.3`, 프로필 미지정 시 `profile=stable` 권장.
  - 참고: 코드 차원의 setdefault는 보류되었습니다. YAML/실험 템플릿에서 위 값을 기본으로 사용하세요.
- 해시 제외 키 보강: `_profile_applied`, `kd_two_view_start_epoch`, `kd_sample_thr`, `auto_tune_target_ratio`, `lr_log_every`를 해시 제외로 추가.
- 로그 배너: 실행 시 `[AUTO] profile=... tv_start=... tau_syn=... kd_unc_w=...` 한 줄 출력.
- channels_last 가드: `need_teachers`일 때만 교사에 적용(학생은 항상 적용)으로 안전화.

### 추가 보완(코드 반영됨)
- A‑Step 항상 실행: `teacher_adapt_epochs > 0`이면 A‑Step을 무조건 수행(이전 IB/kd_target 조건 가드 제거).
- 시너지 게이트 안전화: `last_synergy_acc <= 0`이면 게이트를 강제로 닫음(초기 미학습 시너지 혼입 방지).
- B‑Step 전 초기화: A‑Step 이후에도 `last_synergy_acc<0`이면 `eval_synergy` 1회로 초기값 측정 후 게이트 기준 설정.
- eval_synergy K>2 지원: 기본 2개 사용. 전부 사용하려면 `synergy_eval_use_all_teachers: true`, 특정만 사용하려면 `synergy_eval_teacher_indices: [0,1,...]`.
- CE‑only 권장치: `synergy_only_epochs` 동안 `synergy_ce_alpha: 1.0` 권장(로그로 경고 안내 추가).
- auto_min 운용 팁: `kd_ens_alpha: 0.0` 권장(혼합은 `synergy` 모드에서만 의미 있음).

## 1) models/ib_mbm.py — IB_MBM / SynergyHead 핵심 블록

```1:142:models/ib_mbm.py
# models/ib_mbm.py
import torch
import torch.nn as nn
from typing import List, Optional, Tuple


class IB_MBM(nn.Module):
    """Information‑Bottleneck Manifold‑Bridging Module."""

    def __init__(
        self,
        q_dim: int,
        kv_dim: int,
        d_emb: int,
        beta: float = 1e-2,
        n_head: int = 1,
        logvar_clip: float = 10.0,
        min_std: float = 1e-4,
    ):
        n_head = int(n_head or 1)
        d_emb = int(d_emb)
        if d_emb % n_head != 0:
            raise ValueError("d_emb must be divisible by n_head")
        super().__init__()
        self.q_proj = nn.Linear(q_dim, d_emb)
        self.kv_proj = nn.Linear(kv_dim, d_emb)
        self.q_norm = nn.LayerNorm(d_emb)
        self.kv_norm = nn.LayerNorm(d_emb)
        self.attn = nn.MultiheadAttention(d_emb, n_head, batch_first=True)
        self.out_norm = nn.LayerNorm(d_emb)
        self.mu = nn.Linear(d_emb, d_emb)
        self.logvar = nn.Linear(d_emb, d_emb)
        self.beta = beta
        self.logvar_clip = float(logvar_clip)
        self.min_std = float(min_std)

    def forward(self, q_feat: torch.Tensor, kv_feats: torch.Tensor, sample: bool = True):
        if q_feat.dim() == 2:
            q = self.q_norm(self.q_proj(q_feat)).unsqueeze(1)
        else:
            q = self.q_norm(self.q_proj(q_feat))

        if kv_feats.dim() == 4:
            batch_size = kv_feats.shape[0]
            kv_feats = kv_feats.view(batch_size, -1)
            kv = self.kv_norm(self.kv_proj(kv_feats)).unsqueeze(1)
        elif kv_feats.dim() == 3:
            # Support both per-token projection (in_features == feat_dim)
            # and flattened projection (in_features == tokens * feat_dim)
            bsz, tokens, feat_dim = kv_feats.shape
            in_features = self.kv_proj.in_features
            if in_features == feat_dim:
                kv = self.kv_norm(self.kv_proj(kv_feats))  # (B, tokens, d_emb)
            elif in_features == tokens * feat_dim:
                kv = self.kv_norm(self.kv_proj(kv_feats.reshape(bsz, tokens * feat_dim))).unsqueeze(1)
            else:
                raise RuntimeError(
                    f"KV projection in_features={in_features} mismatches KV shape (tokens={tokens}, feat={feat_dim})."
                )
        else:
            kv = self.kv_norm(self.kv_proj(kv_feats)).unsqueeze(1)

        syn_raw, _ = self.attn(q, kv, kv)
        # Residual connection with pre-normed q
        syn = self.out_norm(syn_raw + q).squeeze(1)
        mu, logvar = self.mu(syn), torch.clamp(self.logvar(syn), -self.logvar_clip, self.logvar_clip)
        std = torch.exp(0.5 * logvar).clamp_min(self.min_std)
        if self.training and sample:
            z = mu + std * torch.randn_like(std)
        else:
            z = mu
        return z, mu, logvar


class SynergyHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int = 100,
        p: float = 0.0,
        learnable_temp: bool = False,
        temp_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(in_dim, num_classes),
        )
        self.learnable_temp = bool(learnable_temp)
        if self.learnable_temp:
            # log_temp=0 -> temperature=1.0
            self.log_temp = nn.Parameter(torch.tensor(float(temp_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if getattr(self, "learnable_temp", False):
            out = out / torch.exp(self.log_temp).clamp_min(1e-4)
        return out


def build_ib_mbm_from_teachers(
    teachers: List[nn.Module],
    cfg: dict,
    query_dim: Optional[int] = None,
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

## 2) core/builder.py — 학생/교사 빌더 핵심 블록

```1:96:core/builder.py
# core/builder.py
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union

from models.common.base_wrapper import MODEL_REGISTRY
from models.common import registry as _reg
from models import build_ib_mbm_from_teachers as build_from_teachers
from modules.partial_freeze import (
    partial_freeze_teacher_resnet,
    partial_freeze_teacher_efficientnet,
    partial_freeze_student_resnet,
)


def build_model(name: str, **kwargs: Any) -> nn.Module:
    """Build model from registry."""
    # 요청 key 가 아직 없으면 그때 가서 import-scan
    if name not in MODEL_REGISTRY:
        _reg.ensure_scanned()
    
    # timm의 INFO 메시지 억제
    import logging
    original_level = logging.getLogger().level
    logging.getLogger().setLevel(logging.WARNING)
    
    try:
        model = MODEL_REGISTRY[name](**kwargs)
        # 로깅 레벨 복원
        logging.getLogger().setLevel(original_level)
        return model
    except KeyError as exc:
        # 로깅 레벨 복원
        logging.getLogger().setLevel(original_level)
        known = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(
            f"[build_model] Unknown model key '{name}'. "
            f"Available: {known}"
        ) from exc


def create_student_by_name(
    student_name: str,
    pretrained: bool = True,
    small_input: bool = False,
    num_classes: int = 100,
    cfg: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """Create student from :data:`MODEL_REGISTRY`."""

    # Friendly aliases: allow base names to map to scratch registry keys
    NAME_ALIASES: Dict[str, str] = {
        "resnet50": "resnet50_scratch",
        "mobilenet_v2": "mobilenet_v2_scratch",
        "efficientnet_b0": "efficientnet_b0_scratch",
        "shufflenet_v2": "shufflenet_v2_scratch",
    }
    student_name = NAME_ALIASES.get(student_name, student_name)

    try:
        return build_model(
            student_name,
            pretrained=pretrained,
            num_classes=num_classes,
            small_input=small_input,
            cfg=cfg,
        )
    except ValueError as exc:
        raise ValueError(
            f"[create_student_by_name] '{student_name}' not in registry"
        ) from exc


def create_teacher_by_name(
    teacher_name: str,
    num_classes: int = 100,
    pretrained: bool = True,
    small_input: bool = False,
    cfg: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """Create teacher from :data:`MODEL_REGISTRY`."""

    try:
        return build_model(
            teacher_name,
            num_classes=num_classes,
            pretrained=pretrained,
            small_input=small_input,
            cfg=cfg,
        )
    except ValueError as exc:
        raise ValueError(
            f"[create_teacher_by_name] '{teacher_name}' not in registry"
        ) from exc
```

## 3) modules/trainer_teacher.py — eval_synergy / A‑Step 핵심 블록

```133:197:modules/trainer_teacher.py
@torch.no_grad()
def eval_synergy(
    teacher_wrappers,
    ib_mbm,
    synergy_head,
    loader,
    device="cuda",
    cfg=None,
    student_model=None,
):
    """Evaluate synergy accuracy (AMP off, float32)."""
    ...
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with autocast_ctx:
            t1_out = teacher_wrappers[0](x)
            t2_out = teacher_wrappers[1](x)
            t1_dict = t1_out[0] if isinstance(t1_out, tuple) else t1_out
            t2_dict = t2_out[0] if isinstance(t2_out, tuple) else t2_out
            key = ("distill_feat" if (cfg or {}).get("use_distillation_adapter", False) else "feat_2d")
            t1_feat = t1_dict[key].float()
            t2_feat = t2_dict[key].float()
            assert student_model is not None, "student_model required for IB_MBM"
            s_feat = student_model(x)[0][(cfg or {}).get("feat_kd_key", "feat_2d")].float()
            fsyn, _, _ = ib_mbm(s_feat, torch.stack([t1_feat, t2_feat], dim=1))
            zsyn = synergy_head(fsyn).float()
    ...

def teacher_adaptive_update(...):
    ...
    # (C) IB_MBM + synergy_head
    kv = torch.stack([t1, t2], dim=1)
    syn_feat, mu, logvar = ib_mbm(s_feat, kv)
    ...
    # CE/KD/IB, CCCP, certainty weights, clamping
    ...
    # per‑epoch synergy eval and logging (last_synergy_acc)
    synergy_test_acc = eval_synergy(...)
    logger.update_metric("last_synergy_acc", float(synergy_test_acc) / 100.0)
    ...
```

### A‑Step 로그/EMA/스냅샷 관련 추가 사항 (동기화됨)

- EMA 업데이트 가드: 평가를 건너뛴 에폭(`do_eval=false`)은 EMA 업데이트 생략하여 음수 값 유입 방지
- zero_grad 최적화: `optimizer.zero_grad(set_to_none=True)` 적용
- backup_state CPU 저장: best/backup 스냅샷 모두 `_cpu_state_dict`로 저장하여 VRAM 파편화 방지
- 첫 스텝 요약 로그(ep=1, step=1):
  `[A-Step ep=1/step=1] kd_gate_on=... kd_weight=... synergy_ce_alpha=... cw_mean/std ... raw_kld ... ib_beta ...`

라벨 스무딩 키 구분
- A-Step: `label_smoothing` (교사 적응 CE에 사용)
- B-Step: `ce_label_smoothing` (학생 CE에 사용)
설정 파일에서 두 키를 구분해 사용하세요(통일하려면 한 키로 리팩토링 가능).

### 2025-08-26 업데이트(ablations/runners 동기화)
- Ablation 공통 안정화(모든 L0/L1/L2/L3/L4/side):
  - DataLoader workers: `dataset.num_workers=8`
  - Clamp(A‑Step): `use_loss_clamp=true`, `loss_clamp_mode=soft`, `disable_loss_clamp_in_a=false`, `loss_clamp_warmup_epochs=0`
  - Adapter: `use_distillation_adapter=true`, `distill_out_dim=512`, `feat_kd_key=distill_feat`
  - Optim/Schedule: `optimizer=adamw`, `student_lr=0.001`, `student_weight_decay=0.0003`, `schedule: {type: cosine, lr_warmup_epochs: 5, min_lr: 1e-6}`
  - KD 합치기: `ce_alpha: 0.70`, `kd_alpha: 0.30`
  - 시너지 예열: `synergy_only_epochs: 8`
  - Rung A‑Step 길이: `teacher_adapt_epochs: 8`
- 사다리(rung)별 차이(대표):
  - L1: `synergy_logit_scale=0.85`, `a_step_lr=5e-5`, `ib_mbm_lr_factor=1`, `loss_clamp_max=30.0`
  - L2: `+CCCP(in_a=true)`, `a_step_lr=1e-4`, `ib_mbm_lr_factor=2`, `loss_clamp_max=35.0`
  - L3: `+two_view(start=20, stop=80)`, `adapter_only`, `a_step_lr=1e-4`, `ib_mbm_lr_factor=2`, `loss_clamp_max=40.0`
  - L4: `+EMA/uncertainty/featKD/PPF`, `two_view start=30/stop=80`, `a_step_lr=5e-5`, `loss_clamp_max=50.0`, `synergy_logit_scale=0.80`
  - side_asib_cccp: `ASIB+CCCP`, `ib_epochs_per_stage=4`, `PPF OFF`, `loss_clamp_max=35.0`
- Runner(ablation): Slurm 배열/DRY_RUN/샤딩 동일. DataLoader workers는 YAML로 4로 통일(경고 제거). 배열 제출 전 DRY_RUN으로 매핑 확인 권장.

## 4) configs/experiment/method/asib.yaml — ASIB 주요 하이퍼

```1:48:configs/experiment/method/asib.yaml
name: asib

# KD losses
ce_alpha: 0.70
kd_alpha: 0.40
kd_ens_alpha: 0.25
kd_ens_alpha_max: 0.8
tau: 4.0

# ASIB core
use_ib: true
ib_beta: 5.0e-05
ib_beta_warmup_epochs: 5
# KD target policy
# Use 'auto' to prefer avg early and switch/mix to synergy when good
kd_target: auto
synergy_logit_scale: 0.7
tau_syn: 6.0

# A-Step (teacher adaptation) and gating
teacher_adapt_epochs: 6
synergy_only_epochs: 2
enable_kd_after_syn_acc: 0.80
teacher_adapt_kd_warmup: 4
kd_warmup_epochs: 5

# Partial freeze policy (light stabilization)
use_partial_freeze: true
student_freeze_level: 1
student_freeze_bn: true

# Optimizer suggestion
optimizer: sgd
student_lr: 0.1
student_weight_decay: 0.0005

# Misc
feat_kd_alpha: 0.05
teacher_weights: [0.7, 0.3]
use_distillation_adapter: true
min_cw: 0.5
max_cw: 1.5
use_mu_for_kd: true
kd_max_ratio: 1.75
```

## 5) data/cifar100.py — Transform/Loader 요약

```120:171:data/cifar100.py
def get_cifar100_loaders(root="./data", batch_size=128, num_workers=2, augment=True, use_spawn_dl: bool = False):
    # 표준 CIFAR-100 증강: RandomCrop(32,pad=4) + HorizontalFlip + Normalize
    if augment:
        transform_train = T.Compose([
            T.RandomCrop(32, padding=4, padding_mode='reflect'),
            T.RandomHorizontalFlip(p=0.5),
            CustomToTensor(),
            T.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))
        ])
    else:
        transform_train = T.Compose([
            CustomToTensor(),
            T.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))
        ])
    transform_test = T.Compose([
        CustomToTensor(),
        T.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))
    ])
    ...
    train_loader = torch.utils.data.DataLoader(...)
    test_loader  = torch.utils.data.DataLoader(...)
    return train_loader, test_loader
```

---

이 문서는 분석/튜닝을 위한 핵심 코드 조각을 모아둔 것입니다. 필요 시 어댑터 정의(`models/adapters.py`), KD 손실/클램프(`modules/trainer_student.py`, `modules/losses.py`)도 추가로 정리할 수 있습니다.

### 2025-08-22 업데이트 요약(코드 ↔ 문서 동기화)

- 학생 MobileNetV2 수정: 분류 경로 1280ch 유지, distill 분기는 1D 어댑터 분리. CIFAR‑size 입력 시 첫 conv stride=1 적용. 레지스트리 별칭 `@register("mobilenet_v2_scratch")` 추가.
- asib_stage 기본 레시피/안정화 YAML로 이전: mixup=0.2, cutmix=1.0, ce_label_smoothing=0.05, dataset.backend=torchvision, `ib_mbm_lr_factor=2`, A‑Step `label_smoothing=0.05`, `synergy_head_dropout=0.05`.
- anchor 공정 락 정렬: `ce_label_smoothing=0.05`, `mixup_alpha=0.2`, `cutmix_alpha_distill=1.0`로 상향(Protected 충돌 해소).
- 데이터셋 호환 보강: `CIFAR100Dataset`가 `classes`/`num_classes`를 노출. `main.py`는 Subset/ConcatDataset/래퍼를 벗겨가며 클래스 수를 추론(최후 가드=100).

클래스 수 추론(변경 후) 발췌:
```10040:10080:main.py
def _infer_num_classes_from(ds) -> int | None:
    nc = getattr(ds, "num_classes", None)
    if isinstance(nc, int):
        return nc
    cls = getattr(ds, "classes", None)
    if cls is not None:
        return (len(cls) if not isinstance(cls, int) else cls)
    return None

base_ds = train_loader.dataset
num_classes = None
import torch.utils.data as _tud
while ...:
    n = _infer_num_classes_from(base_ds)
    if n is not None: num_classes = int(n); break
    if isinstance(base_ds, _tud.Subset): base_ds = base_ds.dataset
    elif isinstance(base_ds, _tud.ConcatDataset):
        for ch in base_ds.datasets:
            n = _infer_num_classes_from(ch)
            if n is not None: num_classes = int(n); break
        break
    elif hasattr(base_ds, "dataset"): base_ds = base_ds.dataset
    else: break
if num_classes is None: num_classes = 100
```

SynergyHead 드롭아웃: `synergy_head_dropout`(기본 0.05)을 사용해 드롭아웃 확률을 구성합니다.

## 6) modules/trainer_student.py — KD/클램프/게이팅 핵심 블록

```1:130:modules/trainer_student.py
# modules/trainer_student.py
import torch
import torch.nn.functional as F
import copy
import logging
from utils.common import smart_tqdm, mixup_data, cutmix_data, get_amp_components
from utils.training import get_tau, get_beta, StageMeter

from modules.losses import (
    soft_clip_loss
)
from modules.disagreement import sample_weights_from_disagreement
from torch.amp import autocast

...
class StudentTrainer:
    ...

def ce_safe_vec(...):
    ...

def kl_safe_vec(...):
    ...
```

```178:246:modules/trainer_student.py
def get_kd_target(...):
    """Select KD target with synergy gating and warmup mixing."""
    ...
```

### 6.x B‑Step 성능 최적화(A–D) — 최신 반영

- A) KD/Synergy clean‑view 재사용: `kd_view == syn_view`인 경우 교사 clean‑view를 한 번만 추론하고, 시너지용은 실제 교사 2개 분량만 재사용(EMA 제외). 중복 추론 제거.
- B) two_view B‑view 비용 절감: B‑view에서 EMA 제외 기본값. `kd_two_view_include_ema_in_b: false`(기본). 필요 시 true로 토글.
- C) 쿨다운 완전 스킵: 에폭 단위 `kd_alpha_eff_epoch`를 계산해 0이면 KD/IB 전체 경로를 스킵. `student_ep{N}_kd_alpha_eff_epoch` 로깅.
- D) two_view 구간 제한: `kd_two_view_stop_epoch` 이후에는 clean 모드로 고정(전반부만 two_view 사용).

### 6.y B‑Step 추가 최적화(T7) — 샘플 게이팅 및 재가중(2025‑08)

- 샘플 게이팅(`kd_sample_gate`): 학생 확신이 높은 샘플은 KD에서 제외하고, 선택된 부분집합에 대해서만 교사/IB 경로 수행.
  - 선택 비율/임계치: `kd_sample_thr`, `kd_sample_max_ratio`, `kd_sample_min_ratio`
  - 두‑뷰 호환: subset에서 A‑view는 재사용, B‑view는 파트너 인덱스만 추가 추론 → `kd_vec_sel`을 full로 scatter
  - 가중치/손실 정합: disagreement, IB‑certainty(cw), KD 불확실도(`kd_uncertainty_weight`), μ‑MSE 모두 subset→full 확장 처리
  - KD 재가중 평균: `kd_sample_reweight: true`면 선택 샘플 평균으로 KD 손실을 산출해 스케일 보존
- 쿨다운 연동: `kd_cooldown_epochs`로 말기 KD 스킵(need_teachers=False)
- 운영 스위치(요약): `kd_sample_gate`, `kd_sample_thr`, `kd_sample_max_ratio`, `kd_sample_min_ratio`, `kd_sample_reweight`, `kd_two_view_stop_epoch`, `kd_cooldown_epochs`, `ema_update_every`

부가 업데이트(학습 신호 품질)
- auto_min 라벨‑CE 라우팅: `kd_auto_policy: label_ce`일 때 mixup‑aware 라벨 CE로 승자(target) 선택 후, 선택된 타깃에 대한 KL을 최적화.
- DKD 하이브리드 바인딩: `use_dkd_with_synergy: true`일 때 auto_min에서 per‑sample 선택 로짓을 `kd_tgt`로 바인딩해 DKD 적용.
- 라벨 정확도 가드: `kd_correct_min`을 통해 teacher 타깃의 라벨 확률 `p_y`로 KD 가중을 하한 보정(mixup‑aware 집계 지원).
- kdSyn 일원화: per‑sample 선택률(`syn_chosen_samples/total_kd_samples`)로 에폭 요약 및 W&B 모두 통일.

```270:866:modules/trainer_student.py
def student_distillation_update(...):
    """Train the student model via knowledge distillation."""
    ...
```

## 7) modules/losses.py — CE/KD/IB 및 보조 손실

```1:343:modules/losses.py
# modules/losses.py
import torch
import torch.nn.functional as F
from typing import Optional

def soft_clip_loss(loss: torch.Tensor, max_val: float) -> torch.Tensor:
    ...

def ce_loss_fn(student_logits, labels, label_smoothing: float = 0.0, reduction: str = "mean"):
    ...

def kd_loss_fn(student_logits, teacher_logits, T=4.0, reduction="batchmean"):
    ...

def hybrid_kd_loss_fn(student_logits, teacher_logits, labels, alpha=0.5, T=4.0):
    ...

def feat_mse_loss(s_feat, t_feat, norm: str = "none", reduction="mean"):
    ...

def ib_loss(mu, logvar, beta: float = 1e-3):
    ...

def kl_loss(student_logits, teacher_logits, temperature=4.0):
    ...

def mse_loss(student_feat, teacher_feat):
    ...

def contrastive_loss(student_feat, teacher_feat, temperature=0.1):
    ...

def attention_loss(student_attn, teacher_attn):
    ...

def factor_transfer_loss(student_factor, teacher_factor):
    ...

def certainty_weights(logvar: torch.Tensor) -> torch.Tensor:
    ...

def dkd_loss(student_logits, teacher_logits, labels, alpha=1.0, beta=1.0, temperature=4.0):
    ...

def rkd_distance_loss(student_feat, teacher_feat, eps: float = 1e-12, reduction: str = "mean", max_clip: Optional[float] = None):
    ...

def rkd_angle_loss(student_feat, teacher_feat, eps: float = 1e-12, reduction: str = "mean"):
    ...

def masked_kd_loss(student_logits, teacher_logits, seen_classes, T=4.0, reduction="batchmean"):
    ...

def class_il_total_loss(student_logits, teacher_logits, targets, seen_classes, ce_weight=1.0, kd_weight=1.0, T=4.0):
    ...
```



