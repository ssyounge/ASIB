## 0) Runtime invariants — Config Lock/Hash

- 효과적 구성은 정규화/정책 적용 후 해시로 잠금됩니다. 검증 시점: `before_run`, `before_safe_retry`, `after_run`.
- 해시 계산 전 sanitize 단계에서 숫자/문자 타입을 실제 런타임과 동일하게 `cast_numeric_configs`로 정규화한 다음, 아래 키/패턴을 제외합니다(허용된 런타임 변이):
  - 단일 키: `config_sha256`, `locked`, `use_amp`, `teacher1_ckpt`, `teacher2_ckpt`, `ib_mbm_out_dim`, `ib_mbm_query_dim`, `auto_align_ib_out_dim`, `_locked_config`, `csv_filename`, `total_time_sec`, `final_student_acc`, `last_synergy_acc`, `last_synergy_acc_pct`, `kd_gate_on`, `optimizer`, `hydra_method`, `cur_stage`, `effective_teacher_lr`, `effective_teacher_wd`, `num_classes`
  - 접두어 패턴: `student_ep*`, `teacher_ep*`, `epoch*`, `csv_*`
- 따라서 SAFE‑RETRY에서 `use_amp` 토글, 교사 ckpt 경로 주입, IB/쿼리/아웃 차원 자동 정렬, 에폭별 로깅/메트릭 추가는 락 위반이 아닙니다.
- 디버그를 위해 잠금 시점의 sanitize된 구성을 `_locked_config`로 보관합니다(해시 제외).

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



