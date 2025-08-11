# methods/asib.py

import copy
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import torch.nn.functional as F

from modules.losses import (
    kd_loss_fn,
    ce_loss_fn,
    rkd_distance_loss,
    rkd_angle_loss,
    feat_mse_loss,
)
from utils.training import get_tau, get_beta
from utils.common import get_amp_components
# LightweightAttnMBM is ignored

class ASIBDistiller(nn.Module):
    """
    Adaptive Synergy Information-Bottleneck (ASIB) Distiller
    - Teacher가 2명(teacher1, teacher2)
    - Student
    - IB-MBM(Information-Bottleneck Manifold Bridging Module), synergy_head
    - 여러 stage에 걸쳐 (A) teacher update, (B) student distillation
    - 최종적으로 student의 Acc를 높이는 KD 프레임워크
    """

    def __init__(
        self,
        teacher1,
        teacher2,
        student,
        mbm,
        synergy_head,
        alpha=0.5,               # student CE vs. KL 비율
        synergy_ce_alpha=0.3,    # teacher 시너지 CE 비중
        temperature=4.0,
        reg_lambda=1e-4,
        mbm_reg_lambda=1e-4,
        num_stages=2,
        device="cuda",
        config=None
    ):
        super().__init__()
        self.teacher1 = teacher1
        self.teacher2 = teacher2
        self.student  = student
        self.mbm = mbm
        self.synergy_head = synergy_head
        # LightweightAttnMBM 기능은 제거되었습니다.
        self.la_mode = False
        assert not self.la_mode, "LA\u2011MBM support has been completely removed."

        # 하이퍼파라미터
        cfg = config or {}
        self.alpha = cfg.get("ce_alpha", alpha)
        self.synergy_ce_alpha = cfg.get("synergy_ce_alpha", synergy_ce_alpha)
        # adaptive 단계에서 KL(syn‖student)을 언제부터 켤지 결정한다
        self.kd_warmup_stage = cfg.get("teacher_adapt_kd_warmup", 2)
        self.T = cfg.get("tau_start", temperature)
        self.reg_lambda = cfg.get("reg_lambda", reg_lambda)
        self.mbm_reg_lambda = cfg.get("ib_mbm_reg_lambda", mbm_reg_lambda)
        self.num_stages = cfg.get("num_stages", num_stages)
        self.device = device
        self.config = config if config is not None else {}
        self.log_int = int(self.config.get("log_step_interval", 100))
        self.logger = self.config.get("logger")
        # ─────── DEBUG 토글 ───────
        self.debug = bool(self.config.get("debug_verbose", True))

        # 기본 Loss
        self.ce_loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        """
        (단발성 forward) => Student Loss만 계산 (Teacher는 이미 학습됐다고 가정)
        - MBM 통해 synergy logit 얻고 => KL with student
        - + CE(student, y)
        => total_loss, student_logit
        """
        # 1) teacher feats (extract only features, not logits)
        with torch.no_grad():
            t1_out = self.teacher1.extract_feats(x)
            t2_out = self.teacher2.extract_feats(x)
            if isinstance(t1_out, (list, tuple)) and len(t1_out) >= 2:
                t1_f4d, t1_f2d = t1_out[0], t1_out[1]
            else:
                t1_f4d, t1_f2d = None, t1_out
            if isinstance(t2_out, (list, tuple)) and len(t2_out) >= 2:
                t2_f4d, t2_f2d = t2_out[0], t2_out[1]
            else:
                t2_f4d, t2_f2d = None, t2_out
            # Normalize teacher features before stacking
            norm_mode = str(self.config.get("ib_mbm_feature_norm", "l2")).lower()
            if norm_mode == "l2":
                t1_f2d = F.normalize(t1_f2d, dim=-1)
                t2_f2d = F.normalize(t2_f2d, dim=-1)
            elif norm_mode == "layernorm":
                t1_f2d = torch.nn.functional.layer_norm(t1_f2d, (t1_f2d.shape[-1],))
                t2_f2d = torch.nn.functional.layer_norm(t2_f2d, (t2_f2d.shape[-1],))
            # Use stacked 3D (batch, 2, feat_dim) as KV for MBM
            feats_2d = torch.stack([t1_f2d, t2_f2d], dim=1)

        # 3) student (query feature)
        feat_dict, s_logit, _ = self.student(x)
        key = self.config.get("feat_kd_key", "feat_2d")
        s_feat = feat_dict[key]
        norm_mode = str(self.config.get("ib_mbm_feature_norm", "l2")).lower()
        if norm_mode == "l2":
            s_feat = F.normalize(s_feat, dim=-1)
        elif norm_mode == "layernorm":
            s_feat = torch.nn.functional.layer_norm(s_feat, (s_feat.shape[-1],))

        # 2) MBM → simple MLP head (no VIB in head)
        syn_feat, _, _ = self.mbm(s_feat, feats_2d)
        zsyn = self.synergy_head(syn_feat)

        # CE
        ce_val = (
            self.ce_loss_fn(s_logit, y)
            if y is not None
            else torch.tensor(0.0, device=s_logit.device)
        )

        # KL with stop‑grad to prevent gradients flowing into MBM/Head in this path
        kd_val = kd_loss_fn(s_logit, zsyn.detach(), T=self.T, reduction="batchmean")
        total_loss = (
            self.alpha * ce_val
            + (1 - self.alpha) * kd_val
        )

        return total_loss, s_logit

    def train_distillation(
        self,
        train_loader,
        test_loader=None,
        teacher_lr=1e-4,
        student_lr=5e-4,
        weight_decay=1e-4,
        epochs_per_stage=10,
        logger=None,
    ):
        """
        단순화된 2‑스텝 학습 루프:
         - A‑Step (IB 학습): 교사·학생 동결, IB‑MBM + SynergyHead만 학습 (AdamW, 낮은 LR)
         - B‑Step (학생 학습): 교사·IB 동결, 학생만 학습 (SGD, 높은 LR)
        """
        # GPU로 이동
        self.to(self.device)
        self.teacher1.to(self.device)
        self.teacher2.to(self.device)
        self.mbm.to(self.device)
        self.synergy_head.to(self.device)
        self.student.to(self.device)

        # ---- Optimizers (분리) ----
        # A‑Step: IB‑MBM + Head
        a_lr = self.config.get("a_step_lr", 1e-4)
        a_wd = self.config.get("a_step_weight_decay", 1e-4)
        params_a = list(self.mbm.parameters()) + list(self.synergy_head.parameters())
        optA = optim.AdamW(params_a, lr=a_lr, weight_decay=a_wd)

        # B‑Step: Student SGD
        b_lr = self.config.get("b_step_lr", 0.1)
        b_wd = self.config.get("b_step_weight_decay", 3e-4)
        b_mom = self.config.get("b_step_momentum", 0.9)
        b_nes = self.config.get("b_step_nesterov", True)
        params_b = [p for p in self.student.parameters() if p.requires_grad]
        optB = optim.SGD(params_b, lr=b_lr, momentum=b_mom, nesterov=b_nes, weight_decay=b_wd)

        best_acc = 0.0
        best_student_state = copy.deepcopy(self.student.state_dict())

        for stage in range(1, self.num_stages+1):
            if self.logger:
                self.logger.info(f"\n[ASIB] Stage {stage}/{self.num_stages} 시작.")

            # (A) IB Step – 교사·학생 동결, IB‑MBM + Head 학습
            ib_epochs = self.config.get("ib_epochs_per_stage", max(1, epochs_per_stage // 3))
            self._ib_update(
                train_loader,
                optimizer=optA,
                epochs=ib_epochs,
                logger=self.logger,
                stage=stage,
            )

            # (B) Student Step – 교사·IB 동결, 학생만 학습
            stu_epochs = self.config.get("student_epochs_per_stage", epochs_per_stage)
            student_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optB, T_max=stu_epochs
            )
            acc = self._student_distill_update(
                train_loader,
                test_loader=test_loader,
                optimizer=optB,
                scheduler=student_scheduler,
                epochs=stu_epochs,
                logger=self.logger,
                label_smoothing=self.config.get("label_smoothing", 0.0),
                stage=stage,
            )
            if acc > best_acc:
                best_acc = acc
                best_student_state = copy.deepcopy(self.student.state_dict())

            for m in (self.teacher1, self.teacher2, self.mbm, self.synergy_head):
                for p in m.parameters():
                    p.requires_grad = True

        # 마지막에 best 복원
        self.student.load_state_dict(best_student_state)
        if self.logger:
            self.logger.info(f"[ASIB] Done. best student acc= {best_acc:.2f}")
        return best_acc

    def _ib_update(
        self,
        train_loader,
        optimizer,
        epochs,
        logger=None,
        *,
        stage: int = 1,
    ):
        """
        A‑Step: IB 학습 전용 업데이트.
        - Freeze teacher and student; train only MBM + synergy head.
        - Loss: KL(zsyn || s_logit) + synergy_ce_alpha * CE(zsyn, y) + vib_kl_weight * VIB_KL
        """
        # Freeze
        for m in (self.teacher1, self.teacher2, self.student):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
        self.mbm.train()
        self.synergy_head.train()
        for p in self.mbm.parameters():
            p.requires_grad = True
        for p in self.synergy_head.parameters():
            p.requires_grad = True

        autocast_ctx, scaler = get_amp_components(self.config)
        logger = logger or self.logger

        for ep in range(1, epochs + 1):
            if self.debug:
                logging.debug("[IB] ====== A‑Step ep %s/%s ======", ep, epochs)
            cur_tau = get_tau(
                self.config,
                epoch=(stage - 1) * epochs + (ep - 1),
                total_epochs=self.num_stages * epochs,
            )
            cur_beta = get_beta(self.config, epoch=(stage - 1) * epochs + (ep - 1))
            cur_beta = get_beta(self.config, epoch=(stage - 1) * epochs + (ep - 1))
            total_loss, total_num = 0.0, 0
            for it, (x, y) in enumerate(train_loader):
                x, y = x.to(self.device), y.to(self.device)

                with autocast_ctx:
                    with torch.no_grad():
                        # student feature + logit (frozen)
                        feat_dict, s_logit, _ = self.student(x)
                        s_feat = feat_dict[self.config.get("feat_kd_key", "feat_2d")]
                        # teacher feats (features only)
                        use_da = self.config.get("use_distillation_adapter", False)
                        t1_out = self.teacher1.extract_feats(x)
                        t2_out = self.teacher2.extract_feats(x)
                        if isinstance(t1_out, (list, tuple)) and len(t1_out) >= 2:
                            t1_f4d, t1_f2d = t1_out[0], t1_out[1]
                        else:
                            t1_f4d, t1_f2d = None, t1_out
                        if isinstance(t2_out, (list, tuple)) and len(t2_out) >= 2:
                            t2_f4d, t2_f2d = t2_out[0], t2_out[1]
                        else:
                            t2_f4d, t2_f2d = None, t2_out
                        if use_da and hasattr(self.teacher1, "distillation_adapter") and hasattr(self.teacher2, "distillation_adapter"):
                            t1_kv = self.teacher1.distillation_adapter(t1_f2d)
                            t2_kv = self.teacher2.distillation_adapter(t2_f2d)
                        else:
                            t1_kv, t2_kv = t1_f2d, t2_f2d
                        kv = torch.stack([t1_kv, t2_kv], dim=1)
                        # teacher ensemble logits (target for optional alignment)
                        t1_logit = self.teacher1.classifier(t1_f2d)
                        t2_logit = self.teacher2.classifier(t2_f2d)
                        t_avg_logit = 0.5 * (t1_logit + t2_logit)

                    # MBM + Head forward (VIB solely in MBM)
                    syn_feat, mu, logvar = self.mbm(s_feat, kv)
                    zsyn = self.synergy_head(syn_feat)

                    # Losses
                    ce_val = ce_loss_fn(
                        zsyn,
                        y,
                        label_smoothing=self.config.get("label_smoothing", 0.0),
                    )
                    synergy_ce = self.synergy_ce_alpha * ce_val
                    vib_kl_weight = float(self.config.get("vib_kl_weight", 1.0))
                    # MBM KL (mean-normalized)
                    kl_mbm = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
                    # Optional weak alignment term (default 0): teacher-ensemble or student
                    lambda_TK = float(self.config.get("lambda_tk_align", 0.0))
                    align_target = t_avg_logit if self.config.get("a_step_align_target", "teacher") == "teacher" else s_logit
                    kl_to_target = lambda_TK * kd_loss_fn(zsyn, align_target, T=cur_tau)
                    # β warmup로 VIB(IB) KL의 기여도 조절 – MBM KL only
                    loss = synergy_ce + (vib_kl_weight * cur_beta) * kl_mbm + kl_to_target

                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, self.mbm.parameters()), max_norm=2.0
                    )
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, self.synergy_head.parameters()), max_norm=2.0
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, self.mbm.parameters()), max_norm=2.0
                    )
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, self.synergy_head.parameters()), max_norm=2.0
                    )
                    optimizer.step()

                bs = x.size(0)
                total_loss += loss.item() * bs
                total_num += bs

            avg_loss = total_loss / max(1, total_num)
            if logger:
                logger.info(f"[IB] ep={ep} => loss={avg_loss:.4f}")

    def _teacher_adaptive_update(
        self,
        train_loader,
        optimizer,
        epochs,
        logger=None,
        *,
        stage: int = 1,
    ):
        """
        Teacher adaptive update (BN/Head + MBM).
        - Student는 고정, Teacher만 업데이트
        - synergy 로짓 vs. Student 로짓 => KL
        - synergy 로짓 vs. GT => synergy_ce_alpha * CE
        - L2 정규화(reg_lambda)
        """
        # 1) Teacher/MBM 파라미터만 requires_grad=True 여야 함
        #    └ teach_params / mbm_params 를 별도로 기록해 L2 계수를 분리
        params, teach_params, mbm_params = [], [], []
        use_da = self.config.get("use_distillation_adapter", False)
        # teacher1
        param_src = (
            self.teacher1.distillation_adapter.parameters()
            if use_da and hasattr(self.teacher1, "distillation_adapter")
            else self.teacher1.parameters()
        )
        for p in param_src:
            if p.requires_grad:
                params.append(p)
                teach_params.append(p)          # ← Teacher 전용
        # teacher2
        param_src = (
            self.teacher2.distillation_adapter.parameters()
            if use_da and hasattr(self.teacher2, "distillation_adapter")
            else self.teacher2.parameters()
        )
        for p in param_src:
            if p.requires_grad:
                params.append(p)
                teach_params.append(p)          # ← Teacher 전용
        # mbm, synergy_head
        for p in self.mbm.parameters():
            if p.requires_grad:
                params.append(p)
                mbm_params.append(p)            # ← MBM / synergy 용
        for p in self.synergy_head.parameters():
            if p.requires_grad:
                params.append(p)
                mbm_params.append(p)

        # ``optimizer`` is constructed outside and shared across stages

        self.teacher1.train()
        self.teacher2.train()
        logger = logger or self.logger
        self.mbm.train()
        self.synergy_head.train()
        self.student.eval()   # Student 고정

        autocast_ctx, scaler = get_amp_components(self.config)

        for ep in range(1, epochs+1):
            if self.debug:
                logging.debug(
                    "[Teacher] ====== Stage-Teacher ep %s/%s ======", ep, epochs
                )
            # 전 스테이지 누적 epoch = (stage-1)·epochs + (ep-1)
            cur_tau = get_tau(
                self.config,
                epoch=(stage - 1) * epochs + (ep - 1),
                total_epochs=self.num_stages * epochs,
            )
            total_loss, total_num = 0.0, 0
            # L2 정규화는 매 batch 재계산해 그래프 중복 backward 오류를 방지
            for it, (x, y) in enumerate(train_loader):
                x, y = x.to(self.device), y.to(self.device)

                with autocast_ctx:
                    with torch.no_grad():
                        # student feature + logit
                        feat_dict, s_logit, _ = self.student(x)
                        s_feat = feat_dict[self.config.get("feat_kd_key", "feat_2d")]
                        # teacher feats (features only)
                        use_da = self.config.get("use_distillation_adapter", False)
                        t1_out = self.teacher1.extract_feats(x)
                        t2_out = self.teacher2.extract_feats(x)
                        if isinstance(t1_out, (list, tuple)) and len(t1_out) >= 2:
                            t1_f4d, t1_f2d = t1_out[0], t1_out[1]
                        else:
                            t1_f4d, t1_f2d = None, t1_out
                        if isinstance(t2_out, (list, tuple)) and len(t2_out) >= 2:
                            t2_f4d, t2_f2d = t2_out[0], t2_out[1]
                        else:
                            t2_f4d, t2_f2d = None, t2_out
                        if use_da and hasattr(self.teacher1, "distillation_adapter") and hasattr(self.teacher2, "distillation_adapter"):
                            t1_kv = self.teacher1.distillation_adapter(t1_f2d)
                            t2_kv = self.teacher2.distillation_adapter(t2_f2d)
                        else:
                            t1_kv, t2_kv = t1_f2d, t2_f2d
                        f1 = torch.stack([t1_kv, t2_kv], dim=1)

                    # synergy
                    if self.la_mode:
                        syn_feat, attn, *_ = self.mbm(s_feat, f1)
                        attn_flat = attn.squeeze(1)
                        _, _ = attn_flat[:, 0], attn_flat[:, 1]
                    else:
                        syn_feat, *_ = self.mbm(s_feat, f1)
                        attn = None
                    zsyn_out = self.synergy_head(syn_feat)
                    if isinstance(zsyn_out, tuple):
                        zsyn, vib_kl = zsyn_out
                    else:
                        zsyn, vib_kl = zsyn_out, torch.tensor(0.0, device=syn_feat.device)

                    # (i)  KL(zsyn \u2016 s_logit)
                    if stage <= self.kd_warmup_stage:
                        kl_val = torch.tensor(0.0, device=x.device)
                    else:
                        kl_val = kd_loss_fn(zsyn, s_logit, T=cur_tau)
                    # (ii) synergy CE
                    ce_val = ce_loss_fn(
                        zsyn,
                        y,
                        label_smoothing=self.config.get("label_smoothing", 0.0)
                    )
                    synergy_ce = self.synergy_ce_alpha * ce_val

                    # ---------- Feature-KD (MSE) ----------
                    feat_kd_val = torch.tensor(0.0, device=s_feat.device)
                    if self.config.get("feat_kd_alpha", 0) > 0:
                        fsyn_use = syn_feat
                        if fsyn_use.dim() == 4 and s_feat.dim() == 2:
                            fsyn_use = torch.nn.functional.adaptive_avg_pool2d(
                                fsyn_use, (1, 1)
                            ).flatten(1)
                        feat_kd_val = feat_mse_loss(
                            s_feat,
                            fsyn_use,
                            norm=self.config.get("feat_kd_norm", "none"),
                        )

                    # ── L2 regularization (per‑batch) ──────────────────
                    reg_teach = torch.stack(
                        [(p ** 2).mean() for p in teach_params]
                    ).mean() if teach_params else torch.tensor(0.0, device=x.device)

                    reg_mbm   = torch.stack(
                        [(p ** 2).mean() for p in mbm_params]
                    ).mean() if mbm_params else torch.tensor(0.0, device=x.device)

                    vib_kl_weight = float(self.config.get("vib_kl_weight", 1.0))
                    loss = (
                        kl_val
                        + synergy_ce
                        + self.config.get("feat_kd_alpha", 0) * feat_kd_val
                        + self.reg_lambda     * reg_teach      # 기존
                        + self.mbm_reg_lambda * reg_mbm        # ★ 추가됨
                        + vib_kl_weight * vib_kl
                    )

                if logger is not None and attn is not None:
                    logger.debug(f"attn_mean={attn.mean().item():.4f}")

                if self.debug and it == 0 and ep == 1:
                    logging.debug(
                        "[TA] stage=%s ep=%s ce=%.3f kl=%.3f zsyn mu=%.2f sigma=%.2f",
                        stage,
                        ep,
                        ce_val.item(),
                        kl_val.item(),
                        zsyn.mean(),
                        zsyn.std(),
                    )

                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(params, max_norm=2.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, max_norm=2.0)
                    optimizer.step()

                bs = x.size(0)
                total_loss += loss.item()*bs
                total_num  += bs

            avg_loss = total_loss / total_num
        if self.logger:
            self.logger.info(f"[TeacherUpdate] ep={ep} => loss={avg_loss:.4f}")

    def _unfreeze_teacher(self):
        """Set ``requires_grad=True`` for teacher components."""
        for p in self.teacher1.parameters():
            p.requires_grad = True
        for p in self.teacher2.parameters():
            p.requires_grad = True
        for p in self.mbm.parameters():
            p.requires_grad = True
        for p in self.synergy_head.parameters():
            p.requires_grad = True

    def _student_distill_update(
        self,
        train_loader,
        test_loader,
        optimizer,
        scheduler,
        epochs,
        logger=None,
        label_smoothing: float = 0.0,
        *,
        stage: int = 1,
    ):
        """
        Student Distillation:
         - Freeze teacher + MBM
         - Student upper layers만 업데이트
         - CE + KL(student vs synergy)
         - Optional vanilla KD blending when ``hybrid_beta > 0``
        """
        # freeze teacher
        self.teacher1.eval()
        self.teacher2.eval()
        logger = logger or self.logger
        for p in self.teacher1.parameters():
            p.requires_grad = False
        for p in self.teacher2.parameters():
            p.requires_grad = False

        self.mbm.eval()
        self.synergy_head.eval()
        for p in self.mbm.parameters():
            p.requires_grad = False
        for p in self.synergy_head.parameters():
            p.requires_grad = False

        autocast_ctx, scaler = get_amp_components(self.config)

        # ``optimizer`` and ``scheduler`` are constructed outside so that the
        # learning rate schedule spans all stages
        best_acc = 0.0
        best_state = copy.deepcopy(self.student.state_dict())

        for ep in range(1, epochs+1):
            if self.debug:
                logging.debug(
                    "[Student] ====== Distill ep %s/%s ======", ep, epochs
                )
            cur_tau = get_tau(
                self.config,
                epoch=(stage - 1) * epochs + (ep - 1),
                total_epochs=self.num_stages * epochs,
            )
            self.student.train()
            total_loss, total_num = 0.0, 0
            for it, (x, y) in enumerate(train_loader):
                x, y = x.to(self.device), y.to(self.device)

                with autocast_ctx:
                    with torch.no_grad():
                        # teacher feats only (with optional normalization)
                        t1_f4d, t1_f2d = self.teacher1.extract_feats(x)
                        t2_f4d, t2_f2d = self.teacher2.extract_feats(x)
                        norm_mode = str(self.config.get("ib_mbm_feature_norm", "l2")).lower()
                        if norm_mode == "l2":
                            t1_f2d = F.normalize(t1_f2d, dim=-1)
                            t2_f2d = F.normalize(t2_f2d, dim=-1)
                        elif norm_mode == "layernorm":
                            t1_f2d = torch.nn.functional.layer_norm(t1_f2d, (t1_f2d.shape[-1],))
                            t2_f2d = torch.nn.functional.layer_norm(t2_f2d, (t2_f2d.shape[-1],))
                        f1 = torch.stack([t1_f2d, t2_f2d], dim=1)

                    # student forward (query)
                    feat_dict, s_logit, _ = self.student(x)
                    s_feat = feat_dict[self.config.get("feat_kd_key", "feat_2d")]
                    norm_mode = str(self.config.get("ib_mbm_feature_norm", "l2")).lower()
                    if norm_mode == "l2":
                        s_feat = F.normalize(s_feat, dim=-1)
                    elif norm_mode == "layernorm":
                        s_feat = torch.nn.functional.layer_norm(s_feat, (s_feat.shape[-1],))

                if self.la_mode:
                    syn_feat, attn, *_ = self.mbm(s_feat, f1)
                    attn_flat = attn.squeeze(1)
                    w1, w2 = attn_flat[:, 0], attn_flat[:, 1]
                else:
                    syn_feat, *_ = self.mbm(s_feat, f1)
                    attn = None
                    w1, w2 = None, None
                # Head is a simple MLP (no VIB in head)
                zsyn = self.synergy_head(syn_feat)

                # CE
                ce_val = ce_loss_fn(
                    s_logit,
                    y,
                    label_smoothing=label_smoothing,
                )
                # KL(s‖zsyn) – 초기 stage(<=warm-up)는 OFF
                if stage <= self.kd_warmup_stage:
                    kd_val = torch.tensor(0.0, device=x.device)
                else:
                    # stop‑grad so student update does not backprop through MBM/Head
                    kd_val = kd_loss_fn(s_logit, zsyn.detach(), T=cur_tau)

                # vanilla KD using teacher logits
                # Teacher logits for vanilla KD – compute from extracted features
                t1_logit = self.teacher1.classifier(t1_f2d)
                t2_logit = self.teacher2.classifier(t2_f2d)
                avg_t_logit = 0.5 * (t1_logit + t2_logit)
                kd_vanilla = kd_loss_fn(s_logit, avg_t_logit, T=cur_tau)

                rkd_val = torch.tensor(0.0, device=s_feat.device)
                if self.config.get("rkd_loss_weight", 0.0) > 0:
                    if s_feat.size(0) <= 2 and logger is not None:
                        logger.warning("batch size <= 2: RKD losses will be zero")
                    rkd_t1 = (
                        rkd_distance_loss(s_feat, t1_f2d.detach(), reduction="none")
                        + rkd_angle_loss(s_feat, t1_f2d.detach(), reduction="none")
                    )
                    rkd_t2 = (
                        rkd_distance_loss(s_feat, t2_f2d.detach(), reduction="none")
                        + rkd_angle_loss(s_feat, t2_f2d.detach(), reduction="none")
                    )
                    rkd_syn = (
                        rkd_distance_loss(s_feat, syn_feat.detach(), reduction="none")
                        + rkd_angle_loss(s_feat, syn_feat.detach(), reduction="none")
                    )
                    gamma = self.config.get("rkd_gamma", 0.5)
                    if w1 is not None:
                        rkd_mix = (w1 * rkd_t1 + w2 * rkd_t2) + gamma * rkd_syn
                    else:
                        rkd_mix = 0.5 * (rkd_t1 + rkd_t2) + gamma * rkd_syn
                    rkd_val = rkd_mix.mean()

                # ───────────── Feature-KD (MSE) ─────────────
                feat_kd_val = torch.tensor(0.0, device=s_feat.device)
                if self.config.get("feat_kd_alpha", 0) > 0:
                    fsyn_use = syn_feat.detach()
                    if fsyn_use.dim() == 4 and s_feat.dim() == 2:
                        fsyn_use = torch.nn.functional.adaptive_avg_pool2d(
                            fsyn_use, (1, 1)
                        ).flatten(1)
                    feat_kd_val = feat_mse_loss(
                        s_feat,
                        fsyn_use,
                        norm=self.config.get("feat_kd_norm", "none"),
                    )

                beta = min(self.config.get("hybrid_beta", 0.0), 1.0)
                alpha_eff = self.alpha
                kd_coeff = 1 - self.alpha
                if beta + alpha_eff > 1:
                    scale = (1 - beta) / (alpha_eff + kd_coeff)
                    alpha_eff *= scale
                    kd_coeff *= scale
                vib_kl_weight = float(self.config.get("vib_kl_weight", 1.0))
                loss = (
                    alpha_eff * ce_val
                    + kd_coeff * kd_val
                    + self.config.get("rkd_loss_weight", 0.0) * rkd_val
                    + self.config.get("feat_kd_alpha", 0.0) * feat_kd_val
                    + beta * kd_vanilla
                )

                if logger is not None and attn is not None:
                    logger.debug(f"attn_mean={attn.mean().item():.4f}")

                # —— DEBUG: first batch summary ——
                if self.debug and it == 0 and ep == 1:
                    logging.debug(
                        "[SD] stage=%s ep=%s ce=%.3f kd=%.3f s_logit μ=%.2f σ=%.2f",
                        stage,
                        ep,
                        ce_val.item(),
                        kd_val.item(),
                        s_logit.mean(),
                        s_logit.std(),
                    )

                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    if self.debug and it == 0:
                        total_norm = torch.nn.utils.clip_grad_norm_(
                            filter(lambda p: p.requires_grad, self.student.parameters()),
                            max_norm=1e9,
                        )
                        logging.debug("[Student] grad-norm=%.3e", total_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if self.debug and it == 0:
                        total_norm = torch.nn.utils.clip_grad_norm_(
                            filter(lambda p: p.requires_grad, self.student.parameters()),
                            max_norm=1e9,
                        )
                        logging.debug("[Student] grad-norm=%.3e", total_norm)
                    optimizer.step()

                bs = x.size(0)
                total_loss += loss.item()*bs
                total_num  += bs

            avg_loss = total_loss / total_num
            # eval
            acc = 0.0
            if test_loader is not None:
                acc = self.evaluate(test_loader)

            if self.logger:
                self.logger.info(f"[StudentDistill] ep={ep} => loss={avg_loss:.4f}, acc={acc:.2f}")
            elif self.debug:
                logging.debug(
                    "[Student] ep=%s avg_loss=%.3f, acc=%.2f", ep, avg_loss, acc
                )

            if acc > best_acc:
                best_acc = acc
                best_state = copy.deepcopy(self.student.state_dict())

            # Step scheduler after optimizer step
            scheduler.step()

        # restore
        self.student.load_state_dict(best_state)
        return best_acc

    @torch.no_grad()
    def evaluate(self, loader):
        self.student.eval()
        correct, total = 0, 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            _, s_logit, _ = self.student(x)
            pred = s_logit.argmax(dim=1)
            correct += (pred==y).sum().item()
            total   += y.size(0)
        return 100.0*correct / total
