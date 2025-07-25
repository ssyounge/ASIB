# methods/asmb.py

import copy
import torch
import torch.nn as nn
import torch.optim as optim

from modules.losses import (
    kd_loss_fn,
    ce_loss_fn,
    rkd_distance_loss,
    rkd_angle_loss,
    feat_mse_loss,
)
from utils.schedule import get_tau
from utils.misc import get_amp_components
# LightweightAttnMBM is deprecated and ignored

class ASMBDistiller(nn.Module):
    """
    Adaptive Synergy Manifold Bridging (ASMB) Distiller
    - Teacher가 2명(teacher1, teacher2)
    - Student
    - MBM(Manifold Bridging Module), synergy_head
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
        self.mbm_reg_lambda = cfg.get("mbm_reg_lambda", mbm_reg_lambda)
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
        # 1) teacher feats
        with torch.no_grad():
            t1 = self.teacher1(x)
            t2 = self.teacher2(x)
            feats_2d = [t1["feat_2d"], t2["feat_2d"]]
            feats_4d = [t1.get("feat_4d"), t2.get("feat_4d")]

        # 3) student (query feature)
        feat_dict, s_logit, _ = self.student(x)
        key = self.config.get("feat_kd_key", "feat_2d")
        s_feat = feat_dict[key]

        # 2) mbm => synergy with query if available
        if self.la_mode:
            syn_feat, _, _, _ = self.mbm(s_feat, feats_2d)
        else:
            syn_feat = self.mbm(feats_2d, feats_4d)
        zsyn = self.synergy_head(syn_feat)

        # CE
        ce_val = (
            self.ce_loss_fn(s_logit, y)
            if y is not None
            else torch.tensor(0.0, device=s_logit.device)
        )

        # KL
        kd_val = kd_loss_fn(s_logit, zsyn, T=self.T, reduction="batchmean")

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
        ASMB Multi-Stage Self-Training:
          for s in [1..num_stages]:
            (A) Teacher Update (adaptive + MBM)
            (B) Student Distillation
        """
        # GPU로 이동
        self.to(self.device)
        self.teacher1.to(self.device)
        self.teacher2.to(self.device)
        self.mbm.to(self.device)
        self.synergy_head.to(self.device)
        self.student.to(self.device)

        # create optimizers / schedulers once so that optimizer state (e.g.,
        # Adam moments) and LR schedule are carried across stages
        teacher_params = []
        use_da = self.config.get("use_distillation_adapter", False)
        src1 = (
            self.teacher1.distillation_adapter.parameters()
            if use_da and hasattr(self.teacher1, "distillation_adapter")
            else self.teacher1.parameters()
        )
        for p in src1:
            if p.requires_grad:
                teacher_params.append(p)
        src2 = (
            self.teacher2.distillation_adapter.parameters()
            if use_da and hasattr(self.teacher2, "distillation_adapter")
            else self.teacher2.parameters()
        )
        for p in src2:
            if p.requires_grad:
                teacher_params.append(p)
        for p in self.mbm.parameters():
            if p.requires_grad:
                teacher_params.append(p)
        for p in self.synergy_head.parameters():
            if p.requires_grad:
                teacher_params.append(p)
        teacher_optimizer = optim.Adam(
            teacher_params, lr=teacher_lr, weight_decay=weight_decay
        )
        teacher_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            teacher_optimizer,
            T_max=epochs_per_stage * self.num_stages,
        )

        student_params = [p for p in self.student.parameters() if p.requires_grad]
        student_optimizer = optim.AdamW(
            student_params, lr=student_lr, weight_decay=weight_decay
        )
        total_epochs = epochs_per_stage * self.num_stages
        student_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            student_optimizer, T_max=total_epochs
        )

        best_acc = 0.0
        best_student_state = copy.deepcopy(self.student.state_dict())

        for stage in range(1, self.num_stages+1):
            if self.logger:
                self.logger.info(f"\n[ASMB] Stage {stage}/{self.num_stages} 시작.")

            # (A) Teacher Update
            self._teacher_adaptive_update(
                train_loader,
                optimizer=teacher_optimizer,
                epochs=epochs_per_stage,
                logger=self.logger,
                stage=stage,
            )
            teacher_scheduler.step()
            # (optional) synergy eval

            # (B) Student Distillation
            acc = self._student_distill_update(
                train_loader,
                test_loader=test_loader,
                optimizer=student_optimizer,
                scheduler=student_scheduler,
                epochs=epochs_per_stage,
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
            self.logger.info(f"[ASMB] Done. best student acc= {best_acc:.2f}")
        return best_acc

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
        #    (partial freeze는 이미 외부에서 했다고 가정, 또는 여기서 처리)
        params = []
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
        # teacher2
        param_src = (
            self.teacher2.distillation_adapter.parameters()
            if use_da and hasattr(self.teacher2, "distillation_adapter")
            else self.teacher2.parameters()
        )
        for p in param_src:
            if p.requires_grad:
                params.append(p)
        # mbm, synergy_head
        for p in self.mbm.parameters():
            if p.requires_grad:
                params.append(p)
        for p in self.synergy_head.parameters():
            if p.requires_grad:
                params.append(p)

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
                print(f"\n[DBG][Teacher] ====== Stage-Teacher ep {ep}/{epochs} ======")
            cur_tau = get_tau(self.config, ep-1)
            total_loss, total_num = 0.0, 0
            # L2 정규화는 매 batch 재계산해 그래프 중복 backward 오류를 방지
            for it, (x, y) in enumerate(train_loader):
                x, y = x.to(self.device), y.to(self.device)

                with autocast_ctx:
                    with torch.no_grad():
                        # student feature + logit
                        feat_dict, s_logit, _ = self.student(x)
                        s_feat = feat_dict[self.config.get("feat_kd_key", "feat_2d")]
                        # teacher feats
                        t1 = self.teacher1(x)
                        t2 = self.teacher2(x)
                        key = "distill_feat" if self.config.get("use_distillation_adapter", False) else "feat_2d"
                        f1 = [t1[key], t2[key]]
                        f2 = [t1.get("feat_4d"), t2.get("feat_4d")]

                    # synergy
                    if self.la_mode:
                        syn_feat, attn, _, _ = self.mbm(s_feat, f1)
                        attn_flat = attn.squeeze(1)
                        _, _ = attn_flat[:, 0], attn_flat[:, 1]
                    else:
                        syn_feat = self.mbm(f1, f2)
                        attn = None
                    zsyn = self.synergy_head(syn_feat)

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

                    # L2 regularization (per-batch)
                    reg_loss = torch.stack([(p ** 2).mean() for p in params]).mean()

                    loss = (
                        kl_val
                        + synergy_ce
                        + self.reg_lambda * reg_loss
                    )

                if logger is not None and attn is not None:
                    logger.debug(f"attn_mean={attn.mean().item():.4f}")

                if self.debug and it == 0 and ep == 1:
                    print(
                        f"[DBG|TA] stage={stage} ep={ep} "
                        f"ce={ce_val.item():.3f} kl={kl_val.item():.3f} "
                        f"zsyn \u03bc={zsyn.mean():.2f} \u03c3={zsyn.std():.2f}"
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
                print(f"\n[DBG][Student] ====== Distill ep {ep}/{epochs} ======")
            cur_tau = get_tau(self.config, ep-1)
            self.student.train()
            total_loss, total_num = 0.0, 0
            for it, (x, y) in enumerate(train_loader):
                x, y = x.to(self.device), y.to(self.device)

                with autocast_ctx:
                    with torch.no_grad():
                        # teacher feats
                        t1 = self.teacher1(x)
                        t2 = self.teacher2(x)
                        f1 = [t1["feat_2d"], t2["feat_2d"]]
                        f2 = [t1.get("feat_4d"), t2.get("feat_4d")]

                    # student forward (query)
                    feat_dict, s_logit, _ = self.student(x)
                    s_feat = feat_dict[self.config.get("feat_kd_key", "feat_2d")]

                if self.la_mode:
                    syn_feat, attn, _, _ = self.mbm(s_feat, f1)
                    attn_flat = attn.squeeze(1)
                    w1, w2 = attn_flat[:, 0], attn_flat[:, 1]
                else:
                    syn_feat = self.mbm(f1, f2)
                    attn = None
                    w1, w2 = None, None
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
                    kd_val = kd_loss_fn(s_logit, zsyn, T=cur_tau)

                # vanilla KD using teacher logits
                avg_t_logit = 0.5 * (t1["logit"] + t2["logit"])
                kd_vanilla = kd_loss_fn(s_logit, avg_t_logit, T=cur_tau)

                rkd_val = torch.tensor(0.0, device=s_feat.device)
                if self.config.get("rkd_loss_weight", 0.0) > 0:
                    if s_feat.size(0) <= 2 and logger is not None:
                        logger.warning("batch size <= 2: RKD losses will be zero")
                    rkd_t1 = (
                        rkd_distance_loss(s_feat, t1["feat_2d"].detach(), reduction="none")
                        + rkd_angle_loss(s_feat, t1["feat_2d"].detach(), reduction="none")
                    )
                    rkd_t2 = (
                        rkd_distance_loss(s_feat, t2["feat_2d"].detach(), reduction="none")
                        + rkd_angle_loss(s_feat, t2["feat_2d"].detach(), reduction="none")
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

                beta = min(self.config.get("hybrid_beta", 0.0), 1.0)
                alpha_eff = self.alpha
                kd_coeff = 1 - self.alpha
                if beta + alpha_eff > 1:
                    scale = (1 - beta) / (alpha_eff + kd_coeff)
                    alpha_eff *= scale
                    kd_coeff *= scale
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
                    print(
                        f"[DBG|SD] stage={stage} ep={ep} "
                        f"ce={ce_val.item():.3f} kd={kd_val.item():.3f} "
                        f"s_logit μ={s_logit.mean():.2f} σ={s_logit.std():.2f}"
                    )

                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    if self.debug and it == 0:
                        total_norm = torch.nn.utils.clip_grad_norm_(
                            filter(lambda p: p.requires_grad, self.student.parameters()),
                            max_norm=1e9,
                        )
                        print(f"[DBG][Student] grad-norm={total_norm:.3e}")
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if self.debug and it == 0:
                        total_norm = torch.nn.utils.clip_grad_norm_(
                            filter(lambda p: p.requires_grad, self.student.parameters()),
                            max_norm=1e9,
                        )
                        print(f"[DBG][Student] grad-norm={total_norm:.3e}")
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
                print(f"[DBG][Student] ep={ep} avg_loss={avg_loss:.3f}, acc={acc:.2f}")

            if acc > best_acc:
                best_acc = acc
                best_state = copy.deepcopy(self.student.state_dict())

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
