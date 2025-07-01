# methods/asmb.py

import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from modules.losses import (
    kd_loss_fn,
    ce_loss_fn,
    rkd_distance_loss,
    rkd_angle_loss,
    adversarial_loss_fn,
)
from utils.schedule import get_tau
from models import LightweightAttnMBM
from models.discriminator import Discriminator

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
        feat_kd_alpha=0.0,       # weight for feature-level KD
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
        self.la_mode = isinstance(mbm, LightweightAttnMBM)

        # 하이퍼파라미터
        self.alpha = alpha
        self.synergy_ce_alpha = synergy_ce_alpha
        self.T = temperature
        self.reg_lambda = reg_lambda
        self.mbm_reg_lambda = mbm_reg_lambda
        self.feat_kd_alpha = feat_kd_alpha
        self.num_stages = num_stages
        self.device = device
        self.config = config if config is not None else {}

        # 기본 Loss
        self.ce_loss_fn = nn.CrossEntropyLoss()

        # discriminator for adversarial distillation
        s_dim = student.get_feat_dim()
        self.discriminator = Discriminator(in_dim=s_dim)
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.get("d_lr", 1e-4),
            betas=(0.5, 0.999),
        )

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
            syn_feat, attn, _, _ = self.mbm(s_feat, feats_2d)
        else:
            syn_feat = self.mbm(feats_2d, feats_4d)
            attn = None
        zsyn = self.synergy_head(syn_feat)

        # CE
        ce_val = 0.0
        if y is not None:
            ce_val = self.ce_loss_fn(s_logit, y)

        # KL
        kd_val = kd_loss_fn(s_logit, zsyn, T=self.T, reduction="batchmean")

        feat_loss = torch.tensor(0.0, device=s_feat.device)
        if self.feat_kd_alpha > 0:
            feat_loss = F.mse_loss(
                s_feat.view(s_feat.size(0), -1),
                syn_feat.detach().view(s_feat.size(0), -1),
            )

        total_loss = (
            self.alpha * ce_val
            + (1 - self.alpha) * kd_val
            + self.feat_kd_alpha * feat_loss
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
            if logger:
                logger.info(f"\n[ASMB] Stage {stage}/{self.num_stages} 시작.")

            # (A) Teacher Update
            self._teacher_adaptive_update(
                train_loader,
                optimizer=teacher_optimizer,
                epochs=epochs_per_stage,
                logger=logger,
            )
            # (optional) synergy eval

            # (B) Student Distillation
            acc = self._student_distill_update(
                train_loader,
                test_loader=test_loader,
                optimizer=student_optimizer,
                scheduler=student_scheduler,
                epochs=epochs_per_stage,
                logger=logger,
                label_smoothing=self.config.get("label_smoothing", 0.0),
            )
            if acc > best_acc:
                best_acc = acc
                best_student_state = copy.deepcopy(self.student.state_dict())

        # 마지막에 best 복원
        self.student.load_state_dict(best_student_state)
        if logger:
            logger.info(f"[ASMB] Done. best student acc= {best_acc:.2f}")
        return best_acc

    def _teacher_adaptive_update(
        self,
        train_loader,
        optimizer,
        epochs,
        logger=None,
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
        self.mbm.train()
        self.synergy_head.train()
        self.student.eval()   # Student 고정

        for ep in range(1, epochs+1):
            cur_tau = get_tau(self.config, ep-1)
            total_loss, total_num = 0.0, 0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)

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
                else:
                    syn_feat = self.mbm(f1, f2)
                    attn = None
                zsyn = self.synergy_head(syn_feat)

                # (i) -KL(s_logit, zsyn)
                kl_val = kd_loss_fn(zsyn, s_logit, T=cur_tau)  # 여기선 sign 주의
                # (ii) synergy CE
                ce_val = ce_loss_fn(
                    zsyn,
                    y,
                    label_smoothing=self.config.get("label_smoothing", 0.0)
                )
                synergy_ce = self.synergy_ce_alpha * ce_val

                feat_loss = torch.tensor(0.0, device=s_feat.device)
                if self.feat_kd_alpha > 0:
                    feat_loss = F.mse_loss(
                        s_feat.view(s_feat.size(0), -1),
                        syn_feat.detach().view(s_feat.size(0), -1),
                    )

                # 정규화
                # (단순 L2) => MBM + teacher head ...
                reg_loss = 0.0
                for p in params:
                    reg_loss += (p**2).sum()

                loss = (
                    (-1.0) * kl_val
                    + synergy_ce
                    + self.feat_kd_alpha * feat_loss
                    + self.reg_lambda * reg_loss
                )

                if logger is not None and attn is not None:
                    logger.debug(f"attn_mean={attn.mean().item():.4f}")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                bs = x.size(0)
                total_loss += loss.item()*bs
                total_num  += bs

            avg_loss = total_loss / total_num
            if logger:
                logger.info(f"[TeacherUpdate] ep={ep} => loss={avg_loss:.4f}")

    def _student_distill_update(
        self,
        train_loader,
        test_loader,
        optimizer,
        scheduler,
        epochs,
        logger=None,
        label_smoothing: float = 0.0,
    ):
        """Student distillation with adversarial training."""
        # Freeze teacher, MBM, and synergy head
        self.teacher1.eval(); self.teacher2.eval(); self.mbm.eval(); self.synergy_head.eval()
        for p in self.teacher1.parameters():
            p.requires_grad = False
        for p in self.teacher2.parameters():
            p.requires_grad = False
        for p in self.mbm.parameters():
            p.requires_grad = False
        for p in self.synergy_head.parameters():
            p.requires_grad = False

        best_acc = 0.0
        best_state = copy.deepcopy(self.student.state_dict())

        self.student.to(self.device)
        self.discriminator.to(self.device)

        for ep in range(1, epochs + 1):
            self.student.train()
            self.discriminator.train()
            total_loss_S, total_loss_D = 0.0, 0.0
            total_num = 0

            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)

                real_labels = torch.ones(x.size(0), 1, device=self.device)
                fake_labels = torch.zeros(x.size(0), 1, device=self.device)

                # --- 1. Discriminator update ---
                self.optimizer_D.zero_grad()

                with torch.no_grad():
                    t1, t2 = self.teacher1(x), self.teacher2(x)
                    s_feat_for_mbm = self.student(x)[0][self.config.get("feat_kd_key", "feat_2d")]
                    syn_feat, _, _, _ = self.mbm(
                        s_feat_for_mbm.detach(), [t1["feat_2d"], t2["feat_2d"]]
                    )

                s_feat = self.student(x)[0][self.config.get("feat_kd_key", "feat_2d")]

                loss_D_real = adversarial_loss_fn(self.discriminator(syn_feat.detach()), real_labels)
                loss_D_fake = adversarial_loss_fn(self.discriminator(s_feat.detach()), fake_labels)
                loss_D = (loss_D_real + loss_D_fake) / 2
                loss_D.backward()
                self.optimizer_D.step()

                # --- 2. Student update ---
                optimizer.zero_grad()

                _, s_logit, _ = self.student(x)
                zsyn = self.synergy_head(syn_feat.detach())
                ce_val = ce_loss_fn(s_logit, y, label_smoothing=label_smoothing)
                kd_val = kd_loss_fn(s_logit, zsyn, T=get_tau(self.config, ep - 1))
                loss_main = self.alpha * ce_val + (1 - self.alpha) * kd_val

                adv_loss_weight = self.config.get("adversarial_loss_weight", 0.1)
                loss_G = adversarial_loss_fn(self.discriminator(s_feat), real_labels)
                total_student_loss = loss_main + adv_loss_weight * loss_G

                total_student_loss.backward()
                optimizer.step()

                bs = x.size(0)
                total_loss_S += total_student_loss.item() * bs
                total_loss_D += loss_D.item() * bs
                total_num += bs

            avg_loss = total_loss_S / total_num
            acc = 0.0
            if test_loader is not None:
                acc = self.evaluate(test_loader)

            if logger:
                logger.info(
                    f"[StudentDistill] ep={ep} => loss={avg_loss:.4f}, acc={acc:.2f}"
                )

            if acc > best_acc:
                best_acc = acc
                best_state = copy.deepcopy(self.student.state_dict())

            scheduler.step()

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
