# methods/asmb.py

import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from modules.losses import kd_loss_fn, ce_loss_fn

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

        # 하이퍼파라미터
        self.alpha = alpha
        self.synergy_ce_alpha = synergy_ce_alpha
        self.T = temperature
        self.reg_lambda = reg_lambda
        self.mbm_reg_lambda = mbm_reg_lambda
        self.num_stages = num_stages
        self.device = device
        self.config = config if config is not None else {}

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

        # 2) mbm => synergy
        fsyn = self.mbm(feats_2d, feats_4d)
        zsyn = self.synergy_head(fsyn)

        # 3) student
        s_feat, s_logit, _ = self.student(x)

        # CE
        ce_val = 0.0
        if y is not None:
            ce_val = self.ce_loss_fn(s_logit, y)

        # KL
        kd_val = kd_loss_fn(s_logit, zsyn, T=self.T, reduction="batchmean")
        total_loss = self.alpha*ce_val + (1-self.alpha)*kd_val

        return total_loss, s_logit

    def train_distillation(
        self,
        train_loader,
        test_loader=None,
        teacher_lr=1e-4,
        student_lr=5e-4,
        weight_decay=1e-4,
        epochs_per_stage=5,
        logger=None
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

        best_acc = 0.0
        best_student_state = copy.deepcopy(self.student.state_dict())

        for stage in range(1, self.num_stages+1):
            if logger:
                logger.info(f"\n[ASMB] Stage {stage}/{self.num_stages} 시작.")

            # (A) Teacher Update
            self._teacher_adaptive_update(
                train_loader,
                teacher_lr=teacher_lr,
                weight_decay=weight_decay,
                epochs=epochs_per_stage,
                logger=logger
            )
            # (optional) synergy eval

            # (B) Student Distillation
            acc = self._student_distill_update(
                train_loader,
                test_loader=test_loader,
                student_lr=student_lr,
                weight_decay=weight_decay,
                epochs=epochs_per_stage,
                logger=logger
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
        teacher_lr,
        weight_decay,
        epochs,
        logger=None
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
        # teacher1
        for p in self.teacher1.parameters():
            if p.requires_grad:
                params.append(p)
        # teacher2
        for p in self.teacher2.parameters():
            if p.requires_grad:
                params.append(p)
        # mbm, synergy_head
        for p in self.mbm.parameters():
            if p.requires_grad:
                params.append(p)
        for p in self.synergy_head.parameters():
            if p.requires_grad:
                params.append(p)

        optimizer = optim.Adam(params, lr=teacher_lr, weight_decay=weight_decay)

        self.teacher1.train()
        self.teacher2.train()
        self.mbm.train()
        self.synergy_head.train()
        self.student.eval()   # Student 고정

        for ep in range(1, epochs+1):
            total_loss, total_num = 0.0, 0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)

                with torch.no_grad():
                    # student logit
                    _, s_logit, _ = self.student(x)
                    # teacher feats
                    t1 = self.teacher1(x)
                    t2 = self.teacher2(x)
                    f1 = [t1["feat_2d"], t2["feat_2d"]]
                    f2 = [t1.get("feat_4d"), t2.get("feat_4d")]

                # synergy
                fsyn = self.mbm(f1, f2)
                zsyn = self.synergy_head(fsyn)

                # (i) -KL(s_logit, zsyn)
                kl_val = kd_loss_fn(zsyn, s_logit, T=self.T)  # 여기선 sign 주의
                # (ii) synergy CE
                ce_val = ce_loss_fn(
                    zsyn,
                    y,
                    label_smoothing=self.config.get("label_smoothing", 0.0)
                )
                synergy_ce = self.synergy_ce_alpha * ce_val

                # 정규화
                # (단순 L2) => MBM + teacher head ... 
                reg_loss = 0.0
                for p in params:
                    reg_loss += (p**2).sum()

                loss = (-1.0)*kl_val + synergy_ce + self.reg_lambda*reg_loss

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
        student_lr,
        weight_decay,
        epochs,
        logger=None
    ):
        """
        Student Distillation:
         - Freeze teacher + MBM
         - Student upper layers만 업데이트
         - CE + KL(student vs synergy)
        """
        # freeze teacher
        self.teacher1.eval()
        self.teacher2.eval()
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

        # student params
        student_params = [p for p in self.student.parameters() if p.requires_grad]
        optimizer = optim.SGD(student_params, lr=student_lr, momentum=0.9, weight_decay=weight_decay)
        best_acc = 0.0
        best_state = copy.deepcopy(self.student.state_dict())

        for ep in range(1, epochs+1):
            self.student.train()
            total_loss, total_num = 0.0, 0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)

                with torch.no_grad():
                    # teacher feats
                    t1 = self.teacher1(x)
                    t2 = self.teacher2(x)
                    f1 = [t1["feat_2d"], t2["feat_2d"]]
                    f2 = [t1.get("feat_4d"), t2.get("feat_4d")]
                    fsyn = self.mbm(f1, f2)
                    zsyn = self.synergy_head(fsyn)

                # student forward
                _, s_logit, _ = self.student(x)

                # CE
                ce_val = ce_loss_fn(s_logit, y)
                # KL
                kd_val = kd_loss_fn(s_logit, zsyn, T=self.T)

                loss = self.alpha*ce_val + (1-self.alpha)*kd_val

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                bs = x.size(0)
                total_loss += loss.item()*bs
                total_num  += bs

            avg_loss = total_loss / total_num
            # eval
            acc = 0.0
            if test_loader is not None:
                acc = self.evaluate(test_loader)

            if logger:
                logger.info(f"[StudentDistill] ep={ep} => loss={avg_loss:.4f}, acc={acc:.2f}")

            if acc > best_acc:
                best_acc = acc
                best_state = copy.deepcopy(self.student.state_dict())

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
