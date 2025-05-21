# methods/crd.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from distillers.losses import ce_loss_fn

class CRDLoss(nn.Module):
    """
    간소화한 CRD(Contrastive Representation Distillation) Loss
    - 실제론 메모리뱅크나 in-batch negatives 필요
    - 여기서는 in-batch만으로 처리하는 데모 버전
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, s_feat, t_feat):
        """
        s_feat, t_feat: [N, D]
         - diag(i==j)를 positive, off-diag(i!=j)를 negative로 취급
         - InfoNCE 유사 계산
        """
        # 1) normalize
        s_norm = F.normalize(s_feat, dim=1)  # [N, D]
        t_norm = F.normalize(t_feat, dim=1)  # [N, D]

        # 2) similarity matrix
        sim = torch.mm(s_norm, t_norm.t())  # [N, N]
        sim = sim / self.temperature

        N = s_feat.size(0)
        pos_mask = torch.eye(N, device=s_feat.device).bool()
        pos_score = sim[pos_mask]    # [N]
        neg_score = sim[~pos_mask].view(N, N-1)

        # 3) InfoNCE
        pos_exp = pos_score.exp()  # [N]
        neg_exp = neg_score.exp().sum(dim=1)  # [N]
        loss = -torch.log(pos_exp / (pos_exp + neg_exp + 1e-7)).mean()
        return loss

class CRDDistiller(nn.Module):
    """
    CRD Distiller:
      - teacher feat vs student feat => CRD
      - student logit => CE
      - total = alpha * CRD + (1-alpha) * CE
    """
    def __init__(self, teacher_model, student_model, alpha=0.5, temperature=0.07):
        super().__init__()
        self.teacher = teacher_model
        self.student = student_model
        self.alpha = alpha
        self.crd_loss_fn = CRDLoss(temperature=temperature)

        # classification loss
        self.ce_loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y):
        """
        1) teacher => (feat, logit, ...)
        2) student => (feat, logit, ...)
        3) CRD(feat_T, feat_S) + CE(S_logit, y)
        """
        with torch.no_grad():
            t_feat, _, _ = self.teacher(x)

        s_feat, s_logit, _ = self.student(x)

        # CRD
        crd_loss_val = self.crd_loss_fn(s_feat, t_feat)
        # CE
        ce_val = self.ce_loss_fn(s_logit, y)

        total_loss = self.alpha * crd_loss_val + (1 - self.alpha)*ce_val
        return total_loss, s_logit

    def train_distillation(
        self,
        train_loader,
        test_loader=None,
        lr=1e-3,
        weight_decay=1e-4,
        epochs=10,
        device="cuda"
    ):
        self.to(device)
        optimizer = optim.SGD(
            self.student.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay
        )
        best_acc = 0.0
        best_state = None

        for epoch in range(1, epochs+1):
            self.train()
            total_loss, total_num = 0.0, 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                loss, _ = self.forward(x, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * x.size(0)
                total_num  += x.size(0)

            avg_loss = total_loss / total_num

            if test_loader is not None:
                acc = self.evaluate(test_loader, device)
                print(f"[Epoch {epoch}] CRD => loss={avg_loss:.4f}, testAcc={acc:.2f}")
                if acc > best_acc:
                    best_acc = acc
                    best_state = {"student": self.student.state_dict()}
            else:
                print(f"[Epoch {epoch}] CRD => loss={avg_loss:.4f}")

        if best_state is not None:
            self.student.load_state_dict(best_state["student"])
        return best_acc

    @torch.no_grad()
    def evaluate(self, loader, device="cuda"):
        self.eval()
        self.to(device)
        correct, total = 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            # forward student
            _, s_logit, _ = self.student(x)
            pred = s_logit.argmax(dim=1)
            correct += (pred==y).sum().item()
            total   += y.size(0)
        return 100.0 * correct / total
