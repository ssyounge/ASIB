import torch, copy

class EWC:
    """
    Diagonal Fisher-based Elastic-Weight-Consolidation.
    사용법
    ------
    ewc = EWC(model, dataloader, device, samples=1024)
    penalty = ewc.penalty(model)
    """

    def __init__(self, model, data_loader, device="cuda",
                 samples: int = 1024, online: bool = False,
                 decay: float = 1.0):
        self.device = device
        self.online = online
        self.decay  = decay

        # θ* 저장
        self.theta_star = {n: p.detach().clone()
                           for n, p in model.named_parameters()
                           if p.requires_grad}

        # Fisher 계산
        self.fisher = {n: torch.zeros_like(p, device=device)
                       for n, p in self.theta_star.items()}
        self._compute_fisher(model, data_loader, samples)

    @torch.no_grad()
    def _compute_fisher(self, model, loader, samples):
        model.eval()
        collected = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            out = model(x)
            loss = torch.nn.functional.cross_entropy(out, y)
            grads = torch.autograd.grad(loss,
                                        [p for p in model.parameters()
                                         if p.requires_grad])
            for (n, _), g in zip(model.named_parameters(), grads):
                if n in self.fisher:          # 일부 파라미터는 제외될 수 있음
                    self.fisher[n] += g.detach()**2
            collected += x.size(0)
            if collected >= samples:
                break

        for n in self.fisher:
            self.fisher[n] /= max(collected, 1)

    def penalty(self, model):
        loss = 0.0
        for n, p in model.named_parameters():
            if n in self.fisher:
                theta_old = self.theta_star[n]
                loss += torch.sum(self.fisher[n] * (p - theta_old) ** 2)
        return loss

    # online EWC: Fisher ← γ·F_old + F_new,  θ* ← θ_now
    def update(self, model, data_loader):
        if not self.online:
            return
        new_ewc = EWC(model, data_loader,
                      device=self.device, samples=len(data_loader),
                      online=False)
        for n in self.fisher:
            self.fisher[n] = (self.decay * self.fisher[n] +
                              new_ewc.fisher[n])
            self.theta_star[n] = model.state_dict()[n].detach().clone()
