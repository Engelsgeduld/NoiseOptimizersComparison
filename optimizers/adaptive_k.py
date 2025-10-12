from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveKLoss(nn.Module):
    def __init__(
        self, base_loss_callable: Callable = F.mse_loss, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8
    ):
        super().__init__()
        self.base_loss_callable = base_loss_callable
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.register_buffer("m", torch.tensor(0.0))
        self.register_buffer("v", torch.tensor(0.0))

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        losses = self.base_loss_callable(y_pred, y_true, reduction="none")
        if losses.dim() > 1:
            losses = losses.mean(dim=tuple(range(1, losses.dim())))
        mu_b = losses.mean().detach()

        self.m: torch.Tensor = self.beta1 * self.m + (1 - self.beta1) * mu_b
        self.v: torch.Tensor = self.beta2 * self.v + (1 - self.beta2) * (mu_b**2)

        with torch.no_grad():
            mu_D = self.m / (torch.sqrt(self.v) + self.eps)

        mask = losses <= mu_D
        if mask.sum() == 0:
            return losses.mean()

        final_loss = losses[mask].mean()
        return final_loss
