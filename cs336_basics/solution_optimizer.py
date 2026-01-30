import math
from collections.abc import Callable, Iterable
from typing import Optional

import torch


class AdamW(torch.optim.Optimizer):

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.01,
    ):

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }

        super().__init__(params, defaults=defaults)

        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["m1"] = torch.zeros_like(p.data)
                self.state[p]["m2"] = torch.zeros_like(p.data)
                self.state[p]["t"] = 1

    def step(self, closure: Optional[Callable] = None):

        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state["t"]
                grad = p.grad.data
                theta = p.data

                m1, m2 = state["m1"], state["m2"]
                m1 = b1 * m1 + (1 - b1) * grad  # Update  first momentum
                m2 = b2 * m2 + (1 - b2) * grad * grad  # Update second momentum

                lr_adj = lr * math.sqrt(1 - b2**t) / (1 - b1**t)# Adjusted learning rate

                theta = theta - lr_adj * m1 / (torch.sqrt(m2) + eps)  # Update parameters
                theta = theta - lr * weight_decay * theta  # Apply weight decay

                p.data = theta
                state["m1"], state["m2"] = m1, m2
                state["t"] += 1
        return loss
