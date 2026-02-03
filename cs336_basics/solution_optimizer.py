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

def lr_cosine_schedule(
    it: int, 
    max_learning_rate: float, 
    min_learning_rate: float, 
    warmup_iters: int, 
    cosine_cycle_iters: int
):

    if it < warmup_iters:
        return it/warmup_iters * max_learning_rate

    if warmup_iters <= it <= cosine_cycle_iters:
        return min_learning_rate + \
               0.5 * (1 + math.cos((it - warmup_iters)/(cosine_cycle_iters - warmup_iters) * math.pi)) * \
               (max_learning_rate - min_learning_rate)

    if it > cosine_cycle_iters:
        return min_learning_rate


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    
    total_norm = 0
    eps = 1e-6

    for parameter in parameters:

        if parameter.grad is not None:
            grad_norm = torch.norm(parameter.grad, p=2)

            total_norm += grad_norm**2

    total_norm = math.sqrt(total_norm)

    if total_norm >= max_l2_norm:
        
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad = parameter.grad *  max_l2_norm/(total_norm + eps)
            

            