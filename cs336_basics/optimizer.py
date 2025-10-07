from collections.abc import Callable
from typing import Any

import torch
import torch.optim as optim


class AdamW(optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data

                state: dict[str, Any] = self.state[p]
                # if state is empty dict, initialize
                if not state:
                    state.update(
                        t=0,
                        m=torch.zeros_like(p.data),
                        v=torch.zeros_like(p.data),
                    )

                state["t"] += 1
                t, m, v = state["t"], state["m"], state["v"]

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1**t
                bias_correction2 = 1 - beta2**t
                step_size = lr * (bias_correction2**0.5) / bias_correction1

                p.data.addcdiv_(m, v.sqrt().add_(eps), value=-step_size)
                p.data.mul_(1 - lr * weight_decay)

        return loss
