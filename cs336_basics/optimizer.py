import math
from collections.abc import Callable, Iterable
from typing import Any

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR


class AdamW(optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
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

                state: dict[str, Any] = self.state[p]
                # if state is empty dict, initialize
                if not state:
                    state.update(t=0, m=torch.zeros_like(p), v=torch.zeros_like(p))
                state["t"] += 1

                self._update_param(
                    p,
                    state["m"],
                    state["v"],
                    state["t"],
                    lr,
                    beta1,
                    beta2,
                    eps,
                    weight_decay,
                )

        return loss

    @staticmethod
    def _update_param(
        p: torch.nn.Parameter,
        m: torch.Tensor,
        v: torch.Tensor,
        t: int,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
    ):
        grad = p.grad.data

        m.mul_(beta1).add_(grad, alpha=1 - beta1)
        v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        bias_correction1 = 1 - beta1**t
        bias_correction2 = 1 - beta2**t
        step_size = lr * (bias_correction2**0.5) / bias_correction1

        p.data.addcdiv_(m, v.add_(eps).sqrt(), value=-step_size)
        p.data.mul_(1 - lr * weight_decay)


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate

    if it > cosine_cycle_iters:
        return min_learning_rate

    decay_ratio = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    cos_factor = 0.5 * (1 + math.cos(decay_ratio * math.pi))
    return min_learning_rate + (max_learning_rate - min_learning_rate) * cos_factor


def get_cosine_scheduler(
    optimizer: optim.Optimizer,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> LambdaLR:
    def lr_lambda(it: int) -> float:
        curr_lr = get_lr_cosine_schedule(
            it,
            max_learning_rate,
            min_learning_rate,
            warmup_iters,
            cosine_cycle_iters,
        )
        return curr_lr / max_learning_rate

    return LambdaLR(optimizer, lr_lambda)


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return

    total_norm = sum(grad.norm().pow(2) for grad in grads).sqrt()
    if total_norm <= max_l2_norm:
        return

    scale_factor = max_l2_norm / (total_norm + 1e-6)

    for grad in grads:
        grad.mul_(scale_factor)
