import einx
import torch
from jaxtyping import Float, Int
from torch import Tensor


def softmax(x: Float[Tensor, " ..."], dim: int = -1) -> Float[Tensor, " ..."]:
    x_max = x.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)

    return x_exp / x_exp.sum(dim=dim, keepdim=True)


def log_softmax(x: Float[Tensor, " ..."], dim: int = -1) -> Float[Tensor, " ..."]:
    x_max = x.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)

    return x - x_max - x_exp.sum(dim=dim, keepdim=True).log()


def cross_entropy(
    inputs: Float[Tensor, " ... seq_len vocab_size"],
    targets: Int[Tensor, " ... seq_len"],
) -> Float[Tensor, ""]:
    log_probs = log_softmax(inputs)
    loss = -log_probs.gather(-1, targets.unsqueeze(-1))

    return loss.mean()
