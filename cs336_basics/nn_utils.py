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
    inputs: Float[Tensor, " batch vocab_size"],
    targets: Int[Tensor, " batch"],
) -> Float[Tensor, ""]:
    log_probs = log_softmax(inputs)
    loss = -einx.get_at("batch [vocab], batch -> batch", log_probs, targets)

    return loss.mean()
