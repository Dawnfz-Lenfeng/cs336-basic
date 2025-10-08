import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Int
from torch import Tensor


def get_batch(
    dataset: npt.NDArray[np.long],
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[Int[Tensor, " batch cxt_len"], Int[Tensor, " batch cxt_len"]]:
    start_indices = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    indices = np.add.outer(start_indices, np.arange(context_length))

    x = dataset[indices]
    y = dataset[indices + 1]

    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
