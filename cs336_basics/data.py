import os
import typing

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


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    model_state = model.state_dict()
    optim_state = optimizer.state_dict()

    checkpoint = dict(
        model_state=model_state,
        optim_state=optim_state,
        iteration=iteration,
    )
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src)

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optim_state"])

    return checkpoint["iteration"]
