import os
import typing
from collections.abc import Iterator

import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Int
from torch import Tensor


def get_batch(
    dataset: npt.NDArray[np.uint16],
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[Int[Tensor, " batch cxt_len"], Int[Tensor, " batch cxt_len"]]:
    start_indices = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    indices = np.add.outer(start_indices, np.arange(context_length))

    x = dataset[indices]
    y = dataset[indices + 1]

    return torch.from_numpy(x).to(device, dtype=torch.long), torch.from_numpy(y).to(
        device, dtype=torch.long
    )


def save_checkpoint(
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    iteration: int,
    wandb_run_id: str | None = None,
):
    checkpoint = dict(
        model_state=model.state_dict(),
        optim_state=optimizer.state_dict(),
        iteration=iteration,
        scheduler_state=scheduler.state_dict(),
        wandb_run_id=wandb_run_id,
    )

    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> tuple[int, str | None]:
    checkpoint = torch.load(src)

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optim_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])

    return checkpoint["iteration"], checkpoint.get("wandb_run_id", None)


class DataLoader:
    def __init__(
        self,
        dataset: npt.NDArray[np.uint16],
        batch_size: int,
        context_length: int,
        device: str,
        shuffle=True,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device

    def __iter__(
        self,
    ) -> Iterator[tuple[Int[Tensor, " batch cxt_len"], Int[Tensor, " batch cxt_len"]]]:
        while True:
            yield get_batch(
                self.dataset,
                self.batch_size,
                self.context_length,
                self.device,
            )
