import sys

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import yaml
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from cs336_basics.data import DataLoader
from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW, get_cosine_scheduler
from scripts.config import Config


def load_config(config_path: str) -> Config:
    with open(config_path) as f:
        data = yaml.safe_load(f)

    return Config.model_validate(data)


def load_data(data_path: str, dtype=np.uint16) -> npt.NDArray[np.uint16]:
    return np.memmap(data_path, dtype=dtype, mode="r")


def setup_training(
    config: Config,
) -> tuple[TransformerLM, DataLoader, Optimizer, LRScheduler]:
    dataset = load_data(config.data.data_path)
    data_loader = DataLoader(
        dataset,
        config.training.batch_size,
        config.model.context_length,
        config.training.device,
    )

    model = TransformerLM(**config.model.model_dump()).to(config.training.device)

    optimizer = AdamW(
        model.parameters(),
        lr=config.scheduler.max_learning_rate,
        **config.optimizer.model_dump(),
    )

    scheduler = get_cosine_scheduler(optimizer, **config.scheduler.model_dump())

    return model, data_loader, optimizer, scheduler


@torch.no_grad()
def estimate_loss(
    model: TransformerLM,
    train_data: np.memmap,
    val_data: np.memmap,
) -> dict[str, float]:
    """
    Estimate loss on training and validation sets.

    This function should:
    1. Set the model to evaluation mode
    2. Run multiple iterations on train and val data
    3. Compute average loss for each split
    4. Restore model to training mode

    Args:
        model: The language model
        train_data: Memory-mapped training data
        val_data: Memory-mapped validation data
        args: Parsed arguments containing batch_size, context_length, eval_iters, etc.

    Returns:
        Dictionary containing 'train' and 'val' loss values
    """
    pass


def save_checkpoint_with_metadata(
    model: TransformerLM,
    optimizer: AdamW,
    epoch: int,
    checkpoint_name: str = "checkpoint.pt",
):
    """
    Save a checkpoint with model, optimizer state, and metadata.

    Args:
        model: The language model
        optimizer: The optimizer
        epoch: Current epoch number
        args: Training configuration arguments
        checkpoint_name: Name of the checkpoint file
    """
    pass


def log_metrics(
    epoch: int,
    loss: float,
    learning_rate: float,
    is_eval: bool = False,
    eval_losses: dict[str, float] | None = None,
):
    """
    Log training metrics to console and optionally to Weights & Biases.

    Args:
        epoch: Current epoch number
        loss: Training loss value
        learning_rate: Current learning rate
        args: Training configuration arguments
        is_eval: Whether this is an evaluation step
        eval_losses: Dictionary of evaluation losses (train/val) if is_eval is True
    """
    pass


def train_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: Optimizer,
) -> float:
    for data, targets in data_loader:
        optimizer.zero_grad()

        outputs = model(data)
        loss = cross_entropy(outputs, targets)

        loss.backward()
        optimizer.step()

    return loss.item()


def train(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    num_epochs: int,
    save_interval: int,
):
    for epoch in range(num_epochs):
        loss = train_epoch(model, data_loader, optimizer)
        scheduler.step()

        log_metrics(epoch, loss, scheduler.get_last_lr()[0])
        if epoch % save_interval == 0:
            save_checkpoint_with_metadata(model, optimizer, epoch)

    save_checkpoint_with_metadata(model, optimizer, num_epochs)


def main():
    if len(sys.argv) != 2:
        print("Usage: python train.py <config_file.yaml>")
        sys.exit(1)

    config = load_config(sys.argv[1])
    print(config.model_dump_json(indent=2))

    model, data_loader, optimizer, scheduler = setup_training(config)

    train(
        model,
        data_loader,
        optimizer,
        scheduler,
        config.training.num_epochs,
        config.training.save_interval,
    )


if __name__ == "__main__":
    main()
