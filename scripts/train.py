import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from jaxtyping import Float
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler

from cs336_basics.data import get_batch
from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW, get_cosine_schedule_with_warmup
from scripts.config import Config


def load_config(config_path: str) -> Config:
    with open(config_path) as f:
        data = yaml.safe_load(f)

    return Config.model_validate(data)


def load_data(data_path: str, dtype=np.uint16) -> np.memmap:
    """
    Load data using memory-mapped file for memory efficiency.

    Args:
        data_path: Path to the binary data file
        dtype: Data type of the tokens (default: np.uint16)

    Returns:
        Memory-mapped array of token IDs
    """
    return np.memmap(data_path, dtype=dtype, mode="r")


def setup_training(
    config: Config,
) -> tuple[
    TransformerLM,
    np.memmap,
    nn.Module,
    optim.Optimizer,
    LRScheduler,
]:
    dataset = load_data(config.data.data_path)
    train_data = get_batch(
        dataset,
        config.training.batch_size,
        config.model.context_length,
        config.training.device,
    )

    model = TransformerLM(**config.model.model_dump()).to(config.training.device)

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=config.scheduler.max_learning_rate,
        **config.optimizer.model_dump(),
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, **config.scheduler.model_dump()
    )

    return model, train_data, criterion, optimizer, scheduler


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
    model: TransformerLM,
    train_loader: Float[Tensor, " batch cxt_len"],
    criterion: nn.Module,
    optimizer: AdamW,
) -> float:
    for data, targets in train_loader:
        outputs = model(data)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


def train(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    num_epochs: int,
    save_interval: int,
):
    for epoch in range(num_epochs):
        loss = train_epoch(model, train_loader, criterion, optimizer)
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
    print(config)

    model, train_data, criterion, optimizer, scheduler = setup_training(config)

    train(
        model,
        train_data,
        criterion,
        optimizer,
        scheduler,
        config.training.num_epochs,
        config.training.save_interval,
    )


if __name__ == "__main__":
    main()
