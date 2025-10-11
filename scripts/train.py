import os
import sys

import numpy as np
import numpy.typing as npt
import torch.nn as nn
import yaml
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

import wandb
from cs336_basics.data import DataLoader, load_checkpoint, save_checkpoint
from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW, get_cosine_scheduler
from scripts.config import Config


def load_config(config_path: str) -> Config:
    with open(config_path) as f:
        data = yaml.safe_load(f)

    return Config.model_validate(data)


def load_data(data_path: str, dtype=np.uint16) -> npt.NDArray[np.uint16]:
    return np.memmap(data_path, dtype=dtype, mode="r")[0:100000]


def setup_training(
    config: Config,
) -> tuple[TransformerLM, DataLoader, Optimizer, LRScheduler, int]:
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

    resume_epoch = 0
    if config.training.resume_from is not None:
        resume_epoch = load_checkpoint(
            config.training.resume_from,
            model,
            optimizer,
            scheduler,
        )
        # resume from next epoch
        resume_epoch += 1

    return model, data_loader, optimizer, scheduler, resume_epoch


def train_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    epoch: int,
    log_interval: int,
) -> float:
    total_loss = 0.0

    for step, (data, targets) in enumerate(data_loader, start=1):
        optimizer.zero_grad()

        outputs = model(data)
        loss = cross_entropy(outputs, targets)

        loss.backward()
        optimizer.step()
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        total_loss += loss.item()
        print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}, LR: {lr:.6f}")
        if wandb.run is not None and step % log_interval == 0:
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/learning_rate": lr,
                },
                step=epoch * len(data_loader) + step,
            )

    return total_loss / step if total_loss > 0 else 0.0


def train(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    num_epochs: int,
    save_interval: int,
    log_interval: int,
    resume_epoch: int,
    save_dir: str,
):
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(resume_epoch, num_epochs + 1):
        train_epoch(model, data_loader, optimizer, scheduler, epoch, log_interval)

        if epoch % save_interval == 0 or epoch == num_epochs:
            checkpoint_name = (
                f"checkpoint_{epoch}.pt"
                if epoch != num_epochs
                else "checkpoint_final.pt"
            )
            checkpoint_path = os.path.join(save_dir, checkpoint_name)
            save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python train.py <config_file.yaml>")
        sys.exit(1)

    config = load_config(sys.argv[1])
    print(config.model_dump_json(indent=2))

    # initialize wandb
    if config.wandb.enabled:
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            name=config.wandb.name,
            tags=config.wandb.tags,
            config=config.model_dump(),
        )

    model, data_loader, optimizer, scheduler, resume_epoch = setup_training(config)

    train(
        model,
        data_loader,
        optimizer,
        scheduler,
        config.training.num_epochs,
        config.training.save_interval,
        config.training.log_interval,
        resume_epoch,
        config.training.save_dir,
    )

    # finish wandb run
    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
