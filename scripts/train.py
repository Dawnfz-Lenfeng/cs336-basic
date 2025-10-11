import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch.nn as nn
import yaml
from jaxtyping import Int
from torch import Tensor
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
    return np.memmap(data_path, dtype=dtype, mode="r")


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

    resume_step = 1
    if config.training.resume_from is not None:
        resume_step = load_checkpoint(
            config.training.resume_from,
            model,
            optimizer,
            scheduler,
        )
        # resume from next epoch
        resume_step += 1

    return model, data_loader, optimizer, scheduler, resume_step


def train_step(
    model: nn.Module,
    data: Int[Tensor, " batch cxt_len"],
    targets: Int[Tensor, " batch cxt_len"],
    optimizer: Optimizer,
    scheduler: LRScheduler,
) -> float:
    optimizer.zero_grad()

    outputs = model(data)
    loss = cross_entropy(outputs, targets)

    loss.backward()
    optimizer.step()
    scheduler.step()
    lr = scheduler.get_last_lr()[0]

    return loss.item(), lr


def train(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    num_steps: int,
    save_interval: int,
    log_interval: int,
    resume_step: int,
    save_dir: str,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for step, (data, targets) in enumerate(data_loader, start=resume_step):
        loss, lr = train_step(model, data, targets, optimizer, scheduler)

        if step % log_interval == 0:
            print(f"Step {step}, Loss: {loss:.4f}, LR: {lr:.6f}")
            if wandb.run is not None:
                wandb.log(
                    {
                        "train/loss": loss,
                        "train/learning_rate": lr,
                    },
                    step=step,
                )

        if step == num_steps:
            checkpoint_name = "checkpoint_final.pt"
            checkpoint_path = save_dir / checkpoint_name
            save_checkpoint(model, optimizer, scheduler, step, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            break

        if step % save_interval == 0:
            checkpoint_name = f"checkpoint_{step}.pt"
            checkpoint_path = save_dir / checkpoint_name
            save_checkpoint(model, optimizer, scheduler, step, checkpoint_path)
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

    model, data_loader, optimizer, scheduler, resume_step = setup_training(config)

    train(
        model,
        data_loader,
        optimizer,
        scheduler,
        config.training.num_steps,
        config.training.save_interval,
        config.training.log_interval,
        resume_step,
        config.training.save_dir,
    )

    # finish wandb run
    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
