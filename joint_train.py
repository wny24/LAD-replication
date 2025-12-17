#!/usr/bin/env python3
import os

import hydra
import torch
import wandb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from robot_clip.data_loading.dataset import get_dataloaders
from robot_clip.model import RobotCLIP
from robot_clip.utils import create_optimizer, get_temperature, save_model

"""
Joint training script for RobotCLIP.

This script performs joint training of encoders and decoders together,
optimizing a single loss that combines contrastive and reconstruction terms.

Config:
- Uses the hydra config located in the `config` directory.
- By default it expects to be run with `two_step_config.yaml` (which provides modality,
  optimizer settings, temperature schedule, and training params). You can pass a different
  config when invoking via hydra.

Behavior:
- Creates a single optimizer that updates all trainable parameters (encoders + decoders).
- Uses model.training_step(...) which already computes both contrastive and reconstruction losses.
- Logs to wandb (unless debug mode is set).
- Saves checkpoints using the project's `save_model` helper.
"""


def train_epoch_joint(model, optimizer, train_loader, config, device, current_epoch):
    model.train()
    epoch_losses = {}

    # Get current temperature for this epoch (if config provides it)
    try:
        current_temp = get_temperature(config, current_epoch)
    except Exception:
        # Fallback to a fixed temperature if schedule isn't provided
        current_temp = getattr(config, "temperature", None)
        if current_temp is None:
            current_temp = 0.07

    for batch in tqdm(train_loader, desc="Joint Training"):
        # move batch to device; datasets used in this repo return torch tensors in many cases
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        # Use the model's training_step which returns combined loss and components
        loss_dict = model.training_step(batch)
        # training_step returns "loss" key for total loss
        loss = loss_dict.get("loss", None)
        if loss is None:
            # If older interface used different key names, try to reconstruct
            contrastive = loss_dict.get("contrastive_loss", 0)
            # sum all reconstruction losses (keys end with '_reconstruction_loss' in model implementation)
            reconstruction_sum = sum(
                v for k, v in loss_dict.items() if k.endswith("_reconstruction_loss")
            )
            # Try to use configured weight if present
            weight = getattr(config.model, "contrastive_loss_weight", 1.0)
            loss = weight * contrastive + reconstruction_sum

        # Backward and step
        loss.backward()
        optimizer.step()

        # Accumulate loss values (convert tensors to scalars)
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                val = v.item()
            else:
                try:
                    val = float(v)
                except Exception:
                    continue
            epoch_losses[k] = epoch_losses.get(k, 0.0) + val

        # Also track temperature for logging convenience
        epoch_losses["temperature"] = current_temp

    # Average losses across batches (skip temperature)
    num_batches = len(train_loader) if len(train_loader) > 0 else 1
    for k in list(epoch_losses.keys()):
        if k == "temperature":
            continue
        epoch_losses[k] /= num_batches

    return epoch_losses


def evaluate_joint(model, test_loader, config, device, current_epoch):
    model.eval()
    epoch_losses = {}

    # Get temperature for evaluation
    try:
        current_temp = get_temperature(config, current_epoch)
    except Exception:
        current_temp = getattr(config, "temperature", None)
        if current_temp is None:
            current_temp = 0.07

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}

            # Use model.validation_step which returns multiple losses
            step_losses = model.validation_step(batch)

            for k, v in step_losses.items():
                if isinstance(v, torch.Tensor):
                    val = v.item()
                else:
                    try:
                        val = float(v)
                    except Exception:
                        continue
                epoch_losses[k] = epoch_losses.get(k, 0.0) + val

    num_batches = len(test_loader) if len(test_loader) > 0 else 1
    for k in list(epoch_losses.keys()):
        epoch_losses[k] /= num_batches

    epoch_losses["temperature"] = current_temp
    return epoch_losses


@hydra.main(config_path="config", config_name="joint_config", version_base="1.1")
def train(config: DictConfig):
    debug_mode = config.training.debug

    # Resolve save path to absolute path (hydra may change cwd)
    save_path = to_absolute_path(config.training.save_path)
    os.makedirs(save_path, exist_ok=True)

    wandb_config = OmegaConf.to_container(config, resolve=True)
    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        config=wandb_config,
        mode="disabled" if debug_mode else None,
        dir=save_path,
    )

    device = torch.device(
        f"cuda:{config.training.gpu_id}" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    train_loader, test_loader, normalization_params = get_dataloaders(config)

    model = RobotCLIP(config).to(device)

    # Decide number of joint epochs (allow config override)
    joint_epochs = getattr(config.training, "joint_epochs", None)
    if joint_epochs is None:
        # fall back to decoder_epochs for convenience
        joint_epochs = getattr(config.training, "decoder_epochs", 50)

    # Create a single optimizer for joint training.
    # Preference order:
    # 1) config.optimizer.joint (explicit joint optimizer config)
    # 2) config.optimizer.name (top-level optimizer)
    # 3) fall back to encoder config if present
    optimizer = None
    if hasattr(config.optimizer, "joint"):
        optimizer = create_optimizer(config, model.parameters(), config.optimizer.joint)
    elif "name" in config.optimizer:
        # top-level optimizer config exists (like in config.yaml)
        optimizer = create_optimizer(config, model.parameters(), config.optimizer)
    elif hasattr(config.optimizer, "encoder"):
        # no joint / top-level config: create optimizer using encoder config but for all params
        optimizer = create_optimizer(
            config, model.parameters(), config.optimizer.encoder
        )
    else:
        # final fallback: use Adam with small lr
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Main training loop
    for epoch in range(joint_epochs):
        train_losses = train_epoch_joint(
            model, optimizer, train_loader, config, device, epoch
        )
        test_losses = evaluate_joint(model, test_loader, config, device, epoch)

        print(f"Joint Epoch {epoch + 1}/{joint_epochs}")
        print("Train Losses:", {k: f"{v:.6f}" for k, v in train_losses.items()})
        print("Test Losses:", {k: f"{v:.6f}" for k, v in test_losses.items()})

        if not debug_mode:
            wandb.log(
                {
                    "joint/epoch": epoch,
                    **{
                        f"joint/train/{k}": v
                        for k, v in train_losses.items()
                        if k != "temperature"
                    },
                    **{
                        f"joint/test/{k}": v
                        for k, v in test_losses.items()
                        if k != "temperature"
                    },
                    "joint/temperature": train_losses.get("temperature", None),
                }
            )

        # Save model checkpoint periodically
        if (epoch + 1) % config.training.save_interval == 0:
            # save_model expects optimizer object (not tuple) when not two-step
            save_model(
                model,
                optimizer,
                epoch + 1,
                config,
                wandb.run.name,
                normalization_params,
                is_two_step=False,
            )

    if not debug_mode:
        wandb.finish()


if __name__ == "__main__":
    train()
