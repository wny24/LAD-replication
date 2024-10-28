import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import wandb
from tqdm import tqdm
from robot_clip.model import RobotCLIP
from robot_clip.data_loading.dataset import get_dataloaders
from robot_clip.utils import create_optimizer, save_model
import os

def train_epoch(model, optimizer, train_loader, config, device):
    model.train()
    epoch_losses = {}

    for batch in tqdm(train_loader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        loss_dict = model.training_step(batch)
        loss = loss_dict["loss"]
        loss.backward()
        optimizer.step()

        for k, v in loss_dict.items():
            if k not in epoch_losses:
                epoch_losses[k] = 0
            epoch_losses[k] += v.item()

    # Calculate average losses
    for k in epoch_losses:
        epoch_losses[k] /= len(train_loader)

    return epoch_losses

def evaluate(model, test_loader, config, device):
    model.eval()
    epoch_losses = {}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss_dict = model.training_step(batch)

            for k, v in loss_dict.items():
                if k not in epoch_losses:
                    epoch_losses[k] = 0
                epoch_losses[k] += v.item()

    # Calculate average losses
    for k in epoch_losses:
        epoch_losses[k] /= len(test_loader)

    return epoch_losses

@hydra.main(config_path="config", config_name="config", version_base="1.1")
def train(config: DictConfig):
    debug_mode = config.training.debug

    os.makedirs(config.training.save_path, exist_ok=True)

    wandb_config = OmegaConf.to_container(config, resolve=True)
    wandb.init(project=config.wandb.project,
                entity=config.wandb.entity, 
                config=wandb_config, 
                mode="disabled" if debug_mode else None,
                dir=config.training.save_path)

    device = torch.device(f"cuda:{config.training.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader, normalization_params = get_dataloaders(config)

    model = RobotCLIP(config).to(device)
    optimizer = create_optimizer(config, model.parameters())

    for epoch in range(config.training.num_epochs):
        train_losses = train_epoch(model, optimizer, train_loader, config, device)
        test_losses = evaluate(model, test_loader, config, device)

        print(f"Epoch {epoch+1}/{config.training.num_epochs}")
        print("Train Losses:", {k: f"{v:.4f}" for k, v in train_losses.items()})
        print("Test Losses:", {k: f"{v:.4f}" for k, v in test_losses.items()})

        if not debug_mode:
            wandb.log({
                **{f"train/{k}": v for k, v in train_losses.items()},
                **{f"test/{k}": v for k, v in test_losses.items()}
            })

        # Save model checkpoint
        if (epoch + 1) % config.training.save_interval == 0:
            save_model(model, optimizer, epoch + 1, config, wandb.run.name, normalization_params)

    if not debug_mode:
        wandb.finish()

if __name__ == "__main__":
    train()
