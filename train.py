import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import wandb
from tqdm import tqdm
from robot_clip.model import RobotCLIP
import os

def load_data(config: DictConfig):
    data = np.load(config.data.source_file, allow_pickle=True).item()
    return {
        "mano": torch.tensor(data["local_representation"], dtype=torch.float32),
        "faive": torch.tensor(data["faive_angles"], dtype=torch.float32),
        "simple_gripper": torch.tensor(data["simple_gripper"], dtype=torch.float32),
    }

def create_optimizer(config: DictConfig, model_params):
    optimizer_name = config.optimizer.name
    optimizer_params = {k: v for k, v in config.optimizer.items() if k != "name" and k != "optimizer"}
    
    if optimizer_name == "Adam":
        return torch.optim.Adam(model_params, **optimizer_params)
    elif optimizer_name == "SGD":
        return torch.optim.SGD(model_params, **optimizer_params)
    elif optimizer_name == "RMSprop":
        return torch.optim.RMSprop(model_params, **optimizer_params)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def train_epoch(model, optimizer, train_loader, config, device):
    model.train()
    epoch_losses = {}

    for batch in tqdm(train_loader, desc="Training"):
        batch = [t.to(device) for t in batch]
        batch_dict = {
            "mano": batch[0],
            "faive": batch[1],
            "simple_gripper": batch[2]
        }
        optimizer.zero_grad()
        loss_dict = model.training_step(batch_dict)
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
            batch = [t.to(device) for t in batch]
            batch_dict = {
                "mano": batch[0],
                "faive": batch[1],
                "simple_gripper": batch[2]
            }
            loss_dict = model.training_step(batch_dict)

            for k, v in loss_dict.items():
                if k not in epoch_losses:
                    epoch_losses[k] = 0
                epoch_losses[k] += v.item()

    # Calculate average losses
    for k in epoch_losses:
        epoch_losses[k] /= len(test_loader)

    return epoch_losses

def save_model(model, optimizer, epoch, config, run_name):
    # Use only the wandb run name for the directory
    save_dir = os.path.join(config.training.save_path, run_name)
    print(f"Saving model to {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config
    }
    torch.save(checkpoint, os.path.join(save_dir, f'model_epoch_{epoch}.pth'))

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = load_data(config)
    
    # Create a TensorDataset
    dataset = TensorDataset(*[tensor for tensor in data.values()])
    
    # Split the dataset
    train_size = int(config.data.train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size)

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
            save_model(model, optimizer, epoch + 1, config, wandb.run.name) 

    if not debug_mode:
        wandb.finish()

if __name__ == "__main__":
    train()
