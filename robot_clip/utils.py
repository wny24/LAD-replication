import os
import torch
from omegaconf import DictConfig, OmegaConf
from robot_clip.model import RobotCLIP


def _to_absolute_path(path: str) -> str:
    from hydra.utils import to_absolute_path
    return to_absolute_path(path)

def create_optimizer(config: DictConfig, model_params, optimizer_config=None):
    if optimizer_config is None:
        optimizer_config = config.optimizer
    optimizer_name = optimizer_config.name
    optimizer_params = {k: v for k, v in optimizer_config.items() if k != "name"}
    
    if optimizer_name == "Adam":
        return torch.optim.Adam(model_params, **optimizer_params)
    elif optimizer_name == "SGD":
        return torch.optim.SGD(model_params, **optimizer_params)
    elif optimizer_name == "RMSprop":
        return torch.optim.RMSprop(model_params, **optimizer_params)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def save_model(model, optimizer, epoch, config, run_name, normalization_params, is_two_step=False):
    # Use hydra's to_absolute_path for save directory
    save_dir = _to_absolute_path(os.path.join(config.training.save_path, run_name))
    print(f"Saving model to {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'config': OmegaConf.to_container(config, resolve=True),
        'normalization_params': normalization_params
    }
    
    if is_two_step:
        checkpoint['encoder_optimizer_state_dict'] = optimizer[0].state_dict()
        checkpoint['decoder_optimizer_state_dict'] = optimizer[1].state_dict()
    else:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    checkpoint_path = os.path.join(save_dir, f'model_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)

def load_checkpoint_file(checkpoint_file: str):
    """Load model, config, and train normalization stats from a .pth file."""
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_file}")

    checkpoint = torch.load(
        checkpoint_file, map_location=torch.device("cpu"), weights_only=False
    )
    if "config" not in checkpoint:
        raise ValueError("Checkpoint does not contain configuration")
    if "normalization_params" not in checkpoint:
        raise ValueError("Checkpoint does not contain normalization parameters")

    config = OmegaConf.create(checkpoint["config"])
    model = RobotCLIP(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, config, checkpoint["normalization_params"]


def load_model_and_config(checkpoint_path: str, run_name: str, epoch: int):
    """
    Load the model, configuration, and normalization parameters from a checkpoint.
    """
    # Use hydra's to_absolute_path for loading
    full_path = _to_absolute_path(os.path.join(checkpoint_path, run_name, f'model_epoch_{epoch}.pth'))
    return load_checkpoint_file(full_path)

def get_temperature(config, current_epoch):
    """Calculate temperature based on schedule and current epoch."""
    temp_config = config.temperature
    initial_temp = temp_config.initial
    final_temp = temp_config.final
    
    # If before start epoch, return initial temperature
    if current_epoch < temp_config.start_epoch:
        return initial_temp
    
    # If after end epoch, return final temperature
    if current_epoch >= temp_config.end_epoch:
        return final_temp
    
    # Calculate progress
    progress = (current_epoch - temp_config.start_epoch) / (temp_config.end_epoch - temp_config.start_epoch)
    
    # Apply different schedules
    if temp_config.schedule == 'linear':
        current_temp = initial_temp + progress * (final_temp - initial_temp)
    elif temp_config.schedule == 'exponential':
        current_temp = initial_temp * (final_temp / initial_temp) ** progress
    elif temp_config.schedule == 'cosine':
        current_temp = final_temp + 0.5 * (initial_temp - final_temp) * (1 + torch.cos(torch.tensor(progress * torch.pi)))
    else:
        raise ValueError(f"Unknown temperature schedule: {temp_config.schedule}")
    
    return current_temp
