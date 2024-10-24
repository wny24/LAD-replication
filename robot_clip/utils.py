import os
import torch
from omegaconf import DictConfig, OmegaConf
from robot_clip.model import RobotCLIP

def load_model_and_config(checkpoint_path: str, run_name: str, epoch: int):
    """
    Load the model, configuration, and normalization parameters from a checkpoint.
    
    Args:
    checkpoint_path (str): Base path to the checkpoints
    run_name (str): Name of the wandb run
    epoch (int): Epoch number of the checkpoint to load
    
    Returns:
    tuple: (model, config, normalization_params)
    """
    full_path = os.path.join(checkpoint_path, run_name, f'model_epoch_{epoch}.pth')
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"No checkpoint found at {full_path}")
    
    checkpoint = torch.load(full_path, map_location=torch.device('cpu'))
    
    if 'config' not in checkpoint:
        raise ValueError("Checkpoint does not contain configuration")
    
    if 'normalization_params' not in checkpoint:
        raise ValueError("Checkpoint does not contain normalization parameters")
    
    config = OmegaConf.create(checkpoint['config'])
    
    model = RobotCLIP(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, config, checkpoint['normalization_params']
