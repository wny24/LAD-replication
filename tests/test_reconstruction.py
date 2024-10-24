import hydra
from omegaconf import DictConfig
import torch
import numpy as np
import matplotlib.pyplot as plt
from robot_clip.utils import load_model_and_config
from train import load_and_normalize_data
import os

@hydra.main(config_path="../config", config_name="config")
def test_reconstruction(config: DictConfig):
    # Load and normalize the data
    data, _ = load_and_normalize_data(config)
    
    # Load the trained model, its config, and normalization parameters
    model, loaded_config, normalization_params = load_model_and_config(
        config.training.save_path,
        config.wandb.run_name,
        config.training.num_epochs
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Select a random sample
    modalities = list(data.keys())

    data_keys_to_batch_names = {
        'mano': 'local_representation',
        'faive': 'faive_angles'
    }
    
    index = np.random.randint(len(data['faive_angles']))
    input_data = {modality: data[modality][index:index+50].to(device) for modality in data.keys()}
    
    for key, batch_name in data_keys_to_batch_names.items():
        input_data[key] = input_data[batch_name] 
        del input_data[batch_name]

    with torch.no_grad():
        _, reconstructions = model(input_data)

    # Visualize reconstructions
    for modality_name, modality_batch_name in data_keys_to_batch_names.items():
        original = input_data[modality_name].cpu().numpy().squeeze()
        reconstructed = reconstructions[modality_name].cpu().numpy().squeeze()

        # Denormalize the data
        mean = normalization_params[modality_batch_name]['mean']
        std = normalization_params[modality_batch_name]['std']
        original = original * std + mean
        reconstructed = reconstructed * std + mean

        num_dims = min(15, len(original))
        fig, axes = plt.subplots(num_dims, 1, figsize=(10, 3*num_dims))
        fig.suptitle(f"{modality_name} Reconstruction")

        for i in range(num_dims):
            ax = axes[i] if num_dims > 1 else axes
            ax.plot(original[i], label='Original', color='blue')
            ax.plot(reconstructed[i], label='Reconstructed', color='red', linestyle='--')
            ax.set_title(f"Dimension {i+1}")
            ax.legend()

        plt.tight_layout()
        print(f'Saving at {config.training.save_path}')
        plt.savefig(f"{config.training.save_path}/{modality_name}_reconstruction.png")
        plt.close()

if __name__ == "__main__":
    test_reconstruction()
