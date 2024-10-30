import hydra
from omegaconf import DictConfig
import torch
import numpy as np
import matplotlib.pyplot as plt
from robot_clip.utils import load_model_and_config
from robot_clip.data_loading import get_dataloaders, normalize_data, denormalize_data
import os
from hydra.utils import to_absolute_path

def plot_cross_reconstruction(original, reconstructed, src_modality, tgt_modality, normalization_params, save_path):
    # Denormalize the data
    mean = normalization_params[tgt_modality]['mean']
    std = normalization_params[tgt_modality]['std']
    original = original * std + mean
    reconstructed = reconstructed * std + mean

    num_dims = min(15, original.shape[1])
    fig, axes = plt.subplots(num_dims, 1, figsize=(10, 3*num_dims))
    fig.suptitle(f"{src_modality} to {tgt_modality} Cross-Reconstruction")

    for i in range(num_dims):
        ax = axes[i] if num_dims > 1 else axes
        ax.plot(original[:, i], label='Original', color='blue')
        ax.plot(reconstructed[:, i], label='Reconstructed', color='red', linestyle='--')
        ax.set_title(f"Dimension {i+1}")
        ax.legend()

    plt.tight_layout()
    save_dir = os.path.join(save_path, "cross_reconstruction")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{src_modality}_to_{tgt_modality}.png"))
    plt.close()

@hydra.main(config_path="../config", config_name="two_step_config")
def test_cross_reconstruction(config: DictConfig):
    # Load the data
    data_loader, _, normalization_params = get_dataloaders(config)
    
    # Load the trained model, its config, and normalization parameters
    model, loaded_config, _ = load_model_and_config(
        to_absolute_path(config.training.save_path),
        config.wandb.run_name,
        config.training.test_epoch
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Select random samples from the test loader
    data = next(iter(data_loader))
    index = np.random.randint(config.training.batch_size)
    length = 100
    input_data = {k: v[index:index+length].to(device) for k, v in data.items()}

    # Change the save path to a 'tests' directory
    save_path = to_absolute_path(os.path.join("tests", "outputs", config.wandb.run_name))
    os.makedirs(save_path, exist_ok=True)

    with torch.no_grad():
        # Encode all modalities
        embeddings = model.encode(input_data)
        
        # Perform cross-reconstruction
        for src_modality in embeddings.keys():
            for tgt_modality in model.decoders.keys():
                if src_modality != tgt_modality:
                    # Use the embedding from src_modality to reconstruct tgt_modality
                    cross_reconstructed = model.decoders[tgt_modality](embeddings[src_modality])
                    
                    # Plot and save the cross-reconstruction
                    plot_cross_reconstruction(
                        input_data[tgt_modality].cpu().numpy(),
                        cross_reconstructed.cpu().numpy(),
                        src_modality,
                        tgt_modality,
                        normalization_params,
                        save_path
                    )

    print(f"Cross-reconstruction plots saved in {os.path.join(save_path, 'cross_reconstruction')}")

if __name__ == "__main__":
    test_cross_reconstruction()
