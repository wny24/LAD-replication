import hydra
from omegaconf import DictConfig
import torch
import numpy as np
import matplotlib.pyplot as plt
from robot_clip.utils import load_model_and_config
from robot_clip.data_loading import get_dataloaders, normalize_data, denormalize_data
from tests.utils import load_test_data, display_test_sequence
import os
from hydra.utils import to_absolute_path

@hydra.main(config_path="../config", config_name="two_step_config")
def test_reconstruction(config: DictConfig):
    # Load the training data (for normalization parameters)
    _, _, train_normalization_params = get_dataloaders(config)
    
    # Load and display the test data
    test_data, test_normalization_params = load_test_data(config)
    save_path = to_absolute_path(os.path.join("tests", "outputs", config.wandb.run_name))
    display_test_sequence(test_data, test_normalization_params, save_path)
    
    # Load the trained model and its config
    model, loaded_config, _ = load_model_and_config(
        to_absolute_path(config.training.save_path),
        config.wandb.run_name,
        config.training.encoder_epochs + config.training.decoder_epochs
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Convert test data to tensors and move to device
    input_data = {k: torch.tensor(v, dtype=torch.float32).to(device) for k, v in test_data.items()}

    with torch.no_grad():
        _, reconstructions = model(input_data)

    # Visualize reconstructions
    for modality in input_data.keys():
        original = input_data[modality].cpu().numpy()
        reconstructed = reconstructions[modality].cpu().numpy()

        # Denormalize the data using training normalization parameters
        original = denormalize_data({modality: original}, {modality: train_normalization_params[modality]})[modality]
        reconstructed = denormalize_data({modality: reconstructed}, {modality: train_normalization_params[modality]})[modality]

        num_dims = min(15, original.shape[1])
        fig, axes = plt.subplots(num_dims, 1, figsize=(15, 3*num_dims))
        fig.suptitle(f"{modality} Reconstruction")

        for i in range(num_dims):
            ax = axes[i] if num_dims > 1 else axes
            ax.plot(original[:, i], label='Original', color='blue')
            ax.plot(reconstructed[:, i], label='Reconstructed', color='red', linestyle='--')
            ax.set_title(f"Dimension {i+1}")
            ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{modality}_reconstruction.png"))
        plt.close()

    print(f"Reconstruction plots saved in {save_path}")

if __name__ == "__main__":
    test_reconstruction()
