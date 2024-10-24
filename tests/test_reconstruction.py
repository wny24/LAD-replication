import hydra
from omegaconf import DictConfig
import torch
import numpy as np
import matplotlib.pyplot as plt
from robot_clip.model import RobotCLIP
from train import load_data

@hydra.main(config_path="../config", config_name="config")
def test_reconstruction(config: DictConfig):
    # Load the data
    data = load_data(config)
    
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RobotCLIP(config).to(device)
    checkpoint = torch.load(f"{config.training.save_path}/model_epoch_{config.training.num_epochs}.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Select a random sample
    index = np.random.randint(len(data['mano']))
    
    input_data = {modality: data[modality][index:index+1].to(device) for modality in data.keys()}
    
    with torch.no_grad():
        _, reconstructions = model(input_data)

    # Visualize reconstructions
    for modality in data.keys():
        original = input_data[modality].cpu().numpy().squeeze()
        reconstructed = reconstructions[modality].cpu().numpy().squeeze()

        num_dims = min(15, len(original))
        fig, axes = plt.subplots(num_dims, 1, figsize=(10, 3*num_dims))
        fig.suptitle(f"{modality} Reconstruction")

        for i in range(num_dims):
            ax = axes[i] if num_dims > 1 else axes
            ax.plot(original[i], label='Original', color='blue')
            ax.plot(reconstructed[i], label='Reconstructed', color='red', linestyle='--')
            ax.set_title(f"Dimension {i+1}")
            ax.legend()

        plt.tight_layout()
        plt.savefig(f"{modality}_reconstruction.png")
        plt.close()

if __name__ == "__main__":
    test_reconstruction()
