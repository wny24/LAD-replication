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

from dataset_parser.faive.h5_tools import load_topic_arrays_h5
from dataset_parser.mano_utils import HandKinematicsInverse

def load_test_data_h5(file_path: str):
    topic_arrays = load_topic_arrays_h5(file_path, verbose=True, selected_topics=["/faive/policy_output", "/ingress/mano"])
    gc_angles = topic_arrays["/faive/policy_output"]["message"]
    mano_data = topic_arrays["/ingress/mano"]["message"]

    mano_local_rep = HandKinematicsInverse().forward(torch.from_numpy(mano_data))

    print(f"Faive max and mins: {np.max(gc_angles, axis=0)}, {np.min(gc_angles, axis=0)}")

    output = {
        "faive": torch.from_numpy(gc_angles).squeeze(2),
        "mano": mano_local_rep
    }

    return output

@hydra.main(config_path="../config", config_name="two_step_config")
def test_reconstruction(config: DictConfig):
    test_data = load_test_data_h5(to_absolute_path(config.data.test_file))

    save_path = to_absolute_path(os.path.join("tests", "outputs", config.wandb.run_name))
    os.makedirs(save_path, exist_ok=True)
    
    # Load the trained model and its config
    model, loaded_config, normalization_params = load_model_and_config(
        to_absolute_path(config.training.save_path),
        config.wandb.run_name,
        config.training.test_epoch
    )

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    input_data = normalize_data(test_data, normalization_params)

    for modality in input_data.keys():
        input_data[modality] = input_data[modality].to(device)
    # Convert test data to tensors and move to device
    with torch.no_grad():
        latents = model.encode(input_data)
        reconstructions = model.decode(latents)
        for modality in reconstructions.keys():
            reconstructions[modality] = reconstructions[modality].cpu()
            print(f'{modality} shape: {reconstructions[modality].shape}')
        reconstructions = denormalize_data(reconstructions, normalization_params)

    # Visualize reconstructions
    for modality in input_data.keys():
        original = test_data[modality].cpu().numpy()
        reconstructed = reconstructions[modality].cpu().numpy()

        print(original.shape, reconstructed.shape)

        # Denormalize the data using training normalization parameters
        # original = denormalize_data({modality: original}, {modality: train_normalization_params[modality]})[modality]
        # reconstructed = denormalize_data({modality: reconstructed}, {modality: train_normalization_params[modality]})[modality]

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
        plt.savefig(os.path.join(save_path, f"{modality}_h5_reconstruction.png"))
        plt.close()

    print(f"Reconstruction plots saved in {save_path}")

if __name__ == "__main__":
    test_reconstruction()
