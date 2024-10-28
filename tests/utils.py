import numpy as np
import os
import matplotlib.pyplot as plt
from robot_clip.data_loading import normalize_data, denormalize_data, adapt_data_keys
from robot_clip.data_loading.dataset import MODALITY_MAPPING
from hydra.utils import to_absolute_path

def load_test_data(config):
    # Load the test data
    test_file_path = to_absolute_path(config.data.test_file)
    test_data = np.load(test_file_path, allow_pickle=True).item()
    
    # Adapt the keys
    test_data = adapt_data_keys(test_data, MODALITY_MAPPING)
    
    # Compute normalization parameters
    normalization_params = {}
    for key, tensor in test_data.items():
        mean = np.mean(tensor, axis=0)
        std = np.std(tensor, axis=0)
        normalization_params[key] = {'mean': mean, 'std': std}
    
    # Normalize the data
    normalized_data = normalize_data(test_data, normalization_params)
    
    return normalized_data, normalization_params

def display_test_sequence(data, normalization_params, save_path):
    denormalized_data = denormalize_data(data, normalization_params)
    
    for modality, sequence in denormalized_data.items():
        num_dims = min(15, sequence.shape[1])
        fig, axes = plt.subplots(num_dims, 1, figsize=(15, 3*num_dims))
        fig.suptitle(f"{modality} Test Sequence")
        
        for i in range(num_dims):
            ax = axes[i] if num_dims > 1 else axes
            ax.plot(sequence[:, i])
            ax.set_title(f"Dimension {i+1}")
        
        plt.tight_layout()
        abs_save_path = to_absolute_path(save_path)
        os.makedirs(abs_save_path, exist_ok=True)
        plt.savefig(os.path.join(abs_save_path, f"{modality}_test_sequence.png"))
        plt.close()

    print(f"Test sequence plots saved in {abs_save_path}")
