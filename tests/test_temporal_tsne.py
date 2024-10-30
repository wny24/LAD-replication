import hydra
from omegaconf import DictConfig
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from robot_clip.utils import load_model_and_config
from robot_clip.data_loading import get_dataloaders, denormalize_data
from tests.utils import load_test_data
import os
from hydra.utils import to_absolute_path

@hydra.main(config_path="../config", config_name="two_step_config")
def visualize_temporal_tsne(config: DictConfig):
    # Load the test data
    test_data, test_normalization_params = load_test_data(config)
    
    # Load the trained model and its config
    model, loaded_config, _ = load_model_and_config(
        to_absolute_path(config.training.save_path),
        config.wandb.run_name,
        config.training.test_epoch
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Convert test data to tensors and get embeddings
    embeddings = {}
    with torch.no_grad():
        # Get sequence length and ensure we can take 10 elements
        seq_length = min(len(next(iter(test_data.values()))), 100)  # assuming all modalities have same length
        # Random start index that ensures we can take 10 elements after it
        start_idx = np.random.randint(0, seq_length - 10)
        end_idx = start_idx + 10
        
        for modality, data in test_data.items():
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data).float()
            data = data.to(device)
            # Take 10 consecutive points starting from start_idx
            data = data[start_idx:end_idx]
            # Get embeddings for each timestep
            embeddings[modality] = model.encoders[modality](data).cpu().numpy()

    # Perform t-SNE
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=5,  # Reduced perplexity for small sample size
        n_iter=1000,   # Increase iterations for better convergence
        learning_rate='auto'  # Let TSNE choose appropriate learning rate
    )
    combined_embeddings = np.concatenate(list(embeddings.values()), axis=0)
    tsne_results = tsne.fit_transform(combined_embeddings)

    # Get the save path
    save_path = to_absolute_path(os.path.join("tests", "outputs", config.wandb.run_name))
    os.makedirs(save_path, exist_ok=True)

    # Create color gradients for temporal visualization
    plt.figure(figsize=(12, 8))
    start = 0
    
    # Define distinct base colors for each modality
    base_colors = {
        'image': 'red',
        'state': 'blue',
        'action': 'green',
        'language': 'purple'  # in case there's language data
    }
    
    for i, (modality, emb) in enumerate(embeddings.items()):
        end = start + len(emb)
        
        # Plot just the points
        plt.scatter(
            tsne_results[start:end, 0],
            tsne_results[start:end, 1],
            c=base_colors.get(modality, plt.cm.Set1(i)),
            label=modality
        )
        
        # Add just the connecting lines
        for j in range(len(emb)-1):
            plt.plot(
                [tsne_results[start+j, 0], tsne_results[start+j+1, 0]],
                [tsne_results[start+j, 1], tsne_results[start+j+1, 1]],
                color=base_colors.get(modality, plt.cm.Set1(i)),
                alpha=0.5
            )
        
        start = end

    plt.legend()
    plt.title("t-SNE visualization of embeddings")
    print(f'Saving at {save_path}')
    plt.savefig(f"{save_path}/simple_tsne_embeddings.png")
    plt.close()

    # Create additional plot showing temporal distances
    plt.figure(figsize=(10, 6))
    for modality, emb in embeddings.items():
        # Calculate distances between consecutive timesteps
        distances = np.sqrt(np.sum(np.diff(emb, axis=0)**2, axis=1))
        plt.plot(distances, label=f'{modality} temporal distances')
    
    plt.xlabel('Timestep')
    plt.ylabel('Distance between consecutive embeddings')
    plt.title('Temporal distances in embedding space')
    plt.legend()
    plt.savefig(f"{save_path}/temporal_distances.png")
    plt.close()

    # Create additional plot showing distances between modalities
    plt.figure(figsize=(10, 6))
    modalities = list(embeddings.keys())
    
    for i in range(len(modalities)):
        for j in range(i+1, len(modalities)):
            mod1, mod2 = modalities[i], modalities[j]
            # Calculate distances between corresponding timesteps of different modalities
            distances = np.sqrt(np.sum((embeddings[mod1] - embeddings[mod2])**2, axis=1))
            plt.plot(distances, label=f'{mod1} vs {mod2} distances')
    
    plt.xlabel('Timestep')
    plt.ylabel('Distance between modality pairs')
    plt.title('Inter-modality distances in embedding space')
    plt.legend()
    plt.savefig(f"{save_path}/modality_distances.png")
    plt.close()

if __name__ == "__main__":
    visualize_temporal_tsne()
