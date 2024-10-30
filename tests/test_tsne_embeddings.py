import hydra
from omegaconf import DictConfig
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from robot_clip.utils import load_model_and_config
from robot_clip.data_loading import get_dataloaders, normalize_data, denormalize_data
import os
from hydra.utils import to_absolute_path

@hydra.main(config_path="../config", config_name="two_step_config")
def visualize_tsne(config: DictConfig):
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

    # Select a random subset of data from the test loader
    num_samples = min(1000, len(data_loader.dataset))
    indices = np.random.choice(len(data_loader.dataset), num_samples, replace=False)
    
    embeddings = {}
    with torch.no_grad():
        for idx in indices:
            sample = data_loader.dataset[idx]
            for modality, data in sample.items():
                if modality not in embeddings:
                    embeddings[modality] = []
                # Convert numpy array to torch tensor if needed
                if isinstance(data, np.ndarray):
                    data = torch.from_numpy(data).float()
                embeddings[modality].append(model.encoders[modality](data.unsqueeze(0).to(device)).cpu().numpy())

    for modality in embeddings:
        embeddings[modality] = np.concatenate(embeddings[modality], axis=0)

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    combined_embeddings = np.concatenate(list(embeddings.values()), axis=0)
    tsne_results = tsne.fit_transform(combined_embeddings)

    # Visualize t-SNE results
    plt.figure(figsize=(10, 8))
    colors = ['r', 'g', 'b']
    start = 0
    for i, (modality, emb) in enumerate(embeddings.items()):
        end = start + len(emb)
        plt.scatter(tsne_results[start:end, 0], tsne_results[start:end, 1], c=colors[i], label=modality, alpha=0.6)
        start = end

    plt.legend()
    plt.title("t-SNE visualization of embeddings")

    # Change the save path to a 'tests' directory
    save_path = to_absolute_path(os.path.join("tests", "outputs", config.wandb.run_name))
    os.makedirs(save_path, exist_ok=True)

    print(f'Saving at {save_path}')
    plt.savefig(f"{save_path}/tsne_embeddings.png")
    plt.close()

if __name__ == "__main__":
    visualize_tsne()
