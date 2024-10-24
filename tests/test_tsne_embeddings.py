import hydra
from omegaconf import DictConfig
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from robot_clip.utils import load_model_and_config
from train import load_and_normalize_data

@hydra.main(config_path="../config", config_name="config")
def visualize_tsne(config: DictConfig):
    # Load and normalize the data
    data, _ = load_and_normalize_data(config)
    
    # Load the trained model, its config, and normalization parameters
    model, loaded_config, _ = load_model_and_config(
        config.training.save_path,
        config.wandb.run_name,
        config.training.num_epochs
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Select a random subset of data
    num_samples = 1000
    indices = np.random.choice(len(data['mano']), num_samples, replace=False)
    
    embeddings = {}
    with torch.no_grad():
        for modality in data.keys():
            input_data = data[modality][indices].to(device)
            embeddings[modality] = model.encoders[modality](input_data).cpu().numpy()

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

    print(f'Saving at {config.training.save_path}')
    plt.savefig(f"{config.training.save_path}/tsne_embeddings.png")
    plt.close()

if __name__ == "__main__":
    visualize_tsne()
