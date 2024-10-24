import hydra
import os
from omegaconf import DictConfig
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from robot_clip.model import RobotCLIP
from train import load_data

@hydra.main(config_path="../config", config_name="config")
def visualize_tsne(config: DictConfig):
    # Load the data
    data = load_data(config)
    
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RobotCLIP(config).to(device)
    model_path = os.path.abspath(f"{config.training.save_path}/model_epoch_{config.training.num_epochs}.pth")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
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
    plt.savefig("tsne_embeddings.png")
    plt.close()

if __name__ == "__main__":
    visualize_tsne()
