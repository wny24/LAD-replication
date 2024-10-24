import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from typing import Dict, List

class ModalityEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class ModalityDecoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class RobotCLIP(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.modalities = config.model.modalities.modalities  # Note the nested structure
        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()

        for modality, params in self.modalities.items():
            self.encoders[modality] = ModalityEncoder(
                params.input_dim, params.encoder_hidden_dims, config.model.embedding_dim
            )
            self.decoders[modality] = ModalityDecoder(
                config.model.embedding_dim, params.decoder_hidden_dims, params.input_dim
            )

    def encode(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {modality: self.encoders[modality](inputs[modality]) for modality in self.modalities}

    def decode(self, embeddings: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {modality: self.decoders[modality](embeddings[modality]) for modality in self.modalities}

    def forward(self, inputs: Dict[str, torch.Tensor]):
        embeddings = self.encode(inputs)
        reconstructions = self.decode(embeddings)
        return embeddings, reconstructions

    def contrastive_loss(self, embeddings: Dict[str, torch.Tensor], temperature: float = 0.07):
        loss = 0
        num_modalities = len(embeddings)

        for i, (mod1, emb1) in enumerate(embeddings.items()):
            for j, (mod2, emb2) in enumerate(embeddings.items()):
                if i < j:
                    sim_matrix = F.cosine_similarity(emb1.unsqueeze(1), emb2.unsqueeze(0), dim=2) / temperature
                    labels = torch.arange(sim_matrix.size(0)).to(sim_matrix.device)
                    loss += F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.t(), labels)

        return loss / (num_modalities * (num_modalities - 1))

    def reconstruction_loss(self, inputs: Dict[str, torch.Tensor], reconstructions: Dict[str, torch.Tensor]):
        return {modality: F.mse_loss(inputs[modality], reconstructions[modality]) for modality in self.modalities}

    def training_step(self, batch: Dict[str, torch.Tensor]):
        embeddings, reconstructions = self(batch)
        contrastive_loss = self.contrastive_loss(embeddings)
        reconstruction_losses = self.reconstruction_loss(batch, reconstructions)
        total_reconstruction_loss = sum(reconstruction_losses.values())
        
        total_loss = (self.config.model.contrastive_loss_weight * contrastive_loss) + total_reconstruction_loss

        return {
            "loss": total_loss,
            "contrastive_loss": contrastive_loss,
            **{f"{modality}_reconstruction_loss": loss for modality, loss in reconstruction_losses.items()}
        }
