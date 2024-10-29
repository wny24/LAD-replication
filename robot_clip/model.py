import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from typing import Dict, List
from robot_clip.objectives import info_nce  # Add this import at the top

class ModalityEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
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
            layers.append(nn.LayerNorm(hidden_dim))
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
        return {modality: self.encoders[modality](inputs[modality]) for modality in inputs}

    def decode(self, embeddings: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {modality: self.decoders[modality](embeddings[modality]) for modality in embeddings}

    def forward(self, inputs: Dict[str, torch.Tensor]):
        embeddings = self.encode(inputs)
        reconstructions = self.decode(embeddings)
        return embeddings, reconstructions

    def contrastive_loss(self, embeddings: Dict[str, torch.Tensor], temperature: float = None):
        # Use provided temperature if given, otherwise use default
        temp = temperature if temperature is not None else 0.07
        
        loss = 0
        num_modalities = len(embeddings)
        modalities = list(embeddings.keys())

        # Compute pairwise InfoNCE loss between modalities
        # bidirectional loss
        for i in range(num_modalities):
            for j in range(i + 1, num_modalities):
                # Get embeddings for the pair of modalities
                query = embeddings[modalities[i]]
                positive_key = embeddings[modalities[j]]
                
                # Compute bidirectional InfoNCE loss
                # First direction: modality i -> modality j
                loss += info_nce(
                    query=query,
                    positive_key=positive_key,
                    negative_keys=None,  # Use other samples as negatives implicitly
                    temperature=temp,
                    reduction='mean'
                )

        # Normalize by number of pairs
        return loss / (num_modalities * (num_modalities - 1))

    def reconstruction_loss(self, inputs: Dict[str, torch.Tensor], reconstructions: Dict[str, torch.Tensor]):
        return {modality: F.mse_loss(inputs[modality], reconstructions[modality]) for modality in self.modalities}

    def self_and_cross_reconstruction_loss(self, inputs: Dict[str, torch.Tensor], embeddings: Dict[str, torch.Tensor]):
        """
        Compute self and cross-reconstruction loss where each modality's embedding is used to reconstruct other modalities.
        This is only used for evaluation/testing.
        """
        cross_losses = {}
        
        for source_modality in self.modalities:
            source_embedding = embeddings[source_modality]
            
            for target_modality in self.modalities:
                # Use source modality's embedding to reconstruct target modality
                cross_reconstruction = self.decoders[target_modality](source_embedding)
                cross_loss = F.mse_loss(inputs[target_modality], cross_reconstruction)
                
                cross_losses[f"{source_modality}_to_{target_modality}"] = cross_loss
        
        return cross_losses

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

    def validation_step(self, batch: Dict[str, torch.Tensor]):
        """
        Similar to training step but includes cross-reconstruction losses for evaluation.
        """
        embeddings, reconstructions = self(batch)
        contrastive_loss = self.contrastive_loss(embeddings)
        reconstruction_losses = self.reconstruction_loss(batch, reconstructions)
        cross_reconstruction_losses = self.self_and_cross_reconstruction_loss(batch, embeddings)
        
        return {
            **{f"{modality}_reconstruction_loss": loss for modality, loss in reconstruction_losses.items()},
            "contrastive_loss": contrastive_loss,
            **{f"{key}_loss": value for key, value in cross_reconstruction_losses.items()}
        }
