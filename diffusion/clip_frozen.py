"""Frozen RobotCLIP encoders / decoders for latent-action diffusion."""

from __future__ import annotations

from typing import Dict, Iterable, Union

import numpy as np
import torch
import torch.nn as nn

from robot_clip.data_loading.dataset import denormalize_data, normalize_data
from robot_clip.utils import load_checkpoint_file


class FrozenActionCLIP(nn.Module):
    """Load a trained contrastive action model and freeze every parameter.

    Raw embodiment actions are z-scored with the CLIP training stats, then
    mapped by ``q_i``. Latents are mapped back by ``p_j`` and un-normalized.
    """

    def __init__(self, checkpoint: str, device: torch.device):
        super().__init__()
        model, clip_config, norm = load_checkpoint_file(checkpoint)
        model.to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        self.model = model
        self.clip_config = clip_config
        self.norm = norm
        self.device = device
        self.embedding_dim = int(clip_config.model.embedding_dim)
        self.modalities = list(model.encoders.keys())

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    def _as_dict(self, modality: str, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {modality: tensor}

    def encode(
        self,
        modality: str,
        raw_action: Union[torch.Tensor, np.ndarray],
    ) -> torch.Tensor:
        """``raw_action`` is (..., D_raw) in physical units. Returns (..., D_z)."""
        if modality not in self.model.encoders:
            raise KeyError(f"Unknown modality {modality!r}. Have {self.modalities}")
        tensor = torch.as_tensor(raw_action, dtype=torch.float32, device=self.device)
        lead = tensor.shape[:-1]
        flat = tensor.reshape(-1, tensor.shape[-1])
        normalized = normalize_data(self._as_dict(modality, flat), self.norm)[modality]
        with torch.no_grad():
            latent = self.model.encoders[modality](normalized)
        return latent.reshape(*lead, self.embedding_dim)

    def decode(
        self,
        modality: str,
        latent: Union[torch.Tensor, np.ndarray],
    ) -> torch.Tensor:
        """``latent`` is (..., D_z). Returns physical-unit action (..., D_raw)."""
        if modality not in self.model.decoders:
            raise KeyError(f"Unknown modality {modality!r}. Have {self.modalities}")
        tensor = torch.as_tensor(latent, dtype=torch.float32, device=self.device)
        lead = tensor.shape[:-1]
        flat = tensor.reshape(-1, tensor.shape[-1])
        with torch.no_grad():
            normalized = self.model.decoders[modality](flat)
        raw = denormalize_data(self._as_dict(modality, normalized), self.norm)[modality]
        return raw.reshape(*lead, raw.shape[-1])

    def encode_batch(
        self,
        embodiments: Iterable[str],
        raw_action: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a batch that may mix embodiments. ``raw_action`` is (B, H, D)."""
        embodiments = list(embodiments)
        if len(set(embodiments)) == 1:
            return self.encode(embodiments[0], raw_action)
        latents = []
        for index, name in enumerate(embodiments):
            latents.append(self.encode(name, raw_action[index]))
        return torch.stack(latents, dim=0)
