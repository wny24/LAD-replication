"""Frozen RobotCLIP encoders / decoders for latent-action diffusion."""

from __future__ import annotations

from typing import Dict, Iterable, List, Union

import numpy as np
import torch
import torch.nn as nn

from robot_clip.data_loading.dataset import denormalize_data, normalize_data
from robot_clip.utils import load_checkpoint_file

# Per-hand dims; must match diffusion/dataset.py RAW_ACTION_DIM
_RAW_ACTION_DIM = {"mano": 189, "xhand": 12, "g2": 1}


class FrozenActionCLIP(nn.Module):
    """Load a trained contrastive action model and freeze every parameter.

    Raw embodiment actions are z-scored with the CLIP training stats, then
    mapped by ``q_i``. Latents are mapped back by ``p_j`` and un-normalized.

    For bimanual (``n_arms=2``), ``action`` is ``[left_hand, right_hand]``;
    each side is encoded/decoded independently and latents are concatenated.
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
        n_arms: int = 1,
    ) -> torch.Tensor:
        """Encode a batch. ``raw_action`` is (B, H, n_arms*D) possibly zero-padded.

        Returns ``(B, H, n_arms * embedding_dim)``.
        """
        embodiments = list(embodiments)
        n_arms = int(n_arms)
        latents: List[torch.Tensor] = []
        for index, name in enumerate(embodiments):
            dim = _RAW_ACTION_DIM[name]
            need = dim * n_arms
            hand = raw_action[index, ..., :need]
            if n_arms == 1:
                latents.append(self.encode(name, hand))
                continue
            # (H, n_arms, D_hand) → encode each arm → (H, n_arms*Dz)
            parts = hand.reshape(*hand.shape[:-1], n_arms, dim)
            z_arms = [self.encode(name, parts[..., arm, :]) for arm in range(n_arms)]
            latents.append(torch.cat(z_arms, dim=-1))
        return torch.stack(latents, dim=0)

    def decode_batch(
        self,
        embodiments: Iterable[str],
        latent: torch.Tensor,
        n_arms: int = 1,
    ) -> torch.Tensor:
        """Decode ``(B, H, n_arms*Dz)`` → physical ``(B, H, n_arms*D_hand)``.

        Mixed-embodiment batches are zero-padded on the hand dim to a common width.
        """
        embodiments = list(embodiments)
        n_arms = int(n_arms)
        actions: List[torch.Tensor] = []
        for index, name in enumerate(embodiments):
            z = latent[index]
            if n_arms == 1:
                actions.append(self.decode(name, z[..., : self.embedding_dim]))
                continue
            z_parts = z.reshape(*z.shape[:-1], n_arms, self.embedding_dim)
            hands = [self.decode(name, z_parts[..., arm, :]) for arm in range(n_arms)]
            actions.append(torch.cat(hands, dim=-1))
        width = max(int(a.shape[-1]) for a in actions)
        padded = []
        for a in actions:
            if a.shape[-1] == width:
                padded.append(a)
            else:
                pad = torch.zeros(*a.shape[:-1], width - a.shape[-1], dtype=a.dtype, device=a.device)
                padded.append(torch.cat([a, pad], dim=-1))
        return torch.stack(padded, dim=0)
