"""Per-dimension Gaussian normalizer for the diffusion action target."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch


class GaussianNormalizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray, eps: float = 1e-8):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.eps = float(eps)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
        return (x - mean) / (std + self.eps)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
        return x * (std + self.eps) + mean

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, state: Dict[str, np.ndarray]) -> "GaussianNormalizer":
        return cls(state["mean"], state["std"])

    @classmethod
    def fit(cls, data: np.ndarray, eps: float = 1e-8) -> "GaussianNormalizer":
        arr = np.asarray(data, dtype=np.float64)
        arr = arr.reshape(-1, arr.shape[-1])
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        std = np.maximum(std, eps)
        return cls(mean.astype(np.float32), std.astype(np.float32), eps=eps)
