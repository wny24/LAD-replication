"""DDPM / DDIM noise schedule. Squared-cosine betas match Chi et al. 2024."""

from __future__ import annotations

import numpy as np
import torch


def squaredcos_cap_v2(num_timesteps: int, max_beta: float = 0.999) -> np.ndarray:
    """Nichol & Dhariwal cosine schedule used by Diffusion Policy."""

    def alpha_bar(t: float) -> float:
        return np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2

    betas = []
    for i in range(num_timesteps):
        t1 = i / num_timesteps
        t2 = (i + 1) / num_timesteps
        betas.append(min(1.0 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.asarray(betas, dtype=np.float64)


class DDPMScheduler:
    def __init__(
        self,
        num_train_timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        beta_schedule: str = "squaredcos_cap_v2",
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
        prediction_type: str = "epsilon",
    ):
        if beta_schedule == "squaredcos_cap_v2":
            betas = squaredcos_cap_v2(num_train_timesteps)
        elif beta_schedule == "linear":
            betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float64)
        else:
            raise ValueError(f"Unknown beta_schedule {beta_schedule}")
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)
        self.num_train_timesteps = int(num_train_timesteps)
        self.clip_sample = bool(clip_sample)
        self.clip_sample_range = float(clip_sample_range)
        self.prediction_type = prediction_type
        self.betas = torch.tensor(betas, dtype=torch.float32)
        self.alphas = torch.tensor(alphas, dtype=torch.float32)
        self.alphas_cumprod = torch.tensor(alphas_cumprod, dtype=torch.float32)

    def _move(self, device: torch.device) -> None:
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)

    def add_noise(
        self, original: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        self._move(original.device)
        alpha = self.alphas_cumprod[timesteps].view(-1, *([1] * (original.ndim - 1)))
        return alpha.sqrt() * original + (1.0 - alpha).sqrt() * noise

    def _predict_x0(self, sample: torch.Tensor, noise_pred: torch.Tensor, t: int) -> torch.Tensor:
        alpha = self.alphas_cumprod[t]
        if self.prediction_type == "epsilon":
            x0 = (sample - (1.0 - alpha).sqrt() * noise_pred) / alpha.sqrt()
        elif self.prediction_type == "sample":
            x0 = noise_pred
        else:
            raise ValueError(self.prediction_type)
        if self.clip_sample:
            x0 = x0.clamp(-self.clip_sample_range, self.clip_sample_range)
        return x0

    @torch.no_grad()
    def step(self, noise_pred: torch.Tensor, timestep: int, sample: torch.Tensor) -> torch.Tensor:
        """Single DDPM reverse step. ``timestep`` is the current integer t."""
        self._move(sample.device)
        t = int(timestep)
        x0 = self._predict_x0(sample, noise_pred, t)
        if t == 0:
            return x0
        alpha_t = self.alphas_cumprod[t]
        alpha_prev = self.alphas_cumprod[t - 1]
        beta_t = self.betas[t]
        coef_x0 = (alpha_prev.sqrt() * beta_t) / (1.0 - alpha_t)
        coef_xt = self.alphas[t].sqrt() * (1.0 - alpha_prev) / (1.0 - alpha_t)
        mean = coef_x0 * x0 + coef_xt * sample
        variance = ((1.0 - alpha_prev) / (1.0 - alpha_t)) * beta_t
        return mean + variance.sqrt() * torch.randn_like(sample)

    @torch.no_grad()
    def ddim_step(
        self,
        noise_pred: torch.Tensor,
        timestep: int,
        prev_timestep: int,
        sample: torch.Tensor,
        eta: float = 0.0,
    ) -> torch.Tensor:
        self._move(sample.device)
        t = int(timestep)
        x0 = self._predict_x0(sample, noise_pred, t)
        if prev_timestep < 0:
            return x0
        alpha_t = self.alphas_cumprod[t]
        alpha_prev = self.alphas_cumprod[prev_timestep]
        if self.prediction_type == "epsilon":
            eps = noise_pred
        else:
            eps = (sample - alpha_t.sqrt() * x0) / (1.0 - alpha_t).sqrt()
        sigma = eta * ((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)).sqrt()
        dir_xt = (1.0 - alpha_prev - sigma ** 2).clamp(min=0).sqrt() * eps
        noise = torch.randn_like(sample) if eta > 0 else 0.0
        return alpha_prev.sqrt() * x0 + dir_xt + sigma * noise
