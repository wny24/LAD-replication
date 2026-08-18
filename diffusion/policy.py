"""Latent diffusion policy: frozen CLIP action model + trainable U-Net."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from clip_frozen import FrozenActionCLIP
from networks import ConditionalUnet1D
from normalizer import GaussianNormalizer
from scheduler import DDPMScheduler
from vision import ObservationEncoder


class LatentDiffusionPolicy(nn.Module):
    def __init__(
        self,
        clip: FrozenActionCLIP,
        obs_encoder: ObservationEncoder,
        unet: ConditionalUnet1D,
        scheduler: DDPMScheduler,
        normalizer: GaussianNormalizer,
        embedding_dim: int,
        wrist_dim: int,
        horizon: int,
    ):
        super().__init__()
        self.clip = clip
        self.obs_encoder = obs_encoder
        self.unet = unet
        self.scheduler = scheduler
        self.normalizer = normalizer
        self.embedding_dim = int(embedding_dim)
        self.wrist_dim = int(wrist_dim)
        self.horizon = int(horizon)
        self.action_dim = self.embedding_dim + self.wrist_dim

    def encode_obs(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.obs_encoder(image=batch.get("obs_image"), lowdim=batch.get("obs_lowdim"))

    def latent_target(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """CLIP-encode raw eef action and concat non-latent wrist pose. (B, H, Dz+Dw)."""
        embodiments: List[str] = batch["embodiment"]
        z = self.clip.encode_batch(embodiments, batch["action"])
        if self.wrist_dim == 0:
            return z
        wrist = batch["wrist_pose"]
        if wrist.shape[-1] != self.wrist_dim:
            raise ValueError(f"wrist_pose dim {wrist.shape[-1]} != config {self.wrist_dim}")
        return torch.cat([z, wrist], dim=-1)

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        cond = self.encode_obs(batch)
        target = self.normalizer.normalize(self.latent_target(batch)).detach()
        noise = torch.randn_like(target)
        timesteps = torch.randint(
            0,
            self.scheduler.num_train_timesteps,
            (target.shape[0],),
            device=target.device,
        )
        noisy = self.scheduler.add_noise(target, noise, timesteps)
        pred = self.unet(noisy, timesteps, global_cond=cond)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(
        self,
        batch: Dict[str, torch.Tensor],
        num_inference_steps: Optional[int] = None,
        use_ddim: bool = True,
        eta: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """Denoise a latent action sequence, then decode with the frozen CLIP decoder.

        Returns physical-unit ``action`` (B, H, D_raw), ``wrist_pose`` (B, H, Dw),
        and ``latent`` (B, H, Dz).
        """
        self.eval()
        cond = self.encode_obs(batch)
        b = cond.shape[0]
        device = cond.device
        sample = torch.randn(b, self.horizon, self.action_dim, device=device)
        steps = int(num_inference_steps or self.scheduler.num_train_timesteps)
        if use_ddim:
            times = torch.linspace(
                self.scheduler.num_train_timesteps - 1, 0, steps, device=device
            ).long()
            for i, t in enumerate(times):
                pred = self.unet(sample, t.expand(b), global_cond=cond)
                prev = int(times[i + 1].item()) if i + 1 < len(times) else -1
                sample = self.scheduler.ddim_step(pred, int(t.item()), prev, sample, eta=eta)
        else:
            for t in range(self.scheduler.num_train_timesteps - 1, -1, -1):
                pred = self.unet(sample, torch.full((b,), t, device=device, dtype=torch.long), global_cond=cond)
                sample = self.scheduler.step(pred, t, sample)

        denorm = self.normalizer.denormalize(sample)
        latent = denorm[..., : self.embedding_dim]
        wrist = denorm[..., self.embedding_dim :]
        embodiments: List[str] = batch["embodiment"]
        if len(set(embodiments)) != 1:
            actions = [self.clip.decode(name, latent[i]) for i, name in enumerate(embodiments)]
            action = torch.stack(actions, dim=0)
        else:
            action = self.clip.decode(embodiments[0], latent)
        return {"action": action, "wrist_pose": wrist, "latent": latent}


class EMAModel:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def copy_to(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.clone() for k, v in state.items()}
