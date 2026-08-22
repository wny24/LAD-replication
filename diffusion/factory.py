"""Shared construction helpers for latent diffusion train / infer."""

from __future__ import annotations

import torch
from omegaconf import DictConfig

from clip_frozen import FrozenActionCLIP
from networks import ConditionalUnet1D
from normalizer import GaussianNormalizer
from policy import LatentDiffusionPolicy
from scheduler import DDPMScheduler
from vision import ObservationEncoder


def device_from_config(config: DictConfig) -> torch.device:
    if config.training.device == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{config.training.gpu_id}")
    return torch.device("cpu")


def move_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def trainable_params(policy: LatentDiffusionPolicy):
    return list(policy.obs_encoder.parameters()) + list(policy.unet.parameters())


def fit_normalizer(
    loader,
    clip: FrozenActionCLIP,
    wrist_dim: int,
    max_samples: int,
    device: torch.device,
    n_arms: int = 1,
):
    import numpy as np

    rows = []
    seen = 0
    clip.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            z = clip.encode_batch(batch["embodiment"], batch["action"], n_arms=n_arms)
            target = z if wrist_dim == 0 else torch.cat([z, batch["wrist_pose"]], dim=-1)
            rows.append(target.detach().cpu().numpy())
            seen += target.shape[0]
            if seen >= max_samples:
                break
    if not rows:
        raise RuntimeError("Normalizer fit saw zero batches")
    data = np.concatenate([row.reshape(-1, row.shape[-1]) for row in rows], axis=0)
    return GaussianNormalizer.fit(data)


def build_policy(
    config: DictConfig, clip: FrozenActionCLIP, normalizer: GaussianNormalizer
) -> LatentDiffusionPolicy:
    n_arms = int(getattr(config, "n_arms", 1))
    obs = ObservationEncoder(
        use_image=bool(config.obs.use_image),
        n_obs_steps=int(config.obs.n_obs_steps),
        n_cameras=int(config.obs.n_cameras),
        image_size=int(config.obs.image_size),
        lowdim_dim=int(config.obs.lowdim_dim),
    )
    action_dim = int(clip.embedding_dim) * n_arms + int(config.wrist_dim)
    unet = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs.feature_dim,
        down_dims=list(config.unet.down_dims),
        diffusion_step_embed_dim=int(config.unet.diffusion_step_embed_dim),
        kernel_size=int(config.unet.kernel_size),
        n_groups=int(config.unet.n_groups),
    )
    scheduler = DDPMScheduler(
        num_train_timesteps=int(config.scheduler.num_train_timesteps),
        beta_start=float(config.scheduler.beta_start),
        beta_end=float(config.scheduler.beta_end),
        beta_schedule=str(config.scheduler.beta_schedule),
        clip_sample=bool(config.scheduler.clip_sample),
        prediction_type=str(config.scheduler.prediction_type),
    )
    return LatentDiffusionPolicy(
        clip=clip,
        obs_encoder=obs,
        unet=unet,
        scheduler=scheduler,
        normalizer=normalizer,
        embedding_dim=clip.embedding_dim,
        wrist_dim=int(config.wrist_dim),
        horizon=int(config.horizon),
        n_arms=n_arms,
    )
