#!/usr/bin/env python3
"""Train a Chi-style diffusion policy in the frozen CLIP latent action space."""

from __future__ import annotations

import sys
from pathlib import Path

DIFFUSION_ROOT = Path(__file__).resolve().parent
REPO_ROOT = DIFFUSION_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(DIFFUSION_ROOT))

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from clip_frozen import FrozenActionCLIP
from dataset import build_dataloader
from factory import (
    build_policy,
    device_from_config,
    fit_normalizer,
    move_batch,
    trainable_params,
)
from policy import EMAModel


@hydra.main(config_path="config", config_name="train", version_base="1.1")
def main(config: DictConfig) -> None:
    device = device_from_config(config)
    print(OmegaConf.to_yaml(config))
    print(f"device={device}")

    try:
        loader = build_dataloader(config, DIFFUSION_ROOT, drop_last=True)
    except FileNotFoundError as exc:
        raise SystemExit(f"{exc}\nSee {DIFFUSION_ROOT / 'README.md'}") from exc

    clip_path = Path(config.clip.checkpoint).expanduser()
    if not clip_path.is_absolute():
        clip_path = (DIFFUSION_ROOT / clip_path).resolve()
    clip = FrozenActionCLIP(str(clip_path), device)
    config.clip.checkpoint = str(clip_path)

    save_dir = Path(config.training.save_path)
    if not save_dir.is_absolute():
        save_dir = DIFFUSION_ROOT / save_dir
    config.training.save_path = str(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Fitting latent+wrist normalizer on a data subset (CLIP frozen)...")
    normalizer = fit_normalizer(
        loader,
        clip,
        int(config.wrist_dim),
        int(config.training.normalizer_samples),
        device,
        n_arms=int(getattr(config, "n_arms", 1)),
    )

    policy = build_policy(config, clip, normalizer).to(device)
    for param in policy.clip.parameters():
        param.requires_grad = False
    ema = EMAModel(policy.unet, decay=float(config.training.ema_decay))
    ema_obs = EMAModel(policy.obs_encoder, decay=float(config.training.ema_decay))
    optimizer = torch.optim.AdamW(
        trainable_params(policy),
        lr=float(config.optimizer.lr),
        weight_decay=float(config.optimizer.weight_decay),
    )

    steps = int(config.training.steps)
    log_every = int(config.training.log_interval)
    save_every = int(config.training.save_interval)
    running = 0.0
    data_iter = iter(loader)

    policy.train()
    policy.clip.eval()
    progress = tqdm(range(1, steps + 1), desc="diffusion")
    for step in progress:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss = policy.compute_loss(batch)
        loss.backward()
        if config.training.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params(policy), float(config.training.grad_clip))
        optimizer.step()
        ema.update(policy.unet)
        ema_obs.update(policy.obs_encoder)
        running += float(loss.item())
        if step % log_every == 0:
            avg = running / log_every
            running = 0.0
            progress.set_postfix(loss=f"{avg:.4f}")
            print(f"step {step}/{steps}  loss={avg:.4f}")
        if step % save_every == 0 or step == steps:
            ckpt = {
                "step": step,
                "unet": policy.unet.state_dict(),
                "obs_encoder": policy.obs_encoder.state_dict(),
                "ema_unet": ema.state_dict(),
                "ema_obs": ema_obs.state_dict(),
                "optimizer": optimizer.state_dict(),
                "normalizer": normalizer.state_dict(),
                "config": OmegaConf.to_container(config, resolve=True),
                "clip_checkpoint": str(clip_path),
            }
            path = save_dir / f"policy_step_{step}.pth"
            torch.save(ckpt, path)
            torch.save(ckpt, save_dir / "policy_latest.pth")
            print(f"saved {path}")


if __name__ == "__main__":
    main()
