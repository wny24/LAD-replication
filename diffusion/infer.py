#!/usr/bin/env python3
"""Sample a latent action sequence from a trained policy and decode with frozen CLIP."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DIFFUSION_ROOT = Path(__file__).resolve().parent
REPO_ROOT = DIFFUSION_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(DIFFUSION_ROOT))

import numpy as np
import torch
from omegaconf import OmegaConf

from checkpointing import load_ema_weights
from clip_frozen import FrozenActionCLIP
from dataset import RAW_ACTION_DIM, _pad_time
from factory import build_policy, device_from_config, move_batch
from normalizer import GaussianNormalizer


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episode", type=Path, required=True, help="One .npz demo to take obs from")
    parser.add_argument("--embodiment", required=True, choices=sorted(RAW_ACTION_DIM))
    parser.add_argument("--t", type=int, default=0, help="Start index in the episode")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--ddim-steps", type=int, default=16)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _obs_from_episode(path: Path, config, t: int, embodiment: str) -> dict:
    raw = np.load(path, allow_pickle=False)
    action = np.asarray(raw["action"], dtype=np.float32)
    t_len = action.shape[0]
    t = int(np.clip(t, 0, t_len - 1))
    n_obs = int(config.obs.n_obs_steps)
    horizon = int(config.horizon)
    pad_before = n_obs - 1
    pad_after = horizon - 1
    action_p = _pad_time(action, pad_before, pad_after)
    center = t + pad_before
    batch = {
        "embodiment": [embodiment],
        "action": torch.from_numpy(action_p[center : center + horizon][None].copy()),
    }
    if int(config.wrist_dim) > 0:
        wrist = np.asarray(raw["wrist_pose"], dtype=np.float32)
        wrist_p = _pad_time(wrist, pad_before, pad_after)
        batch["wrist_pose"] = torch.from_numpy(wrist_p[center : center + horizon][None].copy())
    else:
        batch["wrist_pose"] = torch.zeros(1, horizon, 0)
    if config.obs.use_image:
        image = np.asarray(raw["image"])
        if image.ndim == 4:
            image = image[:, None]
        image_p = _pad_time(image, pad_before, pad_after)
        frames = image_p[center - n_obs + 1 : center + 1]
        image_t = np.transpose(frames, (0, 1, 4, 2, 3))
        batch["obs_image"] = torch.from_numpy(np.ascontiguousarray(image_t))[None]
    if int(config.obs.lowdim_dim) > 0:
        low = np.asarray(raw["lowdim_obs"], dtype=np.float32)
        low_p = _pad_time(low, pad_before, pad_after)
        batch["obs_lowdim"] = torch.from_numpy(low_p[center - n_obs + 1 : center + 1][None].copy())
    return batch


def main() -> None:
    args = _parse()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = OmegaConf.create(ckpt["config"])
    if args.device is not None:
        config.training.device = args.device
    device = device_from_config(config)
    clip = FrozenActionCLIP(str(ckpt.get("clip_checkpoint") or config.clip.checkpoint), device)
    normalizer = GaussianNormalizer.from_state_dict(ckpt["normalizer"])
    policy = build_policy(config, clip, normalizer).to(device)
    load_ema_weights(policy, ckpt)

    batch = move_batch(_obs_from_episode(args.episode, config, args.t, args.embodiment), device)
    policy.eval()
    out = policy.sample(batch, num_inference_steps=args.ddim_steps, use_ddim=True)
    result = {
        "action": out["action"][0].cpu().numpy(),
        "wrist_pose": out["wrist_pose"][0].cpu().numpy(),
        "latent": out["latent"][0].cpu().numpy(),
        "gt_action": batch["action"][0].cpu().numpy(),
        "embodiment": args.embodiment,
    }
    dest = args.out or Path(args.checkpoint).with_name("sample.npz")
    np.savez_compressed(dest, **result)
    err = np.sqrt(np.mean((result["action"] - result["gt_action"]) ** 2))
    print(f"wrote {dest}")
    print(f"decoded action RMSE vs demo window: {err:.4f}")


if __name__ == "__main__":
    main()
