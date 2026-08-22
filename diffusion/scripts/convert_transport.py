#!/usr/bin/env python3
"""Convert raw transport teleop into **bimanual** diffusion episode .npz files.

Raw layout (after unzip into ``data/raw_*``)::

    left_action / right_action
        xhand: (T, 19) = 7 arm joints + 12 finger joints
        g2:    (T, 8)  = 7 arm joints + 1 gripper (0..1000)
    qpos similarly
    rgb_video_file + rgb_timestamp  (images live in sibling .mp4)

Writes one episode per demo under ``data/{xhand,g2}/episode_N.npz``::

    action      (T, 2*D_hand)   [left_hand, right_hand] → frozen CLIP (each side)
    wrist_pose  (T, 14)         [left_arm_cmd, right_arm_cmd] (not through CLIP)
    lowdim_obs  (T, 14)         [left_arm_qpos, right_arm_qpos]
    image       (T, H, W, 3)    shared camera, timestamp-aligned

    conda activate LAD
    cd robot_clip/diffusion
    python scripts/convert_transport.py --overwrite
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

G2_GRIPPER_SCALE = 1000.0
ARM_DIM = 7
XHAND_HAND_DIM = 12
G2_HAND_DIM = 1
N_ARMS = 2


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--xhand-raw",
        type=Path,
        default=DATA / "raw_xhand" / "transport-test",
    )
    p.add_argument(
        "--g2-raw",
        type=Path,
        default=DATA / "raw_g2" / "transport_g2_controller",
    )
    p.add_argument("--out-root", type=Path, default=DATA)
    p.add_argument("--image-size", type=int, default=84, help="Resize RGB to SxS (0 = keep native)")
    p.add_argument("--max-episodes", type=int, default=0, help="0 = all")
    p.add_argument("--skip-video", action="store_true", help="Write arrays without image key")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _episode_index(path: Path) -> int:
    match = re.search(r"episode_(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def _list_episodes(raw_dir: Path) -> list[Path]:
    return sorted(raw_dir.glob("episode_*.npz"), key=_episode_index)


def _split_side(arr: np.ndarray, embodiment: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (arm_7, hand_D) from a single-side action/qpos row vector."""
    arr = np.asarray(arr, dtype=np.float32)
    if embodiment == "xhand":
        if arr.shape[-1] != ARM_DIM + XHAND_HAND_DIM:
            raise ValueError(f"xhand side dim {arr.shape[-1]}, expected {ARM_DIM + XHAND_HAND_DIM}")
        return arr[..., :ARM_DIM], arr[..., ARM_DIM:]
    if embodiment == "g2":
        if arr.shape[-1] != ARM_DIM + G2_HAND_DIM:
            raise ValueError(f"g2 side dim {arr.shape[-1]}, expected {ARM_DIM + G2_HAND_DIM}")
        arm = arr[..., :ARM_DIM]
        grip = np.clip(arr[..., ARM_DIM:] / G2_GRIPPER_SCALE, 0.0, 1.0)
        return arm, grip.astype(np.float32)
    raise ValueError(embodiment)


def _load_video_frames(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")
    frames: list[np.ndarray] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames in {video_path}")
    return np.stack(frames, axis=0)


def _align_images(
    frames: np.ndarray,
    rgb_timestamp: np.ndarray,
    action_timestamp: np.ndarray,
) -> np.ndarray:
    rgb_timestamp = np.asarray(rgb_timestamp, dtype=np.float64)
    action_timestamp = np.asarray(action_timestamp, dtype=np.float64)
    n_rgb = min(len(frames), len(rgb_timestamp))
    frames = frames[:n_rgb]
    rgb_timestamp = rgb_timestamp[:n_rgb]
    order = np.argsort(rgb_timestamp)
    ts_sorted = rgb_timestamp[order]
    idx = np.searchsorted(ts_sorted, action_timestamp, side="left")
    idx = np.clip(idx, 0, n_rgb - 1)
    left = np.clip(idx - 1, 0, n_rgb - 1)
    choose_left = np.abs(ts_sorted[left] - action_timestamp) <= np.abs(ts_sorted[idx] - action_timestamp)
    nearest = np.where(choose_left, left, idx)
    return frames[order[nearest]]


def _resize(images: np.ndarray, size: int) -> np.ndarray:
    if size <= 0 or (images.shape[1] == size and images.shape[2] == size):
        return images
    out = np.empty((images.shape[0], size, size, 3), dtype=np.uint8)
    for i, frame in enumerate(images):
        out[i] = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    return out


def convert_episode_bimanual(
    npz_path: Path,
    embodiment: str,
    image_size: int,
    skip_video: bool,
) -> dict[str, np.ndarray]:
    raw = np.load(npz_path, allow_pickle=False)
    for side in ("left", "right"):
        if f"{side}_action" not in raw.files or f"{side}_qpos" not in raw.files:
            raise KeyError(f"{npz_path.name} missing {side}_action / {side}_qpos")

    left_arm_cmd, left_hand = _split_side(raw["left_action"], embodiment)
    right_arm_cmd, right_hand = _split_side(raw["right_action"], embodiment)
    left_arm_state, _ = _split_side(raw["left_qpos"], embodiment)
    right_arm_state, _ = _split_side(raw["right_qpos"], embodiment)

    out: dict[str, np.ndarray] = {
        "action": np.ascontiguousarray(
            np.concatenate([left_hand, right_hand], axis=-1), dtype=np.float32
        ),
        "wrist_pose": np.ascontiguousarray(
            np.concatenate([left_arm_cmd, right_arm_cmd], axis=-1), dtype=np.float32
        ),
        "lowdim_obs": np.ascontiguousarray(
            np.concatenate([left_arm_state, right_arm_state], axis=-1), dtype=np.float32
        ),
    }

    if not skip_video:
        video_path = npz_path.with_name(str(raw["rgb_video_file"]))
        if not video_path.exists():
            raise FileNotFoundError(video_path)
        frames = _load_video_frames(video_path)
        action_ts = np.asarray(raw["action_timestamp"], dtype=np.float64)
        rgb_ts = np.asarray(raw["rgb_timestamp"], dtype=np.float64)
        if len(rgb_ts) == len(action_ts) and len(frames) == len(action_ts):
            images = frames
        else:
            images = _align_images(frames, rgb_ts, action_ts)
        if images.shape[0] != left_arm_cmd.shape[0]:
            raise RuntimeError(
                f"{npz_path.name}: image T={images.shape[0]} != action T={left_arm_cmd.shape[0]}"
            )
        out["image"] = np.ascontiguousarray(_resize(images, image_size), dtype=np.uint8)

    return out


def _clear_legacy_single_arm(out_dir: Path) -> None:
    """Remove older left/right-split episodes if present."""
    for path in out_dir.glob("episode_*_left.npz"):
        path.unlink(missing_ok=True)
    for path in out_dir.glob("episode_*_right.npz"):
        path.unlink(missing_ok=True)


def convert_dataset(
    raw_dir: Path,
    embodiment: str,
    out_dir: Path,
    image_size: int,
    max_episodes: int,
    skip_video: bool,
    overwrite: bool,
) -> list[Path]:
    episodes = _list_episodes(raw_dir)
    if max_episodes > 0:
        episodes = episodes[:max_episodes]
    out_dir.mkdir(parents=True, exist_ok=True)
    _clear_legacy_single_arm(out_dir)
    written: list[Path] = []
    for npz_path in tqdm(episodes, desc=f"{embodiment}-bimanual"):
        dest = out_dir / f"{npz_path.stem}.npz"
        if dest.exists() and not overwrite:
            written.append(dest)
            continue
        payload = convert_episode_bimanual(npz_path, embodiment, image_size, skip_video)
        np.savez_compressed(dest, **payload)
        written.append(dest)
        shapes = {k: tuple(v.shape) for k, v in payload.items()}
        tqdm.write(f"  wrote {dest.name}  {shapes}")
    return written


def write_manifest(path: Path, xhand_glob: str, g2_glob: str) -> None:
    text = f"""# Auto-generated by scripts/convert_transport.py (bimanual)
# Relative to robot_clip/diffusion/

datasets:
  - name: transport_xhand
    glob: {xhand_glob}
    embodiment: xhand
    weight: 1.0
  - name: transport_g2
    glob: {g2_glob}
    embodiment: g2
    weight: 1.0
"""
    path.write_text(text)


def main() -> None:
    args = _parse()
    for name, path in (("xhand-raw", args.xhand_raw), ("g2-raw", args.g2_raw)):
        if not path.is_dir():
            raise FileNotFoundError(f"--{name} not found: {path}")

    convert_dataset(
        args.xhand_raw,
        "xhand",
        args.out_root / "xhand",
        args.image_size,
        args.max_episodes,
        args.skip_video,
        args.overwrite,
    )
    convert_dataset(
        args.g2_raw,
        "g2",
        args.out_root / "g2",
        args.image_size,
        args.max_episodes,
        args.skip_video,
        args.overwrite,
    )

    write_manifest(
        args.out_root / "manifest.yaml",
        xhand_glob="data/xhand/episode_*.npz",
        g2_glob="data/g2/episode_*.npz",
    )
    print(
        "\nDone (bimanual). Train with:\n"
        "  n_arms=2\n"
        "  wrist_dim=14\n"
        "  obs.lowdim_dim=14\n"
        "  obs.use_image=true\n"
        f"  obs.image_size={args.image_size if args.image_size > 0 else 84}\n"
        f"Manifest: {args.out_root / 'manifest.yaml'}\n"
        "Note: previous single-arm policy checkpoints are incompatible; retrain diffusion."
    )


if __name__ == "__main__":
    main()
