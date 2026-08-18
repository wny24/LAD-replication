"""Episode window dataset for latent diffusion policy.

Each demonstration is one ``.npz`` with time-aligned arrays. Embodiment is
declared in ``data/manifest.yaml``, not inside the file.

Required keys
-------------
action : float32 (T, D_raw)
    Embodiment-specific end-effector action in physical units. Encoded online
    by the frozen CLIP encoder (``xhand`` 12-d, ``g2`` 1-d, ``mano`` 189-d).
wrist_pose : float32 (T, D_wrist)
    Wrist / EEF pose that is **not** passed through CLIP. Use zeros of shape
    ``(T, 0)`` only if ``wrist_dim: 0`` in the config.

Optional keys
-------------
image : uint8 (T, H, W, 3) or (T, Ncam, H, W, 3)
lowdim_obs : float32 (T, D_low)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


RAW_ACTION_DIM = {"mano": 189, "xhand": 12, "g2": 1}


@dataclass
class DatasetSpec:
    name: str
    glob: str
    embodiment: str
    weight: float = 1.0


def load_manifest(path: Path) -> List[DatasetSpec]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Copy data/manifest.yaml and fill in dataset globs."
        )
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True) or {}
    rows = payload.get("datasets") or []
    specs = []
    for row in rows:
        specs.append(
            DatasetSpec(
                name=str(row["name"]),
                glob=str(row["glob"]),
                embodiment=str(row["embodiment"]),
                weight=float(row.get("weight", 1.0)),
            )
        )
    return specs


def _resolve_glob(root: Path, pattern: str) -> List[Path]:
    path = Path(pattern)
    if not path.is_absolute():
        path = root / pattern
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.glob("*.npz"))
    return sorted(path.parent.glob(path.name))


def _pad_time(array: np.ndarray, pad_before: int, pad_after: int) -> np.ndarray:
    if pad_before == 0 and pad_after == 0:
        return array
    before = np.repeat(array[:1], pad_before, axis=0) if pad_before else np.empty((0, *array.shape[1:]), dtype=array.dtype)
    after = np.repeat(array[-1:], pad_after, axis=0) if pad_after else np.empty((0, *array.shape[1:]), dtype=array.dtype)
    return np.concatenate([before, array, after], axis=0)


class EpisodeWindowDataset(Dataset):
    def __init__(
        self,
        files: Sequence[Path],
        embodiment: str,
        horizon: int,
        n_obs_steps: int,
        use_image: bool,
        lowdim_dim: int,
        wrist_dim: int,
        image_size: int,
    ):
        if embodiment not in RAW_ACTION_DIM:
            raise ValueError(f"embodiment must be one of {list(RAW_ACTION_DIM)}, got {embodiment}")
        self.embodiment = embodiment
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps)
        self.use_image = bool(use_image)
        self.lowdim_dim = int(lowdim_dim)
        self.wrist_dim = int(wrist_dim)
        self.image_size = int(image_size)
        self.episodes: List[Dict[str, Any]] = []
        self.index: List[tuple[int, int]] = []
        pad_before = self.n_obs_steps - 1
        pad_after = self.horizon - 1
        expected_action = RAW_ACTION_DIM[embodiment]

        for file in files:
            raw = np.load(file, allow_pickle=False)
            if "action" not in raw.files:
                raise KeyError(f"{file} missing required key 'action'")
            action = np.asarray(raw["action"], dtype=np.float32)
            if action.ndim != 2 or action.shape[1] != expected_action:
                raise ValueError(
                    f"{file} action shape {action.shape}, expected (T, {expected_action}) for {embodiment}"
                )
            t_len = action.shape[0]
            if "wrist_pose" in raw.files:
                wrist = np.asarray(raw["wrist_pose"], dtype=np.float32)
            elif self.wrist_dim == 0:
                wrist = np.zeros((t_len, 0), dtype=np.float32)
            else:
                raise KeyError(f"{file} missing 'wrist_pose' (T, {self.wrist_dim})")
            if wrist.shape[0] != t_len or wrist.shape[1] != self.wrist_dim:
                raise ValueError(f"{file} wrist_pose {wrist.shape}, expected (T, {self.wrist_dim})")

            image = None
            if self.use_image:
                if "image" not in raw.files:
                    raise KeyError(f"{file} missing 'image' but obs.use_image=true")
                image = np.asarray(raw["image"])
                if image.ndim == 4:
                    image = image[:, None]
                if image.ndim != 5:
                    raise ValueError(f"{file} image must be (T,H,W,3) or (T,Ncam,H,W,3), got {image.shape}")
                if image.shape[0] != t_len:
                    raise ValueError(f"{file} image T={image.shape[0]} != action T={t_len}")

            lowdim = None
            if self.lowdim_dim > 0:
                if "lowdim_obs" not in raw.files:
                    raise KeyError(f"{file} missing 'lowdim_obs'")
                lowdim = np.asarray(raw["lowdim_obs"], dtype=np.float32)
                if lowdim.shape != (t_len, self.lowdim_dim):
                    raise ValueError(f"{file} lowdim_obs {lowdim.shape}, expected ({t_len}, {self.lowdim_dim})")

            episode = {
                "action": _pad_time(action, pad_before, pad_after),
                "wrist_pose": _pad_time(wrist, pad_before, pad_after),
            }
            if image is not None:
                episode["image"] = _pad_time(image, pad_before, pad_after)
            if lowdim is not None:
                episode["lowdim"] = _pad_time(lowdim, pad_before, pad_after)
            ep_id = len(self.episodes)
            self.episodes.append(episode)
            for start in range(t_len):
                self.index.append((ep_id, start + pad_before))

        if not self.index:
            raise ValueError(f"No windows from {len(files)} files for {embodiment}")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ep_id, center = self.index[idx]
        ep = self.episodes[ep_id]
        obs_start = center - self.n_obs_steps + 1
        obs_end = center + 1
        act_end = center + self.horizon
        item: Dict[str, Any] = {
            "action": torch.from_numpy(ep["action"][center:act_end].copy()),
            "wrist_pose": torch.from_numpy(ep["wrist_pose"][center:act_end].copy()),
            "embodiment": self.embodiment,
        }
        if "image" in ep:
            frames = ep["image"][obs_start:obs_end]  # (To, Ncam, H, W, 3)
            # (To, Ncam, C, H, W)
            image = np.transpose(frames, (0, 1, 4, 2, 3))
            item["obs_image"] = torch.from_numpy(np.ascontiguousarray(image))
        if "lowdim" in ep:
            item["obs_lowdim"] = torch.from_numpy(ep["lowdim"][obs_start:obs_end].copy())
        return item


def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "embodiment": [row["embodiment"] for row in batch],
        "action": torch.stack([row["action"] for row in batch], dim=0),
        "wrist_pose": torch.stack([row["wrist_pose"] for row in batch], dim=0),
    }
    if "obs_image" in batch[0]:
        out["obs_image"] = torch.stack([row["obs_image"] for row in batch], dim=0)
    if "obs_lowdim" in batch[0]:
        out["obs_lowdim"] = torch.stack([row["obs_lowdim"] for row in batch], dim=0)
    return out


def build_dataset(spec: DatasetSpec, data_root: Path, config) -> EpisodeWindowDataset:
    files = _resolve_glob(data_root, spec.glob)
    if not files:
        raise FileNotFoundError(f"Dataset {spec.name!r}: no files for glob {spec.glob}")
    obs = config.obs
    return EpisodeWindowDataset(
        files=files,
        embodiment=spec.embodiment,
        horizon=int(config.horizon),
        n_obs_steps=int(obs.n_obs_steps),
        use_image=bool(obs.use_image),
        lowdim_dim=int(obs.lowdim_dim),
        wrist_dim=int(config.wrist_dim),
        image_size=int(obs.image_size),
    )


def build_dataloader(config, data_root: Path, drop_last: bool = True) -> DataLoader:
    manifest = Path(config.data.manifest)
    if not manifest.is_absolute():
        manifest = data_root / manifest
    specs = load_manifest(manifest)
    if not specs:
        raise FileNotFoundError(
            "data/manifest.yaml has an empty datasets list. Add episode globs before training."
        )
    pieces = []
    weights = []
    for spec in specs:
        ds = build_dataset(spec, data_root, config)
        pieces.append(ds)
        weights.extend([spec.weight] * len(ds))

    if len(pieces) == 1:
        dataset = pieces[0]
        sampler = None
        shuffle = True
    else:
        dataset = torch.utils.data.ConcatDataset(pieces)
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(dataset),
            replacement=True,
        )
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=int(config.training.batch_size),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(config.training.num_workers),
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=_collate,
    )


def iter_windows(loader: DataLoader) -> Iterable[Dict[str, Any]]:
    while True:
        for batch in loader:
            yield batch
