#!/usr/bin/env python3
"""Run a trained RobotCLIP checkpoint on a contiguous slice of aligned .npy.

Writes vis-ready dicts (keys ``local_representation`` / ``xhand_angles`` /
``g2_width``) that ``retargeting/scripts/visualize_aligned_npy.py`` can open.

    conda activate LAD
    python scripts/infer_aligned.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robot_clip.data_loading import adapt_data_keys, denormalize_data, normalize_data
from robot_clip.utils import load_checkpoint_file

def _parse() -> argparse.Namespace:
    repo = ROOT.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "model_epoch_250.pth",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=repo / "retargeting" / "data" / "aligned_mano_xhand_g2.npy",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo / "retargeting" / "data",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _to_npy(modalities: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {
        "local_representation": np.asarray(modalities["mano"], dtype=np.float32),
        "xhand_angles": np.asarray(modalities["xhand"], dtype=np.float32),
        "g2_width": np.clip(np.asarray(modalities["g2"], dtype=np.float32), 0.0, 1.0),
    }
    if out["g2_width"].ndim == 1:
        out["g2_width"] = out["g2_width"][:, None]
    return out


def _rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def _as_numpy(batch: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {key: value.detach().cpu().numpy() for key, value in batch.items()}


def _decode_from(model, embedding: torch.Tensor, names: list[str]) -> dict[str, torch.Tensor]:
    return {name: model.decoders[name](embedding) for name in names}


def main() -> None:
    args = _parse()
    model, _config, norm = load_checkpoint_file(str(args.checkpoint))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    raw = np.load(args.source, allow_pickle=True).item()
    full = adapt_data_keys(raw)
    n = int(next(iter(full.values())).shape[0])
    start = max(0, int(args.start))
    stop = min(n, start + int(args.length))
    if stop <= start:
        raise ValueError(f"Empty slice: start={start} length={args.length} n={n}")
    slice_np = {key: np.asarray(value[start:stop]) for key, value in full.items()}
    names = list(slice_np.keys())

    normalized = normalize_data(slice_np, norm)
    batch = {
        key: torch.as_tensor(value, dtype=torch.float32, device=device)
        for key, value in normalized.items()
    }

    with torch.no_grad():
        embeddings = model.encode(batch)
        self_norm = model.decode(embeddings)
        from_mano_norm = _decode_from(model, embeddings["mano"], names)
        from_xhand_norm = _decode_from(model, embeddings["xhand"], names)

    self_recon = denormalize_data(_as_numpy(self_norm), norm)
    from_mano = denormalize_data(_as_numpy(from_mano_norm), norm)
    from_xhand = denormalize_data(_as_numpy(from_xhand_norm), norm)

    # Human-to-robot: keep GT skeleton so viser shows the query pose next to
    # the translated XHand / G2.
    from_mano_vis = dict(from_mano)
    from_mano_vis["mano"] = slice_np["mano"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = "clip_infer"
    exports = {
        f"{stem}_gt.npy": slice_np,
        f"{stem}_self_recon.npy": self_recon,
        f"{stem}_from_mano.npy": from_mano_vis,
        f"{stem}_from_xhand.npy": from_xhand,
    }
    for name, payload in exports.items():
        path = args.output_dir / name
        np.save(path, _to_npy(payload), allow_pickle=True)
        print(f"wrote {path}  frames {start}:{stop}")

    print(f"\nslice [{start}:{stop}] of {args.source.name}  (n={n})")
    print("RMSE vs GT")
    print("  self-recon     " + "  ".join(f"{k}={_rmse(self_recon[k], slice_np[k]):.4f}" for k in names))
    print("  from mano      " + "  ".join(f"{k}={_rmse(from_mano[k], slice_np[k]):.4f}" for k in names))
    print("  from xhand     " + "  ".join(f"{k}={_rmse(from_xhand[k], slice_np[k]):.4f}" for k in names))
    print(
        "\nVisualize (conda LAD):\n"
        "  cd retargeting && python scripts/visualize_aligned_npy.py "
        f"--file {args.output_dir / (stem + '_from_mano.npy')} --port 8095"
    )


if __name__ == "__main__":
    main()
