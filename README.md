# robot-clip - Contrastive Action Model for Cross-Embodiment Learning

This is the public implementation for the contrastive action model from "Latent Action Diffusion for Cross-Embodiment Manipulation". This guide is a brief walkthrough on how to use the model.

### Data Preprocessing

Your data should be preprocessed in a NumPy (.npy) file, containing a dictionary with keys that are the modality names and values that are arrays of shape (N, d\_action), where each row entry across modalities should be semantically aligned.

For human / XHand / G2 data, generate that file with `../retargeting` (keys `local_representation`, `xhand_angles`, `g2_width`) and train with `two_step_xhand_g2.yaml` or `joint_xhand_g2.yaml`.

data path: /home/wny24/LAD-replication/retargeting/data/aligned_mano_xhand_g2.npy

### Model Training

Two-step training (encoders then decoders):
- Run `python two_step_train.py --config-name two_step_config.yaml` to first train encoders and then decoders using the provided Hydra config.

Joint training (train encoders and decoders together):
- Run `python joint_train.py --config-name joint_config.yaml` to train encoders and decoders jointly. The joint training script uses `config/joint_config.yaml` (or another Hydra config of your choice) and logs training metrics to wandb when enabled.

Notes:
- Both training flows use the repository's NumPy-based dataset loader. Your dataset must be a NumPy `.npy` file containing a dictionary that maps modality names to arrays of shape `(N, d_action)`. Point the config's `data.source_file` to that `.npy` file.
- Config files live in the `config/` directory. Adjust optimizer, batch size, temperature schedule, and save paths via those configs.

### Inference

`scripts/infer_aligned.py` loads a `.pth` checkpoint, runs it on a contiguous slice of the aligned `.npy`, and writes vis-ready dicts (`local_representation` / `xhand_angles` / `g2_width`). Normalization uses the **training** mean/std stored in the checkpoint, not stats recomputed on the slice.

```bash
conda activate LAD
cd /home/wny24/LAD-replication/robot_clip

python scripts/infer_aligned.py \
  --checkpoint checkpoints/model_epoch_250.pth \
  --source ../retargeting/data/aligned_mano_xhand_g2.npy \
  --output-dir ../retargeting/data \
  --start 0 \
  --length 256 \
  --device cpu
```

Important arguments:

| Argument | Default | Meaning |
|---|---|---|
| `--checkpoint` | `checkpoints/model_epoch_250.pth` | Trained weights. Must contain `model_state_dict`, `config`, and `normalization_params`. |
| `--source` | `../retargeting/data/aligned_mano_xhand_g2.npy` | Row-aligned training/eval file. Same keys as training. |
| `--output-dir` | `../retargeting/data` | Where the four `clip_infer_*.npy` files are written. |
| `--start` | `0` | First frame index in `--source`. |
| `--length` | `256` | How many consecutive frames to run. Keep this modest for viser playback. |
| `--device` | `cpu` | `cpu` or `cuda`. Falls back to CPU if CUDA is unavailable. |

Outputs (overwrite on each run):

| File | Contents |
|---|---|
| `clip_infer_gt.npy` | The raw slice, for side-by-side comparison. |
| `clip_infer_self_recon.npy` | Each modality decoded from its own embedding (autoencoder check). |
| `clip_infer_from_mano.npy` | Encode human, decode XHand + G2; human skeleton kept as GT so viser shows the query pose next to the translated robots. |
| `clip_infer_from_xhand.npy` | Encode XHand, decode human + G2. |

The script also prints per-modality RMSE vs GT (`mano` is 189-d local pose, `xhand` is radians, `g2` is opening in `[0, 1]`).

Visualize in the existing retargeting viewer (conda `LAD`):

```bash
cd /home/wny24/LAD-replication/retargeting
python scripts/visualize_aligned_npy.py --file data/clip_infer_gt.npy --port 8095
python scripts/visualize_aligned_npy.py --file data/clip_infer_from_mano.npy --port 8096
```

`from_mano` is the cross-embodiment check: same human pose, XHand/G2 come from CLIP instead of retargeting.

### Latent diffusion policy

After the contrastive action model is trained, freeze its encoders/decoders and train a Chi-style diffusion policy in latent space. Code, data format, and commands: [`diffusion/README.md`](diffusion/README.md).

### Usage

You can install the package using `pip install -e .` and easily use it in other packages.

### Customization

For custom modalities, the encoder and decoder classes can be easily modified. 

### Logging

For logging, wandb is used, though it can be easily changed to other providers.
