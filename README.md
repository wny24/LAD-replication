# robot-clip - Contrastive Action Model for Cross-Embodiment Learning

This is the public implementation for the contrastive action model from "Latent Action Diffusion for Cross-Embodiment Manipulation". This guide is a brief walkthrough on how to use the model.

### Data Preprocessing

Your data should be preprocessed in a NumPy (.npy) file, containing a dictionary with keys that are the modality names and values that are arrays of shape (N, d\_action), where each row entry across modalities should be semantically aligned.

### Model Training

Two-step training (encoders then decoders):
- Run `python two_step_train.py --config-name two_step_config.yaml` to first train encoders and then decoders using the provided Hydra config.

Joint training (train encoders and decoders together):
- Run `python joint_train.py --config-name joint_config.yaml` to train encoders and decoders jointly. The joint training script uses `config/joint_config.yaml` (or another Hydra config of your choice) and logs training metrics to wandb when enabled.

Notes:
- Both training flows use the repository's NumPy-based dataset loader. Your dataset must be a NumPy `.npy` file containing a dictionary that maps modality names to arrays of shape `(N, d_action)`. Point the config's `data.source_file` to that `.npy` file.
- Config files live in the `config/` directory. Adjust optimizer, batch size, temperature schedule, and save paths via those configs.

### Usage

You can install the package using `pip install -e .` and easily use it in other packages.

### Customization

For custom modalities, the encoder and decoder classes can be easily modified. 

### Logging

For logging, wandb is used, though it can be easily changed to other providers.
