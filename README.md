# robot-clip - Contrastive Action Model for Cross-Embodiment Learning

This is the public implementation for the contrastive action model from "Latent Action Diffusion for Cross-Embodiment Manipulation". This guide is a brief walkthrough on how to use the model.

### Data Preprocessing

Your data should be preprocessed in a NumPy (.npy) file, containing a dictionary with keys that are the modality names and values that are arrays of shape (N, d\_action), where each row entry across modalities should be semantically aligned.

### Model Training

For first training encoders, then decoders, run `python two_step_train.py --config-name two_step_config.yaml` with a reference to the Hydra config of your choice. For joint training, run `python train.py --config-name config.yaml`.

### Usage

You can install the package using `pip install -e .` and easily use it in other packages.

### Customization

For custom modalities, the encoder and decoder classes can be easily modified. 

### Logging

For logging, wandb is used, though it can be easily changed to other providers.
