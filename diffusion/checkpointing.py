"""Load EMA weights into a trained policy."""

from policy import EMAModel


def load_ema_weights(policy, ckpt: dict) -> None:
    policy.unet.load_state_dict(ckpt["unet"])
    policy.obs_encoder.load_state_dict(ckpt["obs_encoder"])
    if "ema_unet" in ckpt:
        ema = EMAModel(policy.unet)
        ema.load_state_dict(ckpt["ema_unet"])
        ema.copy_to(policy.unet)
    if "ema_obs" in ckpt:
        ema_obs = EMAModel(policy.obs_encoder)
        ema_obs.load_state_dict(ckpt["ema_obs"])
        ema_obs.copy_to(policy.obs_encoder)
