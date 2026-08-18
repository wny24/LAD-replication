"""Observation encoders: ResNet18 images (Chi et al.) and optional low-dim MLP."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def replace_bn_with_gn(module: nn.Module, max_groups: int = 16) -> nn.Module:
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            groups = min(max_groups, child.num_features)
            while child.num_features % groups != 0:
                groups -= 1
            setattr(module, name, nn.GroupNorm(groups, child.num_features))
        else:
            replace_bn_with_gn(child, max_groups=max_groups)
    return module


class ResNet18Encoder(nn.Module):
    def __init__(self, image_size: int = 84):
        super().__init__()
        from torchvision.models import resnet18

        backbone = resnet18(weights=None)
        backbone.fc = nn.Identity()
        self.backbone = replace_bn_with_gn(backbone)
        self.image_size = int(image_size)
        self.feature_dim = 512

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """``image`` is (B, C, H, W) in [0, 1] or uint8-scaled float."""
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        elif image.max() > 1.5:
            image = image / 255.0
        if image.shape[-2] != self.image_size or image.shape[-1] != self.image_size:
            image = F.interpolate(
                image, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
            )
        return self.backbone(image)


class LowdimEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.feature_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ObservationEncoder(nn.Module):
    def __init__(
        self,
        use_image: bool,
        n_obs_steps: int,
        n_cameras: int,
        image_size: int,
        lowdim_dim: int,
        lowdim_out: int = 64,
    ):
        super().__init__()
        self.use_image = bool(use_image)
        self.n_obs_steps = int(n_obs_steps)
        self.n_cameras = int(n_cameras)
        self.lowdim_dim = int(lowdim_dim)
        self.image_encoder = ResNet18Encoder(image_size) if self.use_image else None
        self.lowdim_encoder = (
            LowdimEncoder(self.lowdim_dim * self.n_obs_steps, output_dim=lowdim_out)
            if self.lowdim_dim > 0
            else None
        )
        feat = 0
        if self.image_encoder is not None:
            feat += self.image_encoder.feature_dim * self.n_obs_steps * self.n_cameras
        if self.lowdim_encoder is not None:
            feat += self.lowdim_encoder.feature_dim
        if feat == 0:
            raise ValueError("ObservationEncoder needs image and/or low-dim obs")
        self.feature_dim = feat

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        lowdim: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        parts = []
        if self.image_encoder is not None:
            if image is None:
                raise ValueError("use_image=True but image is None")
            # (B, To, C, H, W) or (B, To, Ncam, C, H, W)
            if image.ndim == 5:
                image = image.unsqueeze(2)
            b, to, ncam, c, h, w = image.shape
            flat = image.reshape(b * to * ncam, c, h, w)
            feat = self.image_encoder(flat).reshape(b, to * ncam * self.image_encoder.feature_dim)
            parts.append(feat)
        if self.lowdim_encoder is not None:
            if lowdim is None:
                raise ValueError("lowdim_dim>0 but lowdim is None")
            # (B, To, D)
            parts.append(self.lowdim_encoder(lowdim.reshape(lowdim.shape[0], -1)))
        return torch.cat(parts, dim=-1)
