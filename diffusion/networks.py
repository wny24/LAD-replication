"""1D temporal U-Net with FiLM conditioning (Chi et al., Diffusion Policy)."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10000) / max(half - 1, 1)
        freq = torch.exp(torch.arange(half, device=x.device, dtype=x.dtype) * -scale)
        emb = x.float().unsqueeze(-1) * freq.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class Conv1dBlock(nn.Module):
    def __init__(self, inp: int, out: int, kernel_size: int, n_groups: int = 8):
        super().__init__()
        groups = min(n_groups, out)
        while out % groups != 0:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv1d(inp, out, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(groups, out),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_channels * 2),
        )
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.blocks[0](x)
        scale, bias = self.cond_encoder(cond).unsqueeze(-1).chunk(2, dim=1)
        out = out * (1.0 + scale) + bias
        out = self.blocks[1](out)
        return out + self.residual(x)


class ConditionalUnet1D(nn.Module):
    """Noise-prediction network over a length-H action sequence.

    Input / output: ``(B, H, input_dim)``. ``global_cond`` is ``(B, cond_dim)``.
    """

    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        down_dims=(256, 512, 1024),
        diffusion_step_embed_dim: int = 256,
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        super().__init__()
        down_dims = list(down_dims)
        all_dims = [input_dim] + down_dims
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        self.down_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(dim_in, dim_out, cond_dim, kernel_size, n_groups),
                        ConditionalResidualBlock1D(dim_out, dim_out, cond_dim, kernel_size, n_groups),
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid = down_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(mid, mid, cond_dim, kernel_size, n_groups),
                ConditionalResidualBlock1D(mid, mid, cond_dim, kernel_size, n_groups),
            ]
        )

        self.up_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim, kernel_size, n_groups),
                        ConditionalResidualBlock1D(dim_in, dim_in, cond_dim, kernel_size, n_groups),
                        nn.ConvTranspose1d(dim_in, dim_in, 4, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if timestep.ndim == 0:
            timestep = timestep.expand(sample.shape[0])
        x = sample.transpose(1, 2)
        t_emb = self.diffusion_step_encoder(timestep)
        cond = t_emb if global_cond is None else torch.cat([t_emb, global_cond], dim=-1)

        skips = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, cond)
            x = resnet2(x, cond)
            skips.append(x)
            x = downsample(x)

        for mid in self.mid_modules:
            x = mid(x, cond)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, skips.pop()), dim=1)
            x = resnet(x, cond)
            x = resnet2(x, cond)
            x = upsample(x)

        x = self.final_conv(x)
        return x.transpose(1, 2)
