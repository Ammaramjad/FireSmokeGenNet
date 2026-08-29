"""Hierarchical dual-branch latent U-Net with JCA injection (paper Tables II and IV)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import DualBranchEncoder, ScaleProjector
from .jca import JointCrossAttention, CrossAttention, flatten_tokens
from .schedule import sinusoidal_timestep_embedding


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class ConditioningBlock(nn.Module):
    def __init__(self, channels: int, mode: str, heads: int = 4):
        super().__init__()
        self.mode = mode
        self.norm = nn.GroupNorm(8, channels)
        if mode == "jca":
            self.attn = JointCrossAttention(channels, channels, heads=heads)
        else:
            self.attn = CrossAttention(channels, channels, heads=heads)

    def forward(
        self,
        x: torch.Tensor,
        mask_feat: torch.Tensor,
        image_feat: torch.Tensor,
    ) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = flatten_tokens(self.norm(x))
        if self.mode == "jca":
            out = self.attn(tokens, flatten_tokens(mask_feat), flatten_tokens(image_feat))
        elif self.mode == "mask":
            out = tokens + self.attn(tokens, flatten_tokens(mask_feat))
        else:
            out = tokens + self.attn(tokens, flatten_tokens(image_feat))
        return out.transpose(1, 2).contiguous().view(b, c, h, w)


class FireSmokeUNet(nn.Module):
    def __init__(
        self,
        latent_channels: int = 4,
        channels: list[int] | None = None,
        time_dim: int = 256,
        text_dim: int = 64,
        mask_backbone: str = "resnet18",
        image_backbone: str = "resnet18",
        pretrained_encoders: bool = True,
        use_jca: bool = True,
    ):
        super().__init__()
        channels = channels or [64, 128, 192, 192]
        self.channels = channels
        self.use_jca = use_jca
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.text_proj = nn.Linear(text_dim, time_dim)
        self.in_conv = nn.Conv2d(latent_channels, channels[0], 3, padding=1)
        self.encoder = DualBranchEncoder(
            mask_backbone=mask_backbone,
            image_backbone=image_backbone,
            pretrained=pretrained_encoders,
            freeze=True,
        )
        self.mask_proj = ScaleProjector(self.encoder.mask_channels, channels)
        self.image_proj = ScaleProjector(self.encoder.image_channels, channels)

        self.down_res = nn.ModuleList()
        self.down_cond = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch_in = channels[0]
        down_modes = ["mask", "mask", "jca"]
        for i, ch in enumerate(channels):
            self.down_res.append(ResBlock(ch_in, ch, time_dim))
            mode = down_modes[min(i, 2)]
            if not use_jca and mode == "jca":
                mode = "mask"
            self.down_cond.append(ConditioningBlock(ch, mode))
            if i < len(channels) - 1:
                self.downsamples.append(nn.Conv2d(ch, channels[i + 1], 3, stride=2, padding=1))
            ch_in = channels[min(i + 1, len(channels) - 1)] if i < len(channels) - 1 else ch

        self.mid_res1 = ResBlock(channels[-1], channels[-1], time_dim)
        self.mid_cond = ConditioningBlock(channels[-1], "mask")
        self.mid_res2 = ResBlock(channels[-1], channels[-1], time_dim)

        self.up_res = nn.ModuleList()
        self.up_cond = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        up_modes = ["jca", "image", "image", "image"]
        ch_in = channels[-1]
        for i, ch in enumerate(reversed(channels)):
            skip_ch = ch
            self.up_res.append(ResBlock(ch_in + skip_ch, ch, time_dim))
            mode = up_modes[min(i, 3)]
            if not use_jca and mode == "jca":
                mode = "image"
            self.up_cond.append(ConditioningBlock(ch, mode))
            if i < len(channels) - 1:
                prev = list(reversed(channels))[i + 1]
                self.upsamples.append(nn.ConvTranspose2d(ch, prev, 4, stride=2, padding=1))
            ch_in = list(reversed(channels))[min(i + 1, len(channels) - 1)]

        self.out_norm = nn.GroupNorm(8, channels[0])
        self.out_conv = nn.Conv2d(channels[0], latent_channels, 3, padding=1)

    def _projected(self, mask: torch.Tensor, masked_image: torch.Tensor, spatial: tuple[int, int], stage: int):
        f_m, f_i = self.encoder(mask, masked_image)
        return self.mask_proj(f_m, spatial, stage), self.image_proj(f_i, spatial, stage)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor,
        masked_image: torch.Tensor,
        text_emb: torch.Tensor,
        cond_drop: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        t_emb = self.time_mlp(sinusoidal_timestep_embedding(t, self.time_dim))
        t_emb = t_emb + self.text_proj(text_emb)
        if cond_drop is not None:
            keep = (1.0 - cond_drop.float()).view(-1, 1, 1, 1)
            mask = mask * keep
            masked_image = masked_image * keep
            t_emb = t_emb * keep.view(-1, 1)

        f_m, f_i = self.encoder(mask, masked_image)
        x = self.in_conv(z_t)
        skips = []
        for i, block in enumerate(self.down_res):
            x = block(x, t_emb)
            spatial = tuple(x.shape[-2:])
            stage = min(i, 3)
            m_feat = self.mask_proj.project(f_m, spatial, stage)
            i_feat = self.image_proj.project(f_i, spatial, stage)
            x = self.down_cond[i](x, m_feat, i_feat)
            skips.append(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)

        spatial = tuple(x.shape[-2:])
        m_feat = self.mask_proj.project(f_m, spatial, 2)
        i_feat = self.image_proj.project(f_i, spatial, 2)
        x = self.mid_res1(x, t_emb)
        x = self.mid_cond(x, m_feat, i_feat)
        x = self.mid_res2(x, t_emb)

        for i, block in enumerate(self.up_res):
            skip = skips[-(i + 1)]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)
            x = block(x, t_emb)
            spatial = tuple(x.shape[-2:])
            stage = min(max(len(self.channels) - 1 - i, 0), 3)
            m_feat = self.mask_proj.project(f_m, spatial, stage)
            i_feat = self.image_proj.project(f_i, spatial, stage)
            x = self.up_cond[i](x, m_feat, i_feat)
            if i < len(self.upsamples):
                x = self.upsamples[i](x)
        return self.out_conv(F.silu(self.out_norm(x)))
