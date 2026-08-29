"""Tiny frozen-style VAE analogue of the SD-2 latent encoder (f=4 compact / f=8 paper)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyVAE(nn.Module):
    def __init__(self, in_channels: int = 3, latent_channels: int = 4, base: int = 32):
        super().__init__()
        self.enc = nn.Sequential(
            ConvBlock(in_channels, base, stride=2),
            ConvBlock(base, base * 2, stride=2),
            nn.Conv2d(base * 2, 2 * latent_channels, 1),
        )
        self.dec = nn.Sequential(
            nn.Conv2d(latent_channels, base * 2, 3, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvBlock(base * 2, base),
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvBlock(base, base),
            nn.Conv2d(base, in_channels, 3, padding=1),
            nn.Tanh(),
        )
        self.latent_channels = latent_channels

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        mean, logvar = torch.chunk(h, 2, dim=1)
        logvar = logvar.clamp(-10, 10)
        std = torch.exp(0.5 * logvar)
        z = mean + std * torch.randn_like(std)
        return z, mean, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z, mean, logvar = self.encode(x)
        recon = self.decode(z)
        rec_loss = F.mse_loss(recon, x)
        kl = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        return {"recon": recon, "z": z, "rec_loss": rec_loss, "kl": kl}
