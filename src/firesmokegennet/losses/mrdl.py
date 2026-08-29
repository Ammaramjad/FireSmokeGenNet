"""Mask Random Difference Loss (MRDL), paper Eqs. (28)-(34)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    k = 2 * radius + 1
    return F.max_pool2d(mask, kernel_size=k, stride=1, padding=radius)


def erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    return 1.0 - dilate(1.0 - mask, radius)


def morph_perturb(
    mask: torch.Tensor,
    k_min: int = 1,
    k_max: int = 3,
    n_min: int = 1,
    n_max: int = 3,
) -> torch.Tensor:
    """Stochastic dilation/erosion sequence, Eq. (28)."""
    b = mask.shape[0]
    out = mask.clone()
    for i in range(b):
        n_ops = int(torch.randint(n_min, n_max + 1, (1,)).item())
        cur = out[i : i + 1]
        for _ in range(n_ops):
            k = int(torch.randint(k_min, k_max + 1, (1,)).item())
            if torch.rand(1).item() < 0.5:
                cur = dilate(cur, k)
            else:
                cur = erode(cur, k)
        out[i : i + 1] = cur
    return out.clamp(0, 1)


def boundary_band(mask: torch.Tensor, mask_pert: torch.Tensor, latent_hw: tuple[int, int]) -> torch.Tensor:
    """Symmetric difference resized to latent resolution, Eq. (32)."""
    band = (mask - mask_pert).abs()
    return F.interpolate(band, size=latent_hw, mode="nearest")


def mrdl_loss(
    eps_orig: torch.Tensor,
    eps_pert: torch.Tensor,
    band: torch.Tensor,
    eta: float = 1e-6,
) -> torch.Tensor:
    """Normalized one-sided consistency loss, Eq. (33)."""
    delta = band * (eps_orig - eps_pert.detach())
    numer = delta.pow(2).flatten(1).sum(dim=1)
    denom = band.flatten(1).sum(dim=1) + eta
    return (numer / denom).mean()


def total_loss(diff_loss: torch.Tensor, mrd_loss: torch.Tensor, omega: float) -> torch.Tensor:
    """Eq. (34): (1-omega) L_diff + omega L_MRDL."""
    return (1.0 - omega) * diff_loss + omega * mrd_loss
