"""Cosine noise schedule and DDPM/DDIM utilities (paper Sec. III)."""

from __future__ import annotations

import torch


def cosine_alpha_bar(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Nichol & Dhariwal cosine schedule, Eqs. (7)-(8)."""
    t = torch.arange(timesteps + 1, dtype=torch.float64)
    g = torch.cos(((t / timesteps) + s) / (1.0 + s) * torch.pi * 0.5) ** 2
    alpha_bar = g / g[0]
    return alpha_bar.clamp(min=1e-8, max=1.0)


def make_schedule(
    timesteps: int,
    s: float = 0.008,
    beta_max: float = 0.999,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Build cosine betas/alphas used throughout FireSmokeGenNet."""
    alpha_bar = cosine_alpha_bar(timesteps, s=s)
    betas = (1.0 - (alpha_bar[1:] / alpha_bar[:-1])).clamp(max=beta_max)
    alphas = 1.0 - betas
    alpha_bar_t = torch.cumprod(alphas, dim=0)
    tensors = {
        "betas": betas.float(),
        "alphas": alphas.float(),
        "alpha_bar": alpha_bar_t.float(),
        "sqrt_alpha_bar": torch.sqrt(alpha_bar_t).float(),
        "sqrt_one_minus_alpha_bar": torch.sqrt(1.0 - alpha_bar_t).float(),
        "sqrt_recip_alphas": torch.sqrt(1.0 / alphas).float(),
    }
    return {k: v.to(device) for k, v in tensors.items()}


def q_sample(
    z0: torch.Tensor,
    t: torch.Tensor,
    schedule: dict[str, torch.Tensor],
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward diffusion, Eq. (5): z_t = sqrt(alpha_bar) z0 + sqrt(1-alpha_bar) eps."""
    if noise is None:
        noise = torch.randn_like(z0)
    sqrt_ab = schedule["sqrt_alpha_bar"][t].view(-1, 1, 1, 1)
    sqrt_om = schedule["sqrt_one_minus_alpha_bar"][t].view(-1, 1, 1, 1)
    return sqrt_ab * z0 + sqrt_om * noise, noise


def predicted_x0(
    z_t: torch.Tensor,
    t: torch.Tensor,
    eps: torch.Tensor,
    schedule: dict[str, torch.Tensor],
) -> torch.Tensor:
    sqrt_ab = schedule["sqrt_alpha_bar"][t].view(-1, 1, 1, 1)
    sqrt_om = schedule["sqrt_one_minus_alpha_bar"][t].view(-1, 1, 1, 1)
    return (z_t - sqrt_om * eps) / sqrt_ab.clamp_min(1e-8)


def classifier_free_guidance(
    eps_uncond: torch.Tensor,
    eps_cond: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Eq. (11): eps_uncond + gamma * (eps_cond - eps_uncond)."""
    return eps_uncond + gamma * (eps_cond - eps_uncond)


@torch.no_grad()
def ddim_step(
    z_t: torch.Tensor,
    t: torch.Tensor,
    t_prev: torch.Tensor,
    eps: torch.Tensor,
    schedule: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Deterministic DDIM update (eta = 0)."""
    x0 = predicted_x0(z_t, t, eps, schedule)
    alpha_prev = schedule["alpha_bar"][t_prev].view(-1, 1, 1, 1)
    alpha_prev = torch.where(t_prev < 0, torch.ones_like(alpha_prev), alpha_prev)
    return torch.sqrt(alpha_prev) * x0 + torch.sqrt(1.0 - alpha_prev) * eps


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -torch.log(torch.tensor(10000.0, device=timesteps.device))
        * torch.arange(half, device=timesteps.device)
        / max(half - 1, 1)
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb
