"""DDIM sampling for synthetic wildfire-smoke images."""

from __future__ import annotations

import numpy as np
import torch

from ..models.schedule import make_schedule, classifier_free_guidance, ddim_step


@torch.no_grad()
def generate_image(
    unet,
    vae,
    mask: torch.Tensor,
    masked_image: torch.Tensor,
    text: torch.Tensor,
    cfg: dict,
    device: torch.device,
    seed: int | None = None,
) -> np.ndarray:
    if seed is not None:
        torch.manual_seed(seed)
    schedule = make_schedule(
        cfg["diffusion"]["timesteps"],
        s=cfg["diffusion"]["cosine_s"],
        beta_max=cfg["diffusion"]["beta_max"],
        device=device,
    )
    steps = cfg["inference"]["steps"]
    gamma = cfg["inference"]["guidance_scale"]
    T = cfg["diffusion"]["timesteps"]
    times = torch.linspace(T - 1, 0, steps, device=device).long()
    z = torch.randn(
        1,
        cfg["vae_channels"],
        cfg["latent_size"],
        cfg["latent_size"],
        device=device,
    )
    unet.eval()
    vae.eval()
    mask = mask.to(device)
    masked_image = masked_image.to(device)
    text = text.to(device)
    null = torch.zeros_like(text)
    zero_mask = torch.zeros_like(mask)
    zero_img = torch.zeros_like(masked_image)
    for i, t in enumerate(times):
        t_batch = t.view(1)
        t_prev = times[i + 1].view(1) if i + 1 < len(times) else torch.tensor([-1], device=device)
        eps_c = unet(z, t_batch, mask, masked_image, text)
        eps_u = unet(z, t_batch, zero_mask, zero_img, null)
        eps = classifier_free_guidance(eps_u, eps_c, gamma)
        z = ddim_step(z, t_batch, t_prev, eps, schedule)
    x = vae.decode(z).clamp(-1, 1)
    img = ((x[0].permute(1, 2, 0).cpu().numpy() + 1) / 2).clip(0, 1)
    # Composite: keep known background pixels from the masked image.
    m = mask[0, 0].cpu().numpy()
    bg = ((masked_image[0].permute(1, 2, 0).cpu().numpy() + 1) / 2).clip(0, 1)
    return img * m[..., None] + bg * (1.0 - m[..., None])


def alpha_blend_baseline(background: np.ndarray, mask: np.ndarray, seed: int = 0) -> np.ndarray:
    """Training-free compositing baseline used in the paper's related-work comparisons."""
    rng = np.random.default_rng(seed)
    h, w = mask.shape
    yy, xx = np.mgrid[0:h, 0:w]
    noise = rng.normal(0, 1, (h, w))
    # Smooth noise via a cheap box filter.
    k = max(3, h // 16)
    kernel = np.ones((k, k)) / (k * k)
    import cv2

    field = cv2.filter2D(noise.astype(np.float32), -1, kernel)
    field = (field - field.min()) / (np.ptp(field) + 1e-8)
    smoke_color = np.array([0.62, 0.62, 0.65]) + rng.normal(0, 0.04, 3)
    alpha = (0.35 + 0.45 * field) * mask
    return (background * (1 - alpha[..., None]) + smoke_color * alpha[..., None]).clip(0, 1)
