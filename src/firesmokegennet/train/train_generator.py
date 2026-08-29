"""Train TinyVAE and FireSmokeGenNet (diffusion + MRDL)."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..losses.mrdl import boundary_band, morph_perturb, mrdl_loss, total_loss
from ..models.schedule import make_schedule, q_sample
from ..models.unet import FireSmokeUNet
from ..models.vae import TinyVAE
from ..utils.io import ensure_dir


def train_vae(dataset, cfg: dict, device: torch.device, out_dir: Path) -> TinyVAE:
    vae = TinyVAE(latent_channels=cfg["vae_channels"]).to(device)
    opt = torch.optim.AdamW(vae.parameters(), lr=cfg["train_vae"]["lr"])
    loader = DataLoader(dataset, batch_size=cfg["train_vae"]["batch_size"], shuffle=True)
    it = iter(loader)
    vae.train()
    pbar = tqdm(range(cfg["train_vae"]["iterations"]), desc="vae")
    for _ in pbar:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        x = batch["image"].to(device) * 2 - 1
        out = vae(x)
        loss = out["rec_loss"] + cfg["train_vae"]["kl_weight"] * out["kl"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        pbar.set_postfix(loss=float(loss))
    vae.eval()
    ensure_dir(out_dir)
    torch.save(vae.state_dict(), out_dir / "vae.pt")
    return vae


def build_unet(cfg: dict, use_jca: bool = True, pretrained_encoders: bool = True) -> FireSmokeUNet:
    enc = cfg["encoders"]
    return FireSmokeUNet(
        latent_channels=cfg["vae_channels"],
        channels=list(cfg["unet"]["channels"]),
        mask_backbone=enc["mask_backbone"],
        image_backbone=enc["image_backbone"],
        pretrained_encoders=pretrained_encoders and enc.get("pretrained", True),
        use_jca=use_jca,
    )


def train_generator(
    dataset,
    vae: TinyVAE,
    cfg: dict,
    device: torch.device,
    out_dir: Path,
    use_jca: bool = True,
    omega: float | None = None,
    pretrained_encoders: bool = True,
    iterations: int | None = None,
    tag: str = "full",
) -> FireSmokeUNet:
    unet = build_unet(cfg, use_jca=use_jca, pretrained_encoders=pretrained_encoders).to(device)
    vae = vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False
    params = [p for p in unet.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        params,
        lr=cfg["train_generator"]["lr"],
        betas=tuple(cfg["train_generator"]["betas"]),
        weight_decay=cfg["train_generator"]["weight_decay"],
    )
    schedule = make_schedule(
        cfg["diffusion"]["timesteps"],
        s=cfg["diffusion"]["cosine_s"],
        beta_max=cfg["diffusion"]["beta_max"],
        device=device,
    )
    omega = cfg["diffusion"]["mrdl_weight"] if omega is None else omega
    steps = iterations or cfg["train_generator"]["iterations"]
    loader = DataLoader(dataset, batch_size=cfg["train_generator"]["batch_size"], shuffle=True)
    it = iter(loader)
    unet.train()
    # Keep frozen encoder in eval.
    unet.encoder.eval()
    pbar = tqdm(range(steps), desc=f"gen:{tag}")
    T = cfg["diffusion"]["timesteps"]
    img_size = cfg["image_size"]
    mrdl_cfg = cfg.get("mrdl", {})
    if "k_min" in mrdl_cfg and "k_max" in mrdl_cfg:
        k_min = int(mrdl_cfg["k_min"])
        k_max = int(mrdl_cfg["k_max"])
    else:
        # Paper uses k ~ U(10, 20) at 512 px; scale for compact resolutions.
        k_max = max(1, int(round(20 * img_size / 512)))
        k_min = max(1, int(round(10 * img_size / 512)))
    for step in pbar:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        x = batch["image"].to(device) * 2 - 1
        mask = batch["mask"].to(device)
        masked = batch["masked"].to(device) * 2 - 1
        text = batch["text"].to(device)
        with torch.no_grad():
            z0, _, _ = vae.encode(x)
        t = torch.randint(0, T, (x.size(0),), device=device)
        z_t, noise = q_sample(z0, t, schedule)
        drop = (torch.rand(x.size(0), device=device) < cfg["diffusion"]["cfg_dropout"]).float()
        eps = unet(z_t, t, mask, masked, text, cond_drop=drop)
        diff = torch.mean((eps - noise) ** 2)
        if omega > 0:
            mask_p = morph_perturb(mask, k_min=k_min, k_max=max(k_min, k_max))
            # Resample empty bands.
            band = boundary_band(mask, mask_p, z0.shape[-2:])
            if band.sum() < 1:
                mask_p = morph_perturb(mask, k_min=k_min, k_max=max(k_min + 1, k_max + 1))
                band = boundary_band(mask, mask_p, z0.shape[-2:])
            eps_p = unet(z_t, t, mask_p, masked, text, cond_drop=drop)
            mrd = mrdl_loss(eps, eps_p, band)
        else:
            mrd = torch.zeros((), device=device)
        loss = total_loss(diff, mrd, omega)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        pbar.set_postfix(loss=float(loss), diff=float(diff), mrd=float(mrd))
    unet.eval()
    ensure_dir(out_dir)
    torch.save(
        {"unet": unet.state_dict(), "cfg": cfg, "tag": tag, "omega": omega, "use_jca": use_jca},
        out_dir / f"generator_{tag}.pt",
    )
    return unet
