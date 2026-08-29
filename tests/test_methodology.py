from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from firesmokegennet.losses.mrdl import boundary_band, morph_perturb, mrdl_loss, total_loss
from firesmokegennet.metrics.stats import ci95, holm_bonferroni, sample_mean, sample_sd
from firesmokegennet.models.jca import JointCrossAttention
from firesmokegennet.models.schedule import cosine_alpha_bar, make_schedule, q_sample, classifier_free_guidance
from firesmokegennet.models.unet import FireSmokeUNet
from firesmokegennet.models.vae import TinyVAE
from firesmokegennet.quality.filter import composite_quality


def test_cosine_schedule_monotonic():
    ab = cosine_alpha_bar(100)
    assert ab[0] == 1 or abs(float(ab[0]) - 1) < 1e-6
    assert float(ab[-1]) < float(ab[0])
    assert torch.all(ab[1:] <= ab[:-1] + 1e-8)


def test_q_sample_shapes():
    sched = make_schedule(50)
    z0 = torch.randn(2, 4, 8, 8)
    t = torch.tensor([0, 10])
    zt, eps = q_sample(z0, t, sched)
    assert zt.shape == z0.shape and eps.shape == z0.shape


def test_cfg_formula():
    u = torch.ones(1, 2)
    c = torch.ones(1, 2) * 3
    out = classifier_free_guidance(u, c, 7.5)
    # 1 + 7.5*(3-1) = 16
    assert torch.allclose(out, torch.tensor([[16.0, 16.0]]))


def test_mrdl_stop_gradient_and_normalization():
    eps = torch.randn(2, 4, 8, 8, requires_grad=True)
    eps_p = torch.randn(2, 4, 8, 8, requires_grad=True)
    mask = torch.zeros(2, 1, 32, 32)
    mask[:, :, 8:24, 8:24] = 1
    mask_p = morph_perturb(mask, 1, 2, 1, 2)
    band = boundary_band(mask, mask_p, (8, 8))
    loss = mrdl_loss(eps, eps_p, band)
    loss.backward()
    assert eps.grad is not None
    assert eps_p.grad is None
    tot = total_loss(torch.tensor(1.0), torch.tensor(0.5), 0.4)
    assert abs(float(tot) - (0.6 * 1.0 + 0.4 * 0.5)) < 1e-6


def test_quality_weights():
    q = composite_quality(np.array([10.0, 5.0, 0.0]))
    assert abs(q - (0.4 * 10 + 0.4 * 5 + 0.2 * 0)) < 1e-6


def test_jca_residual_shape():
    jca = JointCrossAttention(32, 32, heads=4)
    x = torch.randn(2, 16, 32)
    m = torch.randn(2, 8, 32)
    i = torch.randn(2, 8, 32)
    y = jca(x, m, i)
    assert y.shape == x.shape


def test_unet_and_vae_forward():
    vae = TinyVAE()
    x = torch.rand(1, 3, 64, 64) * 2 - 1
    out = vae(x)
    assert out["recon"].shape == x.shape
    z, _, _ = vae.encode(x)
    assert z.shape[-2:] == (16, 16)
    unet = FireSmokeUNet(pretrained_encoders=False)
    t = torch.tensor([3])
    mask = torch.rand(1, 1, 64, 64)
    masked = torch.rand(1, 3, 64, 64)
    text = torch.randn(1, 64)
    eps = unet(z, t, mask, masked, text)
    assert eps.shape == z.shape


def test_stats_ci_and_holm():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert abs(sample_mean(x) - 3.0) < 1e-8
    assert sample_sd(x) > 0
    lo, hi = ci95(x)
    assert lo < 3 < hi
    adj = holm_bonferroni([0.01, 0.04, 0.03])
    assert adj[0] <= adj[1] or adj[0] <= 0.03
    assert all(0 <= p <= 1 for p in adj)


if __name__ == "__main__":
    tests = [fn for name, fn in list(globals().items()) if name.startswith("test_")]
    for fn in tests:
        fn()
        print("ok", fn.__name__)
    print(f"{len(tests)} tests passed")
