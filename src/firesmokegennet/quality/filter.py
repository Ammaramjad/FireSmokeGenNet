"""Three-axis smoke quality scoring and ranking (paper Sec. IV-E).

A 7B VLM cannot run on this CPU host, so the same axes, weights, and
ranking protocol are implemented with a transparent heuristic teacher plus
a small MLP student — the same 150-annotation → ranker pattern as the paper.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


QUALITY_WEIGHTS = np.array([0.4, 0.4, 0.2], dtype=np.float64)


def _region_stats(image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    mask = mask.astype(bool)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.sum() < 8:
        return {"color": 0.0, "visibility": 0.0, "translucency": 0.0}
    smoke = image[mask]
    bg = image[~mask] if (~mask).sum() > 8 else image
    mean = smoke.mean(axis=0)
    # Color fidelity: gray/brown/blue smoke, penalize neon hues.
    rg = abs(float(mean[0] - mean[1]))
    gb = abs(float(mean[1] - mean[2]))
    chroma = rg + gb
    brownish = float(mean[0] > mean[2]) * 0.15
    color = float(np.clip(10.0 * (1.0 - chroma / 0.55) + 2.0 * brownish, 0, 10))
    # Visibility: smoke vs background contrast, not fully opaque.
    contrast = float(np.abs(smoke.mean() - bg.mean()))
    opacity = float(np.clip((smoke.std() + 1e-6) / 0.25, 0, 1))
    visibility = float(np.clip(20.0 * contrast * (0.4 + 0.6 * opacity), 0, 10))
    # Translucency: residual background structure through the smoke region.
    if image.shape[-1] == 3:
        gx = np.abs(np.diff(image, axis=1, prepend=image[:, :1])).mean()
    else:
        gx = 0.05
    edge = float(np.abs(np.diff(image * mask[..., None], axis=0)).mean())
    translucency = float(np.clip(10.0 * (0.35 + 4.0 * gx - 2.0 * edge), 0, 10))
    return {"color": color, "visibility": visibility, "translucency": translucency}


def heuristic_scores(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    stats = _region_stats(image, mask)
    return np.array([stats["color"], stats["visibility"], stats["translucency"]], dtype=np.float32)


def composite_quality(scores: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Eq. (36): Q = 0.4 color + 0.4 visibility + 0.2 translucency."""
    weights = QUALITY_WEIGHTS if weights is None else np.asarray(weights)
    return float(np.dot(scores, weights))


class QualityMLP(nn.Module):
    def __init__(self, in_dim: int = 24, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def image_features(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if mask.ndim == 3:
        mask = mask[..., 0]
    smoke = image[mask] if mask.any() else image.reshape(-1, 3)
    bg = image[~mask] if (~mask).any() else image.reshape(-1, 3)
    feats = [
        smoke.mean(axis=0),
        smoke.std(axis=0),
        bg.mean(axis=0),
        bg.std(axis=0),
        np.array([mask.mean(), mask.sum() / mask.size, image.mean(), image.std()]),
        np.percentile(smoke, [25, 50, 75], axis=0).reshape(-1)[:8],
    ]
    vec = np.concatenate([np.ravel(f) for f in feats]).astype(np.float32)
    if vec.size < 24:
        vec = np.pad(vec, (0, 24 - vec.size))
    return vec[:24]


def fit_ranker(
    images: list[np.ndarray],
    masks: list[np.ndarray],
    hidden: int = 64,
    epochs: int = 40,
) -> QualityMLP:
    xs = np.stack([image_features(im, m) for im, m in zip(images, masks)])
    ys = np.stack([heuristic_scores(im, m) for im, m in zip(images, masks)])
    ds = TensorDataset(torch.from_numpy(xs), torch.from_numpy(ys))
    loader = DataLoader(ds, batch_size=16, shuffle=True)
    model = QualityMLP(xs.shape[1], hidden)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    model.eval()
    return model


def rank_and_filter(
    samples: list[dict],
    retain_fraction: float = 0.3,
) -> tuple[list[dict], list[dict]]:
    scored = sorted(samples, key=lambda s: s["quality"], reverse=True)
    k = max(1, int(round(retain_fraction * len(scored))))
    return scored[:k], scored
