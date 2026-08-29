"""Repeated-run statistics: mean, SD, t CI, paired t, Holm, Cohen dz (paper Sec. V-C)."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


def sample_mean(x) -> float:
    return float(np.mean(x))


def sample_sd(x) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return 0.0
    return float(np.std(x, ddof=1))


def ci95(x) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = sample_mean(x)
    sd = sample_sd(x)
    tcrit = stats.t.ppf(0.975, n - 1) if n > 1 else 0.0
    half = tcrit * sd / math.sqrt(max(n, 1))
    return mean - half, mean + half


def paired_ttest(a, b) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = b - a
    shapiro_p = float(stats.shapiro(diff).pvalue) if len(diff) >= 3 else 1.0
    t_res = stats.ttest_rel(b, a)
    dz = float(diff.mean() / (diff.std(ddof=1) + 1e-12))
    return {
        "delta": float(diff.mean()),
        "p": float(t_res.pvalue),
        "shapiro_p": shapiro_p,
        "dz": dz,
        "test": "paired_t",
    }


def holm_bonferroni(pvalues: list[float], alpha: float = 0.05) -> list[float]:
    n = len(pvalues)
    order = np.argsort(pvalues)
    adj = [0.0] * n
    running = 0.0
    for rank, idx in enumerate(order):
        scaled = min(1.0, pvalues[idx] * (n - rank))
        running = max(running, scaled)
        adj[idx] = running
    return adj
