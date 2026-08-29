"""IEEE/JSTARS-quality figures from measured compact-run outputs only.

Writes PNG (900 dpi) and PDF (vector) under paper/figures/results/.
Does not modify manuscript, tables, or LaTeX.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
JSON_DIR = ROOT / "results" / "json"
OUT_DIR = ROOT / "paper" / "figures" / "results"

# Colorblind-safe, shared across all figures.
C_PROPOSED = "#0072B2"
C_BLEND = "#E69F00"
C_REAL = "#009E73"
C_POINT = "#1A1A1A"
C_ERR = "#1A1A1A"
C_ANNOTE = "#333333"

COL_W = 3.50  # IEEE single-column width (in)
DBL_W = 7.16  # IEEE two-column width (in)


def apply_ieee_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Liberation Serif", "DejaVu Serif", "Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "legend.frameon": False,
            "legend.handlelength": 1.2,
            "legend.handletextpad": 0.4,
            "legend.borderaxespad": 0.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_both(fig: plt.Figure, stem: str) -> None:
    from PIL import Image

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=900, format="png", bbox_inches="tight", pad_inches=0.04, facecolor="white")
    fig.savefig(pdf, format="pdf", bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    im = Image.open(png).convert("RGB")
    im.save(png, format="PNG", dpi=(900, 900))
    print(f"wrote {png} and {pdf}")


def style_ax(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", which="both", top=False, right=False)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.45, color="#B0B0B0", zorder=0)
    ax.set_axisbelow(True)


def panel_tag(ax: plt.Axes, letter: str, x: float = -0.16, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="left",
        clip_on=False,
    )


def mean_sd(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(x.std(ddof=1)) if x.size > 1 else (float(x.mean()), 0.0)


def load_json(name: str):
    return json.loads((JSON_DIR / name).read_text())


def fig_a_vlm_ranker() -> None:
    folds = load_json("vlm_validation.json")
    mae = np.array([r["mae"] for r in folds], dtype=float)  # (5, 3)
    rho = np.array([r["rho"] for r in folds], dtype=float)
    mae_mean = mae.mean(axis=0)
    mae_sd = mae.std(axis=0, ddof=1)
    rho_m, rho_s = mean_sd(rho)

    fig, axes = plt.subplots(1, 2, figsize=(DBL_W, 2.55), layout="constrained", width_ratios=[1.55, 1.0])

    ax = axes[0]
    xs = np.arange(3)
    ax.bar(
        xs,
        mae_mean,
        width=0.62,
        color=C_PROPOSED,
        edgecolor="none",
        zorder=2,
        yerr=mae_sd,
        error_kw={"ecolor": C_ERR, "elinewidth": 0.9, "capsize": 3.2, "capthick": 0.9, "zorder": 3},
    )
    rng = np.random.default_rng(0)
    for i in range(3):
        jitter = rng.uniform(-0.08, 0.08, size=mae.shape[0])
        ax.scatter(
            np.full(mae.shape[0], xs[i]) + jitter,
            mae[:, i],
            s=14,
            color=C_POINT,
            zorder=4,
            linewidths=0,
        )
    ax.set_xticks(xs, ["Color", "Visibility", "Translucency"])
    ax.set_ylabel("MAE (0–10 scale)")
    ax.set_xlabel("Quality axis")
    ax.set_ylim(0, max(mae.max(), (mae_mean + mae_sd).max()) * 1.18)
    style_ax(ax)
    panel_tag(ax, "a")

    ax = axes[1]
    ax.bar(
        [0],
        [rho_m],
        width=0.45,
        color=C_PROPOSED,
        edgecolor="none",
        zorder=2,
        yerr=[rho_s],
        error_kw={"ecolor": C_ERR, "elinewidth": 0.9, "capsize": 3.2, "capthick": 0.9, "zorder": 3},
    )
    jitter = rng.uniform(-0.06, 0.06, size=rho.size)
    ax.scatter(jitter, rho, s=16, color=C_POINT, zorder=4, linewidths=0)
    ax.axhline(0.0, color="#888888", linewidth=0.6, linestyle="-", zorder=1)
    ax.set_xlim(-0.55, 0.55)
    ax.set_xticks([0], ["Composite $Q$"])
    ax.set_ylabel(r"Spearman $\rho$")
    ax.set_xlabel("Rank agreement")
    y_lo = min(-0.05, float(rho.min()) - 0.08, rho_m - rho_s - 0.08)
    y_hi = max(0.20, float(rho.max()) + 0.08, rho_m + rho_s + 0.08)
    ax.set_ylim(y_lo, y_hi)
    style_ax(ax)
    panel_tag(ax, "b", x=-0.28)

    save_both(fig, "fig_a_vlm_ranker")


def fig_b_image_quality() -> None:
    g = load_json("generative_metrics.json")
    methods = ["Alpha-blend", "FireSmokeGenNet"]
    psnr = [float(g["blend_psnr"]), float(g["psnr"])]
    ssim = [float(g["blend_ssim"]), float(g["ssim"])]
    colors = [C_BLEND, C_PROPOSED]

    fig, axes = plt.subplots(1, 2, figsize=(DBL_W, 2.50), layout="constrained")

    ax = axes[0]
    bars = ax.bar(methods, psnr, width=0.55, color=colors, edgecolor="none", zorder=2)
    ax.set_ylabel("PSNR (dB)")
    ax.set_ylim(0, max(psnr) * 1.18)
    for b, v in zip(bars, psnr):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + max(psnr) * 0.03,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=C_ANNOTE,
        )
    style_ax(ax)
    panel_tag(ax, "a")

    ax = axes[1]
    bars = ax.bar(methods, ssim, width=0.55, color=colors, edgecolor="none", zorder=2)
    ax.set_ylabel("SSIM")
    ax.set_ylim(0, 1.15)
    for b, v in zip(bars, ssim):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.03,
            f"{v:.2f}" if v >= 0.995 else f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=C_ANNOTE,
        )
    style_ax(ax)
    panel_tag(ax, "b")

    save_both(fig, "fig_b_synthetic_image_quality")


def fig_c_feature_distribution() -> None:
    d = load_json("generative_metrics.json")["distribution"]
    fd = [float(d["compositing"]["clip_fd"]), float(d["firesmokegennet"]["clip_fd"])]
    mmd = [float(d["compositing"]["mmd"]), float(d["firesmokegennet"]["mmd"])]

    fig, ax = plt.subplots(figsize=(COL_W, 2.70), layout="constrained")
    x = np.arange(2)
    w = 0.34
    b1 = ax.bar(x - w / 2, [fd[0], mmd[0]], width=w, color=C_BLEND, edgecolor="none", label="Alpha-blend", zorder=2)
    b2 = ax.bar(x + w / 2, [fd[1], mmd[1]], width=w, color=C_PROPOSED, edgecolor="none", label="FireSmokeGenNet", zorder=2)
    ax.set_xticks(x, ["Feature-FD", r"MMD$^2$"])
    ax.set_ylabel("Distance (lower is better)")
    ymax = max(fd + mmd) * 1.22
    ax.set_ylim(0, ymax)
    for bars in (b1, b2):
        for b in bars:
            v = b.get_height()
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + ymax * 0.02,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=C_ANNOTE,
            )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2, columnspacing=1.1)
    style_ax(ax)
    save_both(fig, "fig_c_feature_distribution")


def fig_d_boundary_quality() -> None:
    b = load_json("generative_metrics.json")["boundary"]
    labels = ["Real smoke", "FireSmokeGenNet", "Alpha-blend"]
    vals = [float(b["real"]), float(b["ours"]), float(b["blend"])]
    colors = [C_REAL, C_PROPOSED, C_BLEND]

    fig, ax = plt.subplots(figsize=(COL_W, 2.70), layout="constrained")
    xs = np.arange(3)
    bars = ax.bar(xs, vals, width=0.58, color=colors, edgecolor="none", zorder=2)
    ax.axhline(vals[0], color=C_REAL, linestyle="--", linewidth=0.8, zorder=1, alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels(["Real smoke", "FireSmokeGenNet", "Alpha-blend"], fontsize=7)
    ax.set_ylabel(r"Boundary softness $S$")
    ax.set_ylim(0, max(vals) * 1.22)
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + max(vals) * 0.025,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=C_ANNOTE,
        )
    ax.text(
        0.5,
        -0.28,
        "Closer to the real-smoke reference indicates better boundary realism.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7,
        color=C_ANNOTE,
    )
    style_ax(ax)
    save_both(fig, "fig_d_boundary_quality")


def fig_e_downstream() -> None:
    raw = load_json("classification.json")
    real = np.asarray(raw["real"], dtype=float) * 100.0
    mixed = np.asarray(raw["mixed"], dtype=float) * 100.0
    real_m, real_s = mean_sd(real)
    mixed_m, mixed_s = mean_sd(mixed)
    delta = float((mixed - real).mean())
    t_res = stats.ttest_rel(mixed, real)
    p = float(t_res.pvalue)
    dz = float((mixed - real).mean() / ((mixed - real).std(ddof=1) + 1e-12))

    fig, ax = plt.subplots(figsize=(COL_W, 2.80), layout="constrained")
    xs = np.array([0.0, 1.0])
    means = [real_m, mixed_m]
    sds = [real_s, mixed_s]
    colors = [C_REAL, C_PROPOSED]
    ax.bar(
        xs,
        means,
        width=0.52,
        color=colors,
        edgecolor="none",
        zorder=2,
        yerr=sds,
        error_kw={"ecolor": C_ERR, "elinewidth": 0.9, "capsize": 3.2, "capthick": 0.9, "zorder": 3},
    )
    offsets = np.linspace(-0.08, 0.08, len(real))
    for i in range(len(real)):
        ax.plot(
            xs + offsets[i],
            [real[i], mixed[i]],
            color="#9A9A9A",
            linewidth=0.7,
            zorder=3,
            alpha=0.85,
        )
        ax.scatter(xs + offsets[i], [real[i], mixed[i]], s=16, color=C_POINT, zorder=4, linewidths=0)

    ax.set_xticks(xs, ["Real only\n(baseline)", "Real + FireSmokeGenNet\n(augmented)"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(70, 114)
    ax.set_xlim(-0.45, 1.45)

    box = (
        f"$\\Delta$ = +{delta:.2f} pp\n"
        f"paired $t$-test $p$ = {p:.3f}\n"
        f"$d_z$ = {dz:.2f}"
    )
    ax.text(
        0.04,
        0.97,
        box,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        color=C_ANNOTE,
        linespacing=1.35,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#CCCCCC", linewidth=0.5),
    )
    style_ax(ax)
    save_both(fig, "fig_e_downstream_performance")


def main() -> None:
    apply_ieee_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_a_vlm_ranker()
    fig_b_image_quality()
    fig_c_feature_distribution()
    fig_d_boundary_quality()
    fig_e_downstream()


if __name__ == "__main__":
    main()
