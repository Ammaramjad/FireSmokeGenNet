"""IEEE JSTARS publication figures from measured compact-run JSON only.

Writes PNG (900 dpi) and PDF (vector) under paper/figures/results/.
Does not modify manuscript text, tables, or results/json.
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

# Okabe–Ito (colorblind-safe). Hatches keep the design readable in grayscale.
C_PROPOSED = "#0072B2"
C_BLEND = "#E69F00"
C_REAL = "#009E73"
C_POINT = "#111111"
C_ERR = "#111111"
C_ANNOTE = "#222222"
C_GRID = "#D0D0D0"
C_ZERO = "#666666"

H_PROPOSED = ""
H_BLEND = "///"
H_REAL = "..."

COL_W = 3.45
DBL_W = 7.10


def apply_ieee_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Liberation Serif", "DejaVu Serif", "Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "hatch.linewidth": 0.35,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "legend.frameon": False,
            "legend.handlelength": 1.15,
            "legend.handletextpad": 0.35,
            "legend.borderaxespad": 0.15,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_both(fig: plt.Figure, stem: str) -> None:
    from PIL import Image

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=900, format="png", bbox_inches="tight", pad_inches=0.02, facecolor="white")
    fig.savefig(pdf, format="pdf", bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    im = Image.open(png).convert("RGB")
    im.save(png, format="PNG", dpi=(900, 900))
    print(f"wrote {png} and {pdf}")


def style_ax(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", which="both", top=False, right=False, pad=1.5)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.4, color=C_GRID, zorder=0)
    ax.set_axisbelow(True)


def panel_tag(ax: plt.Axes, letter: str, x: float = -0.12, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="left",
        clip_on=False,
    )


def err_kw() -> dict:
    return {"ecolor": C_ERR, "elinewidth": 0.75, "capsize": 2.4, "capthick": 0.75, "zorder": 3}


def bar_kw(color: str, hatch: str) -> dict:
    return {
        "color": color,
        "edgecolor": "black",
        "linewidth": 0.45,
        "hatch": hatch,
        "zorder": 2,
    }


def mean_sd(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(x.std(ddof=1)) if x.size > 1 else (float(x.mean()), 0.0)


def load_json(name: str):
    return json.loads((JSON_DIR / name).read_text())


def fig_a_vlm_ranker() -> None:
    folds = load_json("vlm_validation.json")
    mae = np.array([r["mae"] for r in folds], dtype=float)
    rho = np.array([r["rho"] for r in folds], dtype=float)
    mae_mean = mae.mean(axis=0)
    mae_sd = mae.std(axis=0, ddof=1)
    rho_m, rho_s = mean_sd(rho)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(DBL_W, 2.12),
        layout="constrained",
        width_ratios=[1.65, 1.0],
    )
    fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.02, wspace=0.06, hspace=0.02)

    ax = axes[0]
    xs = np.arange(3)
    ax.bar(xs, mae_mean, width=0.58, yerr=mae_sd, error_kw=err_kw(), **bar_kw(C_PROPOSED, H_PROPOSED))
    rng = np.random.default_rng(1)
    for i in range(3):
        jitter = rng.uniform(-0.07, 0.07, size=mae.shape[0])
        ax.scatter(
            np.full(mae.shape[0], xs[i]) + jitter,
            mae[:, i],
            s=11,
            facecolors="white",
            edgecolors=C_POINT,
            linewidths=0.55,
            zorder=4,
        )
    ax.set_xticks(xs, ["Color", "Visibility", "Translucency"])
    ax.set_ylabel("MAE (0–10)")
    ax.set_ylim(0, 3.55)
    ax.set_xlim(-0.55, 2.55)
    style_ax(ax)
    panel_tag(ax, "a")

    ax = axes[1]
    ax.bar([0], [rho_m], width=0.42, yerr=[rho_s], error_kw=err_kw(), **bar_kw(C_PROPOSED, H_PROPOSED))
    jitter = np.linspace(-0.09, 0.09, rho.size)
    ax.scatter(
        jitter,
        rho,
        s=13,
        facecolors="white",
        edgecolors=C_POINT,
        linewidths=0.55,
        zorder=4,
    )
    ax.axhline(0.0, color=C_ZERO, linewidth=0.55, linestyle="-", zorder=1)
    ax.set_xlim(-0.42, 0.42)
    ax.set_xticks([0], [r"Composite $Q$"])
    ax.set_ylabel(r"Spearman $\rho$")
    ax.set_ylim(-0.38, 0.58)
    style_ax(ax)
    panel_tag(ax, "b", x=-0.22)

    save_both(fig, "fig_a_vlm_ranker")


def fig_b_image_quality() -> None:
    g = load_json("generative_metrics.json")
    methods = ["Alpha-blend", "FireSmokeGenNet"]
    psnr = [float(g["blend_psnr"]), float(g["psnr"])]
    ssim = [float(g["blend_ssim"]), float(g["ssim"])]
    colors = [C_BLEND, C_PROPOSED]
    hatches = [H_BLEND, H_PROPOSED]

    fig, axes = plt.subplots(1, 2, figsize=(DBL_W, 2.08), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.02, wspace=0.08, hspace=0.02)

    ax = axes[0]
    bars = ax.bar(
        methods,
        psnr,
        width=0.52,
        color=colors,
        edgecolor="black",
        linewidth=0.45,
        zorder=2,
    )
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)
    ax.set_ylabel("PSNR (dB)")
    ax.set_ylim(0, 68)
    for b, v in zip(bars, psnr):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.4, f"{v:.2f}", ha="center", va="bottom", fontsize=7, color=C_ANNOTE)
    style_ax(ax)
    panel_tag(ax, "a", x=-0.14)

    ax = axes[1]
    bars = ax.bar(
        methods,
        ssim,
        width=0.52,
        color=colors,
        edgecolor="black",
        linewidth=0.45,
        zorder=2,
    )
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)
    ax.set_ylabel("SSIM")
    ax.set_ylim(0, 1.12)
    for b, v in zip(bars, ssim):
        label = f"{v:.2f}" if v >= 0.995 else f"{v:.3f}"
        ax.text(b.get_x() + b.get_width() / 2, v + 0.025, label, ha="center", va="bottom", fontsize=7, color=C_ANNOTE)
    style_ax(ax)
    panel_tag(ax, "b", x=-0.14)

    save_both(fig, "fig_b_synthetic_image_quality")


def fig_c_feature_distribution() -> None:
    d = load_json("generative_metrics.json")["distribution"]
    fd = [float(d["compositing"]["clip_fd"]), float(d["firesmokegennet"]["clip_fd"])]
    mmd = [float(d["compositing"]["mmd"]), float(d["firesmokegennet"]["mmd"])]

    fig, ax = plt.subplots(figsize=(COL_W, 2.18), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.02)
    x = np.arange(2)
    w = 0.32
    b1 = ax.bar(x - w / 2, [fd[0], mmd[0]], width=w, label="Alpha-blend", **bar_kw(C_BLEND, H_BLEND))
    b2 = ax.bar(x + w / 2, [fd[1], mmd[1]], width=w, label="FireSmokeGenNet", **bar_kw(C_PROPOSED, H_PROPOSED))
    ax.set_xticks(x, ["Feature-FD", r"MMD$^2$"])
    ax.set_ylabel("Distance (lower is better)")
    ax.set_ylim(0, 56)
    ax.set_xlim(-0.55, 1.55)
    for bars in (b1, b2):
        for b in bars:
            v = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, v + 0.7, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color=C_ANNOTE)
    ax.legend(loc="upper right", frameon=False, borderpad=0.1, labelspacing=0.25, handletextpad=0.35)
    style_ax(ax)
    save_both(fig, "fig_c_feature_distribution")


def fig_d_boundary_quality() -> None:
    b = load_json("generative_metrics.json")["boundary"]
    vals = [float(b["real"]), float(b["ours"]), float(b["blend"])]
    colors = [C_REAL, C_PROPOSED, C_BLEND]
    hatches = [H_REAL, H_PROPOSED, H_BLEND]

    fig, ax = plt.subplots(figsize=(COL_W, 2.18), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.02)
    xs = np.arange(3)
    bars = ax.bar(xs, vals, width=0.55, color=colors, edgecolor="black", linewidth=0.45, zorder=2)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)
    ax.axhline(vals[0], color=C_REAL, linestyle="--", linewidth=0.7, zorder=1)
    ax.set_xticks(xs, ["Real smoke", "FireSmokeGenNet", "Alpha-blend"])
    ax.tick_params(axis="x", labelsize=6.5)
    ax.set_ylabel(r"Boundary softness $S$")
    ax.set_ylim(0, 0.52)
    ax.set_xlim(-0.5, 2.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=7, color=C_ANNOTE)
    ax.text(
        0.02,
        0.97,
        "Closer to real is better",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
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

    fig, ax = plt.subplots(figsize=(COL_W, 2.22), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.02)
    xs = np.array([0.0, 1.0])
    bars = ax.bar(
        xs,
        [real_m, mixed_m],
        width=0.48,
        yerr=[real_s, mixed_s],
        error_kw=err_kw(),
        color=[C_REAL, C_PROPOSED],
        edgecolor="black",
        linewidth=0.45,
        zorder=2,
    )
    for bar, h in zip(bars, [H_REAL, H_PROPOSED]):
        bar.set_hatch(h)

    offsets = np.linspace(-0.07, 0.07, len(real))
    for i in range(len(real)):
        ax.plot(xs + offsets[i], [real[i], mixed[i]], color="#888888", linewidth=0.6, zorder=3, alpha=0.9)
        ax.scatter(
            xs + offsets[i],
            [real[i], mixed[i]],
            s=12,
            facecolors="white",
            edgecolors=C_POINT,
            linewidths=0.55,
            zorder=4,
        )

    ax.set_xticks(xs, ["Real only", "Real + FireSmokeGenNet"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(76, 108)
    ax.set_xlim(-0.42, 1.42)
    ax.text(
        0.03,
        0.97,
        f"$\\Delta$ = +{delta:.2f} pp\n"
        f"paired $t$-test $p$ = {p:.3f}\n"
        f"$d_z$ = {dz:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        color=C_ANNOTE,
        linespacing=1.28,
        bbox=dict(boxstyle="square,pad=0.22", facecolor="white", edgecolor="#BDBDBD", linewidth=0.4),
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
