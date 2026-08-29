"""IEEE publication figures from the manuscript-of-record tables (PDF Tables X–XXII).

Reads results/paper/manuscript_record.json only. Does not use compact-run JSON.
Writes PNG (900 dpi) and PDF under paper/figures/results/.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "results" / "paper" / "manuscript_record.json"
OUT_DIR = ROOT / "paper" / "figures" / "results"
TABLE_DIR = ROOT / "results" / "paper" / "tables"
PAPER_TABLE_DIR = ROOT / "paper" / "tables"

C_PROPOSED = "#0072B2"
C_BASE = "#E69F00"
C_REAL = "#009E73"
C_GRAY = "#4D4D4D"
C_ANNOTE = "#222222"
C_GRID = "#D0D0D0"
C_ERR = "#111111"
C_STAR = "#D55E00"

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
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_record() -> dict:
    rec = json.loads(RECORD.read_text())
    y13 = next(r for r in rec["table_x_detector_ap50"] if r["detector"] == "YOLOv13")
    assert abs(y13["mixed_mean"] - 80.88) < 1e-9
    ours = next(r for r in rec["table_xii_background"] if r["method"] == "FireSmokeGenNet")
    assert abs(ours["psnr"] - 28.10) < 1e-9
    assert abs(ours["ssim"] - 0.88) < 1e-9
    return rec


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
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left", clip_on=False)


def err_kw() -> dict:
    return {"ecolor": C_ERR, "elinewidth": 0.75, "capsize": 2.4, "capthick": 0.75, "zorder": 3}


def write_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        return
    for dest in (TABLE_DIR, PAPER_TABLE_DIR):
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / name
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


def emit_tables(rec: dict) -> None:
    write_csv(
        "table_x_detector_ap50.csv",
        [
            {
                "detector": r["detector"],
                "condition": cond,
                "s1": r[cond][0],
                "s2": r[cond][1],
                "s3": r[cond][2],
                "s4": r[cond][3],
                "s5": r[cond][4],
                "mean": r[f"{cond}_mean"] if cond == "real" else r["mixed_mean"] if cond == "mixed" else "",
                "sd": r[f"{cond}_sd"] if cond == "real" else r["mixed_sd"],
                "delta": r["delta"] if cond == "mixed" else "",
                "p_adj": r["p_adj"] if cond == "mixed" else "",
                "dz": r["dz"] if cond == "mixed" else "",
            }
            for r in rec["table_x_detector_ap50"]
            for cond in ("real", "mixed")
        ],
    )
    vlm = rec["table_xi_vlm"]
    write_csv(
        "table_xi_vlm.csv",
        [
            {"dimension": k, "mae": v["mae"], "mae_sd": v["mae_sd"], "rmse": v["rmse"], "rmse_sd": v["rmse_sd"], "rho": v["rho"], "rho_sd": v["rho_sd"]}
            for k, v in vlm.items()
        ],
    )
    write_csv("table_xii_background.csv", rec["table_xii_background"])
    write_csv("table_xiii_distribution.csv", rec["table_xiii_distribution"])
    write_csv("table_xiv_boundary.csv", rec["table_xiv_boundary"])
    write_csv("table_xv_domain.csv", rec["table_xv_domain"])
    write_csv("table_xvi_fixed_seed_ablation.csv", rec["table_xvi_fixed_seed_ablation"])
    write_csv("table_xvii_filter_threshold.csv", rec["table_xvii_filter_threshold"])
    write_csv("table_xviii_equal_budget.csv", rec["table_xviii_equal_budget"])
    write_csv("table_xix_multiseed_ablation.csv", rec["table_xix_multiseed_ablation"])
    write_csv("table_xx_selection.csv", rec["table_xx_selection"])
    write_csv("table_xxi_leakage.csv", rec["table_xxi_leakage"])
    write_csv("table_xxii_compute.csv", rec["table_xxii_compute"])
    write_csv("fig7_mrdl_omega.csv", rec["fig7_mrdl_omega"])


def fig_a_vlm_ranker(rec: dict) -> None:
    v = rec["table_xi_vlm"]
    fig, axes = plt.subplots(1, 2, figsize=(DBL_W, 2.12), layout="constrained", width_ratios=[1.65, 1.0])
    fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.02, wspace=0.06, hspace=0.02)
    ax = axes[0]
    names = ["Color", "Visibility", "Translucency"]
    keys = ["color", "visibility", "translucency"]
    mae = [v[k]["mae"] for k in keys]
    sd = [v[k]["mae_sd"] for k in keys]
    ax.bar(np.arange(3), mae, width=0.58, yerr=sd, error_kw=err_kw(), color=C_PROPOSED, edgecolor="black", linewidth=0.45, zorder=2)
    ax.set_xticks(np.arange(3), names)
    ax.set_ylabel("MAE (0–10)")
    ax.set_ylim(0, 1.05)
    style_ax(ax)
    panel_tag(ax, "a")
    ax = axes[1]
    ax.bar([0], [v["composite"]["rho"]], width=0.42, yerr=[v["composite"]["rho_sd"]], error_kw=err_kw(), color=C_PROPOSED, edgecolor="black", linewidth=0.45, zorder=2)
    ax.set_xlim(-0.42, 0.42)
    ax.set_xticks([0], [r"Composite $Q$"])
    ax.set_ylabel(r"Spearman $\rho$")
    ax.set_ylim(0, 1.05)
    ax.text(0, v["composite"]["rho"] + 0.06, f'{v["composite"]["rho"]:.2f}', ha="center", va="bottom", fontsize=7, color=C_ANNOTE)
    style_ax(ax)
    panel_tag(ax, "b", x=-0.22)
    save_both(fig, "fig_a_vlm_ranker")


def fig_b_image_quality(rec: dict) -> None:
    rows = rec["table_xii_background"]
    methods = [r["method"].replace("FireSmokeGenNet", "Ours") for r in rows]
    psnr = [r["psnr"] for r in rows]
    ssim = [r["ssim"] for r in rows]
    colors = [C_GRAY] * (len(rows) - 1) + [C_PROPOSED]
    fig, axes = plt.subplots(1, 2, figsize=(DBL_W, 2.35), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.02, wspace=0.08, hspace=0.02)
    ax = axes[0]
    bars = ax.bar(np.arange(len(rows)), psnr, width=0.68, color=colors, edgecolor="black", linewidth=0.45, zorder=2)
    ax.set_xticks(np.arange(len(rows)), methods, rotation=28, ha="right")
    ax.set_ylabel("PSNR (dB)")
    ax.set_ylim(24.4, 29.0)
    for b, v in zip(bars, psnr):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:.2f}", ha="center", va="bottom", fontsize=6.2, color=C_ANNOTE)
    style_ax(ax)
    panel_tag(ax, "a", x=-0.14)
    ax = axes[1]
    bars = ax.bar(np.arange(len(rows)), ssim, width=0.68, color=colors, edgecolor="black", linewidth=0.45, zorder=2)
    ax.set_xticks(np.arange(len(rows)), methods, rotation=28, ha="right")
    ax.set_ylabel("SSIM")
    ax.set_ylim(0.75, 0.91)
    for b, v in zip(bars, ssim):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.2f}", ha="center", va="bottom", fontsize=6.2, color=C_ANNOTE)
    style_ax(ax)
    panel_tag(ax, "b", x=-0.14)
    save_both(fig, "fig_b_synthetic_image_quality")


def fig_c_feature_distribution(rec: dict) -> None:
    rows = rec["table_xiii_distribution"]
    fig, axes = plt.subplots(1, 2, figsize=(DBL_W, 2.18), layout="constrained")
    names = [r["method"].replace("FireSmokeGenNet", "Ours") for r in rows]
    colors = [C_GRAY] * (len(rows) - 1) + [C_PROPOSED]
    ax = axes[0]
    vals = [r["clip_fd"] for r in rows]
    bars = ax.bar(np.arange(len(rows)), vals, width=0.68, color=colors, edgecolor="black", linewidth=0.45, zorder=2)
    ax.set_xticks(np.arange(len(rows)), names, rotation=22, ha="right")
    ax.set_ylabel("CLIP-FD (lower is better)")
    ax.set_ylim(0, 52)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.7, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color=C_ANNOTE)
    style_ax(ax)
    panel_tag(ax, "a", x=-0.14)
    ax = axes[1]
    vals = [r["mmd"] for r in rows]
    bars = ax.bar(np.arange(len(rows)), vals, width=0.68, color=colors, edgecolor="black", linewidth=0.45, zorder=2)
    ax.set_xticks(np.arange(len(rows)), names, rotation=22, ha="right")
    ax.set_ylabel(r"MMD$^2_{\mathrm{linear}}$ (lower is better)")
    ax.set_ylim(0, 0.145)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, color=C_ANNOTE)
    style_ax(ax)
    panel_tag(ax, "b", x=-0.14)
    save_both(fig, "fig_c_feature_distribution")


def fig_d_boundary_quality(rec: dict) -> None:
    rows = rec["table_xiv_boundary"]
    fig, axes = plt.subplots(1, 2, figsize=(DBL_W, 2.18), layout="constrained")
    names = [r["method"].replace("FireSmokeGenNet", "Ours") for r in rows]
    colors = [C_REAL, C_GRAY, C_BASE, C_PROPOSED]
    ax = axes[0]
    s = [r["softness"] for r in rows]
    bars = ax.bar(np.arange(len(rows)), s, width=0.62, color=colors, edgecolor="black", linewidth=0.45, zorder=2)
    ax.axhline(rows[0]["softness"], color=C_REAL, linestyle="--", linewidth=0.7, zorder=1)
    ax.set_xticks(np.arange(len(rows)), names, rotation=18, ha="right")
    ax.set_ylabel(r"Boundary softness $\mathcal{S}$")
    ax.set_ylim(0, 0.24)
    for b, v in zip(bars, s):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=7, color=C_ANNOTE)
    style_ax(ax)
    panel_tag(ax, "a", x=-0.14)
    ax = axes[1]
    kl_rows = [r for r in rows if r["kl"] is not None]
    kl_names = [r["method"].replace("FireSmokeGenNet", "Ours") for r in kl_rows]
    kl = [r["kl"] for r in kl_rows]
    kl_colors = [C_GRAY, C_BASE, C_PROPOSED]
    bars = ax.bar(np.arange(len(kl_rows)), kl, width=0.55, color=kl_colors, edgecolor="black", linewidth=0.45, zorder=2)
    ax.set_xticks(np.arange(len(kl_rows)), kl_names, rotation=18, ha="right")
    ax.set_ylabel(r"KL divergence $\downarrow$")
    ax.set_ylim(0, 0.22)
    for b, v in zip(bars, kl):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=7, color=C_ANNOTE)
    style_ax(ax)
    panel_tag(ax, "b", x=-0.14)
    save_both(fig, "fig_d_boundary_quality")


def fig_e_downstream(rec: dict) -> None:
    rows = rec["table_x_detector_ap50"]
    fig, ax = plt.subplots(figsize=(DBL_W, 2.55), layout="constrained")
    x = np.arange(len(rows))
    w = 0.36
    real = [r["real_mean"] for r in rows]
    mixed = [r["mixed_mean"] for r in rows]
    real_sd = [r["real_sd"] for r in rows]
    mixed_sd = [r["mixed_sd"] for r in rows]
    ax.bar(x - w / 2, real, width=w, yerr=real_sd, error_kw=err_kw(), label="Real only", color=C_REAL, edgecolor="black", linewidth=0.45, zorder=2)
    ax.bar(x + w / 2, mixed, width=w, yerr=mixed_sd, error_kw=err_kw(), label="Real + FireSmokeGenNet", color=C_PROPOSED, edgecolor="black", linewidth=0.45, zorder=2)
    ax.set_xticks(x, [r["detector"].replace("YOLO", "") for r in rows])
    ax.set_ylabel(r"AP$_{50}$ (%)")
    ax.set_ylim(68.5, 84.5)
    ax.legend(loc="upper left")
    y13 = next(r for r in rows if r["detector"] == "YOLOv13")
    ax.text(
        0.98,
        0.06,
        rf"YOLOv13: ${y13['real_mean']:.2f}\!\pm\!{y13['real_sd']:.2f}$ → "
        rf"${y13['mixed_mean']:.2f}\!\pm\!{y13['mixed_sd']:.2f}$ "
        rf"($\Delta$ = +{y13['delta']:.2f} pp)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color=C_ANNOTE,
    )
    style_ax(ax)
    save_both(fig, "fig_e_downstream_performance")


def fig7_mrdl(rec: dict) -> None:
    pts = rec["fig7_mrdl_omega"]
    xs = [p["omega"] for p in pts]
    ys = [p["ap50"] for p in pts]
    fig, ax = plt.subplots(figsize=(COL_W, 2.35), layout="constrained")
    ax.plot(xs, ys, color=C_PROPOSED, linewidth=1.4, marker="o", markersize=6, zorder=3)
    ax.plot([0.4], [0.829], marker="*", markersize=12, color=C_STAR, zorder=4)
    ax.axvline(0.4, color=C_STAR, linestyle="--", linewidth=0.7, zorder=1)
    for x, y, note, dy in (
        (0.0, 0.771, "No boundary\nregularization", -0.028),
        (0.4, 0.829, r"Optimal ($\omega=0.4$)", 0.018),
        (1.0, 0.682, "Excessive boundary\nregularization", 0.018),
    ):
        ax.annotate(f"{y:.3f}\n{note}", xy=(x, y), xytext=(x, y + dy), ha="center", va="bottom" if dy > 0 else "top", fontsize=6, color=C_ANNOTE)
    ax.set_xlabel(r"MRDL weight $\omega$")
    ax.set_ylabel(r"AP$_{50}$")
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(0.64, 0.88)
    style_ax(ax)
    save_both(fig, "fig7_mrdl_sensitivity")


def fig8_ablation(rec: dict) -> None:
    rows = list(reversed(rec["table_xix_multiseed_ablation"]))
    fig, ax = plt.subplots(figsize=(DBL_W, 3.15), layout="constrained")
    y = np.arange(len(rows))
    means = [r["ap50"] for r in rows]
    sds = [r["ap50_sd"] for r in rows]
    colors = [C_PROPOSED if r["config"] == "Full FireSmokeGenNet" else C_GRAY for r in rows]
    ax.errorbar(means, y, xerr=sds, fmt="none", ecolor=C_ERR, elinewidth=0.8, capsize=2.2, zorder=2)
    ax.scatter(means, y, s=28, c=colors, edgecolors="black", linewidths=0.45, zorder=3)
    ax.set_yticks(y, [r["config"] for r in rows])
    ax.set_xlabel(r"AP$_{50}$ (%)")
    ax.set_xlim(74.2, 82.4)
    for m, yi in zip(means, y):
        ax.text(m + 0.18, yi, f"{m:.2f}", va="center", fontsize=6.5, color=C_ANNOTE)
    style_ax(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, linestyle=":", linewidth=0.4, color=C_GRID, zorder=0)
    save_both(fig, "fig8_multiseed_ablation")


def fig9_equal_budget(rec: dict) -> None:
    rows = rec["table_xviii_equal_budget"]
    fig, ax = plt.subplots(figsize=(DBL_W, 2.45), layout="constrained")
    labels = [
        "Real only",
        "Real + GAN",
        "Real +\nSD-Inpainting",
        "Real +\nFlameDiffuser",
        "FireSmokeGenNet\n(random)",
        "FireSmokeGenNet\n(VLM-ranked)",
    ]
    colors = [C_GRAY, "#9ECAE1", "#6BAED6", "#4292C6", "#74C476", "#006D2C"]
    means = [r["ap50"] for r in rows]
    sds = [r["ap50_sd"] for r in rows]
    x = np.arange(len(rows))
    bars = ax.bar(x, means, width=0.62, yerr=sds, error_kw=err_kw(), color=colors, edgecolor="black", linewidth=0.45, zorder=2)
    ax.set_xticks(x, labels)
    ax.set_ylabel(r"AP$_{50}$ (%)")
    ax.set_ylim(76.8, 83.2)
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.55, f"{v:.2f}", ha="center", va="bottom", fontsize=7, color=C_ANNOTE)
    style_ax(ax)
    save_both(fig, "fig9_equal_budget")


def main() -> None:
    apply_ieee_style()
    rec = load_record()
    emit_tables(rec)
    fig_a_vlm_ranker(rec)
    fig_b_image_quality(rec)
    fig_c_feature_distribution(rec)
    fig_d_boundary_quality(rec)
    fig_e_downstream(rec)
    fig7_mrdl(rec)
    fig8_ablation(rec)
    fig9_equal_budget(rec)


if __name__ == "__main__":
    main()
