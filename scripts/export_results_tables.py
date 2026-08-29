"""Rebuild paper tables and a few figures from the latest JSON measurements."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
JSON = ROOT / "results" / "json"
TABLES = ROOT / "results" / "tables"
FIGS = ROOT / "results" / "figures"
PAPER_FIGS = ROOT / "paper" / "figures"


def mean_sd(x):
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else (float(x.mean()), 0.0)


def ci95(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    m, s = mean_sd(x)
    tcrit = stats.t.ppf(0.975, n - 1) if n > 1 else 0.0
    half = tcrit * s / math.sqrt(max(n, 1))
    return m - half, m + half


def write_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    PAPER_FIGS.mkdir(parents=True, exist_ok=True)

    gen = json.loads((JSON / "generative_metrics.json").read_text())
    summary = json.loads((JSON / "generation_summary.json").read_text())
    cls = json.loads((JSON / "classification.json").read_text())
    vlm = json.loads((JSON / "vlm_validation.json").read_text())
    filt = json.loads((JSON / "filter_threshold.json").read_text())
    splits = json.loads((ROOT / "data" / "splits.json").read_text())

    write_csv(
        TABLES / "dataset_splits.csv",
        ["partition", "n"],
        [
            ["train_real_smoke", splits["train"]],
            ["val_real_smoke", splits["val"]],
            ["test_real_smoke", splits["test"]],
            ["eligible_backgrounds", summary["eligible_backgrounds"]],
            ["available_masks", summary["available_masks"]],
            ["generation_pairs", summary["pairs"]],
            ["candidates", summary["candidates"]],
            ["retained_top30", summary["retained"]],
        ],
    )

    write_csv(
        TABLES / "background_preservation.csv",
        ["method", "psnr_db", "ssim", "one_minus_lpips_proxy", "prompt_sim", "notes"],
        [
            [
                "Alpha-blend compositing",
                f"{gen['blend_psnr']:.2f}",
                f"{gen['blend_ssim']:.3f}",
                "",
                "",
                "Copies the unmasked background; not a fair inpainting PSNR",
            ],
            [
                "FireSmokeGenNet (raw decode)",
                f"{gen['psnr']:.2f}",
                f"{gen['ssim']:.3f}",
                f"{gen['lpips_proxy']:.3f}",
                f"{gen['prompt_sim']:.3f}",
                "PSNR/SSIM on 1-M after 160-step 64px generator; not paper-scale SD-2",
            ],
        ],
    )

    write_csv(
        TABLES / "distribution.csv",
        ["method", "feature_fd", "mmd2"],
        [
            [
                "Alpha-blend compositing",
                f"{gen['distribution']['compositing']['clip_fd']:.2f}",
                f"{gen['distribution']['compositing']['mmd']:.2f}",
            ],
            [
                "FireSmokeGenNet",
                f"{gen['distribution']['firesmokegennet']['clip_fd']:.2f}",
                f"{gen['distribution']['firesmokegennet']['mmd']:.2f}",
            ],
        ],
    )

    write_csv(
        TABLES / "boundary.csv",
        ["method", "boundary_softness_S", "kl_real_synth"],
        [
            ["Real smoke (test)", f"{gen['boundary']['real']:.3f}", ""],
            [
                "Alpha-blend compositing",
                f"{gen['boundary']['blend']:.3f}",
                f"{gen['boundary']['kl_blend']:.2f}",
            ],
            [
                "FireSmokeGenNet",
                f"{gen['boundary']['ours']:.3f}",
                f"{gen['boundary']['kl_ours']:.2f}",
            ],
        ],
    )

    real_pct = np.array(cls["real"]) * 100.0
    mixed_pct = np.array(cls["mixed"]) * 100.0
    real_m, real_s = mean_sd(real_pct)
    mixed_m, mixed_s = mean_sd(mixed_pct)
    real_ci = ci95(real_pct)
    mixed_ci = ci95(mixed_pct)
    diff = mixed_pct - real_pct
    t_res = stats.ttest_rel(mixed_pct, real_pct)
    dz = float(diff.mean() / (diff.std(ddof=1) + 1e-12))
    shapiro_p = float(stats.shapiro(diff).pvalue)

    write_csv(
        TABLES / "classification.csv",
        [
            "condition",
            "s1",
            "s2",
            "s3",
            "s4",
            "s5",
            "mean",
            "sd",
            "ci_lo",
            "ci_hi",
            "delta_pp",
            "test",
            "p",
            "shapiro_p",
            "dz",
        ],
        [
            [
                "Real only",
                *[f"{v:.2f}" for v in real_pct],
                f"{real_m:.2f}",
                f"{real_s:.2f}",
                f"{real_ci[0]:.2f}",
                f"{real_ci[1]:.2f}",
                "",
                "",
                "",
                "",
                "",
            ],
            [
                "Real + quality-filtered FireSmokeGenNet",
                *[f"{v:.2f}" for v in mixed_pct],
                f"{mixed_m:.2f}",
                f"{mixed_s:.2f}",
                f"{mixed_ci[0]:.2f}",
                f"{mixed_ci[1]:.2f}",
                f"{diff.mean():.2f}",
                "paired_t",
                f"{float(t_res.pvalue):.3f}",
                f"{shapiro_p:.3f}",
                f"{dz:.2f}",
            ],
        ],
    )

    mae = np.array([r["mae"] for r in vlm])
    rmse = np.array([r["rmse"] for r in vlm])
    rho = np.array([r["rho"] for r in vlm])
    axes = ["Color fidelity", "Visibility", "Translucency"]
    vlm_rows = []
    for i, name in enumerate(axes):
        m, s = mean_sd(mae[:, i])
        rm, rs = mean_sd(rmse[:, i])
        vlm_rows.append([name, f"{m:.2f}", f"{s:.2f}", f"{rm:.2f}", f"{rs:.2f}", "", ""])
    rm, rs = mean_sd(rho)
    vlm_rows.append(["Composite score (Spearman rho)", "", "", "", "", f"{rm:.2f}", f"{rs:.2f}"])
    write_csv(
        TABLES / "vlm_validation.csv",
        ["quality_dimension", "mae_mean", "mae_sd", "rmse_mean", "rmse_sd", "spearman_mean", "spearman_sd"],
        vlm_rows,
    )

    write_csv(
        TABLES / "filter_threshold.csv",
        ["retention_threshold", "candidate_pool_size", "ap50", "recall", "notes"],
        [
            [
                r["threshold"],
                r["size"],
                "",
                "",
                "AP50 left blank: compact detector AP50 was degenerate on 64px Pyro-SDIS boxes",
            ]
            for r in filt
        ],
    )

    write_csv(
        TABLES / "detector_ap50.csv",
        ["detector", "real_mean", "real_sd", "mixed_mean", "mixed_sd", "delta", "p_adj", "dz", "status"],
        [
            [
                f"yolov{v}",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "Not filled. Compact AP50=0 on all eight family stand-ins (tiny boxes). Paper YOLO@640 numbers are not copied.",
            ]
            for v in range(6, 14)
        ],
    )

    write_csv(
        TABLES / "ablation.csv",
        ["configuration", "ap50", "ap50_95", "precision", "recall", "status"],
        [
            [name, "", "", "", "", "AP50 unfilled (degenerate compact detector); architecture still implemented"]
            for name in [
                "No JCA (CA only)",
                "No MRDL (omega=0)",
                "ResNet without ImageNet pretraining",
                "Unfiltered candidate pool",
                "Full FireSmokeGenNet",
            ]
        ],
    )

    write_csv(
        TABLES / "domain_shift.csv",
        ["shift", "real_ap50", "real_recall", "mixed_ap50", "mixed_recall", "status"],
        [
            [
                name,
                "",
                "",
                "",
                "",
                "Not executed: FLAME2 / HPWREN / SMOKE5K registered sources were not downloaded in this public-data run",
            ]
            for name in ["Summer to Winter", "Day to Dusk", "Clear to Haze/Fog"]
        ],
    )

    # Boundary figure from measured JSON
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    labels = ["Real smoke", "FireSmokeGenNet", "Alpha-blend"]
    vals = [gen["boundary"]["real"], gen["boundary"]["ours"], gen["boundary"]["blend"]]
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    ax.bar(labels, vals, color=colors, width=0.62)
    ax.set_ylabel("Boundary softness $S$")
    ax.set_title("Boundary-gradient magnitude in the $\\delta$-band")
    ax.set_ylim(0, max(vals) * 1.25)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "boundary.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    names = ["Color", "Visibility", "Translucency"]
    means = mae.mean(0)
    sds = mae.std(0, ddof=1)
    ax.bar(names, means, yerr=sds, capsize=4, color="#4c78a8", width=0.55)
    ax.set_ylabel("MAE vs heuristic teacher (0–10)")
    ax.set_title("Five-fold student ranker vs three-axis teacher")
    ax.set_ylim(0, max(means + sds) * 1.35)
    fig.tight_layout()
    fig.savefig(FIGS / "vlm_validation.png", dpi=160)
    plt.close(fig)

    for name in FIGS.glob("*.png"):
        dest = PAPER_FIGS / name.name
        dest.write_bytes(name.read_bytes())

    print("Wrote tables under", TABLES)
    print("Wrote extra figures under", FIGS)


if __name__ == "__main__":
    main()
