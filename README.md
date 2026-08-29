# FireSmokeGenNet

IEEE TAI manuscript: **FireSmokeGenNet: Boundary-Aware Diffusion Learning with Multimodal Quality Assessment for Wildfire Smoke Detection**.

This repository now stores the **paper-of-record results and figures** from the submitted manuscript, plus a CPU-scale methodology implementation.

## Paper results (source of truth)

These numbers are transcribed from the compiled manuscript PDF (`paper/FireSmokeGenNet_TAI.pdf`). They are the values that must appear in any publication figure or table on GitHub.

| Claim | Manuscript value |
| --- | --- |
| YOLOv13 AP$_{50}$ real only | $78.16\pm0.27\%$ |
| YOLOv13 AP$_{50}$ mixed data | $80.88\pm0.70\%$ |
| Absolute gain | $+2.72$ pp |
| Mixed-data 95% CI | $[80.01, 81.75]\%$ |
| Background PSNR / SSIM | $28.10$ dB / $0.88$ |
| CLIP-FD / linear MMD² | $28.15$ / $0.062$ |
| Boundary softness (real / ours) | $0.119$ / $0.124$ |
| VLM composite Spearman $\rho$ | $0.83\pm0.04$ |
| Train / val / test real images | $24{,}000$ / $3{,}000$ / $3{,}000$ |
| Retained synthetic images | $28{,}800$ of $96{,}000$ (top 30%) |

Machine-readable tables: [`results/paper/manuscript_record.json`](results/paper/manuscript_record.json) and [`results/paper/tables/`](results/paper/tables/).
Validation: `PYTHONPATH=src python tests/test_paper_record.py`.

### Figures matching the PDF

Manuscript figures extracted from the PDF (Figs. 1–9), using the same `\includegraphics` names as the IEEEtran source:

| PDF figure | File |
| --- | --- |
| Fig. 1 Forward/reverse diffusion | `paper/figures/fig44.png` |
| Fig. 2 Architecture | `paper/figures/fig11.png` |
| Fig. 3 Dual-branch encoder | `paper/figures/fig22.png` |
| Fig. 4 JCA | `paper/figures/fig33.png` |
| Fig. 5 MRDL | `paper/figures/fig21.png` |
| Fig. 6 Qualitative comparison | `paper/figures/fig_qualitative_comparison.png` |
| Fig. 7 MRDL $\omega$ sensitivity | `paper/figures/fig6.png` |
| Fig. 8 Multi-seed ablation | `paper/figures/fig_multiseed_ablation_900dpi.png` |
| Fig. 9 Equal-budget comparison | `paper/figures/fig_equal_budget_900dpi.png` |

Vector redraws of the result plots from the same table values: `paper/figures/results/` (Figs. A–E, 7–9). Regenerated with:

```bash
python scripts/generate_publication_figures.py
```

## Compact CPU run (not the paper)

`results/json/` is a **64 px / TinyYOLO / Pyro-SDIS** smoke test. Its PSNR (~10 dB) and zero AP$_{50}$ are **not** manuscript results. See [`results/RESULTS.md`](results/RESULTS.md).

## Methodology code

| Paper component | Location |
| --- | --- |
| Cosine schedule, CFG, DDIM | `src/firesmokegennet/models/schedule.py` |
| Dual-branch ResNet-18 / ResNet-50 | `src/firesmokegennet/models/encoders.py` |
| Joint Cross-Attention | `src/firesmokegennet/models/jca.py` |
| Hierarchical U-Net injection | `src/firesmokegennet/models/unet.py` |
| MRDL, $k\sim\mathcal{U}(10,20)$, $\omega=0.4$ | `src/firesmokegennet/losses/mrdl.py` |
| $Q=0.4c+0.4v+0.2t$, top 30% | `src/firesmokegennet/quality/filter.py` |
| Paper hyperparameters (A100 / 512 px) | `configs/paper.yaml` |
| CPU compact hyperparameters | `configs/compact.yaml` |

Full-scale training in `configs/paper.yaml` needs an 80 GB A100, official YOLO v6–v13 at $640\times640$, Qwen2-VL-7B, and the FLAME / HPWREN / SMOKE5K + licensed background collection described in the manuscript. This cloud host cannot re-run that experiment.
