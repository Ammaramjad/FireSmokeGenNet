# Paper vs GitHub alignment

Compiled manuscript: `paper/FireSmokeGenNet_TAI.pdf` (20 pages).

## What now matches 100%

| Manuscript item | GitHub location |
| --- | --- |
| Abstract headline numbers (78.16 → 80.88, PSNR 28.10, SSIM 0.88) | `results/paper/manuscript_record.json` → `headline` |
| Table II splits | `results/paper/tables/` + JSON `table_ii_splits` |
| Table III masks | JSON `table_iii_masks` |
| Table IV generation | JSON `table_iv_generation` |
| Table X eight-detector AP50 | CSV + Fig. E |
| Table XI VLM | CSV + Fig. A |
| Table XII background PSNR/SSIM | CSV + Fig. B |
| Table XIII CLIP-FD / MMD | CSV + Fig. C |
| Table XIV boundary | CSV + Fig. D |
| Table XV domain shift | CSV |
| Table XVI–XVII fixed-seed sweeps | CSV |
| Table XVIII–XXII | `paper/additional_validation_tables.tex` + CSV |
| Fig. 1–9 raster assets | `paper/figures/fig*.png` (extracted from the PDF) |
| Fig. 7 $\omega\in\{0,0.4,1.0\}$ = 0.771 / 0.829 / 0.682 | `fig7_mrdl_omega` |
| Quality weights 0.4 / 0.4 / 0.2, ICC 0.82 | JSON `quality` |
| Generator / detector / inference hyperparams | `configs/paper.yaml` |

`tests/test_paper_record.py` fails if any of those headline values drift.

## What this host cannot reproduce

The manuscript experiments used an NVIDIA A100 (80 GB), 512×512 SD-2 inpainting U-Net, 20,000 generator steps, 50 DDIM steps, Qwen2-VL-7B LoRA, and official YOLO v6–v13 at 640×640 on 24,000 real + 28,800 synthetic images.

The files under `results/json/` are a separate compact CPU run (64 px, 160 generator steps, TinyYOLO). **Do not substitute those measurements for Tables X–XXII.**
