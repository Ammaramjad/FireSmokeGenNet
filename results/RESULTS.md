# FireSmokeGenNet compact reproduction results

This repository implements the paper’s methodology in code and runs it on **public** data. It is a CPU-scale reproduction, not a bit-identical A100 / 512px / 96k-sample replica of the published tables.

## What was implemented (paper → code)

| Paper claim / module | Code | Compact run |
| --- | --- | --- |
| Cosine schedule, Eq. (5) forward process, CFG Eq. (11), DDIM | `models/schedule.py` | T=100, 8 DDIM steps |
| Dual-branch mask/image encoders | `models/encoders.py` | Frozen ResNet-18 / ResNet-18 |
| Joint Cross-Attention Eqs. (22)–(27) | `models/jca.py` | Used at mid-resolution U-Net stages |
| Hierarchical injection (Tables II & IV) | `models/unet.py` | Widths 64–192 |
| MRDL Eqs. (28)–(34), ω=0.4 | `losses/mrdl.py` | Resolution-scaled morphology |
| Q = 0.4 color + 0.4 visibility + 0.2 translucency, keep top 30% | `quality/filter.py` | Heuristic teacher + MLP student (Qwen2-VL-7B does not fit this CPU) |
| Source-level 80/10/10, SHA-256 + perceptual-hash leak control | `data/splits.py` | Camera ID as source key |
| Five-seed paired t, 95% t-interval, Holm, Cohen d_z | `metrics/stats.py` | Same formulas |
| Eight detector families, real vs mixed | `models/detector.py` | Compact single-plume heads, not official YOLO weights |

## Public data

- **Smoke + boxes:** [pyronear/pyro-sdis](https://huggingface.co/datasets/pyronear/pyro-sdis) (Apache-2.0). Role analogue of FLAME / HPWREN / SMOKE5K.
- **Smoke-free backgrounds:** Wikimedia Commons landscape stills (CC), 90 images. Role analogue of the paper’s 8k licensed backgrounds.
- **Masks:** GrabCut inside the YOLO box + largest connected component. Role analogue of SAM + `MaxComponent` + `BBox`.

Split after duplicate control: **train 74 / val 10 / test 23** real smoke images from **28 cameras**. Generation: 90 backgrounds × sampled masks → **96 candidates**, top 30% kept (**29**).

## Results that support the paper’s methodological claims

### 1. Dual-branch generation matches real smoke more closely than compositing

Feature-space distance on a shared 16×16 RGB embedding of real test smoke vs synthetic (lower is better):

| Method | Feature-FD ↓ | MMD² ↓ |
| --- | --- | --- |
| Alpha-blend compositing | 48.55 | 33.28 |
| FireSmokeGenNet | **44.54** | **28.83** |

This is the compact analogue of Table “distribution alignment” (paper CLIP-FD / linear MMD). The dual-branch diffusion samples sit closer to real Pyro-SDIS smoke than training-free compositing.

### 2. Boundary statistics

| Method | Boundary softness S | KL(real ‖ synth) ↓ |
| --- | --- | --- |
| Real smoke | 0.363 | — |
| Alpha-blend | 0.438 | 15.20 |
| FireSmokeGenNet | **0.310** | 15.31 |

Softness is nearer the real reference than compositing (less abrupt edges). KL is comparable at this capacity; the paper’s large gap (0.041 vs 0.187) needs the full 512px model.

### 3. Quality-ranker protocol

Same three axes and weights as Eq. (36). Five-fold student-vs-teacher Spearman ρ on the composite score: **mean ρ ≈ 0.11** (high variance on 80 images). MAE is finite (color ~2.7 / 10). This justifies the *pipeline structure* (annotate → fit ranker → keep top 30%), not the paper’s ρ=0.83 with Qwen2-VL-7B.

### 4. Downstream utility of filtered synthetic images

Official YOLO@640 cannot run here. Image-level smoke vs background classification (frozen ResNet-18, five matched seeds, real test + held-out backgrounds) is the runnable downstream proxy:

| Condition | Five-seed accuracy (%) |
| --- | --- |
| Real only | 93.02 ± 8.37 |
| Real + quality-filtered FireSmokeGenNet | **100.00 ± 0.00** |
| Δ | **+6.98 pp** |

Direction matches the paper’s mixed-data improvement (paper: YOLOv13 AP50 78.16 → 80.88, +2.72 pp). Magnitude and metric differ; the test set is small, so treat this as a directional check, not a replacement for the published detector table.

### 5. Unit tests of the claimed equations

`PYTHONPATH=src python tests/test_methodology.py` checks:

- cosine ᾱ_t is monotone
- CFG formula `ε_u + γ(ε_c − ε_u)`
- MRDL stop-gradient on the perturbed branch and `(1−ω)L_diff + ω L_MRDL`
- quality weights 0.4 / 0.4 / 0.2
- JCA residual shape
- U-Net / VAE forward shapes
- sample mean, SD, t-based 95% CI, Holm–Bonferroni

## Results that do **not** recover the paper’s numbers (and why)

| Paper number | Compact measurement | Why |
| --- | --- | --- |
| Background PSNR 28.10 dB, SSIM 0.88 | Raw decode PSNR **10.08 dB**, SSIM **0.15** | 160 generator steps vs 20k; 64px latent vs 512px SD-2; no pretrained VAE |
| YOLOv6–v13 AP50 ~71–81% | Grid/global compact detectors stay near **0 AP50** | Pyro-SDIS median box area is **0.17%** of the image; at 128px that is ~10 px. IoU 0.5 is unstable. Official YOLO@640 is required |
| 96k candidates / 28.8k retained | 96 / 29 | CPU budget |
| Qwen2-VL-7B LoRA | MLP on heuristic scores | 7B VLM does not fit in 15 GB CPU RAM |

Alpha-blend PSNR is 60 dB only because compositing copies the background exactly; it is not a fair inpainting PSNR.

## How to rerun

```bash
python -m pip install -r requirements.txt
PYTHONPATH=src python tests/test_methodology.py
PYTHONPATH=src python scripts/run_pipeline.py --config configs/compact.yaml
```

Paper-scale hyperparameters (not executed on this host) are in `configs/paper.yaml`.
