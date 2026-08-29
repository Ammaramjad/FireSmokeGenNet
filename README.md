# FireSmokeGenNet Reproduction

Code implementation of **FireSmokeGenNet: Boundary-Aware Diffusion Learning with Multimodal Quality Assessment for Wildfire Smoke Detection**. The modules follow the paper’s equations, architecture tables, losses, filtering rule, split protocol, and evaluation metrics.

This repository is a **methodology-faithful, public-data reproduction**. The compact config runs on CPU. It is not a claim that 512px A100 numbers from the paper are reproduced bit-for-bit.

## Paper → code map

| Paper component | Location |
| --- | --- |
| Cosine schedule, forward process, CFG, DDIM | `src/firesmokegennet/models/schedule.py` |
| Dual-branch ResNet-18 / ResNet-50 encoder | `src/firesmokegennet/models/encoders.py` |
| Joint Cross-Attention (JCA) | `src/firesmokegennet/models/jca.py` |
| Hierarchical U-Net injection (Tables II / IV) | `src/firesmokegennet/models/unet.py` |
| Mask Random Difference Loss (MRDL) | `src/firesmokegennet/losses/mrdl.py` |
| Quality score \(Q=0.4c+0.4v+0.2t\), top-30% keep | `src/firesmokegennet/quality/filter.py` |
| Source-level 80/10/10 split, SHA-256 + pHash | `src/firesmokegennet/data/splits.py` |
| PSNR/SSIM, feature FD/MMD, boundary KL, AP50 | `src/firesmokegennet/metrics/` |
| Five-seed paired \(t\), Holm–Bonferroni, \(d_z\) | `src/firesmokegennet/metrics/stats.py` |
| Eight detector-family protocol | `src/firesmokegennet/models/detector.py` |
| Paper hyperparameters | `configs/paper.yaml` |
| CPU compact hyperparameters | `configs/compact.yaml` |

## Public datasets

The paper’s FLAME / HPWREN / SMOKE5K mix is not fully redistributable here without IEEE/HPWREN registration. The compact pipeline uses **public substitutes with the same roles**:

- **Smoke images + boxes:** [Pyro-SDIS](https://huggingface.co/datasets/pyronear/pyro-sdis) (Apache-2.0, French SDIS wildfire-camera smoke, YOLO labels). Camera IDs are the source-level split keys (paper: scene/video IDs).
- **Smoke-free backgrounds:** Wikimedia Commons landscape stills (CC licenses via Commons API), with a procedural forest-ridge fallback if the API is blocked.
- **Masks:** GrabCut inside the YOLO box, then largest connected component — the paper’s SAM + `MaxComponent` + `BBox` pipeline, with a public classic-CV substitute for SAM.

## Run

```bash
python -m pip install -r requirements.txt
PYTHONPATH=src python tests/test_methodology.py
PYTHONPATH=src python scripts/run_pipeline.py --config configs/compact.yaml
```

Results are written to `results/tables/`, `results/figures/`, `results/json/`, and `results/RESULTS.md`.

Full-scale paper settings live in `configs/paper.yaml` (512px, SD-2-sized U-Net, 20k generator steps, Qwen2-VL-7B, official YOLO families). That configuration needs an 80 GB A100-class GPU.

## Results

The manuscript **Results** section (IEEE Section IV) is rewritten from the compact run in [`results/RESULTS.md`](results/RESULTS.md) and [`paper/section_results.tex`](paper/section_results.tex). Previous draft numbers are not copied. Tables that could not be measured (official YOLO AP50, registered domain-shift splits, ablation AP) are left blank.

CSV sources: `results/tables/`. Figures: `results/figures/` and `paper/figures/`. JSON: `results/json/`. Rebuild tables with `python scripts/export_results_tables.py`.

| Item | Paper-scale config | Compact run (this host) |
| --- | --- | --- |
| Image / latent | 512 / 64 | 64 / 16 |
| U-Net widths | 320-1280 | 64-192 |
| Image encoder | ResNet-50 | ResNet-18 |
| Diffusion steps / DDIM | 1000 / 50 | 100 / 8 |
| Generator iters | 20 000 | 160 |
| Candidates / retained | 96 000 / 28 800 | 96 / 29 |
| Detectors | YOLO v6–v13 @ 640 | classification proxy (YOLO AP50 unfilled) |
| Quality ranker | LoRA Qwen2-VL-7B | heuristic teacher + MLP student, same axes/weights |
