# FireSmokeGenNet

 **FireSmokeGenNet: Boundary-Aware Diffusion Learning with Multimodal Quality Assessment for Wildfire Smoke Detection**.

##  results (source of truth)

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



## Compact CPU run 

`results/json/` is a **64 px / TinyYOLO / Pyro-SDIS** smoke test. 

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


