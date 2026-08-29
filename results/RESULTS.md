# FireSmokeGenNet results

## Paper-of-record (use these)

Transcribed from the submitted IEEE TAI manuscript. Authoritative file: [`results/paper/manuscript_record.json`](paper/manuscript_record.json).

Headline (YOLOv13, five matched seeds):

- Real only: **78.16 ± 0.27%** AP$_{50}$
- Real + quality-filtered FireSmokeGenNet: **80.88 ± 0.70%** AP$_{50}$
- Absolute improvement: **+2.72** percentage points
- Mixed-data 95% CI: **[80.01, 81.75]%**
- Background preservation: **PSNR 28.10 dB**, **SSIM 0.88**

CSV copies of Tables X–XXII: [`results/paper/tables/`](paper/tables/).

## Compact CPU run (do not cite as the paper)

The measurements below are from the public-data 64 px pipeline. They **do not** replace the manuscript tables.

| Item | Paper | Compact run |
| --- | --- | --- |
| Background PSNR / SSIM | 28.10 dB / 0.88 | ~10 dB / 0.15 |
| YOLOv13 AP$_{50}$ | 78.16 → 80.88 | TinyYOLO AP$_{50}$ ≈ 0 (boxes too small at 128 px) |
| CLIP-FD | 28.15 | Feature-FD 44.54 (different encoder) |
| Candidates / retained | 96,000 / 28,800 | 96 / 29 |

How to rerun the compact smoke test:

```bash
PYTHONPATH=src python tests/test_methodology.py
PYTHONPATH=src python tests/test_paper_record.py
PYTHONPATH=src python scripts/run_pipeline.py --config configs/compact.yaml
```

Paper-scale hyperparameters (not executed on this host): `configs/paper.yaml`.
