# Paper drop-in: Section IV (Results)

Replace the manuscript **Results** section with this fragment. Old table values (YOLOv13 AP50 80.88%, PSNR 28.10 dB, CLIP-FD 28.15, Qwen2-VL \(\rho=0.83\), etc.) are **not** copied. Empty cells are intentional: those experiments were not run on this host.

## Files

| File | Use |
| --- | --- |
| `section_results.tex` | IEEE IEEEtran drop-in (`\input{section_results}` after `\graphicspath{{figures/}}`) |
| `section_results.md` | Same section in Markdown |
| `../results/RESULTS.md` | Canonical Markdown copy next to the CSV/JSON |
| `figures/` | Only figures that correspond to measured tables |

## What is filled vs left blank

| Paper table | Status |
| --- | --- |
| XI VLM five-fold | Filled (heuristic teacher + MLP student) |
| XII background PSNR/SSIM | Filled (FireSmokeGenNet vs alpha-blend only) |
| XIII distribution FD/MMD | Filled (same two methods) |
| XIV boundary softness / KL | Filled |
| X five-seed downstream | Filled as **classification accuracy**, not YOLO AP50 |
| XV domain shift | Empty (registered sources not downloaded) |
| XVI component ablation AP | Empty (compact AP50 degenerate) |
| XVII filter-threshold AP | Sizes filled; AP empty |

## Compile a preview

```bash
cd paper
pdflatex compile_preview.tex
```

Requires `graphicx` and a document class (`IEEEtran` if installed, else `article`).
