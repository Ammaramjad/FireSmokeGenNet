"""Verify GitHub paper-of-record numbers match the IEEE TAI manuscript exactly."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RECORD = ROOT / "results" / "paper" / "manuscript_record.json"


def _rec() -> dict:
    return json.loads(RECORD.read_text())


def test_headline_matches_abstract():
    h = _rec()["headline"]
    assert h["yolov13_real_ap50_pct"] == 78.16
    assert h["yolov13_mixed_ap50_pct"] == 80.88
    assert h["delta_pp"] == 2.72
    assert h["psnr_dB"] == 28.10
    assert h["ssim"] == 0.88
    assert h["mixed_ci95"] == [80.01, 81.75]


def test_table_x_seed_means():
    for row in _rec()["table_x_detector_ap50"]:
        real = np.asarray(row["real"], dtype=float)
        mixed = np.asarray(row["mixed"], dtype=float)
        assert abs(real.mean() - row["real_mean"]) < 5e-3
        assert abs(mixed.mean() - row["mixed_mean"]) < 5e-3
        assert abs((mixed - real).mean() - row["delta"]) < 5e-3


def test_yolov13_seeds():
    y13 = next(r for r in _rec()["table_x_detector_ap50"] if r["detector"] == "YOLOv13")
    assert y13["real"] == [78.0, 78.5, 77.8, 78.3, 78.2]
    assert y13["mixed"] == [80.1, 81.8, 80.5, 80.6, 81.4]
    assert y13["p_adj"] == 0.0021
    assert y13["dz"] == 5.12


def test_vlm_table_xi():
    v = _rec()["table_xi_vlm"]
    assert v["composite"]["rho"] == 0.83
    assert v["color"]["mae"] == 0.62
    assert v["visibility"]["mae"] == 0.57
    assert v["translucency"]["mae"] == 0.71


def test_background_and_distribution():
    ours = next(r for r in _rec()["table_xii_background"] if r["method"] == "FireSmokeGenNet")
    assert ours["psnr"] == 28.10 and ours["ssim"] == 0.88 and ours["lpips1"] == 0.92
    dist = next(r for r in _rec()["table_xiii_distribution"] if r["method"] == "FireSmokeGenNet")
    assert dist["clip_fd"] == 28.15 and dist["mmd"] == 0.062


def test_boundary_and_omega():
    b = {r["method"]: r for r in _rec()["table_xiv_boundary"]}
    assert b["Real smoke"]["softness"] == 0.119
    assert b["FireSmokeGenNet"]["softness"] == 0.124
    assert b["FireSmokeGenNet"]["kl"] == 0.041
    omega = {p["omega"]: p["ap50"] for p in _rec()["fig7_mrdl_omega"]}
    assert omega[0.0] == 0.771 and omega[0.4] == 0.829 and omega[1.0] == 0.682


def test_splits_generation_leakage():
    rec = _rec()
    tr = rec["table_ii_splits"]["training"]
    assert tr["real_total"] == 24000 and tr["synthetic_smoke"] == 28800
    assert rec["table_iv_generation"]["generated_candidate_images"] == 96000
    assert rec["table_iv_generation"]["retained_synthetic_images"] == 28800
    leak = {r["category"]: r for r in rec["table_xxi_leakage"]}
    assert leak["Exact duplicates"]["detected"] == 184
    assert leak["Near-duplicate images"]["detected"] == 627


def test_compact_json_is_not_the_paper_record():
    compact = json.loads((ROOT / "results" / "json" / "generative_metrics.json").read_text())
    # Compact CPU PSNR must not be presented as the manuscript 28.10 dB result.
    assert abs(float(compact["psnr"]) - 28.10) > 1.0


def main() -> None:
    tests = [
        test_headline_matches_abstract,
        test_table_x_seed_means,
        test_yolov13_seeds,
        test_vlm_table_xi,
        test_background_and_distribution,
        test_boundary_and_omega,
        test_splits_generation_leakage,
        test_compact_json_is_not_the_paper_record,
    ]
    for fn in tests:
        fn()
        print("ok", fn.__name__)
    print("all paper-record checks passed")


if __name__ == "__main__":
    main()
