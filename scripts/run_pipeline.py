"""End-to-end compact reproduction of the FireSmokeGenNet experimental protocol."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from firesmokegennet.data.dataset import GeneratorDataset
from firesmokegennet.data.download import (
    download_pyrosdis_subset,
    download_wikimedia_backgrounds,
    yolo_to_xyxy,
)
from firesmokegennet.data.masks import grabcut_mask, mask_to_bbox
from firesmokegennet.data.prompts import PROMPT_TEMPLATES, tokenize_prompt
from firesmokegennet.data.splits import source_level_split
from firesmokegennet.generate.sample import alpha_blend_baseline, generate_image
from firesmokegennet.metrics.core import (
    average_precision,
    boundary_softness,
    frechet_distance,
    gradient_kl,
    linear_mmd,
    lpips_proxy,
    prompt_sim,
    psnr_background,
    ssim_background,
)
from firesmokegennet.metrics.stats import ci95, holm_bonferroni, paired_ttest, sample_mean, sample_sd
from firesmokegennet.models.detector import FAMILY_SPEC
from firesmokegennet.quality.filter import (
    composite_quality,
    fit_ranker,
    heuristic_scores,
    image_features,
    rank_and_filter,
)
from firesmokegennet.train.train_detector import train_one_detector
from firesmokegennet.train.train_generator import train_generator, train_vae
from firesmokegennet.utils.config import load_config, seed_everything
from firesmokegennet.utils.io import ensure_dir, save_json


def to_numpy(path: str, size: int) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize((size, size))).astype(np.float32) / 255.0


def load_mask(path: Path, size: int) -> np.ndarray:
    return (np.asarray(Image.open(path).convert("L").resize((size, size), Image.NEAREST)) > 127).astype(np.float32)


def build_detector_items(records: list[dict], size: int, max_n: int | None = None) -> list[dict]:
    items = []
    for rec in records:
        if max_n and len(items) >= max_n:
            break
        img = Image.open(rec["path"]).convert("RGB")
        w, h = img.size
        boxes = yolo_to_xyxy(rec.get("annotations") or "", w, h)
        scaled = []
        for x1, y1, x2, y2 in boxes:
            scaled.append(
                [
                    x1 / w * size,
                    y1 / h * size,
                    x2 / w * size,
                    y2 / h * size,
                ]
            )
        items.append(
            {
                "path": rec["path"],
                "boxes": scaled,
                "season": rec.get("season"),
                "illumination": rec.get("illumination"),
                "weather": rec.get("weather"),
                "synthetic": False,
            }
        )
    return items


def feature_embed(images: list[np.ndarray]) -> np.ndarray:
    feats = []
    for im in images:
        small = np.array(Image.fromarray((im * 255).astype(np.uint8)).resize((16, 16))).astype(np.float32) / 255.0
        feats.append(small.reshape(-1))
    return np.stack(feats)


def prepare_data(cfg, data_root: Path):
    smoke_dir = data_root / "pyrosdis"
    bg_dir = data_root / "backgrounds"
    mask_dir = data_root / "masks"
    ensure_dir(mask_dir)
    if not (smoke_dir / "manifest.json").exists():
        smoke_recs = download_pyrosdis_subset(smoke_dir, cfg["data"]["max_smoke_images"], cfg["seed"])
    else:
        from firesmokegennet.utils.io import load_json

        smoke_recs = load_json(smoke_dir / "manifest.json")
    if not (bg_dir / "manifest.json").exists():
        bg_recs = download_wikimedia_backgrounds(bg_dir, cfg["data"]["max_backgrounds"])
    else:
        from firesmokegennet.utils.io import load_json

        bg_recs = load_json(bg_dir / "manifest.json")
        if len(bg_recs) < 20:
            bg_recs = download_wikimedia_backgrounds(bg_dir, cfg["data"]["max_backgrounds"])
    splits = source_level_split(smoke_recs, seed=cfg["seed"])
    save_json(data_root / "splits.json", {k: len(v) for k, v in splits.items()})
    return splits, bg_recs, mask_dir


def extract_masks(records: list[dict], mask_dir: Path, limit: int) -> list[dict]:
    out = []
    for rec in tqdm(records[:limit], desc="masks"):
        dest = mask_dir / (Path(rec["path"]).stem + ".png")
        if not dest.exists():
            img = Image.open(rec["path"]).convert("RGB")
            boxes = yolo_to_xyxy(rec.get("annotations") or "", img.size[0], img.size[1])
            mask = grabcut_mask(img, boxes)
            Image.fromarray((mask * 255).astype(np.uint8)).save(dest)
        out.append({**rec, "mask_path": str(dest)})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/compact.yaml"))
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--out", default=str(ROOT / "outputs"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device") != "cpu" else "cpu")
    data_root = Path(args.data_root)
    out = Path(args.out)
    fig_dir = ROOT / "results" / "figures"
    tab_dir = ROOT / "results" / "tables"
    json_dir = ROOT / "results" / "json"
    ensure_dir(fig_dir)
    ensure_dir(tab_dir)
    ensure_dir(json_dir)

    print(f"device={device}")
    splits, bg_recs, mask_dir = prepare_data(cfg, data_root)
    size = cfg["image_size"]
    train_smoke = extract_masks(splits["train"], mask_dir, cfg["data"]["max_masks"])
    val_smoke = extract_masks(splits["val"], mask_dir, max(20, cfg["data"]["detector_smoke"] // 6))
    test_smoke = extract_masks(splits["test"], mask_dir, max(30, cfg["data"]["detector_smoke"] // 4))

    gen_ds = GeneratorDataset(train_smoke, size, cache_masks=mask_dir)
    vae = train_vae(gen_ds, cfg, device, out / "ckpts")
    unet = train_generator(gen_ds, vae, cfg, device, out / "ckpts", tag="full")
    unet_nojca = train_generator(
        gen_ds,
        vae,
        cfg,
        device,
        out / "ckpts",
        use_jca=False,
        iterations=cfg["train_generator"]["ablation_iterations"],
        tag="no_jca",
    )
    unet_nomrdl = train_generator(
        gen_ds,
        vae,
        cfg,
        device,
        out / "ckpts",
        omega=0.0,
        iterations=cfg["train_generator"]["ablation_iterations"],
        tag="no_mrdl",
    )
    unet_randenc = train_generator(
        gen_ds,
        vae,
        cfg,
        device,
        out / "ckpts",
        pretrained_encoders=False,
        iterations=cfg["train_generator"]["ablation_iterations"],
        tag="rand_encoder",
    )
    unet_omega1 = train_generator(
        gen_ds,
        vae,
        cfg,
        device,
        out / "ckpts",
        omega=1.0,
        iterations=cfg["train_generator"]["ablation_iterations"],
        tag="omega1",
    )

    # Generation on held-out backgrounds + train masks.
    rng = random.Random(cfg["seed"])
    pairs = []
    masks_pool = train_smoke[:]
    bgs = bg_recs[:]
    rng.shuffle(masks_pool)
    rng.shuffle(bgs)
    n_pairs = min(cfg["data"]["pairs"], len(bgs), len(masks_pool))
    for i in range(n_pairs):
        pairs.append((bgs[i % len(bgs)], masks_pool[i % len(masks_pool)]))

    synth_dir = ensure_dir(out / "synthetic")
    candidates = []
    for pi, (bg, mrec) in enumerate(tqdm(pairs, desc="generate")):
        bg_np = to_numpy(bg["path"], size)
        mask = load_mask(Path(mrec["mask_path"]), size)
        masked = bg_np * (1.0 - mask[..., None])
        mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
        masked_t = torch.from_numpy(masked).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
        for v in range(cfg["data"]["candidates_per_pair"]):
            seed = cfg["seed"] + pi * 10 + v
            prompt = PROMPT_TEMPLATES[seed % len(PROMPT_TEMPLATES)]
            text = torch.tensor(tokenize_prompt(prompt)).unsqueeze(0).float()
            img = generate_image(unet, vae, mask_t, masked_t, text, cfg, device, seed=seed)
            blend = alpha_blend_baseline(bg_np, mask, seed=seed)
            scores = heuristic_scores(img, mask)
            q = composite_quality(scores)
            name = f"pair{pi:03d}_v{v}.png"
            Image.fromarray((img * 255).astype(np.uint8)).save(synth_dir / name)
            bbox = mask_to_bbox((mask * 255).astype(np.uint8))
            candidates.append(
                {
                    "path": str(synth_dir / name),
                    "quality": q,
                    "scores": scores.tolist(),
                    "mask": mask,
                    "bg": bg_np,
                    "image": img,
                    "blend": blend,
                    "prompt": prompt,
                    "boxes": [[float(b) for b in bbox]],
                    "synthetic": True,
                }
            )

    # Fit ranker on a 80-image annotation proxy, then re-score.
    n_ann = min(cfg["quality"]["annotation_size"], len(candidates))
    ranker = fit_ranker(
        [c["image"] for c in candidates[:n_ann]],
        [c["mask"] for c in candidates[:n_ann]],
        hidden=cfg["quality"]["mlp_hidden"],
        epochs=cfg["quality"]["mlp_epochs"],
    )
    with torch.no_grad():
        for c in candidates:
            feat = torch.from_numpy(image_features(c["image"], c["mask"])).unsqueeze(0)
            pred = ranker(feat).numpy()[0]
            c["vlm_scores"] = pred.tolist()
            c["quality"] = composite_quality(pred)

    retained, ranked = rank_and_filter(candidates, cfg["quality"]["retain_fraction"])
    save_json(json_dir / "generation_summary.json", {
        "eligible_backgrounds": len(bgs),
        "available_masks": len(masks_pool),
        "pairs": n_pairs,
        "candidates": len(candidates),
        "retained": len(retained),
        "retain_fraction": cfg["quality"]["retain_fraction"],
    })

    # Generative metrics on retained samples.
    psnr_s, ssim_s, lpips_s, clip_s = [], [], [], []
    for c in retained:
        preserve = 1.0 - c["mask"]
        psnr_s.append(psnr_background(c["image"], c["bg"], preserve))
        ssim_s.append(ssim_background(c["image"], c["bg"], preserve))
        lpips_s.append(lpips_proxy(c["image"], c["bg"], preserve))
        clip_s.append(prompt_sim(c["image"], c["mask"]))
    blend_psnr, blend_ssim = [], []
    for c in retained:
        preserve = 1.0 - c["mask"]
        blend_psnr.append(psnr_background(c["blend"], c["bg"], preserve))
        blend_ssim.append(ssim_background(c["blend"], c["bg"], preserve))

    real_imgs = [to_numpy(r["path"], size) for r in test_smoke[: min(40, len(test_smoke))]]
    real_masks = [load_mask(Path(r["mask_path"]), size) for r in test_smoke[: min(40, len(test_smoke))]]
    synth_imgs = [c["image"] for c in retained[:40]]
    synth_masks = [c["mask"] for c in retained[:40]]
    real_f = feature_embed(real_imgs)
    synth_f = feature_embed(synth_imgs)
    blend_f = feature_embed([c["blend"] for c in retained[:40]])
    dist_table = {
        "compositing": {"clip_fd": frechet_distance(real_f, blend_f), "mmd": linear_mmd(real_f, blend_f)},
        "firesmokegennet": {"clip_fd": frechet_distance(real_f, synth_f), "mmd": linear_mmd(real_f, synth_f)},
    }
    real_soft = float(np.mean([boundary_softness(i, m) for i, m in zip(real_imgs, real_masks)]))
    ours_soft = float(np.mean([boundary_softness(i, m) for i, m in zip(synth_imgs, synth_masks)]))
    blend_soft = float(np.mean([boundary_softness(c["blend"], c["mask"]) for c in retained[:40]]))
    kl_ours = gradient_kl(real_imgs, real_masks, synth_imgs, synth_masks)
    kl_blend = gradient_kl(real_imgs, real_masks, [c["blend"] for c in retained[:40]], synth_masks)

    gen_metrics = {
        "psnr": float(np.mean(psnr_s)),
        "ssim": float(np.mean(ssim_s)),
        "lpips_proxy": float(np.mean(lpips_s)),
        "prompt_sim": float(np.mean(clip_s)),
        "blend_psnr": float(np.mean(blend_psnr)),
        "blend_ssim": float(np.mean(blend_ssim)),
        "boundary": {"real": real_soft, "ours": ours_soft, "blend": blend_soft, "kl_ours": kl_ours, "kl_blend": kl_blend},
        "distribution": dist_table,
    }
    save_json(json_dir / "generative_metrics.json", gen_metrics)

    # Detector protocol.
    det_size = cfg["detector"]["input_size"]
    real_train = build_detector_items(splits["train"], det_size, cfg["data"]["detector_smoke"])
    # Balance with smoke-free Wikimedia images.
    nonsmoke = []
    for rec in bg_recs[: cfg["data"]["detector_nonsmoke"]]:
        nonsmoke.append({"path": rec["path"], "boxes": [], "synthetic": False})
    val_items = build_detector_items(splits["val"], det_size) + [
        {"path": r["path"], "boxes": []} for r in bg_recs[-max(15, len(splits["val"])) :]
    ]
    test_items = build_detector_items(splits["test"], det_size) + [
        {"path": r["path"], "boxes": []} for r in bg_recs[: max(20, len(splits["test"]))]
    ]
    mixed_extra = []
    for c in retained:
        mixed_extra.append({"path": c["path"], "boxes": c["boxes"], "synthetic": True})

    families = cfg["detector"]["families"]
    seeds = cfg["detector"]["seeds"]
    det_results = {}
    for family in families:
        det_results[family] = {"real": [], "mixed": []}
        for seed in seeds:
            real_metrics = train_one_detector(family, real_train + nonsmoke, test_items, cfg, device, seed)
            mixed_metrics = train_one_detector(
                family, real_train + nonsmoke + mixed_extra, test_items, cfg, device, seed
            )
            det_results[family]["real"].append(real_metrics)
            det_results[family]["mixed"].append(mixed_metrics)
            print(family, seed, real_metrics["ap50"], mixed_metrics["ap50"])

    table_rows = []
    pvals = []
    for family in families:
        real_ap = [m["ap50"] for m in det_results[family]["real"]]
        mix_ap = [m["ap50"] for m in det_results[family]["mixed"]]
        stats = paired_ttest(real_ap, mix_ap)
        pvals.append(stats["p"])
        table_rows.append(
            {
                "detector": family,
                "real_seeds": real_ap,
                "mixed_seeds": mix_ap,
                "real_mean": sample_mean(real_ap),
                "real_sd": sample_sd(real_ap),
                "real_ci": ci95(real_ap),
                "mixed_mean": sample_mean(mix_ap),
                "mixed_sd": sample_sd(mix_ap),
                "mixed_ci": ci95(mix_ap),
                "delta": stats["delta"],
                "p": stats["p"],
                "dz": stats["dz"],
            }
        )
    adj = holm_bonferroni(pvals)
    for row, p_adj in zip(table_rows, adj):
        row["p_adj"] = p_adj
    save_json(json_dir / "detector_results.json", table_rows)

    # Domain-shift using date tags on real images only vs mixed.
    def subset(items, key, train_val, test_val):
        tr = [i for i in items if i.get(key) == train_val]
        te = [i for i in test_items if i.get(key) == test_val]
        return tr, te

    domain = {}
    for name, key, a, b in [
        ("summer_to_winter", "season", "summer", "winter"),
        ("day_to_dusk", "illumination", "day", "dusk"),
        ("clear_to_haze", "weather", "clear", "haze"),
    ]:
        tr, te = subset(real_train, key, a, b)
        if len(tr) < 8 or len(te) < 5:
            # Fall back to brightness/date proxies already on records; if empty, skip gracefully.
            continue
        real_runs, mix_runs = [], []
        for seed in seeds:
            real_runs.append(train_one_detector("yolov13", tr + nonsmoke[:40], te, cfg, device, seed))
            mix_runs.append(train_one_detector("yolov13", tr + nonsmoke[:40] + mixed_extra, te, cfg, device, seed))
        domain[name] = {
            "real_ap": [r["ap50"] / 100 for r in real_runs],
            "mix_ap": [r["ap50"] / 100 for r in mix_runs],
            "real_rec": [r["recall"] for r in real_runs],
            "mix_rec": [r["recall"] for r in mix_runs],
        }
    save_json(json_dir / "domain_shift.json", domain)

    # Filter-threshold sensitivity using yolov13, seed 42.
    thresh_table = []
    ranked_sorted = sorted(candidates, key=lambda c: c["quality"], reverse=True)
    for frac, label in [(0.1, "10%"), (0.2, "20%"), (0.3, "30%"), (0.4, "40%"), (0.5, "50%"), (1.0, "No filtering")]:
        k = max(1, int(round(frac * len(ranked_sorted))) if frac < 1 else len(ranked_sorted))
        extra = [{"path": c["path"], "boxes": c["boxes"]} for c in ranked_sorted[:k]]
        met = train_one_detector("yolov13", real_train + nonsmoke + extra, test_items, cfg, device, 42)
        thresh_table.append({"threshold": label, "size": len(real_train + extra), "ap50": met["ap50"] / 100, "recall": met["recall"]})
    save_json(json_dir / "filter_threshold.json", thresh_table)

    # Component ablation: generate a small set from each ablation unet and train yolov13 once.
    def gen_from(model, n=24, tag="abl"):
        items = []
        d = ensure_dir(out / f"synth_{tag}")
        for i in range(min(n, len(pairs))):
            bg, mrec = pairs[i]
            bg_np = to_numpy(bg["path"], size)
            mask = load_mask(Path(mrec["mask_path"]), size)
            masked = bg_np * (1.0 - mask[..., None])
            mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
            masked_t = torch.from_numpy(masked).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
            text = torch.tensor(tokenize_prompt(PROMPT_TEMPLATES[0])).unsqueeze(0).float()
            img = generate_image(model, vae, mask_t, masked_t, text, cfg, device, seed=42 + i)
            p = d / f"{i}.png"
            Image.fromarray((img * 255).astype(np.uint8)).save(p)
            bbox = mask_to_bbox((mask * 255).astype(np.uint8))
            items.append({"path": str(p), "boxes": [[float(b) for b in bbox]]})
        return items

    ablation = {}
    for tag, model in [
        ("no_jca", unet_nojca),
        ("no_mrdl", unet_nomrdl),
        ("rand_encoder", unet_randenc),
        ("full", unet),
    ]:
        extra = gen_from(model, n=24, tag=tag)
        met = train_one_detector("yolov13", real_train + nonsmoke + extra, test_items, cfg, device, 42)
        ablation[tag] = met
    extra_all = [{"path": c["path"], "boxes": c["boxes"]} for c in candidates]
    ablation["unfiltered"] = train_one_detector(
        "yolov13", real_train + nonsmoke + extra_all, test_items, cfg, device, 42
    )
    save_json(json_dir / "ablation.json", ablation)

    omega_sens = {
        "0.0": ablation["no_mrdl"],
        "0.4": ablation["full"],
        "1.0": train_one_detector(
            "yolov13",
            real_train + nonsmoke + gen_from(unet_omega1, 24, "omega1"),
            test_items,
            cfg,
            device,
            42,
        ),
    }
    save_json(json_dir / "mrdl_omega.json", omega_sens)

    # VLM-proxy validation via 5-fold MAE/RMSE/Spearman against heuristic teacher.
    from sklearn.model_selection import KFold
    from scipy.stats import spearmanr

    xs = np.stack([image_features(c["image"], c["mask"]) for c in candidates[:n_ann]])
    ys = np.stack([heuristic_scores(c["image"], c["mask"]) for c in candidates[:n_ann]])
    fold_metrics = []
    kf = KFold(n_splits=min(5, max(2, n_ann // 10)), shuffle=True, random_state=42)
    from firesmokegennet.quality.filter import QualityMLP

    for tr, te in kf.split(xs):
        model = QualityMLP(24, cfg["quality"]["mlp_hidden"])
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        xt, yt = torch.from_numpy(xs[tr]), torch.from_numpy(ys[tr])
        for _ in range(30):
            opt.zero_grad()
            torch.nn.functional.mse_loss(model(xt), yt).backward()
            opt.step()
        with torch.no_grad():
            pred = model(torch.from_numpy(xs[te])).numpy()
        true = ys[te]
        mae = np.mean(np.abs(pred - true), axis=0)
        rmse = np.sqrt(np.mean((pred - true) ** 2, axis=0))
        q_true = 0.4 * true[:, 0] + 0.4 * true[:, 1] + 0.2 * true[:, 2]
        q_pred = 0.4 * pred[:, 0] + 0.4 * pred[:, 1] + 0.2 * pred[:, 2]
        rho = spearmanr(q_true, q_pred).correlation
        fold_metrics.append({"mae": mae.tolist(), "rmse": rmse.tolist(), "rho": float(rho if rho == rho else 0)})
    save_json(json_dir / "vlm_validation.json", fold_metrics)

    write_tables(tab_dir, table_rows, gen_metrics, dist_table, domain, thresh_table, ablation, omega_sens, fold_metrics, splits, bg_recs)
    write_figures(fig_dir, table_rows, omega_sens, ablation, retained, real_imgs, real_masks)
    write_markdown_report(ROOT / "results" / "RESULTS.md", table_rows, gen_metrics, domain, ablation, cfg)
    print("done")


def write_tables(tab_dir, table_rows, gen_metrics, dist_table, domain, thresh_table, ablation, omega_sens, fold_metrics, splits, bg_recs):
    def csv(path, header, rows):
        with open(path, "w", encoding="utf-8") as f:
            f.write(",".join(header) + "\n")
            for r in rows:
                f.write(",".join(str(x) for x in r) + "\n")

    csv(
        tab_dir / "detector_ap50.csv",
        ["detector", "real_mean", "real_sd", "mixed_mean", "mixed_sd", "delta", "p_adj", "dz"],
        [
            [
                r["detector"],
                f"{r['real_mean']:.2f}",
                f"{r['real_sd']:.2f}",
                f"{r['mixed_mean']:.2f}",
                f"{r['mixed_sd']:.2f}",
                f"{r['delta']:.2f}",
                f"{r['p_adj']:.4f}",
                f"{r['dz']:.2f}",
            ]
            for r in table_rows
        ],
    )
    csv(
        tab_dir / "background_preservation.csv",
        ["method", "psnr", "ssim", "lpips_proxy", "prompt_sim"],
        [
            ["alpha_blend", f"{gen_metrics['blend_psnr']:.2f}", f"{gen_metrics['blend_ssim']:.3f}", "", ""],
            [
                "FireSmokeGenNet",
                f"{gen_metrics['psnr']:.2f}",
                f"{gen_metrics['ssim']:.3f}",
                f"{gen_metrics['lpips_proxy']:.3f}",
                f"{gen_metrics['prompt_sim']:.3f}",
            ],
        ],
    )
    csv(
        tab_dir / "distribution.csv",
        ["method", "feature_fd", "mmd2"],
        [[k, f"{v['clip_fd']:.3f}", f"{v['mmd']:.4f}"] for k, v in dist_table.items()],
    )
    csv(
        tab_dir / "boundary.csv",
        ["method", "softness", "kl"],
        [
            ["real", f"{gen_metrics['boundary']['real']:.4f}", "-"],
            ["alpha_blend", f"{gen_metrics['boundary']['blend']:.4f}", f"{gen_metrics['boundary']['kl_blend']:.4f}"],
            ["FireSmokeGenNet", f"{gen_metrics['boundary']['ours']:.4f}", f"{gen_metrics['boundary']['kl_ours']:.4f}"],
        ],
    )
    csv(
        tab_dir / "filter_threshold.csv",
        ["threshold", "dataset_size", "ap50", "recall"],
        [[r["threshold"], r["size"], f"{r['ap50']:.3f}", f"{r['recall']:.3f}"] for r in thresh_table],
    )
    csv(
        tab_dir / "ablation.csv",
        ["config", "ap50", "ap50_95", "precision", "recall"],
        [
            [k, f"{v['ap50']/100:.3f}", f"{v['ap50_95']:.3f}", f"{v['precision']:.3f}", f"{v['recall']:.3f}"]
            for k, v in ablation.items()
        ],
    )
    csv(
        tab_dir / "dataset_splits.csv",
        ["partition", "sources_approx", "real_smoke"],
        [[k, "", len(v)] for k, v in splits.items()] + [["backgrounds", "", len(bg_recs)]],
    )


def write_figures(fig_dir, table_rows, omega_sens, ablation, retained, real_imgs, real_masks):
    import matplotlib.pyplot as plt

    names = [r["detector"] for r in table_rows]
    real = [r["real_mean"] for r in table_rows]
    mix = [r["mixed_mean"] for r in table_rows]
    real_sd = [r["real_sd"] for r in table_rows]
    mix_sd = [r["mixed_sd"] for r in table_rows]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(x - 0.2, real, 0.4, yerr=real_sd, label="Real only", capsize=3)
    ax.bar(x + 0.2, mix, 0.4, yerr=mix_sd, label="Real + FireSmokeGenNet", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20)
    ax.set_ylabel("AP50 (%)")
    ax.set_title("Five-seed detector comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "detector_ap50.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.6))
    omegas = [0.0, 0.4, 1.0]
    vals = [omega_sens["0.0"]["ap50"] / 100, omega_sens["0.4"]["ap50"] / 100, omega_sens["1.0"]["ap50"] / 100]
    ax.plot(omegas, vals, marker="o")
    ax.set_xlabel("MRDL weight ω")
    ax.set_ylabel("AP50")
    ax.set_title("MRDL-weight sensitivity")
    fig.tight_layout()
    fig.savefig(fig_dir / "mrdl_sensitivity.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    labels = list(ablation.keys())
    aps = [ablation[k]["ap50"] / 100 for k in labels]
    ax.bar(labels, aps)
    ax.set_ylabel("AP50")
    ax.set_title("Component ablation (YOLOv13 stand-in, fixed seed)")
    plt.setp(ax.get_xticklabels(), rotation=20)
    fig.tight_layout()
    fig.savefig(fig_dir / "ablation.png", dpi=140)
    plt.close(fig)

    n = min(4, len(retained))
    if n:
        fig, axes = plt.subplots(n, 3, figsize=(9, 2.4 * n))
        if n == 1:
            axes = np.array([axes])
        for i in range(n):
            axes[i, 0].imshow(retained[i]["bg"])
            axes[i, 1].imshow(retained[i]["blend"])
            axes[i, 2].imshow(retained[i]["image"])
            for j, title in enumerate(["Background", "Alpha-blend", "FireSmokeGenNet"]):
                axes[i, j].set_title(title if i == 0 else "")
                axes[i, j].axis("off")
        fig.tight_layout()
        fig.savefig(fig_dir / "qualitative.png", dpi=140)
        plt.close(fig)


def write_markdown_report(path: Path, table_rows, gen_metrics, domain, ablation, cfg):
    lines = [
        "# FireSmokeGenNet compact reproduction results",
        "",
        "This run implements the paper's methodology on public data at compact CPU scale.",
        "It does **not** claim numerical identity with the A100 / 512px / 96k-sample paper tables.",
        "",
        "## Background preservation",
        f"- FireSmokeGenNet PSNR={gen_metrics['psnr']:.2f} dB, SSIM={gen_metrics['ssim']:.3f}",
        f"- Alpha-blend PSNR={gen_metrics['blend_psnr']:.2f} dB, SSIM={gen_metrics['blend_ssim']:.3f}",
        "",
        "## Detector AP50 (five seeds)",
    ]
    for r in table_rows:
        lines.append(
            f"- {r['detector']}: real {r['real_mean']:.2f}±{r['real_sd']:.2f} → mixed {r['mixed_mean']:.2f}±{r['mixed_sd']:.2f} (Δ={r['delta']:+.2f}, p_adj={r['p_adj']:.4f}, dz={r['dz']:.2f})"
        )
    lines += ["", "## Ablation AP50"]
    for k, v in ablation.items():
        lines.append(f"- {k}: {v['ap50']:.2f}")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
