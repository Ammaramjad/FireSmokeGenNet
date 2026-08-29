"""Background preservation, distribution, boundary, and detection metrics."""

from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _masked(a: np.ndarray, b: np.ndarray, keep: np.ndarray):
    keep = keep.astype(bool)
    if keep.ndim == 2:
        keep3 = np.repeat(keep[..., None], 3, axis=2)
    else:
        keep3 = keep
    return a * keep3, b * keep3, keep3


def psnr_background(pred: np.ndarray, ref: np.ndarray, preserve: np.ndarray) -> float:
    a, b, k = _masked(pred, ref, preserve)
    if k.sum() < 16:
        return 0.0
    return float(peak_signal_noise_ratio(b, a, data_range=1.0))


def ssim_background(pred: np.ndarray, ref: np.ndarray, preserve: np.ndarray) -> float:
    a, b, k = _masked(pred, ref, preserve)
    return float(structural_similarity(b, a, channel_axis=2, data_range=1.0))


def lpips_proxy(pred: np.ndarray, ref: np.ndarray, preserve: np.ndarray) -> float:
    """1 - normalized L2 in RGB gradient space (LPIPS stand-in on CPU)."""
    a, b, k = _masked(pred, ref, preserve)
    ga = np.stack(np.gradient(a.mean(axis=2)), axis=-1)
    gb = np.stack(np.gradient(b.mean(axis=2)), axis=-1)
    dist = np.mean((ga - gb) ** 2) ** 0.5
    return float(np.clip(1.0 - dist / 0.25, 0, 1))


def prompt_sim(image: np.ndarray, mask: np.ndarray) -> float:
    smoke = image[mask.astype(bool)]
    if smoke.size == 0:
        return 0.0
    grayness = 1.0 - np.std(smoke.mean(axis=0))
    return float(np.clip(0.15 + 0.2 * grayness + 0.05 * smoke.mean(), 0, 1))


def frechet_distance(x: np.ndarray, y: np.ndarray) -> float:
    mu1, mu2 = x.mean(axis=0), y.mean(axis=0)
    c1 = np.cov(x, rowvar=False) + 1e-6 * np.eye(x.shape[1])
    c2 = np.cov(y, rowvar=False) + 1e-6 * np.eye(y.shape[1])
    diff = mu1 - mu2
    covmean = _sqrtm(c1 @ c2)
    return float(diff.dot(diff) + np.trace(c1 + c2 - 2 * covmean))


def _sqrtm(mat: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh((mat + mat.T) / 2)
    vals = np.clip(vals, 0, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


def linear_mmd(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sum((x.mean(axis=0) - y.mean(axis=0)) ** 2))


def boundary_softness(image: np.ndarray, mask: np.ndarray, delta: int = 2) -> float:
    """Eq. (40): mean gradient magnitude in the morphological boundary band."""
    from scipy.ndimage import binary_dilation, binary_erosion, sobel

    mask_b = mask.astype(bool)
    band = binary_dilation(mask_b, iterations=delta) ^ binary_erosion(mask_b, iterations=delta)
    if band.sum() == 0:
        return 0.0
    gray = image.mean(axis=2)
    grad = np.hypot(sobel(gray, axis=0), sobel(gray, axis=1))
    return float(grad[band].mean())


def gradient_kl(real_images, real_masks, synth_images, synth_masks, bins: int = 40) -> float:
    def hist(images, masks):
        vals = []
        for im, m in zip(images, masks):
            s = boundary_softness(im, m)
            vals.append(s)
        h, _ = np.histogram(vals, bins=bins, range=(0, 0.5), density=True)
        return h + 1e-8

    p = hist(real_images, real_masks)
    q = hist(synth_images, synth_masks)
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def average_precision(pred_boxes: list, pred_scores: list, gt_boxes: list, iou_thr: float = 0.5) -> float:
    """Image-set AP at a single IoU threshold."""
    records = []
    n_gt = 0
    for preds, scores, gts in zip(pred_boxes, pred_scores, gt_boxes):
        n_gt += len(gts)
        order = np.argsort(-np.asarray(scores)) if len(scores) else []
        matched = set()
        for idx in order:
            box, sc = preds[idx], scores[idx]
            best_iou, best_j = 0.0, -1
            for j, gt in enumerate(gts):
                if j in matched:
                    continue
                iou = box_iou(np.asarray(box), np.asarray(gt))
                if iou > best_iou:
                    best_iou, best_j = iou, j
            records.append((sc, best_iou >= iou_thr, best_j))
            if best_iou >= iou_thr:
                matched.add(best_j)
    if n_gt == 0:
        return 0.0
    records.sort(key=lambda r: -r[0])
    tp = fp = 0
    precisions, recalls = [], []
    for _, is_tp, _ in records:
        if is_tp:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / n_gt)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = max([pr for pr, rc in zip(precisions, recalls) if rc >= t] or [0.0])
        ap += p / 11.0
    return float(ap)


def precision_recall(pred_boxes, pred_scores, gt_boxes, iou_thr=0.5, conf=0.25):
    tp = fp = fn = 0
    for preds, scores, gts in zip(pred_boxes, pred_scores, gt_boxes):
        kept = [b for b, s in zip(preds, scores) if s >= conf]
        matched = set()
        for box in kept:
            hit = False
            for j, gt in enumerate(gts):
                if j in matched:
                    continue
                if box_iou(np.asarray(box), np.asarray(gt)) >= iou_thr:
                    matched.add(j)
                    hit = True
                    break
            if hit:
                tp += 1
            else:
                fp += 1
        fn += len(gts) - len(matched)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return float(prec), float(rec)
