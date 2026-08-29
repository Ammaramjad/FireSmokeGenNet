"""GrabCut mask extraction from YOLO boxes, analogue of SAM+bbox in the paper."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def grabcut_mask(image: Image.Image, boxes_xyxy: list[list[float]], work_size: int = 256) -> np.ndarray:
    rgb_full = np.array(image.convert("RGB"))
    h0, w0 = rgb_full.shape[:2]
    scale = work_size / max(h0, w0)
    new_w, new_h = max(32, int(w0 * scale)), max(32, int(h0 * scale))
    rgb = cv2.resize(rgb_full, (new_w, new_h), interpolation=cv2.INTER_AREA)
    sx, sy = new_w / w0, new_h / h0
    boxes_xyxy = [[x1 * sx, y1 * sy, x2 * sx, y2 * sy] for x1, y1, x2, y2 in boxes_xyxy]
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    if not boxes_xyxy:
        return np.zeros((h0, w0), np.uint8)
    gc_mask = np.full((h, w), cv2.GC_BGD, np.uint8)
    for x1, y1, x2, y2 in boxes_xyxy:
        x1, y1 = max(int(x1), 0), max(int(y1), 0)
        x2, y2 = min(int(x2), w - 1), min(int(y2), h - 1)
        if x2 <= x1 + 2 or y2 <= y1 + 2:
            continue
        gc_mask[y1:y2, x1:x2] = cv2.GC_PR_FGD
        inset = 4
        gc_mask[y1 + inset : max(y1 + inset, y2 - inset), x1 + inset : max(x1 + inset, x2 - inset)] = cv2.GC_FGD
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(rgb, gc_mask, None, bgd, fgd, 2, cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            mask[y1:y2, x1:x2] = 1
            continue
        mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1, mask).astype(np.uint8)
    if mask.sum() == 0:
        for x1, y1, x2, y2 in boxes_xyxy:
            mask[max(int(y1), 0) : min(int(y2), h), max(int(x1), 0) : min(int(x2), w)] = 1
    # Keep largest connected component, matching paper MaxComponent(M).
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (labels == largest).astype(np.uint8)
    return cv2.resize(mask, (w0, h0), interpolation=cv2.INTER_NEAREST)


def mask_to_bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return [0, 0, 1, 1]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
