"""Compact YOLO-family stand-ins used for the paper's eight-detector protocol."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Increasing width/depth across v6..v13 mirrors the paper's family sweep.
FAMILY_SPEC = {
    "yolov6": {"width": 0.35, "depth": 1},
    "yolov7": {"width": 0.45, "depth": 1},
    "yolov8": {"width": 0.55, "depth": 2},
    "yolov9": {"width": 0.65, "depth": 2},
    "yolov10": {"width": 0.75, "depth": 2},
    "yolov11": {"width": 0.85, "depth": 3},
    "yolov12": {"width": 0.95, "depth": 3},
    "yolov13": {"width": 1.10, "depth": 3},
}


class TinyYOLO(nn.Module):
    def __init__(self, family: str = "yolov13", grid: int = 8):
        super().__init__()
        spec = FAMILY_SPEC[family]
        w = spec["width"]
        d = spec["depth"]
        c1 = max(8, int(16 * w))
        c2 = max(16, int(32 * w))
        c3 = max(24, int(64 * w))
        layers = [nn.Conv2d(3, c1, 3, stride=2, padding=1), nn.SiLU()]
        for _ in range(d):
            layers += [nn.Conv2d(c1, c1, 3, padding=1), nn.SiLU()]
        layers += [nn.Conv2d(c1, c2, 3, stride=2, padding=1), nn.SiLU()]
        for _ in range(d):
            layers += [nn.Conv2d(c2, c2, 3, padding=1), nn.SiLU()]
        layers += [nn.Conv2d(c2, c3, 3, stride=2, padding=1), nn.SiLU()]
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Conv2d(c3, 5, 1)
        self.grid = grid
        self.family = family

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        h = F.adaptive_avg_pool2d(h, (self.grid, self.grid))
        return self.head(h).permute(0, 2, 3, 1)  # B,G,G,5


def yolo_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    obj = target[..., 4:5]
    bce = F.binary_cross_entropy_with_logits(pred[..., 4:5], obj)
    box = F.mse_loss(torch.sigmoid(pred[..., :4]) * obj, target[..., :4] * obj)
    return bce + box


@torch.no_grad()
def decode_boxes(pred: torch.Tensor, image_size: int, conf_thr: float = 0.25, nms_iou: float = 0.45):
    """Convert grid predictions to xyxy boxes."""
    from ..metrics.core import box_iou
    import numpy as np

    pred = pred.cpu()
    b, g, _, _ = pred.shape
    all_boxes, all_scores = [], []
    for n in range(b):
        boxes, scores = [], []
        logits = pred[n]
        conf = torch.sigmoid(logits[..., 4])
        xywh = torch.sigmoid(logits[..., :4])
        for gj in range(g):
            for gi in range(g):
                c = float(conf[gj, gi])
                if c < conf_thr:
                    continue
                dx, dy, w, h = [float(v) for v in xywh[gj, gi]]
                xc = (gi + dx) / g
                yc = (gj + dy) / g
                bw = max(w, 1e-3)
                bh = max(h, 1e-3)
                x1 = (xc - bw / 2) * image_size
                y1 = (yc - bh / 2) * image_size
                x2 = (xc + bw / 2) * image_size
                y2 = (yc + bh / 2) * image_size
                boxes.append([x1, y1, x2, y2])
                scores.append(c)
        keep = _nms(boxes, scores, nms_iou)
        all_boxes.append([boxes[i] for i in keep])
        all_scores.append([scores[i] for i in keep])
    return all_boxes, all_scores


def _nms(boxes, scores, iou_thr):
    from ..metrics.core import box_iou
    import numpy as np

    if not boxes:
        return []
    order = list(np.argsort(-np.asarray(scores)))
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if box_iou(np.asarray(boxes[i]), np.asarray(boxes[j])) < iou_thr]
    return keep
