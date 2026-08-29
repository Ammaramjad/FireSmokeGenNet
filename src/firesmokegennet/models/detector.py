"""Compact YOLO-family stand-ins: single-plume box heads on a frozen ResNet-18."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

FAMILY_SPEC = {
    "yolov6": {"hidden": 64, "depth": 1},
    "yolov7": {"hidden": 80, "depth": 1},
    "yolov8": {"hidden": 96, "depth": 2},
    "yolov9": {"hidden": 112, "depth": 2},
    "yolov10": {"hidden": 128, "depth": 2},
    "yolov11": {"hidden": 144, "depth": 3},
    "yolov12": {"hidden": 160, "depth": 3},
    "yolov13": {"hidden": 192, "depth": 3},
}


class TinyYOLO(nn.Module):
    """Predicts one smoke box per image (cx, cy, w, h, obj), matching MaxComponent(M)."""

    def __init__(self, family: str = "yolov13", grid: int = 8):
        super().__init__()
        spec = FAMILY_SPEC[family]
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        children = list(backbone.children())
        self.stem = nn.Sequential(*children[:-3])  # through layer3, frozen
        self.layer4 = children[-3]  # trainable
        self.pool = children[-2]
        for p in self.stem.parameters():
            p.requires_grad = False
        hidden, depth = spec["hidden"], spec["depth"]
        layers: list[nn.Module] = [nn.Flatten(), nn.Linear(512, hidden), nn.SiLU()]
        for _ in range(depth):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, 5))
        self.head = nn.Sequential(*layers)
        # Start near a mid-frame 20% box rather than a degenerate 4px box.
        nn.init.constant_(self.head[-1].bias, 0.0)
        self.head[-1].bias.data[2:4] = -1.4  # sigmoid ~0.20
        self.head[-1].bias.data[4] = -1.0
        self.grid = grid
        self.family = family

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            h = self.stem(x)
        h = self.pool(self.layer4(h))
        out = self.head(h)
        return out[:, None, None, :]


def yolo_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Collapse any grid to a single object: take the max-objectness cell as GT if present.
    obj_map = target[..., 4]
    b = target.size(0)
    gt = torch.zeros(b, 5, device=target.device)
    for i in range(b):
        flat = obj_map[i].reshape(-1)
        if flat.max() > 0:
            idx = int(flat.argmax())
            gt[i] = target[i].reshape(-1, 5)[idx]
            # Convert cell-relative xy to image-normalized cx, cy.
            g = target.size(1)
            gj, gi = divmod(idx, g) if False else (idx // g, idx % g)
            dx, dy, w, h, o = gt[i]
            gt[i, 0] = (gi + dx) / g
            gt[i, 1] = (gj + dy) / g
            gt[i, 2] = w
            gt[i, 3] = h
            gt[i, 4] = 1.0
    pred = pred.reshape(b, 5)
    bce = F.binary_cross_entropy_with_logits(pred[:, 4], gt[:, 4])
    pos = gt[:, 4] > 0.5
    if pos.any():
        box = 6.0 * F.mse_loss(torch.sigmoid(pred[pos, :4]), gt[pos, :4])
    else:
        box = pred.new_zeros(())
    return bce + box


@torch.no_grad()
def decode_boxes(pred: torch.Tensor, image_size: int, conf_thr: float = 0.25, nms_iou: float = 0.45):
    pred = pred.cpu().reshape(pred.size(0), -1, 5)
    all_boxes, all_scores = [], []
    for n in range(pred.size(0)):
        logit = pred[n, 0]
        conf = float(torch.sigmoid(logit[4]))
        if conf < conf_thr:
            all_boxes.append([])
            all_scores.append([])
            continue
        cx, cy, w, h = torch.sigmoid(logit[:4]).tolist()
        w, h = max(w, 4.0 / image_size), max(h, 4.0 / image_size)
        x1 = (cx - w / 2) * image_size
        y1 = (cy - h / 2) * image_size
        x2 = (cx + w / 2) * image_size
        y2 = (cy + h / 2) * image_size
        all_boxes.append([[x1, y1, x2, y2]])
        all_scores.append([conf])
    return all_boxes, all_scores
