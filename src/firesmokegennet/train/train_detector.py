"""Train compact detector families under real-only vs mixed-data conditions."""

from __future__ import annotations

import copy

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.dataset import DetectorDataset
from ..metrics.core import average_precision, precision_recall
from ..models.detector import TinyYOLO, decode_boxes, yolo_loss


def train_one_detector(
    family: str,
    items: list[dict],
    val_items: list[dict],
    cfg: dict,
    device: torch.device,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    grid = cfg["detector"]["grid_size"]
    size = cfg["detector"]["input_size"]
    model = TinyYOLO(family, grid=grid).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["detector"]["lr"],
        weight_decay=cfg["detector"]["weight_decay"],
    )
    loader = DataLoader(
        DetectorDataset(items, size, grid),
        batch_size=cfg["detector"]["batch_size"],
        shuffle=True,
        collate_fn=_collate,
    )
    model.train()
    for epoch in range(cfg["detector"]["epochs"]):
        for x, target, _ in loader:
            x, target = x.to(device), target.to(device)
            pred = model(x)
            loss = yolo_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return evaluate_detector(model, val_items, cfg, device)


@torch.no_grad()
def evaluate_detector(model, items: list[dict], cfg: dict, device: torch.device) -> dict:
    model.eval()
    size = cfg["detector"]["input_size"]
    grid = cfg["detector"]["grid_size"]
    loader = DataLoader(
        DetectorDataset(items, size, grid),
        batch_size=8,
        shuffle=False,
        collate_fn=_collate,
    )
    pred_boxes, pred_scores, gt_boxes = [], [], []
    for x, _, batch_items in loader:
        pred = model(x.to(device))
        boxes, scores = decode_boxes(
            pred,
            size,
            conf_thr=cfg["detector"]["conf_threshold"],
            nms_iou=cfg["detector"]["nms_iou"],
        )
        pred_boxes.extend(boxes)
        pred_scores.extend(scores)
        for item in batch_items:
            gt_boxes.append(item.get("boxes") or [])
    ap50 = average_precision(pred_boxes, pred_scores, gt_boxes, 0.5)
    ap_list = [average_precision(pred_boxes, pred_scores, gt_boxes, t) for t in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]]
    prec, rec = precision_recall(
        pred_boxes, pred_scores, gt_boxes, 0.5, cfg["detector"]["conf_threshold"]
    )
    return {
        "ap50": ap50 * 100.0,
        "ap50_95": sum(ap_list) / len(ap_list),
        "precision": prec,
        "recall": rec,
    }


def _collate(batch):
    xs = torch.stack([b[0] for b in batch])
    ys = torch.stack([b[1] for b in batch])
    items = [b[2] for b in batch]
    return xs, ys, items
