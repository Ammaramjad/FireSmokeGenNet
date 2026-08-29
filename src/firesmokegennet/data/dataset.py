"""PyTorch datasets for generator and detector training."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .download import yolo_to_xyxy
from .masks import grabcut_mask, mask_to_bbox
from .prompts import PROMPT_TEMPLATES, tokenize_prompt


def _load_rgb(path: str, size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


class GeneratorDataset(Dataset):
    def __init__(self, records: list[dict], image_size: int, text_dim: int = 64, cache_masks: Path | None = None):
        self.records = records
        self.image_size = image_size
        self.text_dim = text_dim
        self.cache_masks = cache_masks
        if cache_masks:
            cache_masks.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.records)

    def _mask_for(self, rec: dict, image: Image.Image) -> np.ndarray:
        cache = None
        if self.cache_masks:
            cache = self.cache_masks / (Path(rec["path"]).stem + ".png")
            if cache.exists():
                m = np.array(Image.open(cache).convert("L").resize((self.image_size, self.image_size)))
                return (m > 127).astype(np.float32)
        w, h = image.size
        boxes = yolo_to_xyxy(rec.get("annotations") or "", w, h)
        mask = grabcut_mask(image, boxes)
        if cache is not None:
            Image.fromarray((mask * 255).astype(np.uint8)).save(cache)
        mask = np.array(Image.fromarray(mask * 255).resize((self.image_size, self.image_size), Image.NEAREST))
        return (mask > 127).astype(np.float32)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        pil = Image.open(rec["path"]).convert("RGB")
        mask = self._mask_for(rec, pil)
        rgb = _load_rgb(rec["path"], self.image_size)
        masked = rgb * (1.0 - mask[..., None])
        prompt = random.choice(PROMPT_TEMPLATES)
        text = torch.tensor(tokenize_prompt(prompt, self.text_dim), dtype=torch.float32)
        x = torch.from_numpy(rgb).permute(2, 0, 1)
        m = torch.from_numpy(mask).unsqueeze(0)
        xm = torch.from_numpy(masked).permute(2, 0, 1)
        return {"image": x, "mask": m, "masked": xm, "text": text, "prompt": prompt, "path": rec["path"]}


class DetectorDataset(Dataset):
    def __init__(self, items: list[dict], image_size: int, grid: int = 8):
        self.items = items
        self.image_size = image_size
        self.grid = grid

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        rgb = _load_rgb(item["path"], self.image_size)
        x = torch.from_numpy(rgb).permute(2, 0, 1)
        target = torch.zeros(self.grid, self.grid, 5, dtype=torch.float32)
        boxes = item.get("boxes") or []
        for box in boxes:
            x1, y1, x2, y2 = box
            xc = 0.5 * (x1 + x2) / self.image_size
            yc = 0.5 * (y1 + y2) / self.image_size
            w = max(x2 - x1, 1) / self.image_size
            h = max(y2 - y1, 1) / self.image_size
            gi = min(self.grid - 1, max(0, int(xc * self.grid)))
            gj = min(self.grid - 1, max(0, int(yc * self.grid)))
            target[gj, gi] = torch.tensor(
                [xc * self.grid - gi, yc * self.grid - gj, w, h, 1.0]
            )
        return x, target, item
