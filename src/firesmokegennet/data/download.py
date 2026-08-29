"""Public dataset acquisition: Pyro-SDIS (smoke) + Wikimedia Commons (backgrounds)."""

from __future__ import annotations

import hashlib
import io
import json
import random
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

from ..utils.io import ensure_dir, save_json


WIKI_QUERIES = [
    "forest landscape",
    "mountain valley forest",
    "hillside woodland",
    "conifer forest ridge",
    "autumn forest hills",
]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download_pyrosdis_subset(
    out_dir: Path,
    max_images: int = 420,
    seed: int = 42,
) -> list[dict]:
    """Stream a compact subset of https://huggingface.co/datasets/pyronear/pyro-sdis."""
    from datasets import load_dataset

    ensure_dir(out_dir / "images")
    rng = random.Random(seed)
    ds = load_dataset("pyronear/pyro-sdis", split="train", streaming=True)
    records = []
    seen_cameras: dict[str, int] = {}
    for ex in tqdm(ds, desc="pyro-sdis", total=max_images * 3):
        if len(records) >= max_images:
            break
        ann = (ex.get("annotations") or "").strip()
        if not ann:
            continue
        camera = str(ex.get("camera") or "unknown")
        # Cap per-camera frames so the source-level split stays diverse.
        if seen_cameras.get(camera, 0) >= 10:
            continue
        image: Image.Image = ex["image"].convert("RGB")
        name = str(ex.get("image_name") or f"{camera}_{len(records)}.jpg")
        safe = name.replace("/", "_")
        path = out_dir / "images" / safe
        image.save(path, quality=90)
        records.append(
            {
                "path": str(path),
                "annotations": ann,
                "camera": camera,
                "partner": str(ex.get("partner") or ""),
                "date": str(ex.get("date") or ""),
                "image_name": safe,
                "source": "pyronear/pyro-sdis",
                "has_smoke": True,
            }
        )
        seen_cameras[camera] = seen_cameras.get(camera, 0) + 1
    save_json(out_dir / "manifest.json", records)
    return records


def download_wikimedia_backgrounds(out_dir: Path, max_images: int = 140) -> list[dict]:
    """Download CC landscape stills from Wikimedia Commons search."""
    ensure_dir(out_dir / "images")
    headers = {"User-Agent": "FireSmokeGenNetReproduction/1.0 (research pipeline)"}
    records = []
    seen_urls = set()
    for query in WIKI_QUERIES:
        if len(records) >= max_images:
            break
        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": 6,
            "gsrlimit": 40,
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
            "iiurlwidth": 1280,
            "format": "json",
        }
        try:
            resp = requests.get(
                "https://commons.wikimedia.org/w/api.php",
                params=params,
                headers=headers,
                timeout=30,
            )
            resp.raise_for_status()
            pages = (resp.json().get("query") or {}).get("pages") or {}
        except Exception:
            continue
        for page in pages.values():
            if len(records) >= max_images:
                break
            info = (page.get("imageinfo") or [{}])[0]
            mime = info.get("mime") or ""
            if "image" not in mime:
                continue
            url = info.get("thumburl") or info.get("url")
            if not url or url in seen_urls:
                continue
            try:
                raw = requests.get(url, headers=headers, timeout=30).content
                image = Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception:
                continue
            if min(image.size) < 256:
                continue
            digest = _sha256_bytes(raw)[:16]
            path = out_dir / "images" / f"{digest}.jpg"
            image.save(path, quality=90)
            records.append(
                {
                    "path": str(path),
                    "source": "wikimedia-commons",
                    "query": query,
                    "url": url,
                    "has_smoke": False,
                }
            )
            seen_urls.add(url)
    if len(records) < max(12, max_images // 4):
        records.extend(_procedural_landscapes(out_dir, max_images - len(records), start=len(records)))
    save_json(out_dir / "manifest.json", records)
    return records


def _procedural_landscapes(out_dir: Path, n: int, start: int = 0) -> list[dict]:
    """Deterministic CC0-style landscape stand-ins if Wikimedia is unreachable."""
    rng = np.random.default_rng(123)
    records = []
    for i in range(max(0, n)):
        h, w = 360, 640
        yy = np.linspace(0, 1, h)[:, None]
        sky = np.stack([0.45 + 0.25 * (1 - yy), 0.55 + 0.2 * (1 - yy), 0.75 + 0.15 * (1 - yy)], axis=-1)
        ground = np.stack([0.18 + 0.1 * yy, 0.32 + 0.15 * yy, 0.12 + 0.05 * yy], axis=-1)
        ridge = (yy > 0.45 + 0.08 * np.sin(np.linspace(0, 8 * np.pi, w))).astype(np.float32)[:, :, None]
        img = sky * (1 - ridge) + ground * ridge
        img = (img + 0.03 * rng.normal(size=img.shape)).clip(0, 1)
        path = out_dir / "images" / f"procedural_{start + i:03d}.jpg"
        Image.fromarray((img * 255).astype(np.uint8)).save(path, quality=90)
        records.append({"path": str(path), "source": "procedural-landscape", "has_smoke": False})
    return records
    boxes = []
    for line in ann.strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        _, xc, yc, w, h = map(float, parts[:5])
        x1 = (xc - w / 2) * width
        y1 = (yc - h / 2) * height
        x2 = (xc + w / 2) * width
        y2 = (yc + h / 2) * height
        boxes.append([x1, y1, x2, y2])
    return boxes
