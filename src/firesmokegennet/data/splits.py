"""Source-level 80/10/10 splits, duplicate filtering, and environmental tags."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from PIL import Image

from ..utils.io import save_json


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def ahash(path: str) -> int:
    img = Image.open(path).convert("L").resize((8, 8))
    pixels = list(img.getdata())
    avg = sum(pixels) / len(pixels)
    bits = 0
    for i, p in enumerate(pixels):
        if p >= avg:
            bits |= 1 << i
    return bits


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def season_from_date(date_str: str) -> str:
    try:
        dt = datetime.fromisoformat(date_str.replace("T", " ").split(" ")[0])
        month = dt.month
    except Exception:
        return "unknown"
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def illumination_from_date(date_str: str) -> str:
    try:
        hour = int(date_str.split("T")[1].split("-")[0])
    except Exception:
        return "unknown"
    if 7 <= hour <= 16:
        return "day"
    if 17 <= hour <= 20:
        return "dusk"
    return "night"


def weather_proxy(path: str) -> str:
    img = Image.open(path).convert("RGB").resize((64, 64))
    pix = list(img.getdata())
    mean = [sum(c[i] for c in pix) / len(pix) for i in range(3)]
    sat = abs(mean[0] - mean[1]) + abs(mean[1] - mean[2])
    if sat < 18:
        return "haze"
    if mean[0] < 70:
        return "fog"
    return "clear"


def source_level_split(records: list[dict], seed: int = 42, ratios=(0.8, 0.1, 0.1)) -> dict:
    rng = random.Random(seed)
    by_source = defaultdict(list)
    for rec in records:
        by_source[rec.get("camera") or rec.get("path")].append(rec)
    sources = list(by_source.keys())
    rng.shuffle(sources)
    n = len(sources)
    n_train = max(1, int(round(ratios[0] * n)))
    n_val = max(1, int(round(ratios[1] * n)))
    if n_train + n_val >= n:
        n_val = max(1, n // 10)
        n_train = max(1, n - 2 * n_val)
    split_of = {}
    for i, src in enumerate(sources):
        if i < n_train:
            split_of[src] = "train"
        elif i < n_train + n_val:
            split_of[src] = "val"
        else:
            split_of[src] = "test"
    partitioned = {"train": [], "val": [], "test": []}
    hashes, phashes = set(), []
    for rec in records:
        try:
            digest = file_sha256(rec["path"])
            ph = ahash(rec["path"])
        except Exception:
            continue
        if digest in hashes:
            continue
        if any(hamming(ph, other) <= 1 for other in phashes):
            continue
        hashes.add(digest)
        phashes.append(ph)
        rec = dict(rec)
        rec["sha256"] = digest
        rec["split"] = split_of[rec.get("camera") or rec.get("path")]
        rec["season"] = season_from_date(rec.get("date") or "")
        rec["illumination"] = illumination_from_date(rec.get("date") or "")
        rec["weather"] = weather_proxy(rec["path"])
        partitioned[rec["split"]].append(rec)
    return partitioned
