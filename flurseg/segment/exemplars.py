"""Experimental mosaic and same-image visual-exemplar helpers."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


class _BoxBackend(Protocol):
    def score_boxes(self, image: np.ndarray, boxes: Sequence[tuple[int, int, int, int]], prompt: str) -> np.ndarray: ...


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not read exemplar image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _boxes(path: Path) -> list[tuple[int, int, int, int]]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    values = data.get("boxes", data) if isinstance(data, dict) else data
    result: list[tuple[int, int, int, int]] = []
    for value in values:
        if isinstance(value, dict):
            value = value.get("box", value.get("xyxy"))
        if not isinstance(value, Sequence) or len(value) != 4:
            continue
        x0, y0, x1, y1 = (round(float(number)) for number in value)
        if x1 > x0 and y1 > y0:
            result.append((x0, y0, x1, y1))
    return result


def load_exemplar_crops(directory: Path, max_per_class: int) -> list[np.ndarray]:
    """Load annotated exemplar crops deterministically, accepting JPG and PNG source files."""
    crops: list[np.ndarray] = []
    if not directory.is_dir():
        return crops
    images = sorted([*directory.glob("*.jpg"), *directory.glob("*.jpeg"), *directory.glob("*.png")])
    for image_path in images:
        image = _read_rgb(image_path)
        for x0, y0, x1, y1 in _boxes(image_path.with_suffix(".json")):
            x0, x1 = max(0, x0), min(image.shape[1], x1)
            y0, y1 = max(0, y0), min(image.shape[0], y1)
            if x1 > x0 and y1 > y0:
                crops.append(image[y0:y1, x0:x1].copy())
                if len(crops) >= max_per_class:
                    return crops
    return crops


def mosaic_canvas(
    tile: np.ndarray,
    exemplar_directory: Path,
    *,
    tile_size: int,
    strip_fraction: float,
    max_per_class: int,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]], tuple[int, int]]:
    """Build a fixed SAM canvas and return translated strip boxes plus content dimensions."""
    if not 0 < strip_fraction < 1:
        raise ValueError("strip_fraction must be between zero and one")
    content_height = min(tile.shape[0], max(1, round(tile_size * (1 - strip_fraction))))
    content_width = min(tile.shape[1], tile_size)
    canvas = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
    if content_height and content_width:
        canvas[:content_height, :content_width] = tile[:content_height, :content_width]
    strip_height = tile_size - content_height
    crops = load_exemplar_crops(exemplar_directory, max_per_class)
    boxes: list[tuple[int, int, int, int]] = []
    if strip_height <= 0 or not crops:
        return canvas, boxes, (content_height, content_width)
    slots = max(1, min(len(crops), max_per_class))
    slot_width = max(1, tile_size // slots)
    for index, crop in enumerate(crops[:slots]):
        factor = min(slot_width / crop.shape[1], strip_height / crop.shape[0])
        size = (max(1, round(crop.shape[1] * factor)), max(1, round(crop.shape[0] * factor)))
        resized = cv2.resize(crop, size, interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR)
        x0 = index * slot_width + (slot_width - size[0]) // 2
        y0 = content_height + (strip_height - size[1]) // 2
        canvas[y0 : y0 + size[1], x0 : x0 + size[0]] = resized
        boxes.append((x0, y0, x0 + size[0], y0 + size[1]))
    return canvas, boxes, (content_height, content_width)


def score_mosaic(
    tile: np.ndarray,
    backend: _BoxBackend,
    prompt: str,
    exemplar_directory: Path,
    *,
    tile_size: int,
    strip_fraction: float,
    max_per_class: int,
) -> np.ndarray:
    """Score the content region of an experimental mosaic and discard the exemplar strip."""
    canvas, boxes, (height, width) = mosaic_canvas(
        tile, exemplar_directory, tile_size=tile_size, strip_fraction=strip_fraction, max_per_class=max_per_class
    )
    if not boxes:
        return np.zeros((height, width), dtype=np.float32)
    return backend.score_boxes(canvas, boxes, prompt)[:height, :width].astype(np.float32)


def find_similar(
    image: np.ndarray,
    boxes: Sequence[tuple[int, int, int, int]],
    prompt: str,
    backend: _BoxBackend,
    *,
    tile_size: int = 1008,
) -> np.ndarray:
    """Run visual-exemplar prompting in a clamped square window and paste it into full image coordinates."""
    if not boxes:
        return np.zeros(image.shape[:2], dtype=np.float32)
    height, width = image.shape[:2]
    cx = round(sum((x0 + x1) / 2 for x0, _, x1, _ in boxes) / len(boxes))
    cy = round(sum((y0 + y1) / 2 for _, y0, _, y1 in boxes) / len(boxes))
    window_width, window_height = min(tile_size, width), min(tile_size, height)
    left = min(max(0, cx - window_width // 2), max(0, width - window_width))
    top = min(max(0, cy - window_height // 2), max(0, height - window_height))
    right, bottom = left + window_width, top + window_height
    local_boxes = [(max(0, x0 - left), max(0, y0 - top), min(window_width, x1 - left), min(window_height, y1 - top)) for x0, y0, x1, y1 in boxes]
    local_boxes = [box for box in local_boxes if box[2] > box[0] and box[3] > box[1]]
    output = np.zeros((height, width), dtype=np.float32)
    if local_boxes:
        output[top:bottom, left:right] = backend.score_boxes(image[top:bottom, left:right], local_boxes, prompt)
    return output
