"""Final component cleanup for metric and unscaled masks."""

from __future__ import annotations

import cv2
import numpy as np


def _minimum_size(name: str, gsd_m: float | None, config: object) -> float:
    if gsd_m is None:
        return float(getattr(config, "min_component_px", {}).get(name, 0))
    return float(getattr(config, "min_island_m2", {}).get(name, 0)) / (gsd_m * gsd_m)


def _ring_values(result: np.ndarray, labels: np.ndarray, label: int, stats: np.ndarray) -> np.ndarray:
    x, y, width, height, _ = stats[label]
    left, top = max(0, x - 1), max(0, y - 1)
    right, bottom = min(result.shape[1], x + width + 1), min(result.shape[0], y + height + 1)
    component = labels[top:bottom, left:right] == label
    expanded = cv2.dilate(component.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return result[top:bottom, left:right][expanded & ~component]


def _ring_majority(values: np.ndarray) -> int:
    if values.size == 0:
        return 255
    labels, counts = np.unique(values, return_counts=True)
    return int(labels[np.argmax(counts)])


def _remove_small_islands(result: np.ndarray, gsd_m: float | None, config: object, ids: dict[str, int]) -> None:
    for name, target in ids.items():
        if target == 255:
            continue
        minimum = _minimum_size(name, gsd_m, config)
        if minimum <= 0:
            continue
        count, labels, stats, _ = cv2.connectedComponentsWithStats((result == target).astype(np.uint8), connectivity=8)
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] >= minimum:
                continue
            x, y, width, height, _ = stats[label]
            component = labels[y : y + height, x : x + width] == label
            replacement = _ring_majority(_ring_values(result, labels, label, stats))
            result[y : y + height, x : x + width][component] = replacement


def _fill_enclosed_components(result: np.ndarray, gsd_m: float | None, config: object, ids: dict[str, int]) -> None:
    protected = {ids[name] for name in ("utility_cover", "road_marking", "vehicle") if name in ids}
    names = {class_id: name for name, class_id in ids.items()}
    for value in np.unique(result):
        count, labels, stats, _ = cv2.connectedComponentsWithStats((result == value).astype(np.uint8), connectivity=8)
        for label in range(1, count):
            x, y, width, height, area = stats[label]
            if x == 0 or y == 0 or x + width == result.shape[1] or y + height == result.shape[0]:
                continue
            ring = _ring_values(result, labels, label, stats)
            if ring.size == 0 or np.any(ring != ring[0]) or ring[0] == 255:
                continue
            enclosing = int(ring[0])
            enclosing_name = names.get(enclosing)
            if enclosing_name is None or area >= _minimum_size(enclosing_name, gsd_m, config):
                continue
            component_name = names.get(int(value))
            if value in protected and component_name is not None and area >= _minimum_size(component_name, gsd_m, config):
                continue
            component = labels[y : y + height, x : x + width] == label
            result[y : y + height, x : x + width][component] = enclosing


def apply(mask: np.ndarray, gsd_m: float | None, config: object, ids: dict[str, int]) -> np.ndarray:
    """Merge undersized islands and fill enclosed undersized components for every source."""
    result = mask.copy()
    _remove_small_islands(result, gsd_m, config, ids)
    _fill_enclosed_components(result, gsd_m, config, ids)
    return result
