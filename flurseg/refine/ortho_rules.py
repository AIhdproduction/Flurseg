"""Orthophoto-specific metric constraints."""

from __future__ import annotations

import cv2
import numpy as np


def apply(mask: np.ndarray, source: str, gsd_m: float | None, config: object, ids: dict[str, int]) -> np.ndarray:
    if source != "ortho" or gsd_m is None:
        return mask.copy()
    result = mask.copy()
    for name in ("curb", "utility_cover"):
        if name not in ids:
            continue
        count, labels, stats, _ = cv2.connectedComponentsWithStats((result == ids[name]).astype(np.uint8), connectivity=8)
        for label in range(1, count):
            extent = max(stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT]) * gsd_m
            if extent < config.min_small_object_m:
                x, y, width, height, _ = stats[label]
                component = labels[y : y + height, x : x + width] == label
                result[y : y + height, x : x + width][component] = 255
    if "road_marking" in ids:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (result == ids["road_marking"]).astype(np.uint8), connectivity=8
        )
        for label in range(1, count):
            x, y, width, height, _ = stats[label]
            component = (labels[y : y + height, x : x + width] == label).astype(np.uint8)
            padded = np.pad(component, 1)
            component_width = 2 * float(cv2.distanceTransform(padded, cv2.DIST_L2, 5).max()) * gsd_m
            if component_width < config.min_marking_width_m:
                result[y : y + height, x : x + width][component.astype(bool)] = 255
    return result
