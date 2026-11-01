"""Metric component size and compactness constraints."""

from __future__ import annotations

import cv2
import numpy as np


def _invalidate_components(result: np.ndarray, target: int, predicate: object) -> None:
    count, labels, stats, _ = cv2.connectedComponentsWithStats((result == target).astype(np.uint8), connectivity=8)
    for label in range(1, count):
        area = stats[label, cv2.CC_STAT_AREA]
        x, y, width, height, _ = stats[label]
        component = (labels[y : y + height, x : x + width] == label).astype(np.uint8)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
        if predicate(area, perimeter):
            result[y : y + height, x : x + width][component.astype(bool)] = 255


def apply(mask: np.ndarray, gsd_m: float | None, config: object, ids: dict[str, int]) -> np.ndarray:
    if gsd_m is None:
        return mask.copy()
    result = mask.copy()
    if "utility_cover" in ids:
        low, high = config.utility_cover_diameter_m
        def invalid_cover(area: float, perimeter: float) -> bool:
            equivalent = 2 * np.sqrt(area * gsd_m * gsd_m / np.pi)
            compactness = 4 * np.pi * area / (perimeter * perimeter) if perimeter else 0
            return not low <= equivalent <= high or compactness < config.utility_cover_min_compactness
        _invalidate_components(result, ids["utility_cover"], invalid_cover)
    if "vehicle" in ids:
        low, high = config.vehicle_area_m2
        _invalidate_components(result, ids["vehicle"], lambda area, _: not low <= area * gsd_m * gsd_m <= high)
    return result
