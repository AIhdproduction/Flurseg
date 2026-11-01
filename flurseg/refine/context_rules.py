"""Semantic neighbourhood constraints."""

from __future__ import annotations

import cv2
import numpy as np


def _component_slices(labels: np.ndarray, stats: np.ndarray, radius: int):
    for label in range(1, len(stats)):
        x, y, width, height, _ = stats[label]
        left, top = max(0, x - radius), max(0, y - radius)
        right, bottom = min(labels.shape[1], x + width + radius), min(labels.shape[0], y + height + radius)
        yield label, slice(top, bottom), slice(left, right)


def apply(
    mask: np.ndarray, config: object, ids: dict[str, int], paved_reference: np.ndarray | None = None
) -> np.ndarray:
    result = mask.copy()
    paved = result == ids.get("paved_surface", -1)
    if paved_reference is not None:
        if paved_reference.shape != mask.shape:
            raise ValueError("paved reference must match the mask shape")
        paved |= paved_reference
    if "road_marking" in ids:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (result == ids["road_marking"]).astype(np.uint8), connectivity=8
        )
        radius = int(getattr(config, "marking_ring_width_px", 3))
        ratio = float(getattr(config, "marking_paved_ring_ratio", 0.3))
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
        for label, rows, columns in _component_slices(labels, stats[:count], radius):
            component = labels[rows, columns] == label
            ring = cv2.dilate(component.astype(np.uint8), kernel).astype(bool) & ~component
            values = result[rows, columns][ring]
            valid = values != 255
            paved_or_marking = np.isin(values[valid], (ids["paved_surface"], ids["road_marking"]))
            if not valid.any() or paved_or_marking.mean() < ratio:
                result[rows, columns][component] = 255
    if "curb" in ids:
        kernel = np.ones((3, 3), dtype=np.uint8)
        border = cv2.dilate(paved.astype(np.uint8), kernel) != cv2.erode(paved.astype(np.uint8), kernel)
        distance = int(getattr(config, "curb_max_distance_px", 5))
        close = cv2.dilate(border.astype(np.uint8), np.ones((2 * distance + 1, 2 * distance + 1), np.uint8)) > 0
        count, labels, stats, _ = cv2.connectedComponentsWithStats((result == ids["curb"]).astype(np.uint8), connectivity=8)
        for label, rows, columns in _component_slices(labels, stats[:count], 0):
            component = labels[rows, columns] == label
            if not close[rows, columns][component].any():
                result[rows, columns][component] = 255
    if "vehicle" in ids and "building" in ids:
        vehicle = result == ids["vehicle"]
        count, labels, stats, _ = cv2.connectedComponentsWithStats(vehicle.astype(np.uint8), connectivity=8)
        radius = int(getattr(config, "vehicle_ring_width_px", 3))
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
        for label in range(1, count):
            x, y, width, height, _ = stats[label]
            margin = radius
            left, top = max(0, x - margin), max(0, y - margin)
            right, bottom = min(result.shape[1], x + width + margin), min(result.shape[0], y + height + margin)
            component = labels[top:bottom, left:right] == label
            ring = cv2.dilate(component.astype(np.uint8), kernel).astype(bool) & ~component
            if ring.any() and (result[top:bottom, left:right][ring] == ids["building"]).mean() >= float(getattr(config, "vehicle_on_building_ratio", 0.6)):
                result[top:bottom, left:right][component] = 255
    if "utility_cover" in ids:
        utility_cover = result == ids["utility_cover"]
        count, labels, stats, _ = cv2.connectedComponentsWithStats(utility_cover.astype(np.uint8), connectivity=8)
        radius = int(getattr(config, "utility_cover_ring_width_px", 5))
        ground_ids = [
            ids[name]
            for name in ("paved_surface", "ground", "low_vegetation", "road_marking", "curb")
            if name in ids
        ]
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
        for label in range(1, count):
            x, y, width, height, _ = stats[label]
            left, top = max(0, x - radius), max(0, y - radius)
            right, bottom = min(result.shape[1], x + width + radius), min(result.shape[0], y + height + radius)
            component = labels[top:bottom, left:right] == label
            ring = cv2.dilate(component.astype(np.uint8), kernel).astype(bool) & ~component
            values = result[top:bottom, left:right][ring]
            valid = values != 255
            ground_ratio = np.isin(values[valid], ground_ids).mean() if valid.any() else 0.0
            if ground_ratio < float(getattr(config, "utility_cover_ground_ratio", 0.5)):
                result[top:bottom, left:right][component] = 255
    return result
