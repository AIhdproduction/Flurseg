"""Scale-specific image and score helpers."""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np


def resize_image(image: np.ndarray, factor: float) -> np.ndarray:
    """Resize an RGB image with area interpolation when shrinking."""
    if factor <= 0:
        raise ValueError("factor must be positive")
    height, width = image.shape[:2]
    size = (max(1, round(width * factor)), max(1, round(height * factor)))
    interpolation = cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR
    return cv2.resize(image, size, interpolation=interpolation)


def resize_scores(scores: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize ``(classes, height, width)`` float scores with bilinear sampling."""
    height, width = shape
    return np.stack(
        [cv2.resize(score, (width, height), interpolation=cv2.INTER_LINEAR) for score in scores], axis=0
    ).astype(np.float32)


def class_scale_map(assignments: Iterable[object]) -> dict[str, float]:
    """Build the configured class-name to inference-scale lookup."""
    result: dict[str, float] = {}
    for assignment in assignments:
        factor = float(assignment.factor)
        for class_name in assignment.classes:
            if class_name in result:
                raise ValueError(f"class appears at multiple scales: {class_name}")
            result[class_name] = factor
    return result
