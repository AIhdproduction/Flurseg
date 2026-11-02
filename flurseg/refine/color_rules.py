"""RGB colour-evidence refinements for ground, paved surfaces, and low vegetation."""

from __future__ import annotations

from collections.abc import Mapping

import cv2
import numpy as np
from skimage.segmentation import slic


def compute_superpixels(image: np.ndarray, edge_config: object | None = None) -> np.ndarray:
    """Create fine SLIC labels when edge snapping did not provide stored labels."""
    segments_per_mp = int(getattr(edge_config, "slic_n_segments_per_mp", 1000))
    compactness = float(getattr(edge_config, "compactness", 10.0))
    count = max(1, round(image.shape[0] * image.shape[1] / 1_000_000 * segments_per_mp))
    return slic(image, n_segments=count, compactness=compactness, start_label=1, channel_axis=-1).astype(np.int32)


def _majority(mask: np.ndarray, labels: np.ndarray) -> np.ndarray:
    label_count = int(labels.max()) + 1
    valid = mask != 255
    encoded = labels[valid].astype(np.int64) * 256 + mask[valid]
    counts = np.bincount(encoded, minlength=label_count * 256).reshape(label_count, 256)
    return counts.argmax(axis=1).astype(np.uint8)


def _mean_by_label(values: np.ndarray, labels: np.ndarray, areas: np.ndarray) -> np.ndarray:
    """Return one mean value for every superpixel label."""
    return np.divide(
        np.bincount(labels.ravel(), weights=values.ravel(), minlength=len(areas)),
        areas,
        out=np.zeros(len(areas), dtype=np.float64),
        where=areas > 0,
    )


def _revert_small_regions(result: np.ndarray, original: np.ndarray, converted: np.ndarray, minimum: int) -> None:
    """Restore conversion groups that do not meet the configured area threshold."""
    count, components = cv2.connectedComponents(converted.astype(np.uint8), connectivity=8)
    for component_id in range(1, count):
        component = components == component_id
        if int(component.sum()) < minimum:
            result[component] = original[component]


def apply(
    mask: np.ndarray,
    image: np.ndarray,
    config: object,
    ids: dict[str, int],
    superpixels: np.ndarray | None = None,
    edge_config: object | None = None,
    class_scores: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Apply guarded colour evidence to ground, paved surfaces, and low vegetation only."""
    result = mask.copy()
    labels = compute_superpixels(image, edge_config) if superpixels is None else np.asarray(superpixels, dtype=np.int32)
    if labels.shape != mask.shape:
        raise ValueError("superpixels must match the mask shape")

    vegetation = ids.get("low_vegetation")
    ground = ids.get("ground")
    paved = ids.get("paved_surface")
    if vegetation is None or ground is None:
        return result

    label_count = int(labels.max()) + 1
    areas = np.bincount(labels.ravel(), minlength=label_count)
    rgb = image.astype(np.float32)
    chroma = rgb.sum(axis=2, keepdims=True)
    chromatic = np.divide(rgb, chroma, out=np.zeros_like(rgb), where=chroma > 0)
    exg = 2 * chromatic[..., 1] - chromatic[..., 0] - chromatic[..., 2]
    brightness = rgb.max(axis=2)
    mean_exg = _mean_by_label(exg, labels, areas)
    mean_brightness = _mean_by_label(brightness, labels, areas)
    bright = mean_brightness >= float(getattr(config, "min_brightness", 70))
    majority = _majority(result, labels)
    minimum_region_px = int(getattr(config, "min_region_px", 3000))

    surface_sources = [ground]
    if paved is not None:
        surface_sources.append(paved)
    target_labels = bright & (mean_exg >= float(getattr(config, "exg_threshold", 0.15)))
    vegetation_scores = None if class_scores is None else class_scores.get("low_vegetation")
    if vegetation_scores is not None:
        if vegetation_scores.shape != mask.shape:
            raise ValueError("low_vegetation scores must match the mask shape")
        mean_target_score = _mean_by_label(vegetation_scores, labels, areas)
        target_labels &= mean_target_score >= float(getattr(config, "min_target_score", 0.25))

    to_vegetation = np.isin(mask, surface_sources) & target_labels[labels]
    result[to_vegetation] = vegetation
    _revert_small_regions(result, mask, to_vegetation, minimum_region_px)

    to_ground_labels = (
        bright
        & (majority == vegetation)
        & (mean_exg < float(getattr(config, "exg_low_threshold", -0.02)))
    )
    to_ground = (mask == vegetation) & to_ground_labels[labels]
    result[to_ground] = ground
    _revert_small_regions(result, mask, to_ground, minimum_region_px)
    return result
