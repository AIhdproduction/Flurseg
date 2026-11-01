"""RGB colour-evidence refinements for surface and marking labels."""

from __future__ import annotations

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


def apply(
    mask: np.ndarray,
    image: np.ndarray,
    config: object,
    ids: dict[str, int],
    superpixels: np.ndarray | None = None,
    edge_config: object | None = None,
) -> np.ndarray:
    """Apply colour evidence per fine superpixel instead of to individual pixels."""
    result = mask.copy()
    labels = compute_superpixels(image, edge_config) if superpixels is None else np.asarray(superpixels, dtype=np.int32)
    if labels.shape != mask.shape:
        raise ValueError("superpixels must match the mask shape")
    rgb = image.astype(np.float32)
    chroma = rgb.sum(axis=2, keepdims=True)
    chromatic = np.divide(rgb, chroma, out=np.zeros_like(rgb), where=chroma > 0)
    exg = 2 * chromatic[..., 1] - chromatic[..., 0] - chromatic[..., 2]
    vegetation = ids.get("low_vegetation")
    surface_names = ("ground", "paved_surface", "low_vegetation")
    surface_ids = [ids[name] for name in surface_names if name in ids]
    label_count = int(labels.max()) + 1
    areas = np.bincount(labels.ravel(), minlength=label_count)
    mean_exg = np.divide(
        np.bincount(labels.ravel(), weights=exg.ravel(), minlength=label_count),
        areas,
        out=np.zeros(label_count, dtype=np.float64),
        where=areas > 0,
    )
    majority = _majority(result, labels)
    eligible = np.isin(majority, surface_ids)
    if vegetation is not None:
        green_labels = eligible & (mean_exg >= float(config.exg_threshold))
        result[np.isin(result, surface_ids) & green_labels[labels]] = vegetation
        ground = ids.get("ground")
        if ground is not None:
            neutral_vegetation = eligible & (majority == vegetation) & ~green_labels
            result[np.isin(result, surface_ids) & neutral_vegetation[labels]] = ground
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    brightness = hsv[..., 2].astype(np.float32)
    saturation = hsv[..., 1].astype(np.float32)
    paved = result == ids.get("paved_surface", -1)
    marking = ids.get("road_marking")
    if marking is not None and paved.any() and getattr(config, "marking_growth", False):
        mean_brightness = np.divide(
            np.bincount(labels.ravel(), weights=brightness.ravel(), minlength=label_count),
            areas,
            out=np.zeros(label_count, dtype=np.float64),
            where=areas > 0,
        )
        mean_saturation = np.divide(
            np.bincount(labels.ravel(), weights=saturation.ravel(), minlength=label_count),
            areas,
            out=np.zeros(label_count, dtype=np.float64),
            where=areas > 0,
        )
        majority = _majority(result, labels)
        candidate = (majority == ids.get("paved_surface", -1)) & (
            mean_saturation <= float(getattr(config, "marking_max_saturation", 40))
        )
        component_count, components = cv2.connectedComponents((result == marking).astype(np.uint8))
        dilation_kernel = np.ones((7, 7), np.uint8)
        eligible_growth = np.zeros(label_count, dtype=bool)
        for component_id in range(1, component_count):
            component = components == component_id
            dilated = cv2.dilate(component.astype(np.uint8), dilation_kernel) > 0
            ring = dilated & ~component & paved
            if not ring.any():
                continue
            overlap = np.bincount(labels[dilated], minlength=label_count)
            overlaps_half = overlap >= (areas * 0.5)
            ring_median = float(np.median(brightness[ring]))
            eligible_growth |= overlaps_half & (
                mean_brightness >= ring_median + float(getattr(config, "marking_min_contrast", 40))
            )
        candidate &= eligible_growth
        result[paved & candidate[labels]] = marking
    return result
