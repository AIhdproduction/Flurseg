"""SLIC-based boundary snapping with vectorised thin-class protection."""

from __future__ import annotations

import cv2
import numpy as np
from skimage.segmentation import slic


def apply(
    mask: np.ndarray,
    image: np.ndarray,
    config: object,
    ids: dict[str, int],
    superpixels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Snap non-protected pixels by superpixel majority without per-region masks."""
    if superpixels is None:
        count = max(1, round(image.shape[0] * image.shape[1] / 1_000_000 * config.slic_n_segments_per_mp))
        labels = slic(image, n_segments=count, compactness=config.compactness, start_label=1, channel_axis=-1)
    else:
        labels = np.asarray(superpixels, dtype=np.int32)
        if labels.shape != mask.shape:
            raise ValueError("stored superpixels must match the mask shape")
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(
        gray,
        int(getattr(config, "canny_low_threshold", 50)),
        int(getattr(config, "canny_high_threshold", 150)),
    ) > 0
    label_count = int(labels.max()) + 1
    valid = mask != 255
    encoded = labels[valid].astype(np.int64) * 256 + mask[valid]
    counts = np.bincount(encoded, minlength=label_count * 256).reshape(label_count, 256)
    totals = counts.sum(axis=1)
    majority = counts.argmax(axis=1).astype(np.uint8)
    edge_counts = np.bincount(labels[edges], minlength=label_count)
    areas = np.bincount(labels.ravel(), minlength=label_count)
    edge_fraction = np.divide(edge_counts, areas, out=np.zeros(label_count, dtype=np.float64), where=areas > 0)
    can_snap = (totals > 0) & (edge_fraction <= float(getattr(config, "max_edge_fraction", 0.35)))
    result = mask.copy()
    protected_ids = np.asarray([ids[name] for name in config.protected_classes if name in ids], dtype=np.uint8)
    protected = np.isin(mask, protected_ids) if protected_ids.size else np.zeros(mask.shape, dtype=bool)
    replace = can_snap[labels] & ~protected
    result[replace] = majority[labels[replace]]
    return result, labels.astype(np.int32, copy=False)
