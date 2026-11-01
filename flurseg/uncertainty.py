"""Confidence, disagreement and refinement-change uncertainty scoring."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class UncertaintyResult:
    uncertainty: np.ndarray
    artifact: np.ndarray
    mask: np.ndarray


def calculate(
    mask: np.ndarray,
    top_score: np.ndarray,
    margin: np.ndarray,
    priority_conflict: np.ndarray,
    tta_disagreement: np.ndarray,
    auto_change: np.ndarray,
    config: object,
    priority_margin: float,
) -> UncertaintyResult:
    """Compute a normalized weighted uncertainty map and apply its ignore threshold."""
    if not all(array.shape == mask.shape for array in (top_score, margin, priority_conflict, tta_disagreement, auto_change)):
        raise ValueError("uncertainty inputs must share the mask shape")
    weights = config.weights
    raw = np.array(
        [weights.top_score, weights.margin, weights.priority_conflict, weights.tta, weights.rule_change], dtype=np.float32
    )
    if raw.sum() <= 0:
        raise ValueError("at least one uncertainty weight must be positive")
    raw /= raw.sum()
    disagreement = tta_disagreement.astype(np.float32)
    if disagreement.max(initial=0) > 1:
        disagreement /= 255
    uncertainty = (
        raw[0] * (1 - np.clip(top_score, 0, 1))
        + raw[1] * (1 - np.clip(margin, 0, 1))
        + raw[2] * np.clip(priority_conflict, 0, 1)
        + raw[3] * np.clip(disagreement, 0, 1)
        + raw[4] * (auto_change != 0)
    ).astype(np.float32)
    forced_conflict = (priority_conflict > 0) & (priority_conflict >= priority_margin)
    if forced_conflict.any():
        uncertainty = np.maximum(uncertainty, np.where(forced_conflict, np.nextafter(np.float32(config.threshold), np.float32(np.inf)), 0))
    output = mask.copy()
    uncertain = (uncertainty > config.threshold) | forced_conflict
    count, labels, stats, _ = cv2.connectedComponentsWithStats(uncertain.astype(np.uint8), connectivity=8)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= int(getattr(config, "min_region_px", 1)):
            output[labels == label] = 255
    output[mask == 255] = 255
    return UncertaintyResult(uncertainty, np.rint(np.clip(uncertainty, 0, 1) * 255).astype(np.uint8), output)
