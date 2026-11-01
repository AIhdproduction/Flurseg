"""Deterministic image routing and atomic review/pool materialization."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from flurseg.io.atomic import atomic_save_jpg, atomic_save_png
from flurseg.paths import RepoPaths, is_inside


@dataclass(slots=True)
class RoutingDecision:
    status: str
    reason: str | None
    uncertain_fraction: float
    ignore_fraction: float
    classes_present: set[int]
    spot_check: bool


def _spot_check(image_hash: str, seed: int, fraction: float) -> bool:
    value = int.from_bytes(hashlib.sha256(f"{image_hash}{seed}".encode()).digest()[:8], "big") / 2**64
    return value < fraction


def decide(mask: np.ndarray, uncertainty: np.ndarray, image_hash: str, is_test_session: bool, config: object) -> RoutingDecision:
    """Choose a stable route based on threshold, test-session and deterministic sampling."""
    routing = getattr(config, "routing", config)
    uncertainty_config = getattr(config, "uncertainty", None)
    threshold = getattr(config, "threshold", getattr(uncertainty_config, "threshold", 0.5))
    uncertain_fraction = float((uncertainty > threshold).mean())
    ignore_fraction = float((mask == 255).mean())
    classes = {int(value) for value in np.unique(mask) if value != 255}
    if is_test_session:
        return RoutingDecision("needs_review", "test_session", uncertain_fraction, ignore_fraction, classes, False)
    if (
        uncertain_fraction > routing.auto_accept_max_uncertain_fraction
        or ignore_fraction > getattr(routing, "auto_accept_max_ignore_fraction", 0.10)
    ):
        return RoutingDecision("needs_review", "uncertain", uncertain_fraction, ignore_fraction, classes, False)
    sampled = _spot_check(image_hash, routing.seed, routing.spot_check_fraction)
    return RoutingDecision("needs_review" if sampled else "auto_accepted", "spot_check" if sampled else None, uncertain_fraction, ignore_fraction, classes, sampled)


def _inside(path: Path, root: Path) -> bool:
    return is_inside(path, root)


def materialize(paths: RepoPaths, image_id: str, image: np.ndarray, mask: np.ndarray, uncertainty: np.ndarray, decision: RoutingDecision) -> None:
    """Atomically write copies owned by either the pool or the review queue."""
    if decision.status == "auto_accepted":
        image_path, mask_path = paths.pool_images_dir / f"{image_id}.jpg", paths.pool_masks_dir / f"{image_id}.png"
        targets = (image_path, mask_path)
    else:
        directory = paths.review_dir / image_id
        targets = (directory / "image.jpg", directory / "proposed_mask.png", directory / "uncertainty.png")
        image_path, mask_path, uncertainty_path = targets
    if not all(is_inside(target, paths.root) for target in targets):
        raise ValueError("routing output must remain inside the repository root")
    atomic_save_jpg(image_path, image)
    atomic_save_png(mask_path, mask)
    if decision.status != "auto_accepted":
        atomic_save_png(uncertainty_path, uncertainty)
    _remove_stale_route(paths, image_id, decision.status)


def _remove_stale_route(paths: RepoPaths, image_id: str, status: str) -> None:
    """Remove only known stale copies after a successful atomic replacement in the new route."""
    if status == "auto_accepted":
        stale = tuple((paths.review_dir / image_id / name) for name in ("image.jpg", "proposed_mask.png", "uncertainty.png"))
    else:
        stale = (paths.pool_images_dir / f"{image_id}.jpg", paths.pool_masks_dir / f"{image_id}.png")
    for path in stale:
        if not is_inside(path, paths.root):
            raise ValueError("stale routing output must remain inside the repository root")
        path.unlink(missing_ok=True)
