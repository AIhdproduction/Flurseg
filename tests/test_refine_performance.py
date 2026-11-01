from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from flurseg.config import load_config
from flurseg.paths import RepoPaths
from flurseg.refine.edge_snap import apply as edge_snap
from flurseg.refine.registry import RefinementContext, refine


@pytest.mark.slow
def test_refinement_performance_on_20mp_blocky_input() -> None:
    """Keep the vectorised refinement path practical for full drone frames."""
    height, width = 3648, 5472
    rng = np.random.default_rng(12)
    image = cv2.resize(rng.integers(0, 256, (114, 171, 3), dtype=np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    values = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 255], dtype=np.uint8)
    mask = cv2.resize(rng.choice(values, (57, 86)).astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    labels = cv2.resize(np.arange(29 * 43, dtype=np.int32).reshape(29, 43) + 1, (width, height), interpolation=cv2.INTER_NEAREST)
    cfg = load_config(RepoPaths.from_root(Path(__file__).resolve().parents[1]))
    ids = {definition.name: definition.id for definition in cfg.classes.classes}

    started = time.perf_counter()
    edge_snap(mask, image, cfg.pipeline.refinement.edge_snap, ids, labels)
    edge_seconds = time.perf_counter() - started
    print(f"edge_snap_seconds={edge_seconds:.3f}")
    assert edge_seconds < 10

    started = time.perf_counter()
    result = refine(RefinementContext(image, mask, "ortho", 0.10, cfg.pipeline, ids, labels))
    full_seconds = time.perf_counter() - started
    print(f"full_refinement_seconds={full_seconds:.3f}")
    assert result.mask.shape == mask.shape
    assert full_seconds < 60
