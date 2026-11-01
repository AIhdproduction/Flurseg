"""Protocol shared by real and deterministic segmentation backends."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np


class SegmentationBackend(Protocol):
    """Dense prompt scorer used by the prelabel pipeline."""

    name: str

    def score_tile(self, tile: np.ndarray, prompts: Sequence[str]) -> np.ndarray:
        """Return float32 prompt scores shaped ``(prompts, height, width)``."""

    def score_boxes(
        self, image: np.ndarray, boxes: Sequence[tuple[int, int, int, int]], prompt: str
    ) -> np.ndarray:
        """Return a same-image visual-exemplar score map."""

    def close(self) -> None:
        """Release backend resources."""
