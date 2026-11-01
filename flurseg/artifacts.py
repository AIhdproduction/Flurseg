"""Read and write the fixed per-image work artifact set."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import cv2
import numpy as np

from flurseg.io.atomic import atomic_save_jpg, atomic_save_npz, atomic_save_png, atomic_write_json


class ArtifactStore:
    """Typed access to one ``work/<image_id>`` artifact directory."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def _path(self, name: str) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        return self.directory / name

    def write_image(self, image: np.ndarray, quality: int = 95) -> None:
        """Write normalized RGB image data."""
        atomic_save_jpg(self._path("image.jpg"), image, quality)

    def write_scores(
        self,
        scores: Mapping[str, np.ndarray],
        scales: Mapping[str, float],
        winners: Mapping[str, np.ndarray] | None = None,
        tta_std: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        """Store score maps as rounded uint8 plus their computed scales."""
        arrays: dict[str, np.ndarray] = {
            name: np.rint(np.clip(score, 0, 1) * 255).astype(np.uint8) for name, score in scores.items()
        }
        arrays.update({f"scale_{name}": np.asarray(value, dtype=np.float32) for name, value in scales.items()})
        arrays.update({f"winner_{name}": value.astype(np.uint8) for name, value in (winners or {}).items()})
        arrays.update({f"tta_std_{name}": value.astype(np.uint8) for name, value in (tta_std or {}).items()})
        atomic_save_npz(self._path("scores.npz"), **arrays)

    def read_scores(self) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        """Read uint8 disk scores back as float32 values in ``[0, 1]``."""
        with np.load(self.directory / "scores.npz") as data:
            scores = {
                name: data[name].astype(np.float32) / 255
                for name in data.files
                if not name.startswith(("scale_", "winner_", "tta_std_"))
            }
            scales = {name.removeprefix("scale_"): float(data[name]) for name in data.files if name.startswith("scale_")}
        return scores, scales

    def read_score_diagnostics(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Read class-local winner and TTA maps stored at their original inference scales."""
        with np.load(self.directory / "scores.npz") as data:
            winners = {name.removeprefix("winner_"): data[name].astype(np.uint8) for name in data.files if name.startswith("winner_")}
            tta_std = {name.removeprefix("tta_std_"): data[name].astype(np.uint8) for name in data.files if name.startswith("tta_std_")}
        return winners, tta_std

    def write_prompt_winner(self, array: np.ndarray) -> None:
        atomic_save_png(self._path("prompt_winner.png"), array)

    def write_tta_disagreement(self, array: np.ndarray) -> None:
        atomic_save_png(self._path("tta_disagreement.png"), array)

    def write_superpixels(self, labels: np.ndarray) -> None:
        atomic_save_npz(self._path("superpixels.npz"), labels=labels.astype(np.int32))

    def read_superpixels(self) -> np.ndarray:
        with np.load(self.directory / "superpixels.npz") as data:
            return data["labels"].astype(np.int32)

    def write_superpixels_coarse(self, labels: np.ndarray) -> None:
        atomic_save_npz(self._path("superpixels_coarse.npz"), labels=labels.astype(np.int32))

    def read_superpixels_coarse(self) -> np.ndarray | None:
        path = self.directory / "superpixels_coarse.npz"
        if not path.is_file():
            return None
        with np.load(path) as data:
            return data["labels"].astype(np.int32)

    def write_raw_mask(self, mask: np.ndarray) -> None:
        atomic_save_png(self._path("raw_mask.png"), mask)

    def write_proposed_mask(self, mask: np.ndarray) -> None:
        atomic_save_png(self._path("proposed_mask.png"), mask)

    def write_uncertainty(self, array: np.ndarray) -> None:
        atomic_save_png(self._path("uncertainty.png"), array)

    def write_auto_change(self, array: np.ndarray) -> None:
        atomic_save_png(self._path("auto_change.png"), array)

    def write_overlay(self, image: np.ndarray, mask: np.ndarray, colors: Mapping[int, str], quality: int = 95) -> None:
        """Write an RGB preview, drawing ignore pixels dark and subtly hatched."""
        overlay = image.astype(np.float32).copy()
        for class_id, color in colors.items():
            if class_id == 255:
                continue
            rgb = np.array([int(color[index : index + 2], 16) for index in (1, 3, 5)], dtype=np.float32)
            selected = mask == class_id
            overlay[selected] = overlay[selected] * 0.45 + rgb * 0.55
        ignored = mask == 255
        overlay[ignored] *= 0.25
        rows, columns = np.indices(mask.shape)
        overlay[ignored & ((rows + columns) % 8 < 2)] = 110
        atomic_save_jpg(self._path("overlay.jpg"), overlay.astype(np.uint8), quality)

    def write_meta(self, metadata: Mapping[str, object]) -> None:
        atomic_write_json(self._path("meta.json"), dict(metadata))

    def read_mask(self, name: str) -> np.ndarray:
        """Read one fixed grayscale work-mask artifact."""
        return cv2.imread(str(self.directory / name), cv2.IMREAD_UNCHANGED)

    def read_image(self) -> np.ndarray:
        """Read the canonical artifact image as RGB data."""
        image = cv2.imread(str(self.directory / "image.jpg"), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(self.directory / "image.jpg")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def upsample_scores(scores: Mapping[str, np.ndarray], shape: tuple[int, int]) -> dict[str, np.ndarray]:
    """Return full-resolution float32 score maps using bilinear interpolation."""
    height, width = shape
    return {
        name: cv2.resize(score, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        for name, score in scores.items()
    }
