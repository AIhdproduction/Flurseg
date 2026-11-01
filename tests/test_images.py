"""Image normalization, HEIC support, and canonical JPEG coverage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, JpegImagePlugin

from flurseg.config import load_config
from flurseg.io.images import load_image, save_canonical_jpg
from flurseg.paths import RepoPaths


@pytest.mark.parametrize(
    ("orientation", "position"),
    [(3, (1, 2)), (6, (0, 1)), (8, (2, 0))],
    ids=["orientation-3", "orientation-6", "orientation-8"],
)
def test_load_image_places_marked_pixel_after_exif_orientation(
    tmp_path: Path, orientation: int, position: tuple[int, int]
) -> None:
    pixels = np.zeros((2, 3, 3), dtype=np.uint8)
    pixels[0, 0] = (255, 0, 0)
    path = tmp_path / f"orientation-{orientation}.jpg"
    exif = Image.Exif()
    exif[274] = orientation
    Image.fromarray(pixels).save(path, quality=100, subsampling=0, exif=exif)
    loaded = load_image(path)
    marked = np.unravel_index(np.argmax(loaded[..., 0].astype(int) - loaded[..., 1].astype(int)), loaded.shape[:2])
    assert marked == position


def test_load_image_converts_rgba_to_rgb(tmp_path: Path) -> None:
    Image.fromarray(np.full((3, 4, 4), 100, dtype=np.uint8)).save(tmp_path / "rgba.png")
    loaded = load_image(tmp_path / "rgba.png")
    assert loaded.shape == (3, 4, 3)
    assert loaded.dtype == np.uint8


def test_load_image_converts_greyscale_to_rgb(tmp_path: Path) -> None:
    Image.fromarray(np.full((3, 4), 100, dtype=np.uint8)).save(tmp_path / "greyscale.png")
    loaded = load_image(tmp_path / "greyscale.png")
    assert loaded.shape == (3, 4, 3)
    assert np.array_equal(loaded[..., 0], loaded[..., 1])
    assert np.array_equal(loaded[..., 1], loaded[..., 2])


def test_heic_round_trip_preserves_pixels_with_lossy_tolerance(tmp_path: Path) -> None:
    source = np.full((12, 13, 3), (20, 140, 220), dtype=np.uint8)
    path = tmp_path / "source.heic"
    Image.fromarray(source).save(path, format="HEIF", quality=95)
    restored = load_image(path)
    assert restored.shape == source.shape
    assert np.max(np.abs(restored.astype(int) - source.astype(int))) <= 15


def test_canonical_jpg_uses_config_quality_and_444_subsampling(tmp_root: Path) -> None:
    config = load_config(RepoPaths.from_root(tmp_root))
    image = np.full((16, 16, 3), (33, 88, 222), dtype=np.uint8)
    path = tmp_root / "canonical.jpg"
    save_canonical_jpg(path, image, quality=config.pipeline.ingest.jpeg_quality)
    with Image.open(path) as saved:
        assert JpegImagePlugin.get_sampling(saved) == 0
        assert saved.quantization[0][0] == 2
