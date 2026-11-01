"""Image normalization and camera metadata extraction."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

from flurseg.io.atomic import atomic_save_jpg

register_heif_opener()


def load_image(path: Path) -> np.ndarray:
    """Load an image, honour EXIF orientation, and return RGB uint8 pixels."""
    with Image.open(path) as opened:
        normalized = ImageOps.exif_transpose(opened).convert("RGB")
        return np.asarray(normalized, dtype=np.uint8).copy()


def save_canonical_jpg(path: Path, image: np.ndarray, quality: int = 95) -> None:
    """Store an RGB image using Flurseg's canonical JPEG settings."""
    atomic_save_jpg(path, image, quality)


def _rational(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if hasattr(value, "numerator") and hasattr(value, "denominator"):
            return float(value.numerator) / float(value.denominator)
        if isinstance(value, tuple) and len(value) == 2:
            return float(value[0]) / float(value[1])
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def read_image_metadata(path: Path) -> dict[str, float | str]:
    """Read EXIF and DJI XMP values relevant to GSD calculation."""
    metadata: dict[str, float | str] = {}
    with Image.open(path) as opened:
        exif = opened.getexif()
        tags = {
            "focal_length_mm": 37386,
            "focal_length_35mm": 41989,
            "focal_plane_x_resolution": 41486,
            "focal_plane_resolution_unit": 41488,
            "make": 271,
            "model": 272,
        }
        for name, tag in tags.items():
            value = exif.get(tag)
            if name in {"make", "model"}:
                if value:
                    metadata[name] = str(value)
            elif (converted := _rational(value)) is not None:
                metadata[name] = converted
        xmp = opened.info.get("xmp", b"")
        if isinstance(xmp, bytes):
            xmp = xmp.decode("utf-8", errors="ignore")
        match = re.search(r"drone-dji:RelativeAltitude=[\"']([^\"']+)", str(xmp))
        if match:
            try:
                metadata["relative_altitude_m"] = float(match.group(1))
            except ValueError:
                pass
    return metadata
