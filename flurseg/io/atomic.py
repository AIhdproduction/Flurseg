"""Atomic file writers that never expose partial outputs."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _atomic_replace(path: Path, write: Any) -> None:
    """Write a sibling temporary file and atomically replace *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomically write bytes to *path*."""
    _atomic_replace(path, lambda stream: stream.write(data))


def atomic_save_png(path: Path, mask_or_image: np.ndarray) -> None:
    """Atomically save an array as a lossless PNG."""
    _atomic_replace(path, lambda stream: Image.fromarray(mask_or_image).save(stream, format="PNG"))


def atomic_save_jpg(path: Path, image: np.ndarray, quality: int = 95) -> None:
    """Atomically save a canonical RGB JPEG."""
    if not 1 <= quality <= 100:
        raise ValueError("quality must be between 1 and 100")
    _atomic_replace(
        path,
        lambda stream: Image.fromarray(image).convert("RGB").save(
            stream, format="JPEG", quality=quality, subsampling=0
        ),
    )


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    """Atomically save compressed NumPy arrays."""
    _atomic_replace(path, lambda stream: np.savez_compressed(stream, **arrays))


def atomic_write_json(path: Path, content: Any) -> None:
    """Atomically write indented UTF-8 JSON."""
    atomic_write_bytes(path, json.dumps(content, indent=2, sort_keys=True).encode("utf-8") + b"\n")
