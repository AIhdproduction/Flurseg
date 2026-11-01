"""Coverage for each atomic writer and its failure guarantees."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from flurseg.io import atomic


def test_atomic_write_bytes_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    atomic.atomic_write_bytes(path, b"flurseg\x00bytes")
    assert path.read_bytes() == b"flurseg\x00bytes"


def test_atomic_save_png_round_trip(tmp_path: Path) -> None:
    array = np.arange(12, dtype=np.uint8).reshape(3, 4)
    path = tmp_path / "mask.png"
    atomic.atomic_save_png(path, array)
    assert np.array_equal(np.asarray(Image.open(path)), array)


def test_atomic_save_jpg_round_trip(tmp_path: Path) -> None:
    image = np.full((8, 9, 3), (10, 100, 220), dtype=np.uint8)
    path = tmp_path / "image.jpg"
    atomic.atomic_save_jpg(path, image, quality=95)
    restored = np.asarray(Image.open(path).convert("RGB"))
    assert restored.shape == image.shape
    assert np.max(np.abs(restored.astype(int) - image.astype(int))) <= 3


def test_atomic_save_npz_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "arrays.npz"
    atomic.atomic_save_npz(path, integers=np.arange(4), scores=np.array([0.25], dtype=np.float32))
    with np.load(path) as stored:
        assert np.array_equal(stored["integers"], np.arange(4))
        assert np.array_equal(stored["scores"], np.array([0.25], dtype=np.float32))


def test_atomic_write_json_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "meta.json"
    atomic.atomic_write_json(path, {"items": [1, 2], "ok": True})
    assert json.loads(path.read_text(encoding="utf-8")) == {"items": [1, 2], "ok": True}


def test_atomic_writer_failure_preserves_previous_bytes_and_removes_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "payload.bin"
    previous = b"previous content must not change"
    path.write_bytes(previous)

    def fail_writer(stream: object) -> None:
        raise OSError("simulated write failure")

    original_replace = atomic._atomic_replace
    monkeypatch.setattr(atomic, "_atomic_replace", lambda target, writer: original_replace(target, fail_writer))
    with pytest.raises(OSError, match="simulated write failure"):
        atomic.atomic_write_bytes(path, b"new content")
    assert path.read_bytes() == previous
    assert not list(tmp_path.glob("*.tmp"))
