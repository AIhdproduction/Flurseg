"""Lazy, file-backed image pyramids used by the review canvas."""

from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from flurseg.io.atomic import atomic_save_jpg, atomic_save_png, atomic_write_json
from flurseg.paths import RepoPaths, ensure_inside

TILE_SIZE = 256
_GLOBAL_LOCK = threading.Lock()
_LAYER_LOCKS: dict[tuple[str, str], threading.Lock] = {}
_LEVEL_LOCKS: dict[tuple[str, str, int], threading.Lock] = {}
_TILE_LOCKS: dict[tuple[str, str, int, int, int], threading.Lock] = {}
_MASK_COLORS = np.array(
    [
        (158, 158, 158), (160, 82, 45), (92, 107, 192), (156, 204, 101), (27, 94, 32),
        (229, 57, 53), (255, 235, 59), (255, 152, 0), (0, 188, 212), (255, 255, 255),
        (224, 64, 251),
    ],
    dtype=np.uint8,
)
_RULE_COLORS = np.array(
    [(0, 0, 0), (96, 165, 250), (34, 197, 94), (251, 191, 36), (244, 114, 182), (167, 139, 250), (45, 212, 191)],
    dtype=np.uint8,
)


def _read(path: Path, flags: int) -> np.ndarray:
    value = cv2.imread(str(path), flags)
    if value is None:
        raise FileNotFoundError(path)
    return value


def _is_complete(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _get_level_lock(image_id: str, layer: str, level: int) -> threading.Lock:
    with _GLOBAL_LOCK:
        key = (image_id, layer, level)
        if key not in _LEVEL_LOCKS:
            _LEVEL_LOCKS[key] = threading.Lock()
        return _LEVEL_LOCKS[key]


def _get_layer_lock(image_id: str, layer: str) -> threading.Lock:
    with _GLOBAL_LOCK:
        key = (image_id, layer)
        if key not in _LAYER_LOCKS:
            _LAYER_LOCKS[key] = threading.Lock()
        return _LAYER_LOCKS[key]


def _get_tile_lock(image_id: str, layer: str, level: int, x: int, y: int) -> threading.Lock:
    with _GLOBAL_LOCK:
        key = (image_id, layer, level, x, y)
        if key not in _TILE_LOCKS:
            _TILE_LOCKS[key] = threading.Lock()
        return _TILE_LOCKS[key]


class Pyramid:
    """Generate 256-pixel image and overlay tiles on first access."""

    def __init__(self, paths: RepoPaths, image_id: str, work_dir: Path) -> None:
        self.paths = paths
        self.image_id = image_id
        self.work_dir = ensure_inside(work_dir, paths.root)
        self.directory = ensure_inside(paths.work_dir / image_id / "pyramid", paths.root)

    def _image_path(self) -> Path:
        return self.work_dir / "image.jpg"

    def image_shape(self) -> tuple[int, int]:
        image = _read(self._image_path(), cv2.IMREAD_COLOR)
        return image.shape[:2]

    def levels(self) -> int:
        height, width = self.image_shape()
        count = 1
        while max(height, width) > 512:
            height = max(1, (height + 1) // 2)
            width = max(1, (width + 1) // 2)
            count += 1
        return count

    def grid(self, level: int) -> tuple[int, int]:
        height, width = self._shape_for_level(level)
        return ((width + TILE_SIZE - 1) // TILE_SIZE, (height + TILE_SIZE - 1) // TILE_SIZE)

    def _shape_for_level(self, level: int) -> tuple[int, int]:
        if not 0 <= level < self.levels():
            raise ValueError("unknown pyramid level")
        height, width = self.image_shape()
        divisor = 2**level
        return (max(1, (height + divisor - 1) // divisor), max(1, (width + divisor - 1) // divisor))

    def _source_path(self, layer: str) -> Path:
        if layer == "image":
            return self._image_path()
        candidates = {
            "uncertainty": self.work_dir / "uncertainty.png",
            "auto_change": self.work_dir / "auto_change.png",
            "raw_mask": self.work_dir / "raw_mask.png",
        }
        if layer not in candidates:
            raise ValueError("unknown pyramid layer")
        return candidates[layer]

    def _source_signature(self, layer: str) -> dict[str, int]:
        stat = self._source_path(layer).stat()
        return {"mtime_ns": stat.st_mtime_ns, "size": stat.st_size}

    def _layer_directory(self, layer: str) -> Path:
        return ensure_inside(self.directory / layer, self.paths.root)

    @staticmethod
    def _read_source_metadata(path: Path) -> dict[str, object] | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    def _ensure_layer(self, layer: str) -> Path:
        layer_directory = self._layer_directory(layer)
        metadata_path = ensure_inside(layer_directory / "source.json", self.paths.root)
        signature = self._source_signature(layer)
        metadata = self._read_source_metadata(metadata_path)
        if metadata is not None and all(metadata.get(key) == value for key, value in signature.items()):
            version = metadata.get("version")
            if isinstance(version, str):
                active = ensure_inside(layer_directory / version, self.paths.root)
                if active.is_dir():
                    return active
        lock = _get_layer_lock(self.image_id, layer)
        with lock:
            signature = self._source_signature(layer)
            metadata = self._read_source_metadata(metadata_path)
            if metadata is not None and all(metadata.get(key) == value for key, value in signature.items()):
                version = metadata.get("version")
                if isinstance(version, str):
                    active = ensure_inside(layer_directory / version, self.paths.root)
                    if active.is_dir():
                        return active
            previous = metadata.get("version") if metadata is not None else None
            version = f"v-{signature['mtime_ns']}-{signature['size']}-{uuid4().hex}"
            active = ensure_inside(layer_directory / version, self.paths.root)
            active.mkdir(parents=True, exist_ok=False)
            atomic_write_json(metadata_path, {**signature, "version": version})
            if isinstance(previous, str) and previous != version:
                old = ensure_inside(layer_directory / previous, self.directory)
                if old.is_dir():
                    try:
                        shutil.rmtree(old)
                    except OSError:
                        pass
            return active

    def _layer_image(self, layer: str) -> tuple[np.ndarray, str]:
        if layer == "image":
            return cv2.cvtColor(_read(self._source_path(layer), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB), "jpg"
        source = _read(self._source_path(layer), cv2.IMREAD_UNCHANGED)
        if source.ndim == 3:
            source = source[..., 0]
        if layer == "uncertainty":
            return cv2.applyColorMap(source, cv2.COLORMAP_TURBO)[..., ::-1], "png"
        if layer == "raw_mask":
            output = np.zeros((*source.shape, 3), dtype=np.uint8)
            known = source <= 10
            output[known] = _MASK_COLORS[source[known]]
            ignored = source == 255
            rows, columns = np.indices(source.shape)
            output[ignored] = (30, 30, 30)
            output[ignored & ((rows + columns) % 10 < 2)] = (145, 145, 145)
            return output, "png"
        output = np.zeros((*source.shape, 3), dtype=np.uint8)
        valid = source < len(_RULE_COLORS)
        output[valid] = _RULE_COLORS[source[valid]]
        return output, "png"

    def _level_path(self, layer_directory: Path, level: int, extension: str) -> Path:
        return ensure_inside(layer_directory / "levels" / f"{level}.{extension}", self.paths.root)

    def _ensure_level(self, layer: str, level: int) -> tuple[Path, str]:
        extension = "jpg" if layer == "image" else "png"
        layer_directory = self._ensure_layer(layer)
        target = self._level_path(layer_directory, level, extension)
        if _is_complete(target):
            return target, extension
        lock = _get_level_lock(self.image_id, layer, level)
        with lock:
            if _is_complete(target):
                return target, extension
            source, extension = self._layer_image(layer)
            height, width = self._shape_for_level(level)
            if source.shape[:2] != (height, width):
                source = cv2.resize(source, (width, height), interpolation=cv2.INTER_AREA)
            try:
                if extension == "jpg":
                    atomic_save_jpg(target, source)
                else:
                    atomic_save_png(target, source)
            except PermissionError:
                if not _is_complete(target):
                    raise
            return target, extension

    def tile(self, layer: str, level: int, x: int, y: int) -> Path:
        if x < 0 or y < 0:
            raise ValueError("tile coordinates must be non-negative")
        level_path, extension = self._ensure_level(layer, level)
        layer_directory = level_path.parents[1]
        target = ensure_inside(
            layer_directory / "tiles" / str(level) / f"{x}_{y}.{extension}",
            self.paths.root,
        )
        if _is_complete(target):
            return target
        lock = _get_tile_lock(self.image_id, layer, level, x, y)
        with lock:
            if _is_complete(target):
                return target
            image = _read(level_path, cv2.IMREAD_COLOR)
            left, top = x * TILE_SIZE, y * TILE_SIZE
            tile = image[top : top + TILE_SIZE, left : left + TILE_SIZE]
            if tile.size == 0:
                raise FileNotFoundError("tile is outside the image")
            try:
                if extension == "jpg":
                    atomic_save_jpg(target, cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
                else:
                    atomic_save_png(target, cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
            except PermissionError:
                if not _is_complete(target):
                    raise
            return target
