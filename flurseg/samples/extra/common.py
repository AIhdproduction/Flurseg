"""Shared HTTP, image, coordinate, and GeoTIFF helpers for extra sources."""

from __future__ import annotations

import hashlib
import io
import math
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from rasterio.io import MemoryFile
from rasterio.transform import Affine

from flurseg.io.atomic import atomic_write_bytes
from flurseg.samples.fetch import DEFAULT_TIMEOUT, USER_AGENT, open_url_with_retry

WINDOW_PIXELS = 2048
MIN_TILE_BYTES = 1024
_last_request_by_host: dict[str, float] = {}


@dataclass(slots=True)
class ImageProduct:
    """One rendered output image ready to be saved as a GeoTIFF."""

    data: np.ndarray
    transform: Affine
    crs: str
    gsd_m: float
    url: str
    title: str
    licence: str
    attribution: str


@dataclass(slots=True)
class FileProduct:
    """One ordinary image file ready to be saved without GeoTIFF conversion."""

    data: bytes
    filename: str
    gsd_m: float
    url: str
    title: str
    licence: str
    attribution: str
    metadata: dict[str, object] | None = None


def request_bytes(url: str, *, min_interval_s: float = 0.5, data: bytes | None = None) -> bytes:
    """Fetch a URL with the shared retry policy and per-host spacing."""
    host = urllib.parse.urlsplit(url).netloc.lower()
    now = time.monotonic()
    previous = _last_request_by_host.get(host)
    if previous is not None:
        delay = min_interval_s - (now - previous)
        if delay > 0:
            time.sleep(delay)
    _last_request_by_host[host] = time.monotonic()
    if data is None:
        with open_url_with_retry(url, headers={"User-Agent": USER_AGENT}, timeout=DEFAULT_TIMEOUT) as response:
            return response.read()
    with open_url_with_retry(
        url,
        headers={"User-Agent": USER_AGENT, "Content-Type": "application/json"},
        timeout=DEFAULT_TIMEOUT,
        method="POST",
        data=data,
    ) as response:
        return response.read()


def request_bytes_with_content_length(
    url: str, *, min_interval_s: float = 0.5, max_bytes: int | None = None
) -> tuple[bytes, int | None]:
    """Fetch a GET response and return its bytes with a declared Content-Length when present."""
    host = urllib.parse.urlsplit(url).netloc.lower()
    now = time.monotonic()
    previous = _last_request_by_host.get(host)
    if previous is not None:
        delay = min_interval_s - (now - previous)
        if delay > 0:
            time.sleep(delay)
    _last_request_by_host[host] = time.monotonic()
    with open_url_with_retry(url, headers={"User-Agent": USER_AGENT}, timeout=DEFAULT_TIMEOUT) as response:
        header = response.headers.get("Content-Length")
        try:
            content_length = int(header) if header else None
        except ValueError:
            content_length = None
        if max_bytes is not None and content_length is not None and content_length > max_bytes:
            raise ValueError(f"response Content-Length {content_length} exceeds configured limit {max_bytes}")
        payload = response.read()
        if max_bytes is not None and len(payload) > max_bytes:
            raise ValueError(f"response body {len(payload)} exceeds configured limit {max_bytes}")
        return payload, content_length


def decode_rgb(payload: bytes) -> np.ndarray:
    """Decode a server image to a three-channel uint8 array."""
    with Image.open(io.BytesIO(payload)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def is_empty_image(payload: bytes, image: np.ndarray) -> bool:
    """Reject tiny or predominantly blank/transparent tiles."""
    if len(payload) < MIN_TILE_BYTES:
        return True
    pixels = image.reshape(-1, image.shape[-1])
    pure_white = np.all(pixels >= 254, axis=1)
    pure_black = np.all(pixels <= 1, axis=1)
    return float(np.mean(pure_white | pure_black)) > 0.20


def is_low_information_image(image: np.ndarray) -> bool:
    """Reject ortho windows where the image carries almost no structure.

    Criteria:
    - Grayscale standard deviation < 8.0 on a 256-px downscale.
    - > 85% of pixels within a narrow colour cluster (distance < 15 in RGB) from the median colour.
    - > 20% pure/near white or pure/near black pixels.
    """
    if image.shape[0] != 256 or image.shape[1] != 256:
        # Nearest-neighbour sampling retains genuine texture and high-frequency
        # structure; averaging would turn random or textured imagery into a
        # falsely uniform window.
        small = Image.fromarray(image).resize((256, 256), Image.Resampling.NEAREST)
    else:
        small = Image.fromarray(image)

    gray = np.asarray(small.convert("L"), dtype=np.float32)
    if float(np.std(gray)) < 8.0:
        return True

    rgb = np.asarray(small, dtype=np.float32)
    pixels = rgb.reshape(-1, 3)
    median_rgb = np.median(pixels, axis=0)
    dists = np.linalg.norm(pixels - median_rgb, axis=1)
    if float(np.mean(dists < 15.0)) > 0.85:
        return True

    pure_white = np.all(pixels >= 254, axis=1)
    pure_black = np.all(pixels <= 1, axis=1)
    return float(np.mean(pure_white | pure_black)) > 0.20


def jitter_centre(lon: float, lat: float, gsd_m: float, attempt: int) -> tuple[float, float]:
    """Return deterministic candidate positions around a centre for jitter retry."""
    if attempt <= 0:
        return lon, lat
    step_m = WINDOW_PIXELS * gsd_m * 0.8
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = max(111320.0 * math.cos(math.radians(lat)), 1000.0)
    offsets = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)]
    dx, dy = offsets[(attempt - 1) % len(offsets)]
    return lon + (dx * step_m) / meters_per_deg_lon, lat + (dy * step_m) / meters_per_deg_lat


def ground_gsd_for_web_mercator(resolution_m: float, latitude: float, crs: str) -> float:
    """Convert a Web Mercator map-pixel resolution to ground metres per pixel."""
    if crs.upper() != "EPSG:3857":
        return resolution_m
    return resolution_m * math.cos(math.radians(latitude))


def sampled_centres(
    windows: list[tuple[float, float]] | None,
    bbox: tuple[float, float, float, float] | None,
    count: int | None,
) -> list[tuple[float, float]]:
    """Return explicit or deterministic grid-sampled WGS84 window centres."""
    if windows:
        return list(windows)
    if not bbox or not count:
        return []
    west, south, east, north = bbox
    columns = int(np.ceil(np.sqrt(count)))
    rows = int(np.ceil(count / columns))
    result: list[tuple[float, float]] = []
    for row in range(rows):
        for column in range(columns):
            if len(result) >= count:
                return result
            lon = west + (column + 0.5) * (east - west) / columns
            lat = south + (row + 0.5) * (north - south) / rows
            result.append((lon, lat))
    return result


def geotiff_bytes(product: ImageProduct) -> bytes:
    """Encode a 2048 px RGB product as a compressed, georeferenced GeoTIFF."""
    if product.data.shape != (WINDOW_PIXELS, WINDOW_PIXELS, 3):
        raise ValueError(f"expected a {WINDOW_PIXELS}x{WINDOW_PIXELS} RGB image, got {product.data.shape}")
    profile: dict[str, Any] = {
        "driver": "GTiff",
        "height": WINDOW_PIXELS,
        "width": WINDOW_PIXELS,
        "count": 3,
        "dtype": "uint8",
        "crs": product.crs,
        "transform": product.transform,
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "photometric": "RGB",
        "interleave": "pixel",
    }
    with MemoryFile() as memory:
        with memory.open(**profile) as dataset:
            dataset.write(np.moveaxis(product.data, -1, 0))
        return memory.read()


def choose_output_path(session_dir: Path, stem: str, content: bytes) -> tuple[Path, str, bool]:
    """Choose a non-destructive output path, returning path, SHA-256, and existing flag."""
    digest = hashlib.sha256(content).hexdigest()
    number = 0
    while True:
        suffix = "" if number == 0 else f"_{number}"
        candidate = session_dir / f"{stem}{suffix}.tif"
        if not candidate.exists():
            return candidate, digest, False
        existing = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if existing == digest:
            return candidate, digest, True
        number += 1


def save_product(session_dir: Path, stem: str, product: ImageProduct) -> tuple[Path, str, int, bool]:
    """Write a product atomically without replacing a different existing file."""
    content = geotiff_bytes(product)
    path, digest, already_exists = choose_output_path(session_dir, stem, content)
    if not already_exists:
        atomic_write_bytes(path, content)
    return path, digest, len(content), already_exists


def save_file_product(session_dir: Path, product: FileProduct) -> tuple[Path, str, int, bool]:
    """Write a downloaded phone image without overwriting a different file."""
    requested = Path(product.filename).name
    stem = Path(requested).stem or "image"
    suffix = Path(requested).suffix.lower() or ".jpg"
    digest = hashlib.sha256(product.data).hexdigest()
    number = 0
    while True:
        extra = "" if number == 0 else f"_{number}"
        candidate = session_dir / f"{stem}{extra}{suffix}"
        if not candidate.exists():
            atomic_write_bytes(candidate, product.data)
            return candidate, digest, len(product.data), False
        if hashlib.sha256(candidate.read_bytes()).hexdigest() == digest:
            return candidate, digest, len(product.data), True
        number += 1


def intersecting(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    """Return whether two WGS84 bboxes overlap."""
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]
