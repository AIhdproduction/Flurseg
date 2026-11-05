"""Frame extraction from openly licensed Wikimedia Commons videos."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy.fft import dctn

from flurseg.io.atomic import atomic_write_bytes
from flurseg.samples.extra.common import FileProduct, request_bytes
from flurseg.samples.extra.commons_categories import _artist, _info, _licence, commons_files
from flurseg.samples.extra.config import CommonsVideoFramesEntry
from flurseg.samples.extra.licences import is_acceptable_licence
from flurseg.samples.fetch import sanitize_filename

logger = logging.getLogger(__name__)


def _video_url(info: dict[str, Any]) -> str:
    candidates = [item for item in info.get("derivatives", []) if isinstance(item, dict)]
    candidates.extend([info])
    suitable: list[tuple[int, str]] = []
    for candidate in candidates:
        url = str(candidate.get("src") or candidate.get("url") or "")
        width = int(candidate.get("width") or 0)
        if url and (not width or width <= 1080):
            suitable.append((width or 720, url))
    return max(suitable, default=(0, ""))[1]


def _accepted(entry: CommonsVideoFramesEntry) -> list[tuple[dict[str, Any], dict[str, Any], str]]:
    videos: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    cap = max(entry.max_videos * 5, entry.max_videos)
    for page in commons_files(
        entry.categories,
        depth=entry.depth,
        cap=cap,
        exclude_categories=entry.exclude_categories,
        thumb_width=None,
        video=True,
    ):
        info = _info(page)
        if not info or str(info.get("mime", "")).lower() not in {"video/webm", "video/ogg", "video/ogv"}:
            continue
        licence = _licence(info)
        if not is_acceptable_licence(licence) or not _video_url(info):
            continue
        videos.append((page, info, licence))
        if len(videos) >= entry.max_videos:
            break
    return videos


def plan_products(entry: CommonsVideoFramesEntry, limit: int | None = None) -> int:
    """Return an upper-bound frame count using only Commons metadata requests."""
    count = len(_accepted(entry)) * entry.max_frames_per_video
    return min(count, limit) if limit is not None else count


def _cache_path(cache_dir: Path, page: dict[str, Any], url: str) -> Path:
    title = str(page.get("title") or hashlib.sha256(url.encode("utf-8")).hexdigest()[:12])
    filename = sanitize_filename(title).rsplit(".", 1)[0] + Path(url).suffix.lower()
    return cache_dir / filename


def _hash(frame: np.ndarray) -> int:
    """Return the same DCT pHash representation used by ingest deduplication."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hash_image = Image.fromarray(rgb).resize((32, 32), Image.Resampling.LANCZOS)
    coefficients = dctn(np.asarray(hash_image, dtype=np.float32), type=2, norm="ortho")[:8, :9]
    values = coefficients.flatten()[1:65]
    median = float(np.median(values))
    value = 0
    for coefficient in values:
        value = (value << 1) | int(coefficient >= median)
    return value


def _usable(frame: np.ndarray) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()) >= 30 and float(cv2.Laplacian(gray, cv2.CV_64F).var()) >= 12


def _extract(
    entry: CommonsVideoFramesEntry, path: Path, page: dict[str, Any], info: dict[str, Any], licence: str
) -> list[FileProduct]:
    return extract_video_frames(
        path,
        session=entry.session,
        title=str(page.get("title") or path.stem),
        url=str(info.get("descriptionurl") or info.get("url") or ""),
        licence=licence,
        attribution=f"{_artist(info) or 'Wikimedia Commons contributor'}, {licence}, via Wikimedia Commons",
        interval_s=entry.frame_interval_s,
        max_frames=entry.max_frames_per_video,
    )


def extract_video_frames(
    path: Path,
    *,
    session: str,
    title: str,
    url: str,
    licence: str,
    attribution: str,
    interval_s: float = 2.0,
    max_frames: int = 40,
    phash_threshold: int = 10,
    quality: int = 92,
) -> list[FileProduct]:
    """Extract non-dark, non-blurry, visually distinct JPEG frames from a video."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        logger.warning("Could not decode Commons video %s; skipping it.", path.name)
        return []
    slug = sanitize_filename(title).rsplit(".", 1)[0]
    products: list[FileProduct] = []
    previous_hash: int | None = None
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 1.0 / interval_s
    sample_step = max(1, round(interval_s * fps))
    frame_index = 0
    next_sample_index = 0
    try:
        while len(products) < max_frames:
            success, frame = capture.read()
            if not success:
                break
            if frame_index != next_sample_index:
                frame_index += 1
                continue
            timestamp = frame_index / fps
            next_sample_index += sample_step
            current_hash = _hash(frame)
            if _usable(frame) and (previous_hash is None or (previous_hash ^ current_hash).bit_count() > phash_threshold):
                encoded, data = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                if encoded:
                    products.append(
                        FileProduct(
                            data=data.tobytes(),
                            filename=f"{session}_{slug}_{timestamp:g}-seconds.jpg",
                            gsd_m=1.0,
                            url=url,
                            title=title,
                            licence=licence,
                            attribution=attribution,
                        )
                    )
                    previous_hash = current_hash
            frame_index += 1
    finally:
        capture.release()
    return products


def products(entry: CommonsVideoFramesEntry, cache_dir: Path, limit: int | None = None) -> list[FileProduct]:
    """Cache videos outside input and return good, non-duplicate JPEG frames."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: list[FileProduct] = []
    for page, info, licence in _accepted(entry):
        url = _video_url(info)
        cache_path = _cache_path(cache_dir, page, url)
        if not cache_path.exists():
            atomic_write_bytes(cache_path, request_bytes(url, min_interval_s=1.0))
        results.extend(_extract(entry, cache_path, page, info, licence))
        if limit is not None and len(results) >= limit:
            return results[:limit]
    return results
