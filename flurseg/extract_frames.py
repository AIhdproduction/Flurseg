"""Extract useful frames from the operator's own driving videos."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np
import yaml

from flurseg.ingest.session import SessionMetadata
from flurseg.io.atomic import atomic_write_bytes
from flurseg.paths import RepoPaths, ensure_inside
from flurseg.samples.extra.commons_video_frames import _hash
from flurseg.samples.fetch import sanitize_filename

_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
logger = logging.getLogger(__name__)


def video_paths(values: Iterable[Path]) -> list[Path]:
    """Expand video files and folders recursively in a stable order."""
    result: list[Path] = []
    for value in values:
        if value.is_file() and value.suffix.lower() in _VIDEO_SUFFIXES:
            result.append(value)
        elif value.is_dir():
            result.extend(path for path in value.rglob("*") if path.is_file() and path.suffix.lower() in _VIDEO_SUFFIXES)
    return sorted(set(result), key=lambda path: str(path).lower())


def _output_path(directory: Path, stem: str) -> Path:
    number = 0
    while True:
        suffix = "" if number == 0 else f"_{number}"
        result = directory / f"{stem}{suffix}.jpg"
        if not result.exists():
            return result
        number += 1


def _quality(frame: np.ndarray) -> tuple[float, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()), float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_frames(
    paths: RepoPaths,
    inputs: Iterable[Path],
    *,
    session: str,
    source: str = "phone",
    interval_s: float = 2.0,
    max_frames_per_video: int = 0,
    camera: str | None = None,
    date: str | None = None,
    area_type: str = "street",
    phash_threshold: int = 10,
    dry_run: bool = False,
) -> list[dict[str, object]]:
    """Extract full-resolution non-duplicate frames, or return a dry-run plan."""
    records: list[dict[str, object]] = []
    output_dir = paths.input_dir / source / session
    for path in video_paths(inputs):
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            logger.warning("Could not decode own video %s; skipping it.", path)
            records.append({"path": path, "status": "failed", "error": "could not decode video"})
            continue
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if not np.isfinite(fps) or fps <= 0:
            fps = 1.0 / interval_s
        duration = count / fps
        planned = max(1, int(duration / interval_s) + 1) if count else 0
        record: dict[str, object] = {"path": path, "duration": duration, "planned_frames": planned, "status": "dry_run"}
        if dry_run:
            capture.release()
            records.append(record)
            continue
        step = max(1, round(interval_s * fps))
        frames: list[tuple[int, np.ndarray, float, float]] = []
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % step == 0:
                brightness, sharpness = _quality(frame)
                frames.append((index, frame, brightness, sharpness))
            index += 1
        capture.release()
        median_sharpness = float(np.median([item[3] for item in frames])) if frames else 0.0
        previous: int | None = None
        written = 0
        slug = sanitize_filename(path.stem) or "video"
        for index, frame, brightness, sharpness in frames:
            if brightness < 25 or sharpness < 0.25 * median_sharpness:
                continue
            current = _hash(frame)
            if previous is not None and (previous ^ current).bit_count() <= phash_threshold:
                continue
            if max_frames_per_video and written >= max_frames_per_video:
                break
            output_dir.mkdir(parents=True, exist_ok=True)
            ensure_inside(output_dir, paths.input_dir)
            timestamp = index / fps
            target = _output_path(output_dir, f"{slug}_{timestamp:g}")
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                continue
            atomic_write_bytes(target, jpeg.tobytes())
            previous = current
            written += 1
        record.update({"status": "extracted", "written": written})
        records.append(record)
    if not dry_run and records and not (output_dir / "session.yaml").exists():
        metadata = SessionMetadata(camera=camera, date=date, area_type=area_type, test=False, notes="own footage")
        atomic_write_bytes(output_dir / "session.yaml", yaml.safe_dump(metadata.model_dump(exclude_none=True), sort_keys=False).encode("utf-8"))
    return records
