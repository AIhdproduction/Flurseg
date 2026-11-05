"""Selected-image extraction from openly licensed dataset archives."""

from __future__ import annotations

import fnmatch
import io
import json
import logging
import os
import random
import re
import tarfile
import urllib.parse
import zipfile
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from flurseg.io.atomic import atomic_write_bytes
from flurseg.samples.extra.common import FileProduct, request_bytes, request_bytes_with_content_length
from flurseg.samples.extra.commons_video_frames import _hash
from flurseg.samples.extra.config import ArchiveDatasetEntry
from flurseg.samples.extra.licences import normalise_licence

logger = logging.getLogger(__name__)
_MIN_ARCHIVE_BYTES = 1024


class ArchiveDownloadError(ValueError):
    """An archive cache entry cannot safely be used."""


class ArchiveDownloadSkipped(ArchiveDownloadError):
    """An archive is intentionally skipped by the configured download limit."""


def _archive_kind(path: Path) -> str | None:
    """Recognise the archive signatures accepted by the dataset extractor."""
    try:
        with path.open("rb") as stream:
            prefix = stream.read(512)
    except OSError:
        return None
    if path.stat().st_size < _MIN_ARCHIVE_BYTES:
        return None
    if prefix.startswith(b"PK"):
        return "zip"
    if prefix.startswith(b"\x1f\x8b"):
        return "gzip"
    if prefix.startswith(b"BZh"):
        return "bzip2"
    if prefix.startswith(b"\xfd7zXZ\x00"):
        return "xz"
    if prefix[257:262] == b"ustar":
        return "tar"
    return None


def _invalid_preview(path: Path) -> str:
    try:
        with path.open("rb") as stream:
            payload = stream.read(200)
    except OSError:
        return "<unreadable>"
    return "".join(chr(byte) if 32 <= byte <= 126 else "." for byte in payload)


def _validate_archive(path: Path) -> str:
    kind = _archive_kind(path)
    if kind:
        return kind
    raise ArchiveDownloadError(
        f"invalid archive cache {path.name} ({path.stat().st_size if path.exists() else 0} bytes): "
        f"{_invalid_preview(path)}"
    )


def _max_download_bytes(max_download_gb: float) -> int:
    return int(max_download_gb * 1024**3)


def _json(url: str) -> dict[str, Any]:
    return json.loads(request_bytes(url, min_interval_s=0.5).decode("utf-8"))


def _resolved(entry: ArchiveDatasetEntry) -> tuple[list[tuple[str, int | None]], str]:
    if entry.url or entry.urls:
        return [(url, None) for url in ([entry.url] if entry.url else entry.urls)], entry.expected_licence
    if entry.figshare_article is not None:
        response = _json(f"https://api.figshare.com/v2/articles/{entry.figshare_article}")
        return [(str(item["download_url"]), int(item.get("size") or 0)) for item in response.get("files", [])], str(response.get("license", {}).get("name") or "")
    if entry.zenodo_record is not None:
        response = _json(f"https://zenodo.org/api/records/{entry.zenodo_record}")
        return [(str(item["links"]["self"]), int(item.get("size") or 0)) for item in response.get("files", [])], str(response.get("metadata", {}).get("license", {}).get("id") or response.get("metadata", {}).get("license", ""))
    response = _json(f"https://huggingface.co/api/datasets/{entry.hf_dataset}")
    licence = str(response.get("cardData", {}).get("license") or response.get("license") or "")
    siblings = response.get("siblings", [])
    return [(f"https://huggingface.co/datasets/{entry.hf_dataset}/resolve/main/{item['rfilename']}", None) for item in siblings if str(item.get("rfilename", "")).lower().endswith((".zip", ".tar", ".tar.gz", ".tgz"))], licence


def _download(url: str, target: Path, expected_size: int | None, max_download_gb: float) -> Path:
    """Download one archive to a part file and promote it only after validation."""
    limit = _max_download_bytes(max_download_gb)
    if expected_size is not None and expected_size > limit:
        raise ArchiveDownloadSkipped(
            f"archive {target.name} is {expected_size / 1024**3:.2f} GiB, above max_download_gb={max_download_gb:g}"
        )
    if target.is_file():
        try:
            if expected_size is not None and target.stat().st_size != expected_size:
                raise ArchiveDownloadError("cached size differs from metadata")
            _validate_archive(target)
            return target
        except ArchiveDownloadError as error:
            logger.warning("Ignoring invalid cached archive %s: %s", target.name, error)
    part = target.with_suffix(target.suffix + ".part")
    try:
        payload, content_length = request_bytes_with_content_length(url, min_interval_s=0.5, max_bytes=limit)
    except ValueError as error:
        raise ArchiveDownloadSkipped(f"archive {target.name} exceeds max_download_gb={max_download_gb:g}: {error}") from error
    if content_length is not None and content_length > limit:
        raise ArchiveDownloadSkipped(
            f"archive {target.name} is {content_length / 1024**3:.2f} GiB, above max_download_gb={max_download_gb:g}"
        )
    if len(payload) > limit:
        raise ArchiveDownloadSkipped(
            f"archive {target.name} exceeded max_download_gb={max_download_gb:g} while downloading"
        )
    atomic_write_bytes(part, payload)
    try:
        _validate_archive(part)
    except ArchiveDownloadError as error:
        logger.warning("Rejected downloaded archive %s: %s", target.name, error)
        raise
    os.replace(part, target)
    return target


def _members(path: Path) -> Iterable[tuple[str, bytes]]:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for info in sorted(archive.infolist(), key=lambda value: value.filename):
                member_path = Path(info.filename)
                if not info.is_dir() and not member_path.is_absolute() and ".." not in member_path.parts:
                    yield info.filename, archive.read(info)
    else:
        with tarfile.open(path) as archive:
            for member in sorted(archive.getmembers(), key=lambda value: value.name):
                member_path = Path(member.name)
                if (
                    member.isfile()
                    and not member_path.is_absolute()
                    and ".." not in member_path.parts
                    and (stream := archive.extractfile(member))
                ):
                    yield member.name, stream.read()


def _selected(path: str, subset: str | None) -> bool:
    if not path.lower().endswith((".jpg", ".jpeg", ".png")):
        return False
    is_regex = any(char in (subset or "") for char in "^$[]()\\") or ".*" in (subset or "")
    return not subset or bool(re.search(subset, path) if is_regex else fnmatch.fnmatch(path, subset))


def products(entry: ArchiveDatasetEntry, cache_dir: Path, limit: int | None = None) -> list[FileProduct]:
    """Extract a deterministic, licence-verified sample of archive images."""
    urls, actual = _resolved(entry)
    if normalise_licence(actual) != normalise_licence(entry.expected_licence):
        raise ValueError(f"dataset licence mismatch: expected {entry.expected_licence}, got {actual or 'unknown'}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[tuple[str, bytes]] = []
    for url, size in urls:
        path = _download(
            url,
            cache_dir / Path(urllib.parse.urlsplit(url).path).name,
            size,
            entry.max_download_gb,
        )
        candidates.extend((name, data) for name, data in _members(path) if _selected(name, entry.subset))
    rng = random.Random(entry.id)
    rng.shuffle(candidates)
    groups: Counter[str] = Counter()
    result: list[FileProduct] = []
    previous_hash: int | None = None
    maximum = min(entry.max_images, limit) if limit is not None else entry.max_images
    for name, data in candidates:
        try:
            image = Image.open(io.BytesIO(data)).convert("RGB")
        except (OSError, ValueError):
            continue
        if image.width < entry.min_width:
            continue
        group = Path(name).parts[0] if Path(name).parts else "root"
        if entry.max_per_group is not None and groups[group] >= entry.max_per_group:
            continue
        frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        current = _hash(frame)
        if previous_hash is not None and (previous_hash ^ current).bit_count() <= 8:
            continue
        previous_hash = current
        groups[group] += 1
        result.append(FileProduct(data, f"{entry.session}_{len(result):05d}{Path(name).suffix.lower()}", 1.0, "archive://" + name,
                                  name, entry.expected_licence, entry.attribution))
        if len(result) >= maximum:
            break
    return result


def plan_products(entry: ArchiveDatasetEntry, limit: int | None = None) -> int:
    return min(entry.max_images, limit) if limit is not None else entry.max_images
