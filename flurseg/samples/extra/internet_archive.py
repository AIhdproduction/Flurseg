"""Internet Archive movie retrieval with per-item licence verification."""

from __future__ import annotations

import fnmatch
import json
import logging
import re
import urllib.parse
from pathlib import Path
from typing import Any

from flurseg.io.atomic import atomic_write_bytes
from flurseg.samples.extra.common import FileProduct, request_bytes
from flurseg.samples.extra.commons_video_frames import extract_video_frames
from flurseg.samples.extra.config import InternetArchiveEntry
from flurseg.samples.extra.licences import is_acceptable_licence
from flurseg.samples.fetch import open_url_with_retry

logger = logging.getLogger(__name__)


def licence_from_url(value: str | None) -> str:
    """Map common Archive licence URLs to the policy's human-readable names."""
    text = (value or "").lower()
    if "publicdomain/zero/1.0" in text or "cc0" in text:
        return "CC0"
    if "creativecommons.org/licenses/by/" in text and not any(token in text for token in ("by-sa", "by-nd", "by-nc")):
        version = text.rstrip("/").rsplit("/", 1)[-1]
        return f"CC BY {version}" if version else "CC BY"
    if "publicdomain" in text:
        return "Public domain"
    return value or ""


def _json(url: str) -> dict[str, Any]:
    return json.loads(request_bytes(url, min_interval_s=0.5).decode("utf-8"))


def _identifiers(entry: InternetArchiveEntry) -> list[str]:
    query = entry.search_query or "mediatype:movies"
    if entry.creator:
        query = f'({query}) AND creator:"{entry.creator}"'
    rows = max(200, entry.max_videos * 5)

    def search(value: str) -> dict[str, Any]:
        params = urllib.parse.urlencode(
            {"q": value, "fl[]": "identifier", "rows": rows, "sort[]": "identifier asc", "output": "json"}
        )
        return _json(f"https://archive.org/advancedsearch.php?{params}")

    payload = search(query)
    if payload.get("error"):
        message = str(payload["error"])
        logger.warning("Internet Archive search for %s failed: %s", entry.id, message)
        fallback = re.sub(r"\s+AND\s+licenseurl:(?:\"[^\"]+\"|\S+)", "", query, flags=re.IGNORECASE)
        if fallback == query:
            return []
        logger.warning("Internet Archive search for %s is retrying without licenseurl filter", entry.id)
        payload = search(fallback)
        if payload.get("error"):
            logger.warning("Internet Archive fallback search for %s failed: %s", entry.id, payload["error"])
            return []
    docs = payload.get("response", {}).get("docs", [])
    values = [str(doc.get("identifier") or "") for doc in docs if isinstance(doc, dict)]
    if entry.identifier_glob:
        values = [identifier for identifier in values if fnmatch.fnmatchcase(identifier, entry.identifier_glob)]
    return sorted({identifier for identifier in values if identifier})


def _file(metadata: dict[str, Any], prefer: str) -> dict[str, Any] | None:
    files = [value for value in metadata.get("files", []) if isinstance(value, dict)]
    candidates: list[dict[str, Any]] = []
    for item in files:
        name = str(item.get("name") or "")
        fmt = str(item.get("format") or "").lower()
        if not name.lower().endswith((".mp4", ".m4v", ".mov", ".webm", ".avi")):
            continue
        width = int(item.get("width") or 0)
        if width and width > 1080:
            continue
        is_original = str(item.get("source") or "").lower() == "original"
        score = (1 if is_original == (prefer == "original") else 0, 1 if "h.264" in fmt or "mpeg4" in fmt else 0, width)
        candidates.append({**item, "_score": score})
    return max(candidates, key=lambda item: item["_score"], default=None)


def _download(url: str, target: Path, max_download_gb: float = 3.0) -> Path:
    if target.is_file():
        return target
    part = target.with_suffix(target.suffix + ".part")
    with open_url_with_retry(url, timeout=120) as response:
        content_length = getattr(response, "headers", {}).get("Content-Length")
        if content_length and int(content_length) > int(max_download_gb * 1024**3):
            raise ValueError(f"video exceeds max_download_gb={max_download_gb:g}")
        data = response.read()
    if len(data) > int(max_download_gb * 1024**3):
        raise ValueError(f"video exceeds max_download_gb={max_download_gb:g}")
    atomic_write_bytes(part, data)
    part.replace(target)
    return target


def products(entry: InternetArchiveEntry, cache_dir: Path, limit: int | None = None) -> list[FileProduct]:
    """Return frames from Archive movies whose own metadata licence passes the policy."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    result: list[FileProduct] = []
    total = 0
    identifiers = _identifiers(entry)
    if not identifiers:
        logger.warning("Internet Archive %s yielded 0 items: search returned no identifiers", entry.id)
        return result
    rejected_licences = missing_videos = 0
    for identifier in identifiers:
        if len({product.title for product in result}) >= entry.max_videos:
            break
        metadata = _json(f"https://archive.org/metadata/{identifier}")
        details = metadata.get("metadata", {})
        licence = licence_from_url(str(details.get("licenseurl") or ""))
        if not is_acceptable_licence(licence):
            rejected_licences += 1
            continue
        chosen = _file(metadata, entry.prefer)
        if not chosen:
            missing_videos += 1
            continue
        size = int(chosen.get("size") or 0)
        if size > int(entry.max_download_gb * 1024**3):
            logger.warning(
                "Internet Archive %s yielded 0 items: %s is %.2f GiB, above max_download_gb=%g",
                entry.id,
                identifier,
                size / 1024**3,
                entry.max_download_gb,
            )
            return []
        if total + size > int(entry.max_total_gb * 1024**3):
            break
        name = str(chosen["name"])
        url = f"https://archive.org/download/{identifier}/{urllib.parse.quote(name)}"
        path = _download(url, cache_dir / f"{identifier}_{Path(name).name}", entry.max_download_gb)
        total += size
        attribution = ", ".join(value for value in (str(details.get("creator") or ""), str(details.get("title") or identifier), identifier, licence) if value)
        result.extend(extract_video_frames(path, session=entry.session, title=identifier, url=url, licence=licence,
                                           attribution=attribution, interval_s=entry.frame_interval_s,
                                           max_frames=entry.max_frames_per_video))
        if limit is not None and len(result) >= limit:
            return result[:limit]
    if not result:
        logger.warning(
            "Internet Archive %s yielded 0 items: %d identifiers, %d rejected by licence, %d without usable video",
            entry.id,
            len(identifiers),
            rejected_licences,
            missing_videos,
        )
    return result


def plan_products(entry: InternetArchiveEntry, limit: int | None = None) -> int:
    """Return a metadata-only upper bound for Archive frame extraction."""
    count = entry.max_videos * entry.max_frames_per_video
    return min(count, limit) if limit is not None else count
