"""Public sample dataset fetcher and lock manager."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import rasterio
import yaml
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

from flurseg import __version__
from flurseg.ingest.session import SessionMetadata
from flurseg.io.atomic import atomic_write_bytes
from flurseg.io.hashing import sha256_file
from flurseg.paths import RepoPaths, ensure_inside

logger = logging.getLogger(__name__)

USER_AGENT = f"Flurseg/{__version__} (https://github.com/AIhdproduction/Flurseg)"
DEFAULT_TIMEOUT = 60.0
DEFAULT_RETRIES = 3
WIKIMEDIA_REQUEST_INTERVAL = 1.0
_last_wikimedia_request_at: float | None = None


def _is_acceptable_commons_license(license_str: str) -> bool:
    """Check whether a license string matches CC0, CC-BY, CC-BY-SA, or public domain."""
    normalized = license_str.lower().strip()
    # Explicitly exclude non-commercial and no-derivatives variants
    if any(forbidden in normalized for forbidden in ("-nc", " nc", "noncommercial", "non-commercial", "-nd", " nd", "noderiv")):
        return False
    if any(token in normalized for token in ("cc0", "cc-0", "cc zero")):
        return True
    if any(token in normalized for token in ("public domain", "publicdomain", "pd-self", "pd-old", "pd-us")) or normalized == "pd":
        return True
    return any(token in normalized for token in ("cc-by", "cc by", "creative commons attribution"))


def _is_wikimedia_url(url: str) -> bool:
    host = urllib.parse.urlsplit(url).hostname or ""
    return host.lower() in {"commons.wikimedia.org", "upload.wikimedia.org"}


def _wait_for_wikimedia_slot(url: str) -> None:
    """Keep all Commons API and upload requests at or below one request per second."""
    global _last_wikimedia_request_at
    if not _is_wikimedia_url(url):
        return
    now = time.monotonic()
    if _last_wikimedia_request_at is not None:
        wait_seconds = WIKIMEDIA_REQUEST_INTERVAL - (now - _last_wikimedia_request_at)
        if wait_seconds > 0:
            time.sleep(wait_seconds)
    _last_wikimedia_request_at = time.monotonic()


def _retry_after_seconds(value: str | None) -> float | None:
    """Parse a Retry-After header expressed as seconds or an HTTP date."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, IndexError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())


def _clean_text(html_text: str | None) -> str:
    """Strip HTML markup and collapse whitespace from metadata text."""
    if not html_text:
        return ""
    stripped = re.sub(r"<[^>]+>", " ", html_text)
    return " ".join(stripped.split()).strip()


def sanitize_filename(
    name: str,
    existing: set[str] | Sequence[str] | None = None,
    *,
    existing_names: set[str] | Sequence[str] | None = None,
    max_length: int = 120,
) -> str:
    """Sanitize a filename for Windows compatibility with ASCII transliteration and de-duplication."""
    raw = str(name).strip()
    if raw.startswith(("File:", "file:")):
        raw = raw.split(":", 1)[1].strip()

    # Strip URL query string and fragment
    if "?" in raw:
        raw = raw.split("?", 1)[0]
    if "#" in raw:
        raw = raw.split("#", 1)[0]

    # If raw is a URL or file path, take only the basename after unquoting
    unquoted = urllib.parse.unquote(raw)
    raw = Path(unquoted).name or unquoted

    stem, ext = os.path.splitext(raw)
    ext = ext.lower()
    if ext:
        ext = re.sub(r"[^A-Za-z0-9._-]", "", ext)

    # Transliterate German sharp s (ß/ẞ) which NFKD does not decompose into ASCII
    stem = stem.replace("ß", "ss").replace("ẞ", "SS")
    # NFKD separates accents from base letters (e.g. č -> c + caron, ü -> u + diaeresis)
    stem = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode("ascii")

    # Replace characters outside [A-Za-z0-9._-] with '_'
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", stem)
    # Collapse repeated underscores
    stem = re.sub(r"_+", "_", stem)
    stem = stem.strip("._") or "sample"

    max_stem_len = max(1, max_length - len(ext))
    stem = stem[:max_stem_len].rstrip("._") or "sample"
    candidate = f"{stem}{ext}"

    collision_pool: set[str] | None = None
    if existing is not None:
        collision_pool = existing if isinstance(existing, set) else set(existing)
    elif existing_names is not None:
        collision_pool = existing_names if isinstance(existing_names, set) else set(existing_names)

    if collision_pool is not None:
        if candidate in collision_pool:
            counter = 2
            while True:
                suffix = f"_{counter}"
                avail_stem_len = max(1, max_length - len(ext) - len(suffix))
                base = stem[:avail_stem_len].rstrip("._") or "sample"
                candidate = f"{base}{suffix}{ext}"
                if candidate not in collision_pool:
                    break
                counter += 1

        if isinstance(existing, set):
            existing.add(candidate)
        elif isinstance(existing_names, set):
            existing_names.add(candidate)

    return candidate


_sanitize_filename = sanitize_filename


def open_url_with_retry(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    method: str | None = None,
) -> Any:
    """Send an HTTP request with exponential backoff and timeout handling."""
    request_headers = {"User-Agent": USER_AGENT}
    if headers:
        request_headers.update(headers)

    request = urllib.request.Request(url, headers=request_headers, method=method)
    last_error: Exception | None = None

    maximum_retries = min(retries, DEFAULT_RETRIES)
    for attempt in range(maximum_retries + 1):
        try:
            _wait_for_wikimedia_slot(url)
            return urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as error:
            last_error = error
            if attempt >= maximum_retries:
                raise
            if _is_wikimedia_url(url) and error.code in {429, 503}:
                delay = _retry_after_seconds(error.headers.get("Retry-After"))
                time.sleep(delay if delay is not None else 5.0 * (2 ** attempt))
            else:
                time.sleep(0.1 * (2 ** attempt))
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = error
            if attempt < maximum_retries:
                time.sleep(0.1 * (2 ** attempt))
            else:
                raise

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Request failed for {url}")


def probe_file_size(url: str, *, timeout: float = DEFAULT_TIMEOUT) -> int | None:
    """Determine remote file size using HTTP HEAD without downloading."""
    try:
        with open_url_with_retry(url, timeout=timeout, retries=1, method="HEAD") as response:
            header_value = response.headers.get("Content-Length")
            if header_value is not None:
                return int(header_value)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
        logger.warning("Failed to probe file size for %s: %s", url, error)
    return None


def resolve_github_dir(
    entry: dict[str, Any],
    *,
    timeout: float = DEFAULT_TIMEOUT,
) -> list[dict[str, Any]]:
    """Resolve file names in a GitHub directory using the GitHub Contents API."""
    repo = entry["repo"]
    path = entry["path"].strip("/")
    count = int(entry.get("count", 8))
    stride = max(1, int(entry.get("stride", 1)))

    api_url = f"https://api.github.com/repos/{repo}/contents/{path}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    with open_url_with_retry(api_url, headers=headers, timeout=timeout) as response:
        raw_json = response.read().decode("utf-8")
        items = json.loads(raw_json)

    if not isinstance(items, list):
        raise TypeError(f"GitHub contents API returned non-list payload: {items}")

    jpg_items = [
        item for item in items
        if isinstance(item, dict)
        and item.get("type", "file") == "file"
        and item.get("name", "").lower().endswith((".jpg", ".jpeg"))
    ]
    jpg_items.sort(key=lambda item: item["name"])

    selected = jpg_items[::stride][:count]
    resolved: list[dict[str, Any]] = []
    used_names: set[str] = set()
    for item in selected:
        raw_name = item["name"]
        sanitized_name = sanitize_filename(raw_name, used_names)
        download_url = item.get("download_url") or f"https://raw.githubusercontent.com/{repo}/master/{path}/{raw_name}"
        resolved.append({
            "source": entry.get("source", "drone"),
            "session": entry["session"],
            "filename": sanitized_name,
            "url": download_url,
            "title": raw_name,
            "license": entry.get("license", "CC0-1.0"),
            "attribution": entry.get("attribution", "OpenDroneMap"),
            "session_yaml": entry.get("session_yaml", {}),
            "size": item.get("size"),
        })
    return resolved


def resolve_commons_category(
    entry: dict[str, Any],
    *,
    timeout: float = DEFAULT_TIMEOUT,
) -> list[dict[str, Any]]:
    """Resolve files in a Wikimedia Commons category matching license and width criteria."""
    category = entry["category"]
    count = int(entry.get("count", 8))
    min_width = int(entry.get("min_width", 2000))

    category_title = category if category.startswith("Category:") else f"Category:{category}"
    parameters = {
        "action": "query",
        "generator": "categorymembers",
        "gcmtitle": category_title,
        "gcmtype": "file",
        "gcmlimit": "50",
        "prop": "imageinfo",
        "iiprop": "url|size|extmetadata",
        "format": "json",
    }
    query_string = urllib.parse.urlencode(parameters)
    api_url = f"https://commons.wikimedia.org/w/api.php?{query_string}"

    with open_url_with_retry(api_url, timeout=timeout) as response:
        raw_json = response.read().decode("utf-8")
        payload = json.loads(raw_json)

    pages_dict = payload.get("query", {}).get("pages", {})
    pages = sorted(pages_dict.values(), key=lambda page: str(page.get("title", "")))

    resolved: list[dict[str, Any]] = []
    used_names: set[str] = set()
    for page in pages:
        imageinfo = page.get("imageinfo", [])
        if not imageinfo:
            continue
        info = imageinfo[0]
        width = int(info.get("width") or 0)
        if width < min_width:
            continue

        extmetadata = info.get("extmetadata", {})
        license_code = extmetadata.get("License", {}).get("value", "")
        license_short = extmetadata.get("LicenseShortName", {}).get("value", "")
        usage_terms = extmetadata.get("UsageTerms", {}).get("value", "")
        license_summary = f"{license_code} {license_short} {usage_terms}".strip()

        if not _is_acceptable_commons_license(license_summary):
            continue

        author_raw = extmetadata.get("Artist", {}).get("value") or extmetadata.get("Credit", {}).get("value") or ""
        author = _clean_text(author_raw) or "Wikimedia Commons contributor"
        license_name = _clean_text(license_short) or _clean_text(license_code) or "Commons open license"

        file_url = info.get("url", "")
        file_page_url = (
            info.get("descriptionurl")
            or info.get("descriptionshorturl")
            or f"https://commons.wikimedia.org/wiki/{urllib.parse.quote(page.get('title', ''))}"
        )

        page_title = str(page.get("title") or "")
        raw_name = ""
        if page_title:
            if page_title.startswith(("File:", "file:")):
                raw_name = page_title.split(":", 1)[1].strip()
            else:
                raw_name = page_title.strip()
        if not raw_name:
            path_part = urllib.parse.urlsplit(file_url).path
            raw_name = Path(urllib.parse.unquote(path_part)).name

        filename = sanitize_filename(raw_name, used_names)
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
            filename = f"{filename}.jpg"
            if filename in used_names:
                filename = sanitize_filename(filename, used_names)
        used_names.add(filename)

        resolved.append({
            "source": entry.get("source", "phone"),
            "session": entry["session"],
            "filename": filename,
            "url": file_url,
            "title": page_title or raw_name,
            "file_page_url": file_page_url,
            "license": license_name,
            "attribution": author,
            "session_yaml": entry.get("session_yaml", {}),
            "size": info.get("size"),
            "width": width,
            "height": info.get("height"),
        })
        if len(resolved) >= count:
            break

    return resolved


def download_file(
    url: str,
    output_path: Path,
    *,
    expected_sha256: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> tuple[int, str]:
    """Download a remote URL to output_path atomically with SHA-256 verification."""
    target_dir = output_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    temp_file = target_dir / f".{output_path.name}.tmp"

    try:
        with open_url_with_retry(url, timeout=timeout, retries=retries) as response:
            digest = hashlib.sha256()
            with temp_file.open("wb") as stream:
                while chunk := response.read(64 * 1024):
                    stream.write(chunk)
                    digest.update(chunk)

        actual_sha = digest.hexdigest()
        if expected_sha256 and actual_sha.lower() != expected_sha256.lower():
            temp_file.unlink(missing_ok=True)
            raise ValueError(
                f"SHA-256 mismatch for {url}: expected {expected_sha256}, got {actual_sha}"
            )

        file_size = temp_file.stat().st_size
        os.replace(temp_file, output_path)
        return file_size, actual_sha
    except BaseException:
        temp_file.unlink(missing_ok=True)
        raise


def fetch_ortho_window(
    source_url_or_path: str,
    output_path: Path,
    window_coordinates: Sequence[int],
    *,
    expected_sha256: str | None = None,
) -> tuple[int, str]:
    """Extract a window from an orthophoto (remote COG or local file) and save as GeoTIFF."""
    if len(window_coordinates) != 4:
        raise ValueError(f"ortho window must be [col_off, row_off, width, height], got {window_coordinates}")

    col_off, row_off, width, height = window_coordinates
    window = Window(col_off, row_off, width, height)

    if source_url_or_path.startswith(("http://", "https://")):
        cog_source = f"/vsicurl/{source_url_or_path}"
    else:
        cog_source = str(source_url_or_path)

    target_dir = output_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    temp_file = target_dir / f".{output_path.name}.tmp"

    try:
        with rasterio.open(cog_source) as src:
            data = src.read(window=window)
            sub_transform = window_transform(window, src.transform)
            crs = src.crs

            profile = {
                "driver": "GTiff",
                "height": int(height),
                "width": int(width),
                "transform": sub_transform,
                "crs": crs,
                "compress": "deflate",
                "predictor": 2,
                "tiled": True,
                "blockxsize": 512,
                "blockysize": 512,
                "photometric": "RGB",
                "interleave": "pixel",
                "count": 3,
                "dtype": "uint8",
            }

        with rasterio.open(temp_file, "w", **profile) as dst:
            dst.write(data)

        actual_sha = sha256_file(temp_file)
        if expected_sha256 and actual_sha.lower() != expected_sha256.lower():
            temp_file.unlink(missing_ok=True)
            raise ValueError(
                f"SHA-256 mismatch for ortho window: expected {expected_sha256}, got {actual_sha}"
            )

        file_size = temp_file.stat().st_size
        os.replace(temp_file, output_path)
        return file_size, actual_sha
    except BaseException:
        temp_file.unlink(missing_ok=True)
        raise


def _format_session_notes(
    source: str,
    session_files: list[dict[str, Any]],
    default_license: str | None,
    default_attribution: str | None,
) -> str:
    """Format notes containing source, license, and attribution for session.yaml."""
    lines: list[str] = [
        f"Source: {source}",
        f"Licence: {default_license or 'Various open licenses'}",
        f"Attribution: {default_attribution or 'See per-file attribution below'}",
    ]

    has_per_file_metadata = any(
        f.get("file_page_url") or f.get("attribution") or f.get("title") or f.get("url")
        for f in session_files
    )
    if has_per_file_metadata:
        lines.append("")
        lines.append("Files:")
        for item in session_files:
            name = item["filename"]
            lic = item.get("license", default_license or "open")
            author = item.get("attribution", default_attribution or "unknown")
            page = item.get("file_page_url") or item.get("url", "")
            url = item.get("url", "")
            title = item.get("title")

            parts = [f"- {name}: Licence: {lic}, Author: {author}"]
            if title:
                parts.append(f"Title: {title}")
            if url and url != page:
                parts.append(f"URL: {url}")
            if page:
                parts.append(f"Page: {page}")
            lines.append(", ".join(parts))

    return "\n".join(lines)


def _write_session_yaml_if_missing(
    session_dir: Path,
    source: str,
    session_files: list[dict[str, Any]],
    entry_config: dict[str, Any],
) -> None:
    """Write session.yaml in session_dir if it does not already exist."""
    session_yaml_path = session_dir / "session.yaml"
    if session_yaml_path.exists():
        return

    session_fields = entry_config.get("session_yaml") or {}
    notes = _format_session_notes(
        source=source,
        session_files=session_files,
        default_license=entry_config.get("license"),
        default_attribution=entry_config.get("attribution"),
    )

    metadata = SessionMetadata(
        camera=session_fields.get("camera"),
        date=session_fields.get("date"),
        area_type=session_fields.get("area_type"),
        gsd_m=session_fields.get("gsd_m"),
        test=bool(session_fields.get("test", False)),
        notes=notes,
    )
    content = yaml.safe_dump(metadata.model_dump(exclude_none=True), sort_keys=False)
    atomic_write_bytes(session_yaml_path, content.encode("utf-8"))


def _load_samples_config(paths: RepoPaths) -> list[dict[str, Any]]:
    """Load sample definitions from config/samples.yaml with fallback to repository root."""
    config_path = paths.samples_config
    if not config_path.is_file():
        default_config = RepoPaths.from_root().samples_config
        if default_config.is_file():
            config_path = default_config
        else:
            raise FileNotFoundError(f"samples configuration not found at {paths.samples_config}")

    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}

    if isinstance(raw, dict):
        entries = raw.get("samples", [])
    elif isinstance(raw, list):
        entries = raw
    else:
        raise TypeError(f"invalid format in {config_path}: expected dictionary or list")

    return list(entries)


def _write_lock_file(paths: RepoPaths, lock_records: list[dict[str, Any]]) -> None:
    """Write or update config/samples.lock.yaml atomically."""
    lock_file = paths.config_dir / "samples.lock.yaml"
    existing_entries: list[dict[str, Any]] = []

    if lock_file.is_file():
        try:
            with lock_file.open("r", encoding="utf-8") as stream:
                existing_data = yaml.safe_load(stream) or {}
                if isinstance(existing_data, dict):
                    existing_entries = existing_data.get("entries", [])
        except (OSError, yaml.YAMLError) as error:
            logger.warning("Failed to read existing lock file %s: %s", lock_file, error)
            existing_entries = []

    merged_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for entry in existing_entries:
        key = (str(entry.get("source")), str(entry.get("session")), str(entry.get("filename")))
        merged_by_key[key] = entry

    for record in lock_records:
        key = (str(record.get("source")), str(record.get("session")), str(record.get("filename")))
        merged_by_key[key] = record

    sorted_entries = sorted(merged_by_key.values(), key=lambda r: (r.get("source", ""), r.get("session", ""), r.get("filename", "")))
    payload = {"entries": sorted_entries}
    rendered = yaml.safe_dump(payload, sort_keys=False)
    atomic_write_bytes(lock_file, rendered.encode("utf-8"))


def fetch_samples(
    paths: RepoPaths | Path | str | None = None,
    *,
    dry_run: bool = False,
    only: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch sample dataset items, populate input directories, and write lock and session metadata."""
    if paths is None or isinstance(paths, (str, Path)):
        resolved_paths = RepoPaths.from_root(paths)
    elif isinstance(paths, RepoPaths):
        resolved_paths = paths
    else:
        resolved_paths = RepoPaths.from_root()

    raw_entries = _load_samples_config(resolved_paths)
    if only is not None:
        filter_source = only.strip().lower()
        raw_entries = [e for e in raw_entries if str(e.get("source", "")).strip().lower() == filter_source]

    results: list[dict[str, Any]] = []
    lock_records: list[dict[str, Any]] = []

    for entry in raw_entries:
        source = str(entry.get("source", "other"))
        session = str(entry.get("session", "default"))
        entry_type = entry.get("type")

        if entry_type == "github_dir" or "repo" in entry:
            resolved_files = resolve_github_dir(entry)
        elif entry_type == "commons_category" or "category" in entry:
            resolved_files = resolve_commons_category(entry)
        else:
            url = entry.get("url", "")
            default_name = entry.get("filename")
            if not default_name:
                url_path = urllib.parse.urlsplit(url).path
                default_name = Path(urllib.parse.unquote(url_path)).name or "sample"
            default_name = sanitize_filename(default_name)
            resolved_files = [{
                "source": source,
                "session": session,
                "filename": default_name,
                "url": url,
                "title": entry.get("title") or default_name,
                "license": entry.get("license"),
                "attribution": entry.get("attribution"),
                "window": entry.get("window"),
                "sha256": entry.get("sha256"),
                "session_yaml": entry.get("session_yaml", {}),
            }]

        session_dir = resolved_paths.input_dir / source / session

        for file_info in resolved_files:
            filename = file_info["filename"]
            dest_path = session_dir / filename
            ensure_inside(dest_path, resolved_paths.root)

            file_url = file_info.get("url", "")
            expected_sha = file_info.get("sha256")
            window = file_info.get("window")

            if dry_run:
                size = file_info.get("size")
                is_estimate = False
                estimate_str = None
                if window:
                    if len(window) >= 4:
                        window_w = int(window[2])
                        window_h = int(window[3])
                        est_mb = window_w * window_h * 3 / 1e6 * 0.35
                        size = int(window_w * window_h * 3 * 0.35)
                        estimate_str = (
                            f"~{est_mb:.0f} MB (window of a Cloud-Optimized GeoTIFF; "
                            "only the window is transferred)"
                        )
                        is_estimate = True
                elif size is None and file_url.startswith(("http://", "https://")):
                    size = probe_file_size(file_url)

                result_record = {
                    "source": source,
                    "session": session,
                    "filename": filename,
                    "path": dest_path,
                    "url": file_url,
                    "status": "dry_run",
                    "size": size,
                    "is_estimate": is_estimate,
                    "estimate_str": estimate_str,
                    "window": window,
                    "sha256": expected_sha,
                    "license": file_info.get("license"),
                    "attribution": file_info.get("attribution"),
                }
                results.append(result_record)
                continue

            if dest_path.is_file():
                status = "skipped"
                file_size = dest_path.stat().st_size
                sha = sha256_file(dest_path)
            else:
                try:
                    if window:
                        file_size, sha = fetch_ortho_window(
                            file_url,
                            dest_path,
                            window,
                            expected_sha256=expected_sha,
                        )
                    else:
                        file_size, sha = download_file(
                            file_url,
                            dest_path,
                            expected_sha256=expected_sha,
                        )
                except (urllib.error.URLError, OSError, rasterio.errors.RasterioIOError, ValueError) as error:
                    temp_file = dest_path.parent / f".{dest_path.name}.tmp"
                    if temp_file.is_file():
                        temp_file.unlink()
                    results.append({
                        "source": source,
                        "session": session,
                        "filename": filename,
                        "path": dest_path,
                        "url": file_url,
                        "status": "failed",
                        "error": str(error),
                        "size": 0,
                        "license": file_info.get("license"),
                        "attribution": file_info.get("attribution"),
                    })
                    continue
                status = "downloaded"

            record = {
                "source": source,
                "session": session,
                "filename": filename,
                "path": dest_path,
                "url": file_url,
                "status": status,
                "size": file_size,
                "sha256": sha,
                "license": file_info.get("license"),
                "attribution": file_info.get("attribution"),
            }
            results.append(record)

            lock_entry: dict[str, Any] = {
                "source": source,
                "session": session,
                "filename": filename,
                "url": file_url,
                "sha256": sha,
                "size": file_size,
                "license": file_info.get("license"),
                "attribution": file_info.get("attribution"),
            }
            if file_info.get("title"):
                lock_entry["title"] = file_info["title"]
            if file_info.get("file_page_url"):
                lock_entry["file_page_url"] = file_info["file_page_url"]
            lock_records.append(lock_entry)

        if not dry_run and resolved_files:
            _write_session_yaml_if_missing(
                session_dir=session_dir,
                source=source,
                session_files=resolved_files,
                entry_config=entry,
            )

    if not dry_run and lock_records:
        _write_lock_file(resolved_paths, lock_records)

    return results
