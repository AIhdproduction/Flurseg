"""Wikimedia Commons photo-category retrieval for phone training imagery."""

from __future__ import annotations

import json
import urllib.parse
from collections.abc import Iterable
from typing import Any

from flurseg.samples.extra.common import FileProduct, request_bytes
from flurseg.samples.extra.config import CommonsCategoriesEntry
from flurseg.samples.extra.licences import is_acceptable_licence
from flurseg.samples.fetch import _clean_text, open_url_with_retry

_API = "https://commons.wikimedia.org/w/api.php"


def _api(parameters: dict[str, str]) -> dict[str, Any]:
    url = f"{_API}?{urllib.parse.urlencode(parameters)}"
    with open_url_with_retry(url) as response:
        return json.loads(response.read().decode("utf-8"))


def _title(category: str) -> str:
    return category if category.startswith("Category:") else f"Category:{category}"


def _children(category: str) -> list[str]:
    payload = _api(
        {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": _title(category),
            "cmtype": "subcat",
            "cmlimit": "500",
            "format": "json",
        }
    )
    return [
        str(item["title"]) for item in payload.get("query", {}).get("categorymembers", []) if "title" in item
    ]


def _pages(category: str, thumb_width: int | None = 2560, *, video: bool = False) -> list[dict[str, Any]]:
    properties = "url|size|mime|extmetadata" + ("|derivatives" if video else "")
    info_key = "videoinfo" if video else "imageinfo"
    payload = _api(
        {
            "action": "query",
            "generator": "categorymembers",
            "gcmtitle": _title(category),
            "gcmtype": "file",
            "gcmlimit": "500",
            "prop": info_key,
            "viprop" if video else "iiprop": properties,
            "iiurlwidth": str(thumb_width) if thumb_width else "",
            "format": "json",
        }
    )
    pages = payload.get("query", {}).get("pages", {})
    return sorted(
        (page for page in pages.values() if isinstance(page, dict)),
        key=lambda page: str(page.get("title", "")),
    )


def commons_files(
    categories: list[str],
    *,
    depth: int,
    cap: int,
    exclude_categories: list[str] | None = None,
    thumb_width: int | None = 2560,
    video: bool = False,
) -> Iterable[dict[str, Any]]:
    """Yield Commons files from categories and their requested sub-category depth."""
    excluded = {_title(value).lower() for value in exclude_categories or []}
    queue = [(_title(category), 0) for category in categories]
    seen_categories: set[str] = set()
    yielded = 0
    while queue and yielded < cap:
        category, level = queue.pop(0)
        key = category.lower()
        if key in seen_categories or key in excluded:
            continue
        seen_categories.add(key)
        for page in _pages(category, thumb_width, video=video):
            yield page
            yielded += 1
            if yielded >= cap:
                return
        if level < depth:
            queue.extend((child, level + 1) for child in _children(category))


def _info(page: dict[str, Any]) -> dict[str, Any] | None:
    images = page.get("videoinfo", []) or page.get("imageinfo", [])
    return images[0] if images and isinstance(images[0], dict) else None


def _licence(info: dict[str, Any]) -> str:
    metadata = info.get("extmetadata", {})
    return _clean_text(str(metadata.get("LicenseShortName", {}).get("value", "")))


def _artist(info: dict[str, Any]) -> str:
    metadata = info.get("extmetadata", {})
    return _clean_text(
        str(metadata.get("Artist", {}).get("value") or metadata.get("Credit", {}).get("value") or "")
    )


def _download_url(info: dict[str, Any]) -> str:
    return str(info.get("thumburl") or info.get("url") or "")


def accepted_pages(
    entry: CommonsCategoriesEntry,
    limit: int | None = None,
) -> list[tuple[dict[str, Any], dict[str, Any], str]]:
    """Return image pages that are photos of sufficient size and accepted licence."""
    max_count = min(entry.max_per_entry, limit) if limit is not None else entry.max_per_entry
    accepted: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    # The input cap applies to accepted photos, not the raw category stream:
    # Commons commonly intersperses SVGs, tiny files, and incompatible licences.
    for page in commons_files(entry.categories, depth=entry.depth, cap=max_count * 10):
        info = _info(page)
        if not info or str(info.get("mime", "")).lower() not in {"image/jpeg", "image/png"}:
            continue
        if min(int(info.get("width") or 0), int(info.get("height") or 0)) < entry.min_width:
            continue
        licence = _licence(info)
        if not is_acceptable_licence(licence) or not _download_url(info):
            continue
        accepted.append((page, info, licence))
        if len(accepted) >= max_count:
            break
    return accepted


def plan_products(entry: CommonsCategoriesEntry, limit: int | None = None) -> int:
    """Return the number of accepted Commons photos without downloading thumbnails."""
    return len(accepted_pages(entry, limit=limit))


def products(entry: CommonsCategoriesEntry, limit: int | None = None) -> list[FileProduct]:
    """Download scaled Commons thumbnails rather than original high-resolution files."""
    results: list[FileProduct] = []
    for index, (page, info, licence) in enumerate(accepted_pages(entry, limit=limit)):
        artist = _artist(info) or "Wikimedia Commons contributor"
        url = _download_url(info)
        title = str(page.get("title") or "Commons photo")
        extension = ".png" if str(info.get("mime", "")).lower() == "image/png" else ".jpg"
        results.append(
            FileProduct(
                data=request_bytes(url, min_interval_s=1.0),
                filename=f"{entry.session}_{index}{extension}",
                gsd_m=1.0,
                url=url,
                title=title,
                licence=licence,
                attribution=f"{artist}, {licence}, via Wikimedia Commons",
            )
        )
    return results
