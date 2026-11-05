"""Panoramax street-level imagery retrieval."""

from __future__ import annotations

import json
import logging
import math
import re
import time
import urllib.parse
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

import cv2

from flurseg.samples.extra.common import FileProduct, decode_rgb, request_bytes
from flurseg.samples.extra.config import PanoramaxEntry
from flurseg.samples.extra.equirect import equirect_to_perspective
from flurseg.samples.extra.licences import is_acceptable_licence, normalise_licence

logger = logging.getLogger(__name__)


def _json(url: str) -> dict[str, Any]:
    return json.loads(request_bytes(url, min_interval_s=0.5).decode("utf-8"))


def _link_value(item: dict[str, Any], rel: str) -> str:
    for link in item.get("links", []):
        if isinstance(link, dict) and link.get("rel") == rel:
            return str(link.get("title") or link.get("href") or "")
    return ""


def _link_href(item: dict[str, Any], rel: str) -> str:
    for link in item.get("links", []):
        if isinstance(link, dict) and link.get("rel") == rel and link.get("href"):
            return str(link["href"])
    return ""


def _licence_link_value(link: dict[str, Any]) -> str:
    """Extract the machine-readable licence identifier from a Panoramax link."""
    title = str(link.get("title") or "")
    match = re.search(r"\(([^()]+)\)", title)
    if match:
        return match.group(1)
    href = str(link.get("href") or "").lower()
    if "creativecommons.org/licenses/by-sa/" in href:
        return "CC BY-SA"
    if "creativecommons.org/licenses/by-nd/" in href:
        return "CC BY-ND"
    if "creativecommons.org/licenses/by-nc/" in href:
        return "CC BY-NC"
    if "creativecommons.org/licenses/by/" in href:
        return "CC BY 4.0"
    if "creativecommons.org/publicdomain/zero" in href:
        return "CC0"
    if "etalab" in href or "licence-ouverte" in href:
        return "Etalab 2.0"
    return title or str(link.get("href") or "")


def _direct_licence(item: dict[str, Any]) -> str:
    for link in item.get("links", []):
        if isinstance(link, dict) and link.get("rel") == "license":
            return _licence_link_value(link)
    properties = item.get("properties", {})
    return str(properties.get("license") or "") if isinstance(properties, dict) else ""


def _licence(item: dict[str, Any], collections: dict[str, str]) -> str:
    properties = item.get("properties", {})
    direct = _direct_licence(item)
    if direct:
        return direct
    collection_url = _link_href(item, "collection")
    collection_id = str(properties.get("collection") or item.get("collection") or collection_url)
    if collection_id and collection_id not in collections:
        try:
            collection = _json(collection_url) if collection_url else {}
            collections[collection_id] = _link_value(collection, "license") or str(
                collection.get("license") or collection.get("properties", {}).get("license") or ""
            )
        except (OSError, ValueError, json.JSONDecodeError):
            collections[collection_id] = ""
    return collections.get(collection_id, "")


def _is_panorama(item: dict[str, Any]) -> bool:
    properties = item.get("properties", {})
    orientation = properties.get("pers:interior_orientation", {})
    if isinstance(orientation, dict) and orientation.get("field_of_view") == 360:
        return True
    if properties.get("field_of_view") == 360:
        return True
    text = " ".join(str(value).lower() for value in properties.values())
    if "equirectangular" in text:
        return True
    assets = item.get("assets", {})
    asset = assets.get("sd") or assets.get("hd") or {}
    width, height = asset.get("width"), asset.get("height")
    return bool(width and height and abs(float(width) / float(height) - 2.0) < 0.02)


def _coordinates(item: dict[str, Any]) -> tuple[float, float] | None:
    geometry = item.get("geometry", {})
    coordinates = geometry.get("coordinates", []) if isinstance(geometry, dict) else []
    if isinstance(coordinates, list) and len(coordinates) >= 2:
        try:
            return float(coordinates[0]), float(coordinates[1])
        except (TypeError, ValueError):
            return None
    return None


def _distance_m(left: tuple[float, float], right: tuple[float, float]) -> float:
    lon1, lat1, lon2, lat2 = map(math.radians, (*left, *right))
    return (
        6_371_000
        * 2
        * math.asin(
            math.sqrt(
                math.sin((lat2 - lat1) / 2) ** 2
                + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
            )
        )
    )


def _heading(item: dict[str, Any]) -> float | None:
    value = item.get("properties", {}).get("compass_angle", item.get("properties", {}).get("heading"))
    try:
        return float(value) % 360
    except (TypeError, ValueError):
        return None


def _similar_view(item: dict[str, Any], kept: list[dict[str, Any]], spacing: float) -> bool:
    position = _coordinates(item)
    if position is None:
        return False
    current_heading = _heading(item)
    for prior in kept:
        prior_position = _coordinates(prior)
        if prior_position is None or _distance_m(position, prior_position) >= spacing:
            continue
        prior_heading = _heading(prior)
        if current_heading is None or prior_heading is None:
            return True
        delta = abs((current_heading - prior_heading + 180) % 360 - 180)
        if delta < 45:
            return True
    return False


def _items(entry: PanoramaxEntry, max_pages: int = 10) -> Iterable[dict[str, Any]]:
    bbox = ",".join(str(value) for value in entry.bbox)
    url = f"{entry.url}?bbox={bbox}&limit={entry.max_per_entry}"
    pages = 0
    while url and pages < max_pages:
        pages += 1
        payload = _json(url)
        yield from (feature for feature in payload.get("features", []) if isinstance(feature, dict))
        url = ""
        for link in payload.get("links", []):
            if isinstance(link, dict) and link.get("rel") == "next" and link.get("href"):
                url = str(link["href"])
                break


def _legacy_accepted_items_with_stats(
    entry: PanoramaxEntry,
    limit: int | None = None,
    max_pages: int = 10,
) -> tuple[list[tuple[dict[str, Any], str]], dict[str, Any]]:
    """Search, licence-filter, and spatially thin Panoramax items, reporting statistics."""
    target_count = min(entry.max_per_entry, limit) if limit is not None else entry.max_per_entry
    collections: dict[str, str] = {}
    kept: list[dict[str, Any]] = []
    result: list[tuple[dict[str, Any], str]] = []
    rejected_by_licence: Counter[str] = Counter()
    seen = 0
    skipped_360 = 0
    skipped_thinned = 0
    for item in _items(entry, max_pages=max_pages):
        seen += 1
        licence = _licence(item, collections)
        if not is_acceptable_licence(licence):
            rejected_by_licence[normalise_licence(licence) or "unknown"] += 1
            continue
        if _is_panorama(item):
            skipped_360 += 1
            continue
        if _similar_view(item, kept, entry.min_spacing_m):
            skipped_thinned += 1
            continue
        assets = item.get("assets", {})
        if not isinstance(assets.get("sd") or assets.get("hd"), dict):
            continue
        kept.append(item)
        result.append((item, licence))
        if len(result) >= target_count:
            break
    stats = {
        "seen": seen,
        "accepted": len(result),
        "rejected_by_licence": dict(rejected_by_licence),
        "skipped_360": skipped_360,
        "skipped_thinned": skipped_thinned,
        "thinned": skipped_thinned,
    }
    licence_summary = ", ".join(f"{lic}={cnt}" for lic, cnt in sorted(rejected_by_licence.items()))
    logger.info(
        "Panoramax %s: seen %d, accepted %d, rejected by licence: %s, skipped 360: %d, thinned: %d",
        entry.id,
        seen,
        len(result),
        licence_summary or "none",
        skipped_360,
        skipped_thinned,
    )
    return result, stats


def _legacy_accepted_items(
    entry: PanoramaxEntry,
    limit: int | None = None,
    max_pages: int = 10,
) -> list[tuple[dict[str, Any], str]]:
    """Search, licence-filter, and spatially thin Panoramax items."""
    items, _ = accepted_items_with_stats(entry, limit=limit, max_pages=max_pages)
    return items


def _legacy_plan_products(entry: PanoramaxEntry, limit: int | None = None) -> int:
    """Return the accepted image count after metadata-only search requests."""
    return len(accepted_items(entry, limit=limit))


def _legacy_products(entry: PanoramaxEntry, limit: int | None = None) -> list[FileProduct]:
    """Download the 2048 px asset, falling back to HD when SD is unavailable."""
    products: list[FileProduct] = []
    for item, licence in accepted_items(entry, limit=limit):
        assets = item["assets"]
        asset = assets.get("sd") or assets["hd"]
        url = str(asset.get("href") or "")
        if not url:
            continue
        providers = item.get("properties", {}).get("providers", [])
        provider = providers[0].get("name") if providers and isinstance(providers[0], dict) else ""
        provider = provider or item.get("properties", {}).get("producer") or "Panoramax contributor"
        identifier = str(item.get("id") or len(products))
        products.append(
            FileProduct(
                data=request_bytes(url, min_interval_s=0.5),
                filename=f"{entry.session}_{identifier}.jpg",
                gsd_m=1.0,
                url=url,
                title=str(item.get("id") or "Panoramax image"),
                licence=licence,
                attribution=f"© {provider} / Panoramax, {licence}",
            )
        )
    return products


# Part C uses federation collection prefiltering and produces perspective crops.
# The definitions below intentionally replace the compatible Part B helpers above.
_CACHE_TTL_S = 24 * 60 * 60


def _collection_cache_path(entry: PanoramaxEntry, cache_dir: Path) -> Path:
    host = urllib.parse.urlsplit(entry.instance).netloc.lower().replace(":", "_") or "panoramax"
    return cache_dir / f"panoramax_collections_{host}.json"


def _collection_licence(collection: dict[str, Any]) -> str:
    direct = _direct_licence(collection)
    if direct:
        return direct
    properties = collection.get("properties", {})
    return str(collection.get("license") or (properties.get("license") if isinstance(properties, dict) else "") or "")


def _allowed_collections(entry: PanoramaxEntry, cache_dir: Path | None) -> list[str]:
    path = _collection_cache_path(entry, cache_dir) if cache_dir is not None else None
    if path is not None:
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if time.time() - float(cached.get("created", 0)) < _CACHE_TTL_S:
                return [str(value) for value in cached.get("ids", [])]
        except (OSError, TypeError, ValueError):
            pass
    url = f"{entry.instance.rstrip('/')}/collections?limit=1000"
    identifiers: list[str] = []
    for _page in range(entry.max_collection_pages):
        payload = _json(url)
        collections = payload.get("collections", payload.get("features", payload.get("items", [])))
        for collection in collections if isinstance(collections, list) else []:
            if not isinstance(collection, dict) or not is_acceptable_licence(_collection_licence(collection)):
                continue
            identifier = str(collection.get("id") or collection.get("identifier") or "")
            if identifier:
                identifiers.append(identifier)
        url = next(
            (str(link["href"]) for link in payload.get("links", []) if isinstance(link, dict) and link.get("rel") == "next" and link.get("href")),
            "",
        )
        if not url:
            break
    result = sorted(set(identifiers))
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"created": time.time(), "ids": result}, sort_keys=True), encoding="utf-8")
    return result


def _search_url(
    entry: PanoramaxEntry, collection_ids: list[str] | None = None, *, flat_filter: bool = True
) -> str:
    params = {"bbox": ",".join(str(value) for value in entry.bbox), "limit": str(entry.max_per_entry)}
    if collection_ids:
        params["collections"] = ",".join(collection_ids)
    if flat_filter and entry.flat_only and "api.panoramax.xyz" in entry.instance:
        params["filter"] = "field_of_view<360"
    return f"{entry.instance.rstrip('/')}/search?{urllib.parse.urlencode(params, safe=',')}"


def _http_error_body(error: HTTPError) -> str:
    """Return a short useful HTTP error body without masking the original error."""
    try:
        return error.read().decode("utf-8", errors="replace")[:1000]
    except OSError:
        return "<response body unavailable>"


def _items(entry: PanoramaxEntry, max_pages: int = 10, cache_dir: Path | None = None) -> Iterable[dict[str, Any]]:
    chunks: list[list[str] | None] = [None]
    if entry.mode == "collections_prefilter":
        identifiers = _allowed_collections(entry, cache_dir)
        chunks = [identifiers[index : index + 50] for index in range(0, len(identifiers), 50)]
    for collection_ids in chunks:
        url = _search_url(entry, collection_ids)
        for _page in range(max_pages):
            try:
                payload = _json(url)
            except HTTPError as error:
                if error.code != 400 or "filter=" not in url:
                    raise
                logger.warning(
                    "Panoramax filter request returned HTTP 400 for %s: %s; retrying without filter.",
                    url,
                    _http_error_body(error),
                )
                url = _search_url(entry, collection_ids, flat_filter=False)
                payload = _json(url)
            yield from (feature for feature in payload.get("features", []) if isinstance(feature, dict))
            url = next(
                (str(link["href"]) for link in payload.get("links", []) if isinstance(link, dict) and link.get("rel") == "next" and link.get("href")),
                "",
            )
            if not url:
                break


def accepted_items_with_stats(
    entry: PanoramaxEntry,
    limit: int | None = None,
    max_pages: int = 10,
    cache_dir: Path | None = None,
) -> tuple[list[tuple[dict[str, Any], str]], dict[str, Any]]:
    """Return allowed, spatially distinct picture records and audit counters."""
    target = min(entry.max_per_entry, limit) if limit is not None else entry.max_per_entry
    collections: dict[str, str] = {}
    kept: list[dict[str, Any]] = []
    accepted: list[tuple[dict[str, Any], str]] = []
    rejected: Counter[str] = Counter()
    seen = thinned = skipped_360 = 0
    for item in _items(entry, max_pages=max_pages, cache_dir=cache_dir):
        seen += 1
        licence = _licence(item, collections)
        if not is_acceptable_licence(licence):
            rejected[normalise_licence(licence) or "unknown"] += 1
            continue
        if entry.flat_only and _is_panorama(item):
            skipped_360 += 1
            continue
        if _similar_view(item, kept, entry.min_spacing_m):
            thinned += 1
            continue
        if _is_panorama(item):
            image_asset = item.get("assets", {}).get("hd") or {}
            if not isinstance(image_asset, dict) or not image_asset.get("href"):
                skipped_360 += 1
                continue
        if not isinstance(item.get("assets", {}).get("sd") or item.get("assets", {}).get("hd"), dict):
            continue
        kept.append(item)
        accepted.append((item, licence))
        if len(accepted) >= target:
            break
    stats = {"seen": seen, "accepted": len(accepted), "rejected_by_licence": dict(rejected), "skipped_360": skipped_360,
             "skipped_thinned": thinned, "thinned": thinned}
    logger.info("Panoramax %s: seen %d, accepted %d, rejected by licence: %s, thinned: %d", entry.id, seen,
                len(accepted), ", ".join(f"{name}={count}" for name, count in sorted(rejected.items())) or "none", thinned)
    return accepted, stats


def accepted_items(
    entry: PanoramaxEntry, limit: int | None = None, max_pages: int = 10
) -> list[tuple[dict[str, Any], str]]:
    """Return allowed Panoramax records without downloading their image bytes."""
    return accepted_items_with_stats(entry, limit=limit, max_pages=max_pages)[0]


def plan_products(entry: PanoramaxEntry, limit: int | None = None) -> int:
    """Return the number of flat files or projected views planned from metadata."""
    return sum(entry.views if _is_panorama(item) and not entry.flat_only else 1 for item, _licence in accepted_items(entry, limit))


def products(entry: PanoramaxEntry, limit: int | None = None) -> list[FileProduct]:
    """Download flat pictures or JPEG perspective views of 360-degree pictures."""
    result: list[FileProduct] = []
    accepted = accepted_items(entry, limit=limit)
    if not accepted:
        logger.warning("Panoramax %s yielded 0 items: no images remained after metadata filters", entry.id)
        return result
    for item, licence in accepted:
        panorama = _is_panorama(item)
        assets = item["assets"]
        asset = (assets.get("hd") or assets.get("sd")) if panorama else (assets.get("sd") or assets.get("hd"))
        url = str(asset.get("href") or "")
        if not url:
            continue
        payload = request_bytes(url, min_interval_s=0.5)
        identifier = str(item.get("id") or len(result))
        providers = item.get("properties", {}).get("providers", [])
        provider = providers[0].get("name") if providers and isinstance(providers[0], dict) else ""
        attribution = f"© {provider or item.get('properties', {}).get('producer') or 'Panoramax contributor'} / Panoramax, {licence}"
        if not panorama:
            result.append(FileProduct(payload, f"{entry.session}_{identifier}.jpg", 1.0, url, identifier, licence, attribution))
            continue
        image = decode_rgb(payload)
        if abs(image.shape[1] / image.shape[0] - 2.0) >= 0.03:
            continue
        for view in range(entry.views):
            yaw = 360.0 * view / entry.views
            crop = equirect_to_perspective(image, yaw, entry.pitch_deg, entry.fov_deg, entry.crop_size)
            encoded, jpeg = cv2.imencode(".jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            if encoded:
                result.append(FileProduct(jpeg.tobytes(), f"{entry.session}_{identifier}_yaw{int(yaw)}.jpg", 1.0,
                                          url, identifier, licence, attribution))
                if limit is not None and len(result) >= limit:
                    return result
    if not result:
        logger.warning("Panoramax %s yielded 0 items: accepted images had no usable payload", entry.id)
    return result[:limit] if limit is not None else result
