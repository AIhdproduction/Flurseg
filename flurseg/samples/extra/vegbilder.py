"""Statens vegvesen Vegbilder WFS retrieval for planar road imagery."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import urllib.parse
from collections.abc import Iterable
from typing import Any
from xml.etree import ElementTree

import cv2

from flurseg.samples.extra.common import FileProduct, decode_rgb, request_bytes
from flurseg.samples.extra.commons_video_frames import _hash
from flurseg.samples.extra.config import VegbilderBbox, VegbilderWfsEntry

logger = logging.getLogger(__name__)

ATTRIBUTION = "© Statens vegvesen (Vegbilder), NLOD 2.0"


def layer_name(year: int) -> str:
    """Return the planar, never 360-degree, layer name for one year."""
    return f"vegbilder_1_0:Vegbilder_{year}"


def feature_url(
    entry: VegbilderWfsEntry,
    year: int,
    bbox: VegbilderBbox,
    *,
    start_index: int = 0,
    count: int = 1000,
    hits: bool = False,
    feature_id: str | None = None,
) -> str:
    """Build one WFS 2.0 JSON request in longitude/latitude order."""
    params = {
        "service": "WFS", "version": "2.0.0", "request": "GetFeature", "typeNames": layer_name(year),
        "outputFormat": "application/json", "srsName": "EPSG:4326", "bbox": ",".join(map(str, bbox.bbox)) + ",EPSG:4326",
        "count": str(min(count, 1000)), "startIndex": str(start_index),
    }
    if hits:
        params["resultType"] = "hits"
    if feature_id:
        params["resourceId"] = feature_id
    return f"{entry.url}?{urllib.parse.urlencode(params, safe=',:')}"


def _json(url: str) -> dict[str, Any]:
    return json.loads(request_bytes(url, min_interval_s=0.5).decode("utf-8"))


def _hit_count_from_xml(payload: bytes) -> int:
    """Extract WFS 2.0 ``numberMatched`` without depending on namespace prefixes."""
    try:
        root = ElementTree.fromstring(payload)
    except ElementTree.ParseError:
        return 0
    value = root.attrib.get("numberMatched", "0")
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _coordinates(feature: dict[str, Any]) -> tuple[float, float] | None:
    coordinates = feature.get("geometry", {}).get("coordinates", [])
    if isinstance(coordinates, list) and len(coordinates) >= 2:
        try:
            return float(coordinates[0]), float(coordinates[1])
        except (TypeError, ValueError):
            return None
    return None


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    lon1, lat1, lon2, lat2 = map(math.radians, (*left, *right))
    return 6_371_000 * 2 * math.asin(math.sqrt(math.sin((lat2-lat1)/2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2-lon1)/2) ** 2))


def _image_url(feature: dict[str, Any]) -> str:
    return str(feature.get("properties", {}).get("URL") or "")


def _metadata(feature: dict[str, Any], year: int) -> dict[str, object]:
    properties = feature.get("properties", {})
    coordinates = _coordinates(feature)
    return {
        "feature_id": str(feature.get("id") or ""),
        "year": year,
        "capture_date": str(properties.get("TIDSPUNKT") or ""),
        "road_id": str(properties.get("VEGNUMMER") or properties.get("VEG") or ""),
        "heading": str(properties.get("RETNING") or ""),
        "longitude": coordinates[0] if coordinates else "",
        "latitude": coordinates[1] if coordinates else "",
    }


def _refresh_feature(entry: VegbilderWfsEntry, year: int, bbox: VegbilderBbox, feature: dict[str, Any]) -> dict[str, Any] | None:
    """Re-query once when the authority's short-lived signed image URL expires."""
    feature_id = str(feature.get("id") or "")
    if not feature_id:
        return None
    payload = _json(feature_url(entry, year, bbox, count=1, feature_id=feature_id))
    features = [item for item in payload.get("features", []) if isinstance(item, dict)]
    return features[0] if features else None


def _clean_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def _selected(entry: VegbilderWfsEntry, features: Iterable[dict[str, Any]], target: int) -> list[dict[str, Any]]:
    ordered = sorted(features, key=lambda item: str(item.get("id") or ""))
    random.Random(f"{entry.id}:{target}").shuffle(ordered)
    kept: list[dict[str, Any]] = []
    positions: list[tuple[float, float]] = []
    for feature in ordered:
        position = _coordinates(feature)
        if not _image_url(feature) or (position and any(_distance(position, prior) < entry.min_spacing_m for prior in positions)):
            continue
        kept.append(feature)
        if position:
            positions.append(position)
        if len(kept) >= target:
            break
    return kept


def _sample_offsets(entry: VegbilderWfsEntry, year: int, bbox: VegbilderBbox, total: int) -> tuple[list[int], int]:
    """Return many deterministic, widely spread small WFS pages for distinct road locations."""
    wanted = max(entry.per_bbox * 6, 300)
    page_size = min(4, wanted)
    requests = min(100, max(1, math.ceil(wanted / page_size)))
    upper = max(0, total - page_size)
    if requests == 1 or upper == 0:
        return [0], page_size
    generator = random.Random(f"{entry.id}:{year}:{bbox.name}:{total}")
    offsets = [0, upper]
    for index in range(1, requests - 1):
        low = round(index * upper / (requests - 1))
        high = round((index + 1) * upper / (requests - 1))
        offsets.append(generator.randint(low, max(low, high)))
    return sorted(set(offsets)), page_size


def _candidates(entry: VegbilderWfsEntry, year: int, bbox: VegbilderBbox) -> list[dict[str, Any]]:
    """Collect a deliberately oversized raw candidate pool before spatial thinning."""
    total = hit_count(entry, year, bbox)
    if total <= 0:
        return []
    result: list[dict[str, Any]] = []
    offsets, page_size = _sample_offsets(entry, year, bbox, total)
    for start in offsets:
        payload = _json(feature_url(entry, year, bbox, start_index=start, count=page_size))
        features = [item for item in payload.get("features", []) if isinstance(item, dict)]
        result.extend(features)
    return result


def hit_count(entry: VegbilderWfsEntry, year: int, bbox: VegbilderBbox) -> int:
    """Return the WFS hit count without requesting any imagery."""
    payload = request_bytes(feature_url(entry, year, bbox, count=1, hits=True), min_interval_s=0.5)
    return _hit_count_from_xml(payload)


def _year_candidates(entry: VegbilderWfsEntry) -> Iterable[tuple[int, VegbilderBbox, dict[str, Any]]]:
    """Yield one year's selected features at a time so signed image URLs are consumed promptly."""
    for year in entry.layer_year:
        for bbox in entry.bboxes:
            for feature in _selected(entry, _candidates(entry, year, bbox), entry.per_bbox * 6):
                yield year, bbox, feature


def products(
    entry: VegbilderWfsEntry,
    limit: int | None = None,
    *,
    existing_feature_ids: set[str] | None = None,
) -> list[FileProduct]:
    """Fetch new quality-filtered images, counting ``limit`` as new images still needed for the entry cap."""
    maximum = min(entry.max_per_entry, limit) if limit is not None else entry.max_per_entry
    if maximum <= 0:
        return []
    existing = existing_feature_ids or set()
    result: list[FileProduct] = []
    hashes: list[int] = []
    quotient, remainder = divmod(maximum, len(entry.layer_year))
    year_limits = {year: quotient + (index < remainder) for index, year in enumerate(entry.layer_year)}
    year_kept = {year: 0 for year in entry.layer_year}
    for year, bbox, feature in _year_candidates(entry):
        if year_kept[year] >= year_limits[year]:
            continue
        identifier = str(feature.get("id") or "").replace("/", "_")
        if identifier in existing:
            continue
        signed_url = _image_url(feature)
        if not signed_url:
            continue
        try:
            payload = request_bytes(signed_url, min_interval_s=0.2)
            image = decode_rgb(payload)
        except (OSError, ValueError):
            refreshed = _refresh_feature(entry, year, bbox, feature)
            if refreshed is None:
                continue
            feature = refreshed
            identifier = str(feature.get("id") or "").replace("/", "_")
            if identifier in existing:
                continue
            signed_url = _image_url(feature)
            try:
                payload = request_bytes(signed_url, min_interval_s=0.2)
                image = decode_rgb(payload)
            except (OSError, ValueError):
                logger.warning("Vegbilder image %s remained unavailable after one URL refresh; skipping it.", feature.get("id"))
                continue
        if float(image.mean()) < 25:
            continue
        digest = _hash(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if any((digest ^ prior).bit_count() <= 8 for prior in hashes):
            continue
        hashes.append(digest)
        if not identifier:
            identifier = hashlib.sha256(signed_url.encode()).hexdigest()[:12]
        result.append(
            FileProduct(
                payload,
                f"{entry.session}_{year}_{identifier}.jpg",
                1.0,
                _clean_url(signed_url),
                identifier,
                "NLOD-2.0",
                ATTRIBUTION,
                _metadata(feature, year),
            )
        )
        year_kept[year] += 1
        if len(result) >= maximum:
            return result
    return result


def plan_products(entry: VegbilderWfsEntry, limit: int | None = None) -> int:
    """Return the configured upper bound for non-network planning callers."""
    count = len(entry.layer_year) * len(entry.bboxes) * entry.per_bbox
    return min(count, entry.max_per_entry, limit) if limit is not None else min(count, entry.max_per_entry)
