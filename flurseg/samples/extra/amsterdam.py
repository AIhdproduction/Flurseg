"""Amsterdam panorama API source."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterable
from typing import Any

import cv2

from flurseg.samples.extra.common import FileProduct, decode_rgb, request_bytes
from flurseg.samples.extra.config import AmsterdamPanoramaEntry
from flurseg.samples.extra.equirect import equirect_to_perspective
from flurseg.samples.extra.panoramax import _distance_m

_API = "https://api.data.amsterdam.nl/panorama/panoramas/"
_ATTRIBUTION = "© Gemeente Amsterdam, CC BY 4.0"
logger = logging.getLogger(__name__)


def _json(url: str) -> dict[str, Any]:
    return json.loads(request_bytes(url, min_interval_s=0.5).decode("utf-8"))


def _records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    embedded = payload.get("_embedded", {})
    for key in ("panoramas", "results", "items"):
        values = embedded.get(key) if isinstance(embedded, dict) else None
        if isinstance(values, list):
            return [value for value in values if isinstance(value, dict)]
    for key in ("panoramas", "results", "items"):
        values = payload.get(key)
        if isinstance(values, list):
            return [value for value in values if isinstance(value, dict)]
    return []


def _position(record: dict[str, Any]) -> tuple[float, float] | None:
    location = record.get("geometry") or record.get("location") or record.get("coordinates")
    if isinstance(location, dict):
        location = location.get("coordinates")
    if isinstance(location, (tuple, list)) and len(location) >= 2:
        try:
            return float(location[0]), float(location[1])
        except (TypeError, ValueError):
            return None
    return None


def _identifier(record: dict[str, Any]) -> str:
    return str(record.get("id") or record.get("pano_id") or record.get("identifier") or "")


def _image_url(record: dict[str, Any]) -> str:
    links = record.get("_links", {})
    for key in (
        "equirectangular_medium",
        "equirectangular_full",
        "equirectangular_small",
        "equirectangular",
        "panorama",
        "image",
    ):
        value = links.get(key) if isinstance(links, dict) else None
        if isinstance(value, dict) and value.get("href"):
            return str(value["href"])
    identifier = _identifier(record)
    date = str(
        record.get("timestamp") or record.get("recorded_at") or record.get("date") or record.get("created_at") or ""
    )[:10]
    if identifier and len(date) == 10:
        year, month, day = date.split("-")
        return f"https://t1.data.amsterdam.nl/panorama/{year}/{month}/{day}/{identifier}/equirectangular/panorama_8000.jpg"
    return ""


def _records_near(entry: AmsterdamPanoramaEntry, limit: int | None = None) -> Iterable[dict[str, Any]]:
    """Yield nearby API records across the small number of pages needed for the crop cap."""
    lon, lat = entry.near
    target = min(entry.max_per_entry, limit) if limit is not None else entry.max_per_entry
    page_size = min(25, max(1, math.ceil(target / entry.views)))
    url = f"{_API}?near={lon},{lat}&radius={entry.radius_m:g}&page_size={page_size}"
    for _page in range(10):
        payload = _json(url)
        yield from _records(payload)
        links = payload.get("_links", {})
        next_link = links.get("next") if isinstance(links, dict) else None
        url = str(next_link.get("href") or "") if isinstance(next_link, dict) else ""
        if not url:
            break


def products(entry: AmsterdamPanoramaEntry, limit: int | None = None) -> list[FileProduct]:
    """Fetch Amsterdam panoramas and project their road-facing perspective views."""
    result: list[FileProduct] = []
    positions: list[tuple[float, float]] = []
    seen_records = 0
    for record in _records_near(entry, limit):
        seen_records += 1
        position = _position(record)
        if position and any(_distance_m(position, prior) < entry.min_spacing_m for prior in positions):
            continue
        url = _image_url(record)
        identifier = _identifier(record)
        if not url or not identifier:
            continue
        image = decode_rgb(request_bytes(url, min_interval_s=0.5))
        if abs(image.shape[1] / image.shape[0] - 2.0) >= 0.03:
            continue
        positions.append(position) if position else None
        for view in range(entry.views):
            yaw = 360.0 * view / entry.views
            crop = equirect_to_perspective(image, yaw, entry.pitch_deg, entry.fov_deg, entry.crop_size)
            encoded, data = cv2.imencode(".jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            if encoded:
                result.append(FileProduct(data.tobytes(), f"{entry.session}_{identifier}_yaw{int(yaw)}.jpg", 1.0,
                                          url, identifier, "CC BY 4.0", _ATTRIBUTION))
                if limit is not None and len(result) >= limit:
                    return result
    if not result:
        logger.warning(
            "Amsterdam panorama %s yielded 0 items: %s",
            entry.id,
            "API returned no panorama records" if not seen_records else "no records had a usable panorama image",
        )
    return result
