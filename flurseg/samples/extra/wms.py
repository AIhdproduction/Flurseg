"""WMS 1.3.0 window retrieval."""

from __future__ import annotations

import logging
import urllib.parse

import numpy as np
from rasterio.transform import from_bounds
from rasterio.warp import transform

from flurseg.samples.extra.common import (
    WINDOW_PIXELS,
    ImageProduct,
    decode_rgb,
    ground_gsd_for_web_mercator,
    is_empty_image,
    is_low_information_image,
    jitter_centre,
    request_bytes,
)
from flurseg.samples.extra.config import WmsEntry

logger = logging.getLogger(__name__)


def _bbox_string(bounds: tuple[float, float, float, float], axis_order: str) -> str:
    west, south, east, north = bounds
    values = (south, west, north, east) if axis_order == "ne" else bounds
    return ",".join(f"{value:.10f}" for value in values)


def _fetch_wms_window(entry: WmsEntry, lon: float, lat: float) -> ImageProduct | None:
    x, y = transform("EPSG:4326", entry.crs, [lon], [lat])
    half = WINDOW_PIXELS * entry.gsd_m / 2
    bounds = (x[0] - half, y[0] - half, x[0] + half, y[0] + half)
    chunk = entry.max_request_px
    assembled = np.zeros((WINDOW_PIXELS, WINDOW_PIXELS, 3), dtype=np.uint8)
    request_urls: list[str] = []
    for top in range(0, WINDOW_PIXELS, chunk):
        for left in range(0, WINDOW_PIXELS, chunk):
            width = min(chunk, WINDOW_PIXELS - left)
            height = min(chunk, WINDOW_PIXELS - top)
            west = bounds[0] + left * entry.gsd_m
            east = west + width * entry.gsd_m
            north = bounds[3] - top * entry.gsd_m
            south = north - height * entry.gsd_m
            parameters = {
                "service": "WMS",
                "version": "1.3.0",
                "request": "GetMap",
                "layers": entry.layer,
                "styles": "",
                "crs": entry.crs,
                "bbox": _bbox_string((west, south, east, north), entry.axis_order),
                "width": str(width),
                "height": str(height),
                "format": entry.format,
                "transparent": "false",
            }
            url = f"{entry.url}?{urllib.parse.urlencode(parameters)}"
            payload = request_bytes(url, min_interval_s=entry.min_interval_s)
            image = decode_rgb(payload)
            if is_empty_image(payload, image):
                return None
            if image.shape[:2] != (height, width):
                raise ValueError(f"WMS returned {image.shape[:2]} for requested {height}x{width}")
            assembled[top : top + height, left : left + width] = image
            request_urls.append(url)
    return ImageProduct(
        data=assembled,
        transform=from_bounds(*bounds, WINDOW_PIXELS, WINDOW_PIXELS),
        crs=entry.crs,
        gsd_m=ground_gsd_for_web_mercator(entry.gsd_m, lat, entry.crs),
        url=request_urls[0],
        title=entry.id,
        licence=entry.licence,
        attribution=entry.attribution,
    )


def get_wms_window(entry: WmsEntry, lon: float, lat: float, max_tries: int = 3) -> ImageProduct | None:
    """Retrieve and stitch one 2048 px WMS window around a WGS84 centre with jitter retry."""
    for attempt in range(max_tries):
        cur_lon, cur_lat = jitter_centre(lon, lat, entry.gsd_m, attempt)
        product = _fetch_wms_window(entry, cur_lon, cur_lat)
        if product is None or is_low_information_image(product.data):
            logger.info("skipped_low_information wms %s at (%.5f, %.5f)", entry.id, cur_lon, cur_lat)
            continue
        return product
    return None
