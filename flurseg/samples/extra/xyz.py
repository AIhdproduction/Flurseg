"""XYZ and WMTS tile template retrieval."""

from __future__ import annotations

import logging
import math

import numpy as np
from rasterio.transform import from_origin

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
from flurseg.samples.extra.config import XyzEntry

logger = logging.getLogger(__name__)

WEB_MERCATOR_HALF = 20037508.342789244
WEB_MERCATOR_PIXEL = 156543.03392


def tile_range(lon: float, lat: float, zoom: int) -> tuple[int, int, int, int, int, int]:
    """Return tile bounds and pixel offset for a 2048 px window around a centre."""
    n = 2**zoom
    world = 256 * n
    pixel_x = (lon + 180.0) / 360.0 * world
    sine = math.sin(math.radians(max(-85.05112878, min(85.05112878, lat))))
    pixel_y = (0.5 - math.log((1 + sine) / (1 - sine)) / (4 * math.pi)) * world
    left = math.floor(pixel_x - WINDOW_PIXELS / 2)
    top = math.floor(pixel_y - WINDOW_PIXELS / 2)
    return (
        left // 256,
        top // 256,
        (left + WINDOW_PIXELS - 1) // 256,
        (top + WINDOW_PIXELS - 1) // 256,
        left,
        top,
    )


def _fetch_xyz_window(entry: XyzEntry, lon: float, lat: float) -> ImageProduct | None:
    min_x, min_y, max_x, max_y, left, top = tile_range(lon, lat, entry.zoom)
    n = 2**entry.zoom
    canvas = np.zeros(((max_y - min_y + 1) * 256, (max_x - min_x + 1) * 256, 3), dtype=np.uint8)
    first_url = ""
    for tile_y in range(min_y, max_y + 1):
        for tile_x in range(min_x, max_x + 1):
            wrapped_x = tile_x % n
            if tile_y < 0 or tile_y >= n:
                return None
            url = (
                entry.url.replace("{z}", str(entry.zoom))
                .replace("{x}", str(wrapped_x))
                .replace("{y}", str(tile_y))
            )
            payload = request_bytes(url, min_interval_s=entry.min_interval_s)
            image = decode_rgb(payload)
            if is_empty_image(payload, image):
                return None
            if image.shape[:2] != (256, 256):
                raise ValueError(f"XYZ tile must be 256x256, got {image.shape[:2]}")
            offset_y = (tile_y - min_y) * 256
            offset_x = (tile_x - min_x) * 256
            canvas[offset_y : offset_y + 256, offset_x : offset_x + 256] = image
            first_url = first_url or url
    crop_x = left - min_x * 256
    crop_y = top - min_y * 256
    image = canvas[crop_y : crop_y + WINDOW_PIXELS, crop_x : crop_x + WINDOW_PIXELS]
    resolution = WEB_MERCATOR_PIXEL / n
    west = -WEB_MERCATOR_HALF + left * resolution
    north = WEB_MERCATOR_HALF - top * resolution
    ground_gsd = ground_gsd_for_web_mercator(resolution, lat, "EPSG:3857")
    return ImageProduct(
        data=image,
        transform=from_origin(west, north, resolution, resolution),
        crs="EPSG:3857",
        gsd_m=ground_gsd,
        url=first_url,
        title=entry.id,
        licence=entry.licence,
        attribution=entry.attribution,
    )


def get_xyz_window(entry: XyzEntry, lon: float, lat: float, max_tries: int = 3) -> ImageProduct | None:
    """Stitch tile images and crop an exact 2048 px Web Mercator window with jitter retry."""
    n = 2**entry.zoom
    resolution = WEB_MERCATOR_PIXEL / n
    for attempt in range(max_tries):
        ground_gsd = ground_gsd_for_web_mercator(resolution, lat, "EPSG:3857")
        cur_lon, cur_lat = jitter_centre(lon, lat, ground_gsd, attempt)
        product = _fetch_xyz_window(entry, cur_lon, cur_lat)
        if product is None or is_low_information_image(product.data):
            logger.info("skipped_low_information xyz %s at (%.5f, %.5f)", entry.id, cur_lon, cur_lat)
            continue
        return product
    return None
