"""ArcGIS ImageServer and MapServer export window retrieval."""

from __future__ import annotations

import logging
import urllib.parse

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
from flurseg.samples.extra.config import ArcgisEntry

logger = logging.getLogger(__name__)


def _fetch_arcgis_window(entry: ArcgisEntry, lon: float, lat: float) -> ImageProduct | None:
    x, y = transform("EPSG:4326", entry.crs, [lon], [lat])
    half = WINDOW_PIXELS * entry.gsd_m / 2
    bounds = (x[0] - half, y[0] - half, x[0] + half, y[0] + half)
    parameters = {
        "bbox": ",".join(f"{value:.10f}" for value in bounds),
        "bboxSR": entry.crs.removeprefix("EPSG:"),
        "imageSR": entry.crs.removeprefix("EPSG:"),
        "size": f"{WINDOW_PIXELS},{WINDOW_PIXELS}",
        "format": entry.format,
        "f": "image",
    }
    separator = "&" if "?" in entry.url else "?"
    url = f"{entry.url}{separator}{urllib.parse.urlencode(parameters)}"
    payload = request_bytes(url, min_interval_s=entry.min_interval_s)
    image = decode_rgb(payload)
    if is_empty_image(payload, image):
        return None
    if image.shape[:2] != (WINDOW_PIXELS, WINDOW_PIXELS):
        raise ValueError(f"ArcGIS returned {image.shape[:2]} instead of 2048x2048")
    return ImageProduct(
        data=image,
        transform=from_bounds(*bounds, WINDOW_PIXELS, WINDOW_PIXELS),
        crs=entry.crs,
        gsd_m=ground_gsd_for_web_mercator(entry.gsd_m, lat, entry.crs),
        url=url,
        title=entry.id,
        licence=entry.licence,
        attribution=entry.attribution,
    )


def get_arcgis_window(entry: ArcgisEntry, lon: float, lat: float, max_tries: int = 3) -> ImageProduct | None:
    """Request one 2048 px RGB ArcGIS export image with jitter retry."""
    for attempt in range(max_tries):
        cur_lon, cur_lat = jitter_centre(lon, lat, entry.gsd_m, attempt)
        product = _fetch_arcgis_window(entry, cur_lon, cur_lat)
        if product is None or is_low_information_image(product.data):
            logger.info("skipped_low_information arcgis %s at (%.5f, %.5f)", entry.id, cur_lon, cur_lat)
            continue
        return product
    return None
