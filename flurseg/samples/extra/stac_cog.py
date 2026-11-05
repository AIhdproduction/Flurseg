"""STAC static-catalog and API COG window retrieval."""

from __future__ import annotations

import json
import logging
import random
import urllib.parse
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

from flurseg.samples.extra.common import (
    WINDOW_PIXELS,
    ImageProduct,
    ground_gsd_for_web_mercator,
    intersecting,
    is_low_information_image,
    request_bytes,
)
from flurseg.samples.extra.config import StacSearchEntry, StacStaticEntry
from flurseg.samples.extra.licences import is_acceptable_licence

logger = logging.getLogger(__name__)


def _json(url: str, interval: float, *, data: bytes | None = None) -> dict[str, Any]:
    """Load a JSON object over the source request policy."""
    decoded = json.loads(request_bytes(url, min_interval_s=interval, data=data).decode("utf-8"))
    if not isinstance(decoded, dict):
        raise TypeError(f"STAC response from {url} is not an object")
    return decoded


def _absolute(base: str, href: str) -> str:
    """Resolve STAC relative links for HTTP and local test catalogues."""
    if href.startswith(("http://", "https://", "/")) or Path(href).is_absolute():
        return href
    if base.startswith(("http://", "https://")):
        return urllib.parse.urljoin(base, href)
    return str((Path(base).parent / href).resolve())


def _asset_href(item: dict[str, Any], base: str) -> str | None:
    assets = item.get("assets") or {}
    for key in ("visual", "image", "data"):
        asset = assets.get(key)
        if isinstance(asset, dict) and asset.get("href"):
            return _absolute(base, str(asset["href"]))
    for asset in assets.values():
        if isinstance(asset, dict) and asset.get("href"):
            return _absolute(base, str(asset["href"]))
    return None


def _rgb(data: np.ndarray) -> np.ndarray:
    """Convert raster bands to HWC uint8 RGB."""
    if data.shape[0] == 1:
        data = np.repeat(data, 3, axis=0)
    data = data[:3]
    if data.dtype != np.uint8:
        maximum = (
            float(np.iinfo(data.dtype).max)
            if np.issubdtype(data.dtype, np.integer)
            else float(data.max() or 1)
        )
        data = np.clip(data.astype(np.float32) * 255.0 / maximum, 0, 255).astype(np.uint8)
    return np.moveaxis(data, 0, -1)


def read_cog_window(
    href: str,
    *,
    target_gsd_m: float | None,
    rng: random.Random,
    metadata: tuple[str, str, str, str],
    max_tries: int = 4,
) -> ImageProduct | None:
    """Read one seeded, valid COG window and preserve its CRS and transform."""
    source = f"/vsicurl/{href}" if href.startswith(("http://", "https://")) else href
    environment = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
    }
    with rasterio.Env(**environment), rasterio.open(source) as dataset:
        if dataset.count < 1 or dataset.width < WINDOW_PIXELS or dataset.height < WINDOW_PIXELS:
            return None
        native_gsd = max(abs(dataset.transform.a), abs(dataset.transform.e))
        scale = target_gsd_m / native_gsd if target_gsd_m else 1.0
        read_size = max(WINDOW_PIXELS, round(WINDOW_PIXELS * scale))
        if read_size > dataset.width or read_size > dataset.height:
            read_size = min(dataset.width, dataset.height)
        if read_size < WINDOW_PIXELS:
            return None
        for attempt in range(max_tries):
            col = rng.randint(0, dataset.width - read_size)
            row = rng.randint(0, dataset.height - read_size)
            window = Window(col, row, read_size, read_size)
            masks = dataset.read_masks(1, window=window, out_shape=(WINDOW_PIXELS, WINDOW_PIXELS))
            if float(np.mean(masks > 0)) < 0.80:
                logger.info("skipped_low_information stac %s: empty mask (attempt %d)", metadata[0], attempt)
                continue
            data = dataset.read(
                indexes=list(range(1, min(dataset.count, 3) + 1)),
                window=window,
                out_shape=(min(dataset.count, 3), WINDOW_PIXELS, WINDOW_PIXELS),
                resampling=Resampling.bilinear,
            )
            rgb_data = _rgb(data)
            if is_low_information_image(rgb_data):
                logger.info(
                    "skipped_low_information stac %s: low information (attempt %d)", metadata[0], attempt
                )
                continue
            sub_transform = window_transform(window, dataset.transform) * Affine.scale(
                read_size / WINDOW_PIXELS,
                read_size / WINDOW_PIXELS,
            )
            output_gsd = native_gsd * read_size / WINDOW_PIXELS
            if dataset.crs is None:
                return None
            if str(dataset.crs).upper() == "EPSG:3857":
                centre_x = sub_transform.c + sub_transform.a * WINDOW_PIXELS / 2
                centre_y = sub_transform.f + sub_transform.e * WINDOW_PIXELS / 2
                _, latitudes = rasterio.warp.transform("EPSG:3857", "EPSG:4326", [centre_x], [centre_y])
                output_gsd = ground_gsd_for_web_mercator(output_gsd, latitudes[0], str(dataset.crs))
            title, licence, attribution, url = metadata
            return ImageProduct(
                data=rgb_data,
                transform=sub_transform,
                crs=str(dataset.crs),
                gsd_m=output_gsd,
                url=url,
                title=title,
                licence=licence,
                attribution=attribution,
            )
        return None


def static_products(
    entry: StacStaticEntry,
    licence_overrides: tuple[str, ...] = (),
    limit: int | None = None,
) -> list[ImageProduct]:
    """Resolve static STAC item links and read intersecting item windows."""
    with rasterio.Env() as environment:
        if "WEBP" not in environment.drivers():
            raise RuntimeError(
                "GDAL WebP support is required for LINZ WebP-compressed COGs; install a rasterio/GDAL build with WEBP."
            )
    collection = _json(entry.url, entry.min_interval_s)
    links = collection.get("links") or []
    items: list[tuple[dict[str, Any], str]] = []
    for link in links:
        if isinstance(link, dict) and link.get("rel") == "item" and link.get("href"):
            item_url = _absolute(entry.url, str(link["href"]))
            items.append((_json(item_url, entry.min_interval_s), item_url))
    products: list[ImageProduct] = []
    max_count = min(entry.max_per_entry, limit) if limit is not None else entry.max_per_entry
    for item, item_url in items:
        bbox = item.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) >= 4):
            continue
        item_bbox = tuple(float(value) for value in bbox[:4])
        if not any(intersecting(item_bbox, candidate) for candidate in entry.bboxes):
            continue
        href = _asset_href(item, item_url)
        if not href:
            continue
        properties = item.get("properties") or {}
        licence = str(properties.get("license") or entry.licence)
        if not is_acceptable_licence(licence, licence_overrides):
            continue
        product = read_cog_window(
            href,
            target_gsd_m=entry.target_gsd_m,
            rng=random.Random(f"{entry.id}:{item.get('id', href)}"),
            metadata=(str(item.get("id", entry.id)), licence, entry.attribution, href),
        )
        if product:
            products.append(product)
        if len(products) >= max_count:
            break
    return products


def search_products(
    entry: StacSearchEntry,
    licence_overrides: tuple[str, ...] = (),
    limit: int | None = None,
) -> list[ImageProduct]:
    """Search a STAC API and create valid seeded windows from eligible COGs."""
    products: list[ImageProduct] = []
    max_count = min(entry.max_per_entry, limit) if limit is not None else entry.max_per_entry
    for bbox in entry.bboxes:
        query: dict[str, Any] = {
            "collections": [entry.collection],
            "bbox": list(bbox),
            "limit": entry.max_items,
        }
        if entry.max_gsd_m is not None or entry.min_gsd_m is not None:
            condition: dict[str, Any] = {}
            if entry.max_gsd_m is not None:
                condition["lte"] = entry.max_gsd_m
            if entry.min_gsd_m is not None:
                condition["gte"] = entry.min_gsd_m
            query["filter-lang"] = "cql2-json"
            query["filter"] = (
                {"op": "and", "args": [{"op": "<=", "args": [{"property": "gsd"}, entry.max_gsd_m]}]}
                if entry.max_gsd_m is not None
                else condition
            )
        response = _json(entry.url, entry.min_interval_s, data=json.dumps(query).encode("utf-8"))
        for item in response.get("features", []):
            if not isinstance(item, dict):
                continue
            properties = item.get("properties") or {}
            licence = str(properties.get("license") or "")
            gsd = properties.get("gsd")
            if not is_acceptable_licence(licence, licence_overrides) or not isinstance(gsd, (int, float)):
                continue
            if entry.max_gsd_m is not None and gsd > entry.max_gsd_m:
                continue
            if entry.min_gsd_m is not None and gsd < entry.min_gsd_m:
                continue
            href = _asset_href(item, entry.url)
            if not href:
                continue
            providers = properties.get("providers") or []
            provider = (
                providers[0].get("name")
                if providers and isinstance(providers[0], dict)
                else "OpenAerialMap provider"
            )
            attribution = f"© {provider} via OpenAerialMap, CC BY 4.0"
            for number in range(entry.windows_per_item):
                product = read_cog_window(
                    href,
                    target_gsd_m=float(gsd),
                    rng=random.Random(f"{entry.id}:{item.get('id', href)}:{number}"),
                    metadata=(str(item.get("id", entry.id)), licence, attribution, href),
                )
                if product:
                    products.append(product)
                if len(products) >= max_count:
                    return products
    return products
    return products
