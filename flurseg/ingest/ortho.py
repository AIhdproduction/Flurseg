"""GeoTIFF extraction with nodata filtering and georeference preservation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

from flurseg.segment.tiling import tile_starts


@dataclass(frozen=True, slots=True)
class OrthoTile:
    """Extracted RGB ortho tile and its raster metadata."""

    image: np.ndarray
    window: Window
    transform: object
    crs_wkt: str | None
    index: int


def read_ortho_tiles(path: Path, tile_size: int, max_nodata_fraction: float, black_threshold: int = 5) -> list[OrthoTile]:
    """Read kept RGB tiles whose invalid/black fraction is within the configured limit."""
    kept: list[OrthoTile] = []
    with rasterio.open(path) as dataset:
        if dataset.count < 3:
            raise ValueError(f"orthophoto needs at least three bands: {path}")
        index = 0
        for y in tile_starts(dataset.height, tile_size, 0):
            for x in tile_starts(dataset.width, tile_size, 0):
                width = min(tile_size, dataset.width - x)
                height = min(tile_size, dataset.height - y)
                window = Window(x, y, width, height)
                rgb = dataset.read((1, 2, 3), window=window, boundless=False)
                masks = dataset.read_masks((1, 2, 3), window=window)
                image = np.moveaxis(rgb, 0, -1).astype(np.uint8, copy=False)
                invalid = np.any(masks == 0, axis=0) | np.all(image <= black_threshold, axis=2)
                if float(invalid.mean()) <= max_nodata_fraction:
                    kept.append(
                        OrthoTile(
                            image=image.copy(), window=window,
                            transform=window_transform(window, dataset.transform),
                            crs_wkt=dataset.crs.to_wkt() if dataset.crs else None, index=index,
                        )
                    )
                index += 1
    return kept
