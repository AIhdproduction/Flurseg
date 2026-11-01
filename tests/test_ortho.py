"""Orthophoto tiling and metadata preservation coverage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from flurseg.ingest.ortho import read_ortho_tiles


def _write_ortho(path: Path, width: int, height: int, data: np.ndarray, *, nodata: int | None = None) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=3,
        dtype="uint8",
        nodata=nodata,
        transform=from_origin(2600000, 1200000, 0.1, 0.1),
        crs="EPSG:2056",
    ) as dataset:
        dataset.write(data)


def test_5000_by_3000_ortho_has_expected_tiles_covering_full_raster(tmp_path: Path) -> None:
    path = tmp_path / "large.tif"
    _write_ortho(path, 5000, 3000, np.full((3, 3000, 5000), 50, dtype=np.uint8))
    tiles = read_ortho_tiles(path, 2048, 0.5)
    assert len(tiles) == 6
    assert min(tile.window.col_off for tile in tiles) == 0
    assert min(tile.window.row_off for tile in tiles) == 0
    assert max(tile.window.col_off + tile.window.width for tile in tiles) == 5000
    assert max(tile.window.row_off + tile.window.height for tile in tiles) == 3000


def test_nodata_block_tile_is_dropped(tmp_path: Path) -> None:
    path = tmp_path / "nodata.tif"
    data = np.full((3, 4096, 4096), 50, dtype=np.uint8)
    data[:, :2048, :2048] = 0
    _write_ortho(path, 4096, 4096, data, nodata=0)
    tiles = read_ortho_tiles(path, 2048, 0.5)
    assert len(tiles) == 3
    assert all(not (tile.window.col_off == 0 and tile.window.row_off == 0) for tile in tiles)


def test_mostly_black_tile_is_dropped(tmp_path: Path) -> None:
    path = tmp_path / "black.tif"
    data = np.full((3, 4096, 4096), 50, dtype=np.uint8)
    data[:, :2048, :2048] = 0
    _write_ortho(path, 4096, 4096, data)
    tiles = read_ortho_tiles(path, 2048, 0.5, black_threshold=5)
    assert len(tiles) == 3


def test_tile_preserves_window_transform_and_crs(tmp_path: Path) -> None:
    path = tmp_path / "georef.tif"
    _write_ortho(path, 3000, 2500, np.full((3, 2500, 3000), 50, dtype=np.uint8))
    tile = read_ortho_tiles(path, 2048, 0.5)[-1]
    assert tile.crs_wkt is not None and "CH1903+" in tile.crs_wkt
    assert tile.transform.c == 2600000 + tile.window.col_off * 0.1
    assert tile.transform.f == 1200000 - tile.window.row_off * 0.1


def test_smaller_than_tile_size_returns_one_tile(tmp_path: Path) -> None:
    path = tmp_path / "small.tif"
    _write_ortho(path, 100, 200, np.full((3, 200, 100), 50, dtype=np.uint8))
    tiles = read_ortho_tiles(path, 2048, 0.5)
    assert len(tiles) == 1
    assert tiles[0].image.shape == (200, 100, 3)
