from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from flurseg.artifacts import ArtifactStore
from flurseg.paths import RepoPaths
from flurseg.review.pyramid import Pyramid


def test_pyramid_levels_tiles_cache_and_edges(tmp_root) -> None:
    paths = RepoPaths.from_root(tmp_root)
    artifact = ArtifactStore(paths.work_dir / "image")
    artifact.write_image(np.full((600, 900, 3), 120, dtype=np.uint8))
    artifact.write_raw_mask(np.ones((600, 900), dtype=np.uint8))
    pyramid = Pyramid(paths, "image", artifact.directory)
    assert pyramid.levels() == 2
    assert pyramid.grid(0) == (4, 3)
    first = pyramid.tile("image", 0, 0, 0)
    assert first == pyramid.tile("image", 0, 0, 0)
    edge = cv2.imread(str(pyramid.tile("image", 0, 3, 2)), cv2.IMREAD_COLOR)
    assert edge.shape[:2] == (88, 132)
    assert cv2.imread(str(pyramid.tile("raw_mask", 1, 0, 0)), cv2.IMREAD_COLOR).shape[:2] == (256, 256)


def test_concurrent_level_build_is_safe(tmp_root) -> None:
    """16 threads request tiles of the same level of a 3000x2000 image simultaneously."""
    paths = RepoPaths.from_root(tmp_root)
    artifact = ArtifactStore(paths.work_dir / "large_image")
    artifact.write_image(np.full((2000, 3000, 3), 140, dtype=np.uint8))
    pyramid = Pyramid(paths, "large_image", artifact.directory)

    # 16 concurrent requests for tiles of level 1 racing on level generation and tile cache
    coords = [(index % 4, (index // 4) % 4) for index in range(16)]
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(pyramid.tile, "image", 1, x, y) for x, y in coords]
        results = [future.result() for future in futures]

    level_h, level_w = pyramid._shape_for_level(1)
    assert len(results) == 16
    for (x, y), tile_path in zip(coords, results):
        assert tile_path.is_file()
        image = cv2.imread(str(tile_path), cv2.IMREAD_COLOR)
        assert image is not None
        expected_w = min(256, level_w - x * 256)
        expected_h = min(256, level_h - y * 256)
        assert image.shape[:2] == (expected_h, expected_w)


def test_pyramid_rebuilds_a_layer_when_its_source_changes(tmp_root) -> None:
    paths = RepoPaths.from_root(tmp_root)
    artifact = ArtifactStore(paths.work_dir / "image")
    artifact.write_image(np.full((300, 300, 3), 120, dtype=np.uint8))
    artifact.write_raw_mask(np.ones((300, 300), dtype=np.uint8))
    pyramid = Pyramid(paths, "image", artifact.directory)
    first = pyramid.tile("raw_mask", 0, 0, 0)
    source = artifact.directory / "raw_mask.png"
    before = source.stat()
    artifact.write_raw_mask(np.full((300, 300), 2, dtype=np.uint8))
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))
    second = pyramid.tile("raw_mask", 0, 0, 0)
    assert first != second
    assert not first.exists()
    tile = cv2.imread(str(second), cv2.IMREAD_COLOR)
    assert tile is not None and tuple(tile[0, 0]) == (192, 107, 92)


def test_concurrent_access_during_layer_rebuild_is_safe(tmp_root) -> None:
    paths = RepoPaths.from_root(tmp_root)
    artifact = ArtifactStore(paths.work_dir / "image")
    artifact.write_image(np.full((600, 600, 3), 120, dtype=np.uint8))
    artifact.write_raw_mask(np.ones((600, 600), dtype=np.uint8))
    pyramid = Pyramid(paths, "image", artifact.directory)
    pyramid.tile("raw_mask", 0, 0, 0)
    source = artifact.directory / "raw_mask.png"
    before = source.stat()
    artifact.write_raw_mask(np.full((600, 600), 2, dtype=np.uint8))
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))
    coordinates = [(index % 3, index // 3) for index in range(9)]
    with ThreadPoolExecutor(max_workers=9) as pool:
        results = list(pool.map(lambda point: pyramid.tile("raw_mask", 0, *point), coordinates))
    for path in results:
        tile = cv2.imread(str(path), cv2.IMREAD_COLOR)
        assert tile is not None and tuple(tile[0, 0]) == (192, 107, 92)
