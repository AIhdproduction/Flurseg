"""Candidate ingestion, deduplication, naming, and archive guard coverage."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.transform import from_origin

from flurseg.config import load_config
from flurseg.ingest.discover import Candidate
from flurseg.ingest.ingest import archive_original, ingest_candidate
from flurseg.io.images import save_canonical_jpg
from flurseg.manifest.repo import ManifestRepository
from flurseg.paths import RepoPaths


def _repository(paths: RepoPaths) -> ManifestRepository:
    return ManifestRepository(sqlite3.connect(paths.database))


def _image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((24, 25, 3), color, dtype=np.uint8)).save(path)


def test_ingest_sequences_names_within_a_session_and_independently_per_session(tmp_root: Path) -> None:
    paths = RepoPaths.from_root(tmp_root)
    repository = _repository(paths)
    config = load_config(paths)
    first = paths.drone_input_dir / "a" / "first.png"
    second = paths.drone_input_dir / "a" / "second.png"
    other_session = paths.drone_input_dir / "b" / "first.png"
    _image(first, (1, 2, 3))
    _image(second, (4, 5, 6))
    _image(other_session, (7, 8, 9))
    ids = [
        ingest_candidate(Candidate(first, "drone", "a"), paths, repository, config)[0].id,
        ingest_candidate(Candidate(second, "drone", "a"), paths, repository, config)[0].id,
        ingest_candidate(Candidate(other_session, "drone", "b"), paths, repository, config)[0].id,
    ]
    assert ids == ["drone_a_0001", "drone_a_0002", "drone_b_0001"]


def test_ingest_deduplicates_on_second_run(tmp_root: Path) -> None:
    paths = RepoPaths.from_root(tmp_root)
    repository = _repository(paths)
    path = paths.drone_input_dir / "a" / "image.png"
    _image(path, (1, 2, 3))
    candidate = Candidate(path, "drone", "a")
    assert len(ingest_candidate(candidate, paths, repository, load_config(paths))) == 1
    assert ingest_candidate(candidate, paths, repository, load_config(paths)) == []


def test_identical_content_in_second_session_is_duplicate(tmp_root: Path) -> None:
    paths = RepoPaths.from_root(tmp_root)
    repository = _repository(paths)
    first = paths.drone_input_dir / "a" / "image.png"
    second = paths.drone_input_dir / "b" / "image.png"
    _image(first, (1, 2, 3))
    second.parent.mkdir(parents=True)
    second.write_bytes(first.read_bytes())
    config = load_config(paths)
    assert len(ingest_candidate(Candidate(first, "drone", "a"), paths, repository, config)) == 1
    assert ingest_candidate(Candidate(second, "drone", "b"), paths, repository, config) == []


def test_ortho_creates_one_record_per_kept_tile_with_parent_and_index(tmp_root: Path) -> None:
    paths = RepoPaths.from_root(tmp_root)
    path = paths.ortho_input_dir / "session" / "source.tif"
    path.parent.mkdir(parents=True)
    data = np.full((3, 2, 2050), 50, dtype=np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2050,
        height=2,
        count=3,
        dtype="uint8",
        transform=from_origin(0, 0, 0.1, 0.1),
        crs="EPSG:2056",
    ) as dataset:
        dataset.write(data)
    records = ingest_candidate(Candidate(path, "ortho", "session"), paths, _repository(paths), load_config(paths))
    assert len(records) == 2
    assert len({record.parent_hash for record in records}) == 1
    assert {record.tile_index for record in records} == {0, 1}


def test_heic_phone_input_becomes_jpg_in_work_directory(tmp_root: Path) -> None:
    paths = RepoPaths.from_root(tmp_root)
    path = paths.phone_input_dir / "session" / "phone.heic"
    path.parent.mkdir(parents=True)
    Image.fromarray(np.full((20, 21, 3), (40, 100, 220), dtype=np.uint8)).save(path, format="HEIF")
    record = ingest_candidate(Candidate(path, "phone", "session"), paths, _repository(paths), load_config(paths))[0]
    assert record.image_path.endswith(".jpg")
    assert Path(record.image_path).is_file()


def test_source_without_anonymization_is_unchanged(tmp_root: Path) -> None:
    paths = RepoPaths.from_root(tmp_root)
    path = paths.drone_input_dir / "session" / "drone.png"
    source = np.zeros((20, 21, 3), dtype=np.uint8)
    source[5:15, 5:15] = (240, 10, 10)
    path.parent.mkdir(parents=True)
    Image.fromarray(source).save(path)
    reference = tmp_root / "reference.jpg"
    quality = load_config(paths).pipeline.ingest.jpeg_quality
    save_canonical_jpg(reference, source, quality)
    record = ingest_candidate(Candidate(path, "drone", "session"), paths, _repository(paths), load_config(paths))[0]
    assert Path(record.image_path).read_bytes() == reference.read_bytes()


def test_archive_uses_collision_safe_name(tmp_root: Path) -> None:
    paths = RepoPaths.from_root(tmp_root)
    path = paths.drone_input_dir / "session" / "source.jpg"
    _image(path, (1, 2, 3))
    destination = paths.archive_dir / "drone" / "session"
    destination.mkdir(parents=True)
    _image(destination / "source.jpg", (3, 2, 1))
    archived = archive_original(path, paths, "drone", "session")
    assert archived.name == "source_1.jpg"
    assert archived.is_file()


def test_root_containment_guard_rejects_candidate_outside_repository(tmp_root: Path) -> None:
    paths = RepoPaths.from_root(tmp_root)
    outside = tmp_root.parent / "outside.jpg"
    with pytest.raises(ValueError, match="inside the configured repository root"):
        ingest_candidate(Candidate(outside, "drone", "a"), paths, _repository(paths), load_config(paths))
