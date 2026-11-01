"""Tests for flurseg.dataset.build: dataset construction, layout, manifest, guards, and validation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from flurseg.config import load_config
from flurseg.dataset.build import build_dataset
from flurseg.dataset.validate import MaskValidationError
from flurseg.io.atomic import atomic_save_jpg, atomic_save_png
from flurseg.manifest.db import Database
from flurseg.manifest.migrations import migrate
from flurseg.manifest.repo import ImageRecord, ManifestRepository
from flurseg.paths import RepoPaths


def _make_pool_image(
    paths: RepoPaths,
    image_id: str,
    shape: tuple[int, int] = (1024, 1024),
    mask_value: int = 1,
    *,
    source: str | None = None,
    session: str | None = None,
    is_test_session: bool | None = None,
    status: str = "reviewed",
    record: bool = True,
) -> tuple[Path, Path]:
    """Create a valid synthetic pool image, single-channel mask, and manifest record."""
    paths.pool_images_dir.mkdir(parents=True, exist_ok=True)
    paths.pool_masks_dir.mkdir(parents=True, exist_ok=True)

    img_path = paths.pool_images_dir / f"{image_id}.jpg"
    mask_path = paths.pool_masks_dir / f"{image_id}.png"

    height, width = shape
    image = np.full((height, width, 3), 120, dtype=np.uint8)
    atomic_save_jpg(img_path, image)

    mask = np.full((height, width), mask_value, dtype=np.uint8)
    atomic_save_png(mask_path, mask)

    if record:
        parts = image_id.split("_")
        resolved_source = source if source is not None else (parts[0] if len(parts) >= 2 else "drone")
        resolved_session = session if session is not None else (parts[1] if len(parts) >= 2 else "sessionA")
        resolved_is_test = (
            is_test_session
            if is_test_session is not None
            else ("test" in resolved_session.lower())
        )

        with Database(paths.database) as conn:
            migrate(conn)
            repo = ManifestRepository(conn)
            if repo.get(image_id) is None:
                repo.insert(
                    ImageRecord(
                        id=image_id,
                        hash=f"hash_{image_id}",
                        source=resolved_source,
                        session=resolved_session,
                        gsd_method="fixed",
                        original_path=str(img_path),
                        image_path=str(img_path),
                        work_dir=str(paths.root / "work" / image_id),
                        width=width,
                        height=height,
                        status=status,
                        is_test_session=resolved_is_test,
                    )
                )

    return img_path, mask_path


def test_full_build_directory_layout_and_manifest(tmp_root: Path) -> None:
    paths = RepoPaths.from_root(tmp_root)
    cfg = load_config(paths)

    _make_pool_image(paths, "drone_sessionA_0001", mask_value=1)
    _make_pool_image(paths, "drone_sessionA_0002", mask_value=2)
    _make_pool_image(paths, "drone_sessionB_0001", mask_value=3)
    _make_pool_image(paths, "phone_sessionC_0001", mask_value=4)
    _make_pool_image(paths, "phone_testsession_0001", mask_value=5)

    dataset_dict = build_dataset(paths, cfg)

    # 1. Directory layout checks
    dataset_dir = paths.dataset_dir
    assert dataset_dir.is_dir()
    manifest_file = dataset_dir / "dataset.json"
    assert manifest_file.is_file()

    for split in ("train", "val", "test"):
        assert (dataset_dir / split / "images").is_dir()
        assert (dataset_dir / split / "masks").is_dir()

    # Image files are .jpg, mask files are .png
    all_images = list(dataset_dir.glob("*/*/*.jpg"))
    all_masks = list(dataset_dir.glob("*/*/*.png"))
    assert len(all_images) == 5
    assert len(all_masks) == 5
    for img in all_images:
        corresponding_mask = img.parent.parent / "masks" / f"{img.stem}.png"
        assert corresponding_mask.is_file()

    # 2. dataset.json content checks
    with manifest_file.open("r", encoding="utf-8") as f:
        loaded_json = json.load(f)
    assert loaded_json == dataset_dict

    assert dataset_dict["version"] == "1.0.0"
    assert "created_at" in dataset_dict
    assert dataset_dict["tile_size"] == cfg.pipeline.dataset.tile_size
    assert dataset_dict["ignore_index"] == 255

    # Classes catalogue
    classes = dataset_dict["classes"]
    assert len(classes) == 12
    for cls in classes:
        assert "id" in cls
        assert "name" in cls
        assert "color" in cls
        assert "colour" in cls
        assert "cyclone_code" in cls

    # Split per session: test session must be in 'test'
    split_per_session = dataset_dict["split_per_session"]
    assert "testsession" in split_per_session
    assert split_per_session["testsession"] == "test"
    for sp in split_per_session.values():
        assert sp in {"train", "val", "test"}

    # Frequencies sum
    freqs = dataset_dict["class_pixel_frequencies"]
    assert "overall" in freqs
    assert sum(freqs["overall"].values()) == pytest.approx(1.0)
    for split in ("train", "val", "test"):
        assert split in freqs
        if dataset_dict["tile_counts"]["by_split"][split] > 0:
            assert sum(freqs[split].values()) == pytest.approx(1.0)

    # Tile counts
    tile_counts = dataset_dict["tile_counts"]
    assert tile_counts["total"] == 5
    assert sum(tile_counts["by_split"].values()) == 5
    assert tile_counts["by_source"]["drone"] == 3
    assert tile_counts["by_source"]["phone"] == 2


def test_two_sessions_with_no_validation_records_warn_and_mark_manifest(
    tmp_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A missing validation split is explicitly recorded for training fallback."""
    paths = RepoPaths.from_root(tmp_root)
    cfg = load_config(paths)
    _make_pool_image(paths, "drone_train_0001", source="drone", session="train")
    _make_pool_image(
        paths,
        "phone_test_0001",
        source="phone",
        session="test",
        is_test_session=True,
    )

    result = build_dataset(paths, cfg)

    assert result["val_empty"] is True
    assert "WARNING: validation split is empty" in capsys.readouterr().out


def test_rebuild_removes_old_files(tmp_root: Path) -> None:
    paths = RepoPaths.from_root(tmp_root)
    cfg = load_config(paths)

    _make_pool_image(paths, "drone_sessionA_0001")
    build_dataset(paths, cfg)

    # Inject stale artifacts
    stale_file = paths.dataset_dir / "train" / "images" / "stale_image.jpg"
    stale_file.write_text("old", encoding="utf-8")
    stale_root = paths.dataset_dir / "stale_canary.txt"
    stale_root.write_text("canary", encoding="utf-8")
    assert stale_file.is_file()
    assert stale_root.is_file()

    # Rebuild
    build_dataset(paths, cfg)
    assert not stale_file.exists()
    assert not stale_root.exists()
    assert (paths.dataset_dir / "dataset.json").is_file()


def test_invalid_pool_mask_aborts_without_deleting_existing_dataset(tmp_root: Path) -> None:
    paths = RepoPaths.from_root(tmp_root)
    cfg = load_config(paths)

    _make_pool_image(paths, "drone_sessionA_0001")
    build_dataset(paths, cfg)

    # Place a sentinel in the existing dataset
    sentinel = paths.dataset_dir / "preserve_this_dataset.txt"
    sentinel.write_text("do_not_delete", encoding="utf-8")

    # Introduce an invalid mask in the pool (class 11 is outside 0..10 and 255)
    bad_img = paths.pool_images_dir / "drone_bad_0001.jpg"
    bad_mask = paths.pool_masks_dir / "drone_bad_0001.png"
    atomic_save_jpg(bad_img, np.zeros((1024, 1024, 3), dtype=np.uint8))
    atomic_save_png(bad_mask, np.full((1024, 1024), 11, dtype=np.uint8))

    with pytest.raises((MaskValidationError, ValueError)):
        build_dataset(paths, cfg)

    # Existing dataset and sentinel file must remain untouched
    assert sentinel.is_file()
    assert sentinel.read_text(encoding="utf-8") == "do_not_delete"


def test_guard_refuses_target_outside_train_dataset(tmp_root: Path) -> None:
    paths = RepoPaths.from_root(tmp_root)
    _make_pool_image(paths, "drone_sessionA_0001")
    cfg = load_config(paths)

    # Case 1: outside repository root
    outside_paths = SimpleNamespace(
        root=tmp_root,
        dataset_dir=tmp_root.parent / "outside_dataset",
        pool_images_dir=paths.pool_images_dir,
        pool_masks_dir=paths.pool_masks_dir,
    )
    with pytest.raises(ValueError):
        build_dataset(outside_paths, cfg)

    # Case 2: inside root but wrong path
    wrong_path = SimpleNamespace(
        root=tmp_root,
        dataset_dir=tmp_root / "other_dir",
        pool_images_dir=paths.pool_images_dir,
        pool_masks_dir=paths.pool_masks_dir,
    )
    with pytest.raises(ValueError):
        build_dataset(wrong_path, cfg)

    # Case 3: inside root / train but not dataset
    wrong_sub_path = SimpleNamespace(
        root=tmp_root,
        dataset_dir=tmp_root / "train" / "not_dataset",
        pool_images_dir=paths.pool_images_dir,
        pool_masks_dir=paths.pool_masks_dir,
    )
    with pytest.raises(ValueError):
        build_dataset(wrong_sub_path, cfg)


def test_small_image_padded_with_255(tmp_root: Path) -> None:
    paths = RepoPaths.from_root(tmp_root)
    cfg = load_config(paths)

    # Small image 512x512
    _make_pool_image(paths, "drone_sessionA_0001", shape=(512, 512), mask_value=1)
    result = build_dataset(paths, cfg)

    assert result["tile_counts"]["total"] >= 1
    tile_masks = list(paths.dataset_dir.glob("*/*/*.png"))
    assert len(tile_masks) >= 1

    tile_arr = cv2.imread(str(tile_masks[0]), cv2.IMREAD_UNCHANGED)
    assert tile_arr.shape == (1024, 1024)
    # Original content preserved at top-left
    assert np.all(tile_arr[:512, :512] == 1)
    # Padded region filled with 255 (ignore)
    assert np.all(tile_arr[512:, :] == 255)
    assert np.all(tile_arr[:, 512:] == 255)


def test_ignore_heavy_tiles_dropped(tmp_root: Path) -> None:
    paths = RepoPaths.from_root(tmp_root)
    cfg = load_config(paths)

    _img_path, mask_path = _make_pool_image(paths, "drone_sessionA_0001")
    # >95% ignore
    mask = np.full((1024, 1024), 255, dtype=np.uint8)
    mask[:50, :50] = 1
    atomic_save_png(mask_path, mask)

    result = build_dataset(paths, cfg)
    assert result["tile_counts"]["total"] == 0


def test_dataset_build_resolves_metadata_from_database(tmp_root: Path) -> None:
    paths = RepoPaths.from_root(tmp_root)
    cfg = load_config(paths)

    _make_pool_image(paths, "item_alpha_001", record=False)
    _make_pool_image(paths, "item_beta_002", record=False)

    with Database(paths.database) as conn:
        repo = ManifestRepository(conn)
        repo.insert(
            ImageRecord(
                id="item_alpha_001",
                hash="hash_alpha",
                source="ortho",
                session="flight_alpha",
                gsd_method="fixed",
                original_path="dummy",
                image_path="dummy",
                work_dir="dummy",
                width=1024,
                height=1024,
                status="auto_accepted",
                is_test_session=False,
            )
        )
        repo.insert(
            ImageRecord(
                id="item_beta_002",
                hash="hash_beta",
                source="ortho",
                session="flight_beta_test",
                gsd_method="fixed",
                original_path="dummy",
                image_path="dummy",
                work_dir="dummy",
                width=1024,
                height=1024,
                status="auto_accepted",
                is_test_session=True,
            )
        )

    result = build_dataset(paths, cfg)
    assert result["split_per_session"]["flight_beta_test"] == "test"
    assert "item_beta_002" in result["source_image_ids_per_split"]["test"]


def test_pool_image_without_manifest_record_aborts_build(tmp_root: Path) -> None:
    paths = RepoPaths.from_root(tmp_root)
    cfg = load_config(paths)

    # Pre-populate dataset with a canary file
    paths.dataset_dir.mkdir(parents=True, exist_ok=True)
    sentinel = paths.dataset_dir / "preserve_this_dataset.txt"
    sentinel.write_text("do_not_delete", encoding="utf-8")

    # One pool image with a manifest record, one without
    _make_pool_image(paths, "drone_sessionA_0001", record=True)
    _make_pool_image(paths, "drone_sessionA_0002", record=False)

    with pytest.raises(ValueError) as exc_info:
        build_dataset(paths, cfg)

    assert "drone_sessionA_0002" in str(exc_info.value)
    assert sentinel.is_file()
    assert sentinel.read_text(encoding="utf-8") == "do_not_delete"


def test_missing_manifest_database_aborts_build(tmp_root: Path) -> None:
    paths = RepoPaths.from_root(tmp_root)
    cfg = load_config(paths)

    # Pre-populate dataset with a canary file
    paths.dataset_dir.mkdir(parents=True, exist_ok=True)
    sentinel = paths.dataset_dir / "preserve_this_dataset.txt"
    sentinel.write_text("do_not_delete", encoding="utf-8")

    # Pool image present, but manifest database file does not exist
    _make_pool_image(paths, "drone_sessionA_0001", record=False)
    assert not paths.database.exists()

    with pytest.raises((RuntimeError, FileNotFoundError)) as exc_info:
        build_dataset(paths, cfg)

    assert "Manifest database file not found" in str(exc_info.value)
    assert sentinel.is_file()
    assert sentinel.read_text(encoding="utf-8") == "do_not_delete"
