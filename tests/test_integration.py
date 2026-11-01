from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import pillow_heif
import pytest
import rasterio
from fastapi.testclient import TestClient
from PIL import Image
from rasterio.transform import from_origin

from flurseg.config import load_config
from flurseg.dataset.build import build_dataset
from flurseg.manifest.db import Database
from flurseg.manifest.repo import ManifestRepository
from flurseg.paths import RepoPaths
from flurseg.pipeline import prelabel
from flurseg.report.render import generate_report
from flurseg.review.app import create_app
from flurseg.segment.fake_backend import FakeBackend


def _synthetic_content(height: int, width: int, seed: int = 0) -> np.ndarray:
    """Generate a multi-class synthetic RGB image for FakeBackend with seed-derived content."""
    rng = np.random.default_rng(seed)
    base_val = int(110 + rng.integers(-5, 6))
    image = np.full((height, width, 3), base_val, dtype=np.int16)

    offset_y = int(rng.integers(-max(2, height // 10), max(3, height // 10 + 1)))
    offset_x = int(rng.integers(-max(2, width // 10), max(3, width // 10 + 1)))
    split_y = max(4, min(height - 4, height // 2 + offset_y))
    split_x = max(4, min(width - 4, width // 2 + offset_x))

    # Vegetation patch (green) -> low_vegetation / tree
    image[:split_y, :split_x] = [30, 190, 30]
    # Building roof patch (reddish) -> building
    image[:split_y, split_x:] = [210, 55, 45]
    # Road marking stripe (white line) -> road_marking
    stripe_h = max(2, height // 10)
    stripe_y = max(0, min(height - stripe_h, split_y))
    image[stripe_y : stripe_y + stripe_h, :] = [250, 250, 250]

    noise = rng.integers(-2, 3, size=(height, width, 3), dtype=np.int16)
    return np.clip(image + noise, 0, 255).astype(np.uint8)


def _write_jpg(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path, format="JPEG", quality=95)


def _write_heic(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_image = Image.fromarray(image)
    try:
        heif_file = pillow_heif.from_pillow(pil_image)
        heif_file.save(str(path))
    except (AttributeError, ValueError, OSError):
        pil_image.save(path, format="HEIF")


def _write_ortho(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = image.shape[:2]
    transform = from_origin(2600000.0, 1200000.0, 0.10, 0.10)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype="uint8",
        transform=transform,
        crs="EPSG:2056",
    ) as dataset:
        dataset.write(np.moveaxis(image, -1, 0))


@pytest.mark.integration
def test_full_workflow(tmp_root: Path) -> None:
    paths = RepoPaths.from_root(tmp_root)

    # 1. Inputs:
    # 3 drone sessions (2 JPGs each, ~640x480, one session has session.yaml with test: true and gsd_m 0.02)
    drone_s1 = paths.drone_input_dir / "drone_s1"
    _write_jpg(drone_s1 / "img1.jpg", _synthetic_content(480, 640, seed=1))
    _write_jpg(drone_s1 / "img2.jpg", _synthetic_content(480, 640, seed=2))

    drone_s2 = paths.drone_input_dir / "drone_s2"
    _write_jpg(drone_s2 / "img1.jpg", _synthetic_content(480, 640, seed=3))
    _write_jpg(drone_s2 / "img2.jpg", _synthetic_content(480, 640, seed=4))

    drone_test = paths.drone_input_dir / "drone_test"
    _write_jpg(drone_test / "img1.jpg", _synthetic_content(480, 640, seed=5))
    _write_jpg(drone_test / "img2.jpg", _synthetic_content(480, 640, seed=6))
    (drone_test / "session.yaml").write_text("test: true\ngsd_m: 0.02\n", encoding="utf-8")

    # 1 phone session with one HEIC (create with pillow_heif) and one JPG
    phone_s1 = paths.phone_input_dir / "phone_s1"
    _write_heic(phone_s1 / "phone1.heic", _synthetic_content(480, 640, seed=7))
    _write_jpg(phone_s1 / "phone2.jpg", _synthetic_content(480, 640, seed=8))

    # 1 ortho session with a small synthetic GeoTIFF (rasterio, EPSG:2056, 2500x2500, 10 cm)
    ortho_s1 = paths.ortho_input_dir / "ortho_s1"
    _write_ortho(ortho_s1 / "ortho.tif", _synthetic_content(2500, 2500, seed=9))

    # 2. Force everything to review for determinism
    cfg = load_config(paths)
    cfg.pipeline.routing.auto_accept_max_uncertain_fraction = 0.0

    summary = prelabel(paths, cfg, FakeBackend)
    assert summary.failed == 0
    assert summary.auto_accepted == 0
    assert summary.skipped_duplicates == 0
    assert summary.needs_review > 0

    with Database(paths.database) as connection:
        records = ManifestRepository(connection).query()

    assert summary.skipped_duplicates == 0
    assert len(records) == 6 + 2 + 4  # 6 drone + 2 phone + 4 ortho tiles
    assert len(records) == summary.needs_review
    assert all(record["status"] == "needs_review" for record in records)

    # 3. TestClient: GET queue, for every queued image GET mask bytes,
    # change a 20x20 block to class 4 (tree), PUT the mask,
    # then accept all images except one which is rejected.
    client = TestClient(create_app(paths, cfg, FakeBackend))
    queue_res = client.get("/api/queue")
    assert queue_res.status_code == 200
    queue = queue_res.json()
    assert len(queue) == len(records)

    # Select one image from drone_s1 to reject so the test session remains accepted in the pool
    rejected_id: str | None = None
    for item in queue:
        if item["source"] == "drone" and item["session"] == "drone_s1":
            rejected_id = item["id"]
            break
    assert rejected_id is not None

    for item in queue:
        image_id = item["id"]
        mask_response = client.get(f"/api/images/{image_id}/mask")
        assert mask_response.status_code == 200
        mask_bytes = mask_response.content
        h, w = item["height"], item["width"]
        assert len(mask_bytes) == h * w

        mask_array = np.frombuffer(mask_bytes, dtype=np.uint8).reshape((h, w)).copy()
        mask_array[:20, :20] = 4
        put_response = client.put(f"/api/images/{image_id}/mask", content=mask_array.tobytes())
        assert put_response.status_code == 204

        if image_id == rejected_id:
            reject_response = client.post(f"/api/images/{image_id}/reject", json={"review_seconds": 1.0})
            assert reject_response.status_code == 200
            assert reject_response.json()["status"] == "rejected"
        else:
            accept_response = client.post(f"/api/images/{image_id}/accept", json={"review_seconds": 1.0})
            assert accept_response.status_code == 200
            assert accept_response.json()["status"] == "reviewed"

    with Database(paths.database) as connection:
        records_after = {r["id"]: r for r in ManifestRepository(connection).query()}

    assert records_after[rejected_id]["status"] == "rejected"
    assert not (paths.pool_images_dir / f"{rejected_id}.jpg").exists()
    assert not (paths.pool_masks_dir / f"{rejected_id}.png").exists()

    for item in queue:
        image_id = item["id"]
        if image_id == rejected_id:
            continue
        assert records_after[image_id]["status"] == "reviewed"
        assert (paths.pool_images_dir / f"{image_id}.jpg").is_file()
        assert (paths.pool_masks_dir / f"{image_id}.png").is_file()

    # 4. build_dataset -> assert train/val/test directories, dataset.json classes
    # and that the test session ended in the test split, every mask value in {0..10, 255}
    dataset_info = build_dataset(paths, cfg)

    for split in ("train", "val", "test"):
        assert (paths.dataset_dir / split / "images").is_dir()
        assert (paths.dataset_dir / split / "masks").is_dir()

    assert (paths.dataset_dir / "dataset.json").is_file()
    assert "classes" in dataset_info
    assert len(dataset_info["classes"]) >= 11

    assert "drone_test" in dataset_info["split_to_sessions"]["test"]
    assert dataset_info["split_per_session"]["drone_test"] == "test"

    allowed_mask_values = set(range(11)) | {255}
    all_exported_masks = list(paths.dataset_dir.glob("*/*/*.png"))
    assert len(all_exported_masks) > 0

    found_tree_class = False
    for mask_path in all_exported_masks:
        mask_data = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        assert mask_data is not None
        unique_vals = set(np.unique(mask_data))
        assert unique_vals.issubset(allowed_mask_values)
        if 4 in unique_vals:
            found_tree_class = True

    assert found_tree_class, "Expected class 4 (tree) in dataset masks after review edit"

    # 5. generate_report -> HTML exists, contains the word "Spot" or the status table,
    # has no "http://" or "https://"
    report_file = generate_report(paths, cfg)
    assert report_file.is_file()
    assert (paths.reports_dir / "latest.html").is_file()

    report_html = report_file.read_text(encoding="utf-8")
    assert "Spot" in report_html or "<table" in report_html.lower() or "status" in report_html.lower()
    assert "http://" not in report_html
    assert "https://" not in report_html

    latest_html = (paths.reports_dir / "latest.html").read_text(encoding="utf-8")
    assert "http://" not in latest_html
    assert "https://" not in latest_html


@pytest.mark.integration
def test_second_prelabel_run_is_idempotent(tmp_root: Path) -> None:
    paths = RepoPaths.from_root(tmp_root)
    cfg = load_config(paths)
    cfg.pipeline.routing.auto_accept_max_uncertain_fraction = 0.0

    drone_s = paths.drone_input_dir / "drone_idem"
    _write_jpg(drone_s / "drone.jpg", _synthetic_content(480, 640, seed=21))

    phone_s = paths.phone_input_dir / "phone_idem"
    _write_heic(phone_s / "phone.heic", _synthetic_content(480, 640, seed=22))

    ortho_s = paths.ortho_input_dir / "ortho_idem"
    _write_ortho(ortho_s / "ortho.tif", _synthetic_content(100, 100, seed=23))

    summary1 = prelabel(paths, cfg, FakeBackend)
    assert summary1.failed == 0
    assert summary1.skipped_duplicates == 0
    assert summary1.auto_accepted + summary1.needs_review == 3

    assert (paths.archive_dir / "drone" / "drone_idem" / "drone.jpg").is_file()
    assert (paths.archive_dir / "phone" / "phone_idem" / "phone.heic").is_file()
    assert (paths.archive_dir / "ortho" / "ortho_idem" / "ortho.tif").is_file()

    with Database(paths.database) as connection:
        count_before = len(ManifestRepository(connection).query())
    assert count_before == 3

    # Re-add the exact same inputs back to input_dir after archive
    shutil.copy2(paths.archive_dir / "drone" / "drone_idem" / "drone.jpg", drone_s / "drone.jpg")
    shutil.copy2(paths.archive_dir / "phone" / "phone_idem" / "phone.heic", phone_s / "phone.heic")
    shutil.copy2(paths.archive_dir / "ortho" / "ortho_idem" / "ortho.tif", ortho_s / "ortho.tif")

    summary2 = prelabel(paths, cfg, FakeBackend)
    assert summary2.auto_accepted == 0
    assert summary2.needs_review == 0
    assert summary2.failed == 0
    assert summary2.skipped_duplicates == 3

    with Database(paths.database) as connection:
        count_after = len(ManifestRepository(connection).query())
    assert count_after == count_before
