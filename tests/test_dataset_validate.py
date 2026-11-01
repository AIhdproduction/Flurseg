"""Tests for pool mask validation and image consistency."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from flurseg.dataset.validate import (
    MaskValidationError,
    ValidationError,
    check_mask,
    validate_mask,
    validate_pool,
)


def _save_mask_png(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_img = Image.fromarray(mask)
    pil_img.save(path, format="PNG")


def _save_image_jpg(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_img = Image.fromarray(image).convert("RGB")
    pil_img.save(path, format="JPEG", quality=95)


def test_valid_mask_passes_as_array() -> None:
    height, width = 120, 160
    # Values 0..10 and 255 are allowed
    mask = np.zeros((height, width), dtype=np.uint8)
    for cls_id in range(11):
        mask[cls_id * 5 : (cls_id + 1) * 5, :] = cls_id
    mask[100:110, :] = 255

    errors = check_mask(mask)
    assert errors == []
    # Does not raise
    validate_mask(mask)

    # With matching numpy array image
    image_arr = np.zeros((height, width, 3), dtype=np.uint8)
    assert check_mask(mask, image_arr) == []
    validate_mask(mask, image_arr)

    # With matching (height, width) tuple
    assert check_mask(mask, (height, width)) == []
    validate_mask(mask, (height, width))


def test_valid_mask_passes_as_file(tmp_path: Path) -> None:
    height, width = 80, 100
    mask_arr = np.full((height, width), 4, dtype=np.uint8)
    mask_arr[10:20, 10:20] = 255

    mask_file = tmp_path / "valid_mask.png"
    _save_mask_png(mask_file, mask_arr)

    image_arr = np.full((height, width, 3), 120, dtype=np.uint8)
    image_file = tmp_path / "valid_image.jpg"
    _save_image_jpg(image_file, image_arr)

    errors = check_mask(mask_file, image_file)
    assert errors == []
    validate_mask(mask_file, image_file)


def test_wrong_size_array() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    image = np.zeros((100, 120, 3), dtype=np.uint8)

    errors = check_mask(mask, image)
    assert len(errors) == 1
    assert "mask size (100, 100) does not match image size (100, 120)" in errors[0]

    with pytest.raises(MaskValidationError, match="does not match image size"):
        validate_mask(mask, image)


def test_wrong_size_with_tuple() -> None:
    mask = np.zeros((64, 64), dtype=np.uint8)
    errors = check_mask(mask, (128, 128))
    assert len(errors) == 1
    assert "mask size (64, 64) does not match image size (128, 128)" in errors[0]

    with pytest.raises(MaskValidationError):
        validate_mask(mask, (128, 128))


def test_wrong_size_as_files(tmp_path: Path) -> None:
    mask_file = tmp_path / "mask.png"
    _save_mask_png(mask_file, np.zeros((80, 80), dtype=np.uint8))

    image_file = tmp_path / "image.jpg"
    _save_image_jpg(image_file, np.zeros((100, 100, 3), dtype=np.uint8))

    errors = check_mask(mask_file, image_file)
    assert any("does not match image size" in e for e in errors)

    with pytest.raises(MaskValidationError, match="does not match image size"):
        validate_mask(mask_file, image_file)


def test_wrong_channel_count_3d_array() -> None:
    mask_3d = np.zeros((50, 50, 3), dtype=np.uint8)

    errors = check_mask(mask_3d)
    assert any("mask must be 2D single channel" in e for e in errors)

    with pytest.raises(MaskValidationError, match="2D single channel"):
        validate_mask(mask_3d)


def test_wrong_channel_count_rgb_png_file(tmp_path: Path) -> None:
    rgb_file = tmp_path / "rgb_mask.png"
    Image.new("RGB", (64, 64), color=(0, 0, 0)).save(rgb_file, format="PNG")

    errors = check_mask(rgb_file)
    assert any("must be single-channel, got mode 'RGB'" in e for e in errors)

    with pytest.raises(MaskValidationError, match="single-channel"):
        validate_mask(rgb_file)


def test_wrong_channel_count_rgba_png_file(tmp_path: Path) -> None:
    rgba_file = tmp_path / "rgba_mask.png"
    Image.new("RGBA", (64, 64), color=(0, 0, 0, 255)).save(rgba_file, format="PNG")

    errors = check_mask(rgba_file)
    assert any("must be single-channel, got mode 'RGBA'" in e for e in errors)

    with pytest.raises(MaskValidationError, match="single-channel"):
        validate_mask(rgba_file)


def test_invalid_class_value_eleven() -> None:
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[25, 25] = 11

    errors = check_mask(mask)
    assert any("contains invalid class IDs [11]" in e for e in errors)

    with pytest.raises(MaskValidationError, match="contains invalid class IDs \\[11\\]"):
        validate_mask(mask)


def test_multiple_invalid_values() -> None:
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[0, 0] = 12
    mask[0, 1] = 50
    mask[0, 2] = 200

    errors = check_mask(mask)
    assert any("[12, 50, 200]" in e for e in errors)

    with pytest.raises(MaskValidationError):
        validate_mask(mask)


def test_non_uint8_dtype() -> None:
    # Float array
    mask_float = np.zeros((30, 30), dtype=np.float32)
    errors = check_mask(mask_float)
    assert any("mask must be uint8, got dtype float32" in e for e in errors)

    # Int32 with values outside uint8
    mask_outside = np.full((30, 30), 300, dtype=np.int32)
    errors_outside = check_mask(mask_outside)
    assert any("values outside uint8 range" in e for e in errors_outside)


def test_non_png_file(tmp_path: Path) -> None:
    jpg_file = tmp_path / "mask.jpg"
    Image.new("L", (40, 40)).save(jpg_file, format="JPEG")

    errors = check_mask(jpg_file)
    assert any("not a PNG file" in e for e in errors)

    with pytest.raises(MaskValidationError, match="not a PNG file"):
        validate_mask(jpg_file)


def test_missing_mask_file(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.png"
    errors = check_mask(missing)
    assert any("mask file does not exist" in e for e in errors)

    with pytest.raises(MaskValidationError, match="does not exist"):
        validate_mask(missing)


def test_missing_image_file(tmp_path: Path) -> None:
    mask_file = tmp_path / "mask.png"
    _save_mask_png(mask_file, np.zeros((50, 50), dtype=np.uint8))

    missing_image = tmp_path / "missing_image.jpg"
    errors = check_mask(mask_file, missing_image)
    assert any("image file does not exist" in e for e in errors)


def test_unsupported_mask_type() -> None:
    errors = check_mask(12345)  # type: ignore[arg-type]
    assert any("unsupported mask type" in e for e in errors)


def test_validation_error_alias() -> None:
    assert ValidationError is MaskValidationError


def test_validate_pool_all_valid(tmp_path: Path) -> None:
    images_dir = tmp_path / "pool" / "images"
    masks_dir = tmp_path / "pool" / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)

    for i in range(3):
        stem = f"image_{i:03d}"
        _save_image_jpg(images_dir / f"{stem}.jpg", np.zeros((100, 100, 3), dtype=np.uint8))
        _save_mask_png(masks_dir / f"{stem}.png", np.full((100, 100), i, dtype=np.uint8))

    bad_files = validate_pool(images_dir, masks_dir)
    assert bad_files == []


def test_validate_pool_with_missing_and_invalid_masks(tmp_path: Path) -> None:
    images_dir = tmp_path / "pool" / "images"
    masks_dir = tmp_path / "pool" / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)

    # 1. Valid pair
    _save_image_jpg(images_dir / "valid.jpg", np.zeros((100, 100, 3), dtype=np.uint8))
    _save_mask_png(masks_dir / "valid.png", np.zeros((100, 100), dtype=np.uint8))

    # 2. Missing mask
    _save_image_jpg(images_dir / "missing.jpg", np.zeros((100, 100, 3), dtype=np.uint8))

    # 3. Invalid class value 11 in mask
    _save_image_jpg(images_dir / "bad_class.jpg", np.zeros((100, 100, 3), dtype=np.uint8))
    bad_mask = np.zeros((100, 100), dtype=np.uint8)
    bad_mask[10, 10] = 11
    _save_mask_png(masks_dir / "bad_class.png", bad_mask)

    # 4. Size mismatch
    _save_image_jpg(images_dir / "wrong_size.jpg", np.zeros((100, 100, 3), dtype=np.uint8))
    _save_mask_png(masks_dir / "wrong_size.png", np.zeros((80, 80), dtype=np.uint8))

    bad_files = validate_pool(images_dir, masks_dir)

    # Exactly 3 bad reports
    assert len(bad_files) == 3

    error_dict = {path.name: msg for path, msg in bad_files}
    assert "missing.png" in error_dict
    assert "missing mask for image missing.jpg" in error_dict["missing.png"]

    assert "bad_class.png" in error_dict
    assert "invalid class IDs [11]" in error_dict["bad_class.png"]

    assert "wrong_size.png" in error_dict
    assert "mask size (80, 80) does not match image size (100, 100)" in error_dict["wrong_size.png"]


def test_validate_pool_images_dir_does_not_exist(tmp_path: Path) -> None:
    bad_files = validate_pool(tmp_path / "nonexistent", tmp_path / "masks")
    assert len(bad_files) == 1
    assert "pool images directory does not exist" in bad_files[0][1]
