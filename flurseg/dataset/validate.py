"""Validation of pool segmentation masks and image consistency."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_ALLOWED_CLASSES = set(range(11)) | {255}


class MaskValidationError(ValueError):
    """Validation failure for a dataset mask."""


ValidationError = MaskValidationError


def check_mask(
    mask: Path | str | np.ndarray,
    image: Path | str | np.ndarray | tuple[int, int] | None = None,
) -> list[str]:
    """Return a list of validation error descriptions, or an empty list if valid."""
    errors: list[str] = []
    mask_arr: np.ndarray | None = None
    mask_name = "mask"

    if isinstance(mask, (str, Path)):
        mask_path = Path(mask)
        mask_name = mask_path.name
        if not mask_path.is_file():
            return [f"mask file does not exist: {mask_path}"]

        try:
            with Image.open(mask_path) as pil_img:
                if pil_img.format != "PNG":
                    errors.append(f"{mask_name}: not a PNG file (format={pil_img.format})")
                if pil_img.mode not in ("L", "P", "1"):
                    errors.append(f"{mask_name}: mask must be single-channel, got mode '{pil_img.mode}'")
        except (OSError, ValueError) as error:
            errors.append(f"{mask_name}: failed to open image ({error})")
            return errors

        mask_arr = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask_arr is None:
            errors.append(f"{mask_name}: failed to read image data")
            return errors
    elif isinstance(mask, np.ndarray):
        mask_arr = mask
    else:
        errors.append(f"unsupported mask type: {type(mask)}")
        return errors

    if mask_arr.ndim != 2:
        errors.append(f"{mask_name}: mask must be 2D single channel, got shape {mask_arr.shape}")
        return errors

    if mask_arr.dtype != np.uint8:
        if ((mask_arr < 0) | (mask_arr > 255)).any():
            errors.append(f"{mask_name}: values outside uint8 range")
        else:
            errors.append(f"{mask_name}: mask must be uint8, got dtype {mask_arr.dtype}")

    unique_values = set(np.unique(mask_arr))
    invalid_values = unique_values - _ALLOWED_CLASSES
    if invalid_values:
        errors.append(
            f"{mask_name}: contains invalid class IDs {sorted(int(v) for v in invalid_values)}; allowed are 0..10 and 255"
        )

    if image is not None:
        img_h: int | None = None
        img_w: int | None = None

        if isinstance(image, (str, Path)):
            img_path = Path(image)
            if not img_path.is_file():
                errors.append(f"image file does not exist: {img_path}")
            else:
                try:
                    with Image.open(img_path) as pil_img:
                        img_w, img_h = pil_img.size
                except (OSError, ValueError) as error:
                    errors.append(f"failed to read image dimensions from {img_path}: {error}")
        elif isinstance(image, tuple) and len(image) >= 2:
            img_h, img_w = int(image[0]), int(image[1])
        elif isinstance(image, np.ndarray):
            img_h, img_w = image.shape[:2]

        if img_h is not None and img_w is not None:
            mask_h, mask_w = mask_arr.shape[:2]
            if (mask_h, mask_w) != (img_h, img_w):
                errors.append(
                    f"{mask_name}: mask size ({mask_h}, {mask_w}) does not match image size ({img_h}, {img_w})"
                )

    return errors


def validate_mask(
    mask: Path | str | np.ndarray,
    image: Path | str | np.ndarray | tuple[int, int] | None = None,
) -> None:
    """Validate that a mask is a single-channel uint8 PNG with allowed values and matching image size.

    Raises:
        MaskValidationError: If any check fails.
    """
    errors = check_mask(mask, image)
    if errors:
        raise MaskValidationError("; ".join(errors))


def validate_pool(
    pool_images_dir: Path,
    pool_masks_dir: Path,
) -> list[tuple[Path, str]]:
    """Validate all pool masks against pool images, returning a list of (bad_file, error_msg)."""
    bad_files: list[tuple[Path, str]] = []
    if not pool_images_dir.exists():
        return [(pool_images_dir, "pool images directory does not exist")]

    image_extensions = {".jpg", ".jpeg", ".png"}
    image_files = sorted(
        [p for p in pool_images_dir.iterdir() if p.is_file() and p.suffix.lower() in image_extensions]
    )

    for img_path in image_files:
        mask_path = pool_masks_dir / f"{img_path.stem}.png"
        if not mask_path.is_file():
            bad_files.append((mask_path, f"missing mask for image {img_path.name}"))
            continue
        errors = check_mask(mask_path, img_path)
        for err in errors:
            bad_files.append((mask_path, err))

    return bad_files
