"""Pixel-exact test-time augmentation transforms."""

from __future__ import annotations

import numpy as np


def apply(variant: str, image: np.ndarray) -> np.ndarray:
    """Apply a configured spatial transform to HWC images or CHW scores."""
    if variant == "identity":
        return image.copy()
    axes = (0, 1) if image.ndim == 3 and image.dtype == np.uint8 else (-2, -1)
    if variant == "rot90":
        return np.rot90(image, axes=axes).copy()
    if variant == "hflip":
        return np.flip(image, axis=axes[1]).copy()
    if variant == "vflip":
        return np.flip(image, axis=axes[0]).copy()
    raise ValueError(f"unknown TTA variant: {variant}")


def invert(variant: str, array: np.ndarray) -> np.ndarray:
    """Invert :func:`apply` exactly for an image or a score stack."""
    if variant == "identity":
        return array.copy()
    axes = (0, 1) if array.ndim == 3 and array.dtype == np.uint8 else (-2, -1)
    if variant == "rot90":
        return np.rot90(array, k=-1, axes=axes).copy()
    if variant == "hflip":
        return np.flip(array, axis=axes[1]).copy()
    if variant == "vflip":
        return np.flip(array, axis=axes[0]).copy()
    raise ValueError(f"unknown TTA variant: {variant}")
