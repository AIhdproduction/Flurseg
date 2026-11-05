"""Deterministic perspective views from equirectangular panoramas."""

from __future__ import annotations

import cv2
import numpy as np


def equirect_to_perspective(
    image_rgb: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    fov_deg: float,
    out_size: tuple[int, int] = (1600, 1200),
) -> np.ndarray:
    """Project an RGB equirectangular panorama into one pinhole-camera view."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must be an HxWx3 array")
    width, height = out_size
    if width < 1 or height < 1 or not 0 < fov_deg < 180:
        raise ValueError("out_size must be positive and fov_deg must be between 0 and 180")
    source_h, source_w = image_rgb.shape[:2]
    focal = (width / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
    x, y = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    rays = np.stack(((x - (width - 1) / 2.0) / focal, -(y - (height - 1) / 2.0) / focal,
                     np.ones_like(x)), axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    pitch = np.deg2rad(pitch_deg)
    yaw = np.deg2rad(yaw_deg)
    # Camera looks along +Z; apply pitch first, then compass yaw about the vertical axis.
    cp, sp, cy, sy = np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    rx = rays[..., 0]
    ry = cp * rays[..., 1] + sp * rays[..., 2]
    rz = -sp * rays[..., 1] + cp * rays[..., 2]
    world_x = cy * rx + sy * rz
    world_z = -sy * rx + cy * rz
    longitude = np.arctan2(world_x, world_z)
    latitude = np.arcsin(np.clip(ry, -1.0, 1.0))
    map_x = ((longitude / (2.0 * np.pi) + 0.5) * source_w).astype(np.float32)
    map_y = np.clip((0.5 - latitude / np.pi) * source_h, 0, source_h - 1).astype(np.float32)
    return cv2.remap(image_rgb, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
