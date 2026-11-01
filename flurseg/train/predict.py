"""Sliding-window semantic inference and image export."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional


def semantic_scores(output: object, target_size: tuple[int, int]) -> torch.Tensor:
    """Turn EoMT query logits into per-class semantic scores."""
    masks = functional.interpolate(
        output.masks_queries_logits,
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )
    classes = output.class_queries_logits.softmax(-1)[..., :-1]
    return torch.einsum("bqc,bqhw->bchw", classes, masks.sigmoid())


def sliding_window_scores(
    model: torch.nn.Module,
    image: torch.Tensor,
    window_size: int | None = None,
    stride: int | None = None,
) -> torch.Tensor:
    """Predict arbitrary-sized normalized images with Hann-window blending."""
    model_config = getattr(model, "config", None)
    if isinstance(model_config, dict):
        model_image_size = model_config.get("image_size")
    else:
        model_image_size = getattr(model_config, "image_size", None)

    if window_size is None:
        window_size = model_image_size if model_image_size is not None else 768
    elif model_image_size is not None and window_size != model_image_size:
        raise ValueError(
            f"window_size ({window_size}) must equal model.config.image_size ({model_image_size}) "
            "because the EoMT patch grid is fixed to the model image size"
        )
    if stride is None:
        stride = window_size // 2

    if image.ndim == 3:
        image = image.unsqueeze(0)
    _, _, height, width = image.shape
    device = next(model.parameters()).device
    score_sum: torch.Tensor | None = None
    weights = torch.zeros((height, width), device=device)
    window = (
        torch.outer(
            torch.hann_window(window_size, device=device),
            torch.hann_window(window_size, device=device),
        )
        + 1e-6
    )
    starts_y = list(range(0, max(height - window_size, 0) + 1, stride))
    starts_x = list(range(0, max(width - window_size, 0) + 1, stride))
    if starts_y[-1] != max(height - window_size, 0):
        starts_y.append(max(height - window_size, 0))
    if starts_x[-1] != max(width - window_size, 0):
        starts_x.append(max(width - window_size, 0))
    with torch.inference_mode():
        for top in starts_y:
            for left in starts_x:
                bottom, right = min(top + window_size, height), min(left + window_size, width)
                tile = image[:, :, top:bottom, left:right].to(device)
                tile = functional.pad(
                    tile, (0, window_size - tile.shape[-1], 0, window_size - tile.shape[-2])
                )
                output = model(pixel_values=tile)
                scores = semantic_scores(output, (window_size, window_size))[0]
                scores = scores[:, : bottom - top, : right - left]
                if score_sum is None:
                    score_sum = torch.zeros((scores.shape[0], height, width), device=device)
                blend = window[: bottom - top, : right - left]
                score_sum[:, top:bottom, left:right] += scores * blend
                weights[top:bottom, left:right] += blend
    assert score_sum is not None
    return score_sum / weights.clamp_min(1e-6)


def predict_image(
    model: torch.nn.Module,
    image_path: Path,
    out_dir: Path,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    window_size: int | None = None,
    stride: int | None = None,
    classes: list[dict[str, object]] | None = None,
) -> tuple[Path, Path]:
    """Write a class-index PNG and a transparent-looking coloured overlay."""
    model_config = getattr(model, "config", None)
    if isinstance(model_config, dict):
        model_image_size = model_config.get("image_size")
    else:
        model_image_size = getattr(model_config, "image_size", None)

    if window_size is None:
        window_size = model_image_size if model_image_size is not None else 768
    if stride is None:
        stride = window_size // 2
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    values = (rgb - np.asarray(mean)) / np.asarray(std)
    tensor = torch.from_numpy(values.transpose(2, 0, 1)).float()
    mask = (
        sliding_window_scores(
            model,
            tensor,
            window_size,
            stride,
        )
        .argmax(0)
        .cpu()
        .numpy()
        .astype(np.uint8)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_path = out_dir / f"{image_path.stem}_mask.png"
    overlay_path = out_dir / f"{image_path.stem}_overlay.png"
    cv2.imwrite(str(mask_path), mask)
    colours = {int(item["id"]): str(item.get("color", "#000000")) for item in classes or []}
    overlay = np.zeros_like(rgb, dtype=np.uint8)
    for class_id, colour in colours.items():
        overlay[mask == class_id] = tuple(int(colour[index : index + 2], 16) for index in (1, 3, 5))
    blended = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 0.55, overlay, 0.45, 0)
    cv2.imwrite(str(overlay_path), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    return mask_path, overlay_path


def load_run_metadata(run_dir: Path) -> list[dict[str, object]]:
    """Read class colours saved alongside a run's dataset snapshot."""
    with (run_dir / "dataset.json").open("r", encoding="utf-8") as stream:
        return list(json.load(stream).get("classes", []))
