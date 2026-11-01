"""Tile, scale and TTA orchestration for dense segmentation scores."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from flurseg.segment.exemplars import score_mosaic
from flurseg.segment.scales import resize_image
from flurseg.segment.tiling import TileAccumulator, iter_tiles
from flurseg.segment.tta import apply, invert


@dataclass(slots=True)
class ScoreResult:
    """Scores at their configured inference scales plus full-resolution diagnostic maps."""

    scores: dict[str, np.ndarray]
    scales: dict[str, float]
    winner_per_class: dict[str, np.ndarray]
    tta_std_per_class: dict[str, np.ndarray]
    prompt_winner: np.ndarray
    tta_disagreement: np.ndarray
    seconds: float
    timings: dict[str, float]


def _prompt_data(config: Any, class_names: list[str]) -> tuple[list[str], list[int], list[int]]:
    prompts: list[str] = []
    class_ids: list[int] = []
    global_indices: list[int] = []
    global_index = 0
    wanted = set(class_names)
    for definition in config.classes.classes:
        if definition.id == 255:
            continue
        for prompt in definition.prompts:
            if definition.name in wanted:
                prompts.append(prompt)
                class_ids.append(definition.id)
                global_indices.append(global_index)
            global_index += 1
    return prompts, class_ids, global_indices


def _class_maps(prompt_maps: np.ndarray, class_ids: list[int]) -> dict[int, np.ndarray]:
    return {class_id: prompt_maps[[i for i, value in enumerate(class_ids) if value == class_id]].max(axis=0) for class_id in set(class_ids)}


def _tile_feather_weights(tile: Any, width: int, height: int) -> np.ndarray:
    """Compute 2D linear-ramp feathering weights for a tile within an image canvas."""
    y = np.ones(tile.height, dtype=np.float32)
    x = np.ones(tile.width, dtype=np.float32)
    if tile.x > 0:
        x = np.minimum(x, (np.arange(tile.width, dtype=np.float32) + 1) / max(1, tile.width // 4))
    if tile.x + tile.width < width:
        x = np.minimum(x, (np.arange(tile.width, 0, -1, dtype=np.float32)) / max(1, tile.width // 4))
    if tile.y > 0:
        y = np.minimum(y, (np.arange(tile.height, dtype=np.float32) + 1) / max(1, tile.height // 4))
    if tile.y + tile.height < height:
        y = np.minimum(y, (np.arange(tile.height, 0, -1, dtype=np.float32)) / max(1, tile.height // 4))
    return np.outer(y, x).clip(1e-6, 1.0)


def score_image(image: np.ndarray, backend: Any, cfg: Any, *, exemplars_dir: Path | None = None) -> ScoreResult:
    """Score an RGB image over configured scales, tiles and exact inverse TTA transforms."""
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image must be an HxWx3 uint8 RGB array")
    started = time.monotonic()
    pipeline = getattr(cfg, "pipeline", cfg)
    full_height, full_width = image.shape[:2]
    class_scores: dict[str, np.ndarray] = {}
    scale_map: dict[str, float] = {}
    winners_per_class: dict[str, np.ndarray] = {}
    std_per_class: dict[str, np.ndarray] = {}
    winner_full = np.zeros((full_height, full_width), dtype=np.uint8)
    disagreement_full = np.zeros((full_height, full_width), dtype=np.float32)
    timings: dict[str, float] = {}
    sam_seconds = 0.0
    tile_variants = 0
    for assignment in pipeline.scales:
        scale_started = time.monotonic()
        scaled = resize_image(image, float(assignment.factor))
        height, width = scaled.shape[:2]
        prompts, prompt_class_ids, global_indices = _prompt_data(cfg, list(assignment.classes))
        if not prompts:
            continue

        scale_class_ids: list[int] = []
        for class_id in prompt_class_ids:
            if class_id not in scale_class_ids:
                scale_class_ids.append(class_id)
        num_classes = len(scale_class_ids)
        class_prompt_indices = {
            class_id: [idx for idx, cid in enumerate(prompt_class_ids) if cid == class_id]
            for class_id in scale_class_ids
        }

        mosaic = pipeline.exemplars.mosaic
        effective_height = max(1, round(pipeline.tiling.tile_size * (1 - mosaic.strip_fraction))) if mosaic.enabled else pipeline.tiling.tile_size
        tiles = list(iter_tiles(width, height, pipeline.tiling.tile_size, pipeline.tiling.min_overlap))
        if mosaic.enabled:
            tiles = list(iter_tiles(width, height, effective_height, pipeline.tiling.min_overlap))
            tiles = [type(tile)(tile.x, tile.y, min(tile.width, pipeline.tiling.tile_size), tile.height) for tile in tiles]

        score_acc = TileAccumulator(num_classes, width, height, reduction="mean")
        std_acc = TileAccumulator(num_classes, width, height, reduction="mean")

        scale_best_weight = np.full((height, width), -1.0, dtype=np.float32)
        scale_class_winners: dict[int, np.ndarray] = {
            class_id: np.zeros((height, width), dtype=np.uint8) for class_id in scale_class_ids
        }

        for tile in tiles:
            data = scaled[tile.y : tile.y + tile.height, tile.x : tile.x + tile.width]
            per_variant: list[np.ndarray] = []
            for variant in pipeline.tta.variants:
                transformed = apply(variant, data)
                sam_started = time.monotonic()
                backend_scores = backend.score_tile(transformed, prompts)
                sam_seconds += time.monotonic() - sam_started
                tile_variants += 1
                values = invert(variant, backend_scores)
                if values.shape != (len(prompts), tile.height, tile.width):
                    raise ValueError(f"backend returned {values.shape}; expected {(len(prompts), tile.height, tile.width)}")
                if mosaic.enabled:
                    for index, class_id in enumerate(prompt_class_ids):
                        class_name = cfg.class_by_id(class_id).name
                        directory = exemplars_dir / class_name if exemplars_dir is not None else Path("config") / "exemplars" / class_name
                        exemplar_score = score_mosaic(
                            data,
                            backend,
                            prompts[index],
                            directory,
                            tile_size=pipeline.tiling.tile_size,
                            strip_fraction=mosaic.strip_fraction,
                            max_per_class=mosaic.max_per_class,
                        )
                        values[index] = np.maximum(values[index], exemplar_score)
                per_variant.append(values)

            stack = np.stack(per_variant)
            prompt_mean = stack.mean(axis=0)

            feather_weights = _tile_feather_weights(tile, width, height)
            dest_y = slice(tile.y, tile.y + tile.height)
            dest_x = slice(tile.x, tile.x + tile.width)
            best_weight_window = scale_best_weight[dest_y, dest_x]
            higher = feather_weights > best_weight_window

            tile_class_scores = np.empty((num_classes, tile.height, tile.width), dtype=np.float32)
            tile_class_stds = np.empty((num_classes, tile.height, tile.width), dtype=np.float32)

            for c_idx, class_id in enumerate(scale_class_ids):
                idx_c = class_prompt_indices[class_id]
                class_score_tile = prompt_mean[idx_c].max(axis=0)
                class_winner_tile = prompt_mean[idx_c].argmax(axis=0).astype(np.uint8)
                class_std_tile = stack[:, idx_c].max(axis=1).std(axis=0)

                tile_class_scores[c_idx] = class_score_tile
                tile_class_stds[c_idx] = class_std_tile

                winner_window = scale_class_winners[class_id][dest_y, dest_x]
                winner_window[higher] = class_winner_tile[higher]

            best_weight_window[higher] = feather_weights[higher]

            score_acc.add(tile, tile_class_scores)
            std_acc.add(tile, tile_class_stds)

        accumulated_scores = score_acc.scores()
        accumulated_stds = std_acc.scores()

        for c_idx, class_id in enumerate(scale_class_ids):
            class_name = cfg.class_by_id(class_id).name
            class_scores[class_name] = accumulated_scores[c_idx].astype(np.float32)
            scale_map[class_name] = float(assignment.factor)
            winners_per_class[class_name] = scale_class_winners[class_id]
            std_per_class[class_name] = np.rint(np.clip(accumulated_stds[c_idx], 0, 1) * 255).astype(np.uint8)

        best_class_idx = np.argmax(accumulated_scores, axis=0)
        local_winner = np.zeros((height, width), dtype=np.uint8)
        selected_std = np.zeros((height, width), dtype=np.float32)

        for c_idx, class_id in enumerate(scale_class_ids):
            mask = (best_class_idx == c_idx)
            if not np.any(mask):
                continue
            idx_c = class_prompt_indices[class_id]
            class_global_indices = np.asarray([global_indices[i] for i in idx_c], dtype=np.uint8)
            local_winner[mask] = class_global_indices[scale_class_winners[class_id][mask]]
            selected_std[mask] = accumulated_stds[c_idx][mask]

        resized_winner = cv2.resize(local_winner, (full_width, full_height), interpolation=cv2.INTER_NEAREST)
        resized_std = cv2.resize(selected_std, (full_width, full_height), interpolation=cv2.INTER_LINEAR)
        # Scales have distinct configured classes, so writing only where that scale wins is deterministic.
        better = resized_std >= disagreement_full
        winner_full[better] = resized_winner[better]
        disagreement_full = np.maximum(disagreement_full, resized_std)
        timings[f"scale_{assignment.factor}"] = time.monotonic() - scale_started
        timings[f"tiles_{assignment.factor}"] = float(len(tiles))
    timings["sam"] = sam_seconds
    timings["tile_variants"] = float(tile_variants)
    return ScoreResult(
        scores=class_scores,
        scales=scale_map,
        winner_per_class=winners_per_class,
        tta_std_per_class=std_per_class,
        prompt_winner=winner_full,
        tta_disagreement=np.rint(np.clip(disagreement_full, 0, 1) * 255).astype(np.uint8),
        seconds=time.monotonic() - started,
        timings=timings,
    )
