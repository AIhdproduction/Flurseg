"""Class-score upsampling and deterministic priority fusion."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from flurseg.artifacts import upsample_scores


@dataclass(slots=True)
class FusionResult:
    mask: np.ndarray
    top_score: np.ndarray
    second_score: np.ndarray
    margin: np.ndarray
    conflict: np.ndarray
    scores: dict[str, np.ndarray]


def fuse(scores: Mapping[str, np.ndarray], shape: tuple[int, int], cfg: object) -> FusionResult:
    """Fuse class scores with priority limited to candidates near the best score."""
    if not scores:
        zero = np.zeros(shape, dtype=np.float32)
        return FusionResult(np.full(shape, 255, np.uint8), zero, zero, zero, zero, {})
    full = upsample_scores(scores, shape)
    names = list(full)
    stack = np.stack([full[name] for name in names]).astype(np.float32)
    ordered = sorted(names, key=lambda name: cfg.priority_rank(name))
    mask = np.full(shape, 255, dtype=np.uint8)
    unassigned = np.ones(shape, dtype=bool)
    ranks = sorted({cfg.priority_rank(name) for name in names})
    presence_default = float(cfg.pipeline.fusion.presence_threshold)
    per_class = cfg.pipeline.fusion.presence_threshold_per_class
    priority_margin = float(cfg.pipeline.fusion.priority_margin)
    best_score = stack.max(axis=0)
    for rank in ranks:
        contenders = [name for name in ordered if cfg.priority_rank(name) == rank]
        rank_stack = np.stack([full[name] for name in contenders])
        thresholds = np.asarray([per_class.get(name, presence_default) for name in contenders], dtype=np.float32)
        eligible = (rank_stack >= thresholds[:, None, None]) & (rank_stack >= best_score - priority_margin)
        candidate_scores = np.where(eligible, rank_stack, -np.inf)
        rank_index = candidate_scores.argmax(axis=0)
        rank_score = candidate_scores.max(axis=0)
        replace = unassigned & np.isfinite(rank_score)
        if replace.any():
            rank_ids = np.asarray([cfg.class_by_name(name).id for name in contenders], dtype=np.uint8)
            mask[replace] = rank_ids[rank_index[replace]]
            unassigned[replace] = False
    for override in cfg.pipeline.fusion.priority_overrides:
        if override.winner not in full or override.loser not in full:
            continue
        winner_id = cfg.class_by_name(override.winner).id
        loser_id = cfg.class_by_name(override.loser).id
        replace = (
            (mask == loser_id)
            & (full[override.winner] >= override.min_winner_score)
            & (full[override.winner] >= full[override.loser] - priority_margin)
        )
        mask[replace] = winner_id
    argmax = stack.argmax(axis=0)
    fallback = (mask == 255) & (best_score >= cfg.pipeline.fusion.no_prompt_score)
    class_ids = np.asarray([cfg.class_by_name(name).id for name in names], dtype=np.uint8)
    mask[fallback] = class_ids[argmax[fallback]]

    assigned_index = np.full(shape, -1, dtype=np.int16)
    for index, class_id in enumerate(class_ids):
        assigned_index[mask == class_id] = index
    top = np.zeros(shape, dtype=np.float32)
    valid = assigned_index >= 0
    rows, columns = np.nonzero(valid)
    top[valid] = stack[assigned_index[valid], rows, columns]
    other = stack.copy()
    if rows.size:
        other[assigned_index[valid], rows, columns] = -np.inf
    second = np.where(valid, other.max(axis=0), 0).astype(np.float32)
    margin = np.where(valid, np.clip(top - second, 0, 1), 0).astype(np.float32)
    conflict = np.where(valid, np.clip(best_score - top, 0, 1), 0).astype(np.float32)
    return FusionResult(mask, top, second, margin, conflict, full)
