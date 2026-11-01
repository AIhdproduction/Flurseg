"""Dense SAM 3 head fusion independent of the predictor runtime."""

from __future__ import annotations

import torch
from torch.nn import functional


def validate_heads(predictions: dict[str, torch.Tensor]) -> tuple[int, int, int, int]:
    """Validate the raw SAM 3 head contract and return ``(classes, queries, height, width)``.

    The check is intentionally strict: a changed Ultralytics contract must fail rather than
    silently produce incorrectly aligned pseudo labels.
    """
    required = {"pred_logits", "pred_masks", "semantic_seg", "presence_logit_dec", "pred_boxes"}
    missing = required - predictions.keys()
    if missing:
        raise ValueError(f"missing SAM 3 heads: {', '.join(sorted(missing))}")
    logits = predictions["pred_logits"]
    masks = predictions["pred_masks"]
    semantic = predictions["semantic_seg"]
    presence = predictions["presence_logit_dec"]
    boxes = predictions["pred_boxes"]
    if logits.ndim != 3 or logits.shape[-1] != 1:
        raise ValueError("pred_logits must have shape (classes, queries, 1)")
    classes, queries = logits.shape[:2]
    if masks.ndim != 4 or masks.shape[:2] != (classes, queries):
        raise ValueError("pred_masks must have shape (classes, queries, height, width)")
    if semantic.ndim != 4 or semantic.shape[:2] != (classes, 1) or semantic.shape[2:] != masks.shape[2:]:
        raise ValueError("semantic_seg must have shape (classes, 1, height, width)")
    if presence.numel() != classes:
        raise ValueError("presence_logit_dec must contain one value per class")
    if boxes.ndim != 3 or boxes.shape[:2] != (classes, queries) or boxes.shape[-1] != 4:
        raise ValueError("pred_boxes must have shape (classes, queries, 4)")
    return classes, queries, masks.shape[2], masks.shape[3]


def fuse_dense_heads(
    predictions: dict[str, torch.Tensor],
    *,
    query_threshold: float,
    output_size: tuple[int, int],
    query_chunk_size: int = 128,
) -> torch.Tensor:
    """Fuse raw SAM 3 heads into dense class probabilities.

    Query masks are evaluated in chunks, avoiding a large temporary tensor when a model
    exposes many queries.  Returned data is ``(classes, output_height, output_width)``.
    """
    if not 0 <= query_threshold <= 1:
        raise ValueError("query_threshold must be between 0 and 1")
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")
    classes, queries, _, _ = validate_heads(predictions)
    logits = predictions["pred_logits"].squeeze(-1).sigmoid()
    masks = predictions["pred_masks"]
    semantic = predictions["semantic_seg"].squeeze(1).sigmoid()
    presence = predictions["presence_logit_dec"].reshape(classes).sigmoid().view(classes, 1, 1)
    best_query = torch.zeros_like(semantic)
    for start in range(0, queries, query_chunk_size):
        end = min(queries, start + query_chunk_size)
        confidence = logits[:, start:end]
        eligible = confidence >= query_threshold
        values = masks[:, start:end].sigmoid() * confidence[..., None, None]
        values = values.masked_fill(~eligible[..., None, None], 0)
        best_query = torch.maximum(best_query, values.max(dim=1).values)
    merged = torch.maximum(semantic, best_query) * presence
    return functional.interpolate(merged[:, None].float(), size=output_size, mode="bilinear", align_corners=False)[:, 0]
