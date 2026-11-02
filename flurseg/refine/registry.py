"""Ordered refinement-rule registry and automatic-change attribution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flurseg.refine import color_rules, context_rules, edge_snap, morphology, ortho_rules, shape_rules


@dataclass(slots=True)
class RefinementContext:
    image: np.ndarray
    mask: np.ndarray
    source: str
    gsd_m: float | None
    config: object
    ids: dict[str, int]
    superpixels: np.ndarray | None = None
    class_scores: dict[str, np.ndarray] | None = None


@dataclass(slots=True)
class RefinementResult:
    mask: np.ndarray
    auto_change: np.ndarray
    superpixels: np.ndarray | None


def refine(context: RefinementContext) -> RefinementResult:
    """Apply enabled rules in stable ID order, recording the last rule to change each pixel."""
    mask = context.mask.copy()
    changed = np.zeros(mask.shape, dtype=np.uint8)
    superpixels = context.superpixels
    paved_reference: np.ndarray | None = None
    rules = [
        (1, context.config.refinement.edge_snap, lambda value: edge_snap.apply(value, context.image, context.config.refinement.edge_snap, context.ids, superpixels)),
        (
            5,
            context.config.refinement.color,
            lambda value: color_rules.apply(
                value,
                context.image,
                context.config.refinement.color,
                context.ids,
                superpixels,
                context.config.refinement.edge_snap,
                context.class_scores,
            ),
        ),
        (4, context.config.refinement.context, lambda value: context_rules.apply(value, context.config.refinement.context, context.ids, paved_reference)),
        (3, context.config.refinement.shape, lambda value: shape_rules.apply(value, context.gsd_m, context.config.refinement.shape, context.ids)),
        (6, context.config.refinement.ortho, lambda value: ortho_rules.apply(value, context.source, context.gsd_m, context.config.refinement.ortho, context.ids)),
        (2, context.config.refinement.morphology, lambda value: morphology.apply(value, context.gsd_m, context.config.refinement.morphology, context.ids)),
    ]
    for rule_id, rule_config, function in rules:
        if context.source == "phone" and rule_id in {3, 6}:
            continue
        if not rule_config.enabled:
            continue
        if rule_id == 5 and superpixels is None:
            superpixels = color_rules.compute_superpixels(context.image, context.config.refinement.edge_snap)
        if rule_id == 5:
            paved_reference = mask == context.ids.get("paved_surface", -1)
        previous = mask.copy()
        output = function(mask)
        if rule_id == 1:
            mask, superpixels = output
        else:
            mask = output
        changed[mask != previous] = rule_id
    changed[mask == context.mask] = 0
    return RefinementResult(mask, changed, superpixels)
