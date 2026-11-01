"""Ignore-aware EoMT matching and loss computation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import functional

from flurseg.train.config import LossConfig


@dataclass
class LossValues:
    """Weighted loss total and individual components."""

    total: torch.Tensor
    mask: torch.Tensor
    dice: torch.Tensor
    classification: torch.Tensor


def sample_point(
    input_features: torch.Tensor,
    point_coordinates: torch.Tensor,
    align_corners: bool = False,
) -> torch.Tensor:
    """Sample features at normalized coordinates [0, 1] using bilinear interpolation."""
    grid = 2.0 * point_coordinates.unsqueeze(2) - 1.0
    return functional.grid_sample(
        input_features,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=align_corners,
    ).squeeze(3)


def pixel_to_normalized_coords(
    pixel_coords: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Convert (y, x) pixel coordinates to normalized continuous (x, y) coordinates in [0, 1]."""
    if not pixel_coords.numel():
        return pixel_coords.new_empty((0, 2), dtype=torch.float32)
    norm_x = (pixel_coords[..., 1].float() + 0.5) / float(width)
    norm_y = (pixel_coords[..., 0].float() + 0.5) / float(height)
    return torch.stack([norm_x, norm_y], dim=-1)


def sample_valid_pixels(
    valid: torch.Tensor,
    count: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample count pixel (y, x) coordinates uniformly at random over valid pixels."""
    valid_coords = valid.nonzero(as_tuple=False)
    num_valid = valid_coords.shape[0]
    if num_valid == 0:
        return valid_coords.new_empty((0, 2))
    device = valid.device
    if num_valid >= count:
        indices = torch.randperm(num_valid, generator=generator, device=device)[:count]
    else:
        indices = torch.randint(0, num_valid, (count,), generator=generator, device=device)
    return valid_coords[indices]


def calculate_uncertainty(logits: torch.Tensor) -> torch.Tensor:
    """Compute point uncertainty as negative absolute logit distance from 0."""
    return -torch.abs(logits)


def sample_importance_points(
    pred_masks: torch.Tensor,
    valid: torch.Tensor,
    num_points: int,
    oversample_ratio: float,
    importance_sample_ratio: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample points for matched masks via uncertainty-based importance sampling."""
    num_boxes = pred_masks.shape[0]
    height, width = valid.shape
    valid_coords = valid.nonzero(as_tuple=False)
    num_valid = valid_coords.shape[0]
    if num_valid == 0 or num_boxes == 0:
        empty_norm = pred_masks.new_empty((num_boxes, 0, 2))
        empty_pix = torch.empty((num_boxes, 0, 2), dtype=torch.long, device=pred_masks.device)
        return empty_norm, empty_pix

    device = valid.device
    num_sampled = int(num_points * oversample_ratio)
    num_uncertain = int(num_points * importance_sample_ratio)
    num_random = num_points - num_uncertain

    if num_valid >= num_sampled:
        cand_indices = torch.stack(
            [
                torch.randperm(num_valid, generator=generator, device=device)[:num_sampled]
                for _ in range(num_boxes)
            ],
            dim=0,
        )
    else:
        cand_indices = torch.randint(
            0, num_valid, (num_boxes, num_sampled), generator=generator, device=device
        )
    cand_pixels = valid_coords[cand_indices]
    cand_norm = pixel_to_normalized_coords(cand_pixels, height, width)

    with torch.no_grad():
        cand_logits = sample_point(pred_masks, cand_norm).squeeze(1)
        uncertainties = calculate_uncertainty(cand_logits)
        topk_indices = torch.topk(uncertainties, k=num_uncertain, dim=1)[1]
        gather_idx = topk_indices.unsqueeze(-1).expand(-1, -1, 2)
        uncertain_norm = torch.gather(cand_norm, 1, gather_idx)
        uncertain_pixels = torch.gather(cand_pixels, 1, gather_idx)

    if num_random > 0:
        if num_valid >= num_random:
            rand_indices = torch.stack(
                [
                    torch.randperm(num_valid, generator=generator, device=device)[:num_random]
                    for _ in range(num_boxes)
                ],
                dim=0,
            )
        else:
            rand_indices = torch.randint(
                0, num_valid, (num_boxes, num_random), generator=generator, device=device
            )
        rand_pixels = valid_coords[rand_indices]
        rand_norm = pixel_to_normalized_coords(rand_pixels, height, width)
        final_norm = torch.cat([uncertain_norm, rand_norm], dim=1)
        final_pixels = torch.cat([uncertain_pixels, rand_pixels], dim=1)
    else:
        final_norm = uncertain_norm
        final_pixels = uncertain_pixels

    return final_norm, final_pixels


def _single_layer_loss(
    masks_queries_logits: torch.Tensor,
    class_queries_logits: torch.Tensor,
    mask_labels: list[torch.Tensor],
    class_labels: list[torch.Tensor],
    valid_masks: list[torch.Tensor],
    config: LossConfig,
) -> LossValues:
    """Compute Hungarian matching, mask/dice, and classification losses for one prediction pair."""
    device = masks_queries_logits.device
    gen_device = device.type if device.type in ("cpu", "cuda") else "cpu"
    generator = torch.Generator(device=gen_device)
    seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    generator.manual_seed(seed)

    mask_total = masks_queries_logits.new_zeros(())
    dice_total = masks_queries_logits.new_zeros(())
    class_total = masks_queries_logits.new_zeros(())
    total_matched_masks = 0
    batches, queries, labels_plus_null = class_queries_logits.shape
    null_label = labels_plus_null - 1

    for batch_index in range(batches):
        prediction = masks_queries_logits[batch_index]
        targets = mask_labels[batch_index].to(device=device, dtype=prediction.dtype)
        target_classes = class_labels[batch_index].to(device=device)
        valid = valid_masks[batch_index].to(device=device, dtype=torch.bool)
        height, width = valid.shape

        assigned_predictions = torch.empty(0, dtype=torch.long, device=device)
        assigned_targets = torch.empty(0, dtype=torch.long, device=device)

        if len(target_classes) > 0:
            match_pixels = sample_valid_pixels(valid, config.num_points, generator=generator)
            if len(match_pixels) > 0:
                match_norm = pixel_to_normalized_coords(match_pixels, height, width)
                pred_points = sample_point(prediction.unsqueeze(0), match_norm.unsqueeze(0)).squeeze(0)
                target_points = targets[:, match_pixels[:, 0], match_pixels[:, 1]]
                probabilities = class_queries_logits[batch_index].softmax(-1)[:, target_classes]
                bce = functional.binary_cross_entropy_with_logits(
                    pred_points[:, None, :].expand(-1, len(target_classes), -1),
                    target_points[None, :, :].expand(queries, -1, -1),
                    reduction="none",
                ).mean(-1)
                prob = pred_points.sigmoid()[:, None, :]
                dice = 1.0 - (2.0 * (prob * target_points[None]).sum(-1) + 1.0) / (
                    prob.sum(-1) + target_points[None].sum(-1) + 1.0
                )
                cost = (
                    config.mask_weight * bce
                    + config.dice_weight * dice
                    - config.class_weight * probabilities
                )
                cost = torch.nan_to_num(cost, nan=1e9, posinf=1e9, neginf=1e9)
                pred_index, target_index = linear_sum_assignment(cost.detach().float().cpu().numpy())
                assigned_predictions = torch.as_tensor(pred_index, device=device, dtype=torch.long)
                assigned_targets = torch.as_tensor(target_index, device=device, dtype=torch.long)

        if len(assigned_predictions) > 0:
            matched_pred = prediction[assigned_predictions].unsqueeze(1)
            matched_tgt = targets[assigned_targets]
            final_norm, final_pixels = sample_importance_points(
                matched_pred,
                valid,
                config.num_points,
                config.oversample_ratio,
                config.importance_sample_ratio,
                generator=generator,
            )
            point_logits = sample_point(matched_pred, final_norm).squeeze(1)
            num_matched, num_points_actual = final_pixels.shape[:2]
            batch_idx = torch.arange(num_matched, device=device).unsqueeze(1).expand(-1, num_points_actual)
            point_labels = matched_tgt[batch_idx, final_pixels[..., 0], final_pixels[..., 1]]

            bce_per_mask = functional.binary_cross_entropy_with_logits(
                point_logits, point_labels, reduction="none"
            ).mean(-1)
            prob_mask = point_logits.sigmoid()
            dice_num = 2.0 * (prob_mask * point_labels).sum(-1) + 1.0
            dice_den = prob_mask.sum(-1) + point_labels.sum(-1) + 1.0
            dice_per_mask = 1.0 - dice_num / dice_den

            mask_total = mask_total + bce_per_mask.sum()
            dice_total = dice_total + dice_per_mask.sum()
            total_matched_masks += num_matched

        target_queries = torch.full((queries,), null_label, dtype=torch.long, device=device)
        if len(assigned_predictions) > 0:
            target_queries[assigned_predictions] = target_classes[assigned_targets]
        class_weights = torch.ones(labels_plus_null, device=device)
        class_weights[-1] = config.no_object_weight
        class_total = class_total + functional.cross_entropy(
            class_queries_logits[batch_index], target_queries, weight=class_weights
        )

    divisor = max(total_matched_masks, 1)
    mask_value = mask_total / divisor
    dice_value = dice_total / divisor
    class_value = class_total / max(batches, 1)
    total = (
        config.mask_weight * mask_value
        + config.dice_weight * dice_value
        + config.class_weight * class_value
    )
    return LossValues(total=total, mask=mask_value, dice=dice_value, classification=class_value)


def ignore_aware_loss(
    masks_queries_logits: torch.Tensor,
    class_queries_logits: torch.Tensor,
    mask_labels: list[torch.Tensor],
    class_labels: list[torch.Tensor],
    valid_masks: list[torch.Tensor],
    config: LossConfig,
    auxiliary_predictions: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> LossValues:
    """Compute EoMT-style Hungarian losses on valid pixels with deep supervision."""
    if config.deep_supervision and auxiliary_predictions:
        pairs = list(auxiliary_predictions) + [(masks_queries_logits, class_queries_logits)]
    else:
        pairs = [(masks_queries_logits, class_queries_logits)]

    layer_losses = [
        _single_layer_loss(
            pair[0],
            pair[1],
            mask_labels,
            class_labels,
            valid_masks,
            config,
        )
        for pair in pairs
    ]
    total = torch.stack([loss.total for loss in layer_losses]).sum()
    mask_mean = torch.stack([loss.mask for loss in layer_losses]).mean()
    dice_mean = torch.stack([loss.dice for loss in layer_losses]).mean()
    class_mean = torch.stack([loss.classification for loss in layer_losses]).mean()
    return LossValues(total=total, mask=mask_mean, dice=dice_mean, classification=class_mean)
