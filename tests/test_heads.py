from __future__ import annotations

import torch

from flurseg.segment.heads import fuse_dense_heads


def test_dense_head_fusion_honours_threshold_presence_and_chunking() -> None:
    predictions = {
        "pred_logits": torch.tensor([[[8.0], [-8.0]]]),
        "pred_masks": torch.tensor([[[[8.0]], [[8.0]]]]),
        "semantic_seg": torch.tensor([[[[-8.0]]]]),
        "presence_logit_dec": torch.tensor([8.0]),
        "pred_boxes": torch.zeros((1, 2, 4)),
    }
    chunked = fuse_dense_heads(predictions, query_threshold=0.5, output_size=(2, 2), query_chunk_size=1)
    unchunked = fuse_dense_heads(predictions, query_threshold=0.5, output_size=(2, 2), query_chunk_size=8)
    assert torch.allclose(chunked, unchunked)
    assert float(chunked.min()) > 0.99


def test_dense_heads_exclude_low_queries_and_apply_presence_gate() -> None:
    predictions = {
        "pred_logits": torch.tensor([[[-8.0]]]),
        "pred_masks": torch.tensor([[[[20.0]]]]),
        "semantic_seg": torch.tensor([[[[0.0]]]]),
        "presence_logit_dec": torch.tensor([-20.0]),
        "pred_boxes": torch.zeros((1, 1, 4)),
    }
    output = fuse_dense_heads(predictions, query_threshold=0.5, output_size=(1, 1))
    # The semantic term is 0.5 and the near-zero presence gate dominates it.
    assert float(output[0, 0, 0]) < 1e-6


def test_dense_heads_formula_uses_best_eligible_query() -> None:
    predictions = {
        "pred_logits": torch.tensor([[[0.0], [2.0]]]),
        "pred_masks": torch.tensor([[[[0.0]], [[0.0]]]]),
        "semantic_seg": torch.tensor([[[[-20.0]]]]),
        "presence_logit_dec": torch.tensor([0.0]),
        "pred_boxes": torch.zeros((1, 2, 4)),
    }
    output = fuse_dense_heads(predictions, query_threshold=0.6, output_size=(1, 1))
    expected = torch.sigmoid(torch.tensor(0.0)) * torch.sigmoid(torch.tensor(2.0)) * torch.sigmoid(torch.tensor(0.0))
    assert torch.allclose(output.squeeze(), expected)


def test_presence_gate_multiplies_the_retained_semantic_score() -> None:
    predictions = {
        "pred_logits": torch.tensor([[[-8.0]]]),
        "pred_masks": torch.tensor([[[[20.0]]]]),
        "semantic_seg": torch.tensor([[[[0.0]]]]),
        "presence_logit_dec": torch.tensor([0.0]),
        "pred_boxes": torch.zeros((1, 1, 4)),
    }
    output = fuse_dense_heads(predictions, query_threshold=0.5, output_size=(1, 1))
    assert torch.allclose(output.squeeze(), torch.tensor(0.25))


def test_empty_retained_queries_fall_back_to_the_semantic_head_only() -> None:
    predictions = {
        "pred_logits": torch.tensor([[[-8.0], [-7.0]]]),
        "pred_masks": torch.tensor([[[[20.0]], [[20.0]]]]),
        "semantic_seg": torch.tensor([[[[2.0]]]]),
        "presence_logit_dec": torch.tensor([20.0]),
        "pred_boxes": torch.zeros((1, 2, 4)),
    }
    output = fuse_dense_heads(predictions, query_threshold=0.5, output_size=(1, 1))
    assert torch.allclose(output.squeeze(), torch.sigmoid(torch.tensor(2.0)), atol=1e-6)
