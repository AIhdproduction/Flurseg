from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from flurseg.refine import registry
from flurseg.refine.registry import RefinementContext, refine


def _config(*, enabled: set[int] | None = None) -> SimpleNamespace:
    active = set(range(1, 7)) if enabled is None else enabled
    refinement = SimpleNamespace(**{
        name: SimpleNamespace(enabled=rule_id in active)
        for rule_id, name in enumerate(("edge_snap", "morphology", "shape", "context", "color", "ortho"), 1)
    })
    return SimpleNamespace(refinement=refinement)


def _install_marking_rules(monkeypatch: pytest.MonkeyPatch, *, same_pixel: bool = False) -> None:
    def marked(mask: np.ndarray, rule_id: int) -> np.ndarray:
        output = mask.copy()
        output[0, 0 if same_pixel else rule_id - 1] = rule_id
        return output

    monkeypatch.setattr(registry.edge_snap, "apply", lambda mask, *_: (marked(mask, 1), np.ones(mask.shape, np.int32)))
    for rule_id, module in enumerate(
        (registry.morphology, registry.shape_rules, registry.context_rules, registry.color_rules, registry.ortho_rules), 2
    ):
        monkeypatch.setattr(module, "apply", lambda mask, *_, value=rule_id: marked(mask, value))


@pytest.mark.parametrize("disabled", range(1, 7), ids=lambda value: f"rule_{value}")
def test_disabling_each_rule_individually_removes_its_effect(monkeypatch: pytest.MonkeyPatch, disabled: int) -> None:
    _install_marking_rules(monkeypatch)
    result = refine(RefinementContext(np.zeros((1, 6, 3), np.uint8), np.zeros((1, 6), np.uint8), "drone", 1.0, _config(enabled=set(range(1, 7)) - {disabled}), {}))
    assert result.mask[0, disabled - 1] == 0
    assert all(result.mask[0, index - 1] == index for index in set(range(1, 7)) - {disabled})


def test_auto_change_records_the_last_changing_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_marking_rules(monkeypatch, same_pixel=True)
    result = refine(RefinementContext(np.zeros((1, 1, 3), np.uint8), np.zeros((1, 1), np.uint8), "drone", 1.0, _config(), {}))
    assert result.mask[0, 0] == 2 and result.auto_change[0, 0] == 2


def test_phone_source_skips_only_metre_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_marking_rules(monkeypatch)
    result = refine(RefinementContext(np.zeros((1, 6, 3), np.uint8), np.zeros((1, 6), np.uint8), "phone", 0.2, _config(), {}))
    assert np.array_equal(result.mask, np.array([[1, 2, 0, 4, 5, 0]], np.uint8))
    assert np.array_equal(result.auto_change, result.mask)


def test_context_mask_is_not_mutated() -> None:
    mask = np.array([[1, 2]], dtype=np.uint8)
    context = RefinementContext(np.zeros((1, 2, 3), np.uint8), mask, "phone", None, _config(enabled=set()), {})
    refine(context)
    assert np.array_equal(context.mask, np.array([[1, 2]], dtype=np.uint8))


def test_all_disabled_keeps_mask_and_zero_auto_change() -> None:
    mask = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    result = refine(RefinementContext(np.zeros((2, 2, 3), np.uint8), mask, "drone", 1.0, _config(enabled=set()), {}))
    assert np.array_equal(result.mask, mask) and not result.auto_change.any()


def test_stored_superpixels_are_returned_when_edge_snap_is_disabled() -> None:
    stored = np.array([[3, 4]], dtype=np.int32)
    result = refine(RefinementContext(np.zeros((1, 2, 3), np.uint8), np.ones((1, 2), np.uint8), "drone", 1.0, _config(enabled=set()), {}, stored))
    assert result.superpixels is stored


def test_registry_executes_the_required_quality_order(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr(registry.edge_snap, "apply", lambda mask, *_: (mask, np.ones(mask.shape, np.int32)))
    for rule_id, module in ((2, registry.morphology), (3, registry.shape_rules), (4, registry.context_rules), (5, registry.color_rules), (6, registry.ortho_rules)):
        monkeypatch.setattr(module, "apply", lambda mask, *_, value=rule_id: calls.append(value) or mask)
    refine(RefinementContext(np.zeros((2, 2, 3), np.uint8), np.zeros((2, 2), np.uint8), "drone", 1.0, _config(), {}))
    assert calls == [5, 4, 3, 6, 2]
