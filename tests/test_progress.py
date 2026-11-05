"""Plain-console progress and deterministic ETA coverage."""

from __future__ import annotations

from datetime import datetime

from flurseg.progress import ProgressReporter


def test_progress_eta_math_uses_injected_clock() -> None:
    values = iter([10.0, 12.0, 16.0])
    messages: list[str] = []
    reporter = ProgressReporter(4, messages.append, clock=lambda: next(values))
    assert reporter.advance() == "1/4 | last 2.0s | average 2.0s | ETA 6.0s"
    assert reporter.advance() == "2/4 | last 4.0s | average 3.0s | ETA 6.0s"


def test_progress_output_contains_count_seconds_per_image_and_eta() -> None:
    messages: list[str] = []
    reporter = ProgressReporter(2, messages.append, clock=lambda: 1.0)
    output = reporter.advance()
    assert "1/2" in output
    assert "last" in output
    assert "s" in output
    assert "ETA" in output


def test_weighted_eta_uses_completed_weight() -> None:
    values = iter([0.0, 4.0])
    weighted = ProgressReporter(5, clock=lambda: next(values), total_weight=8)
    output = weighted.advance(weight=4)
    unweighted_eta = (5 - 1) * 4.0 / 1
    assert output.endswith("ETA 4.0s")
    assert 4.0 != unweighted_eta


def test_human_progress_message_contains_weighted_eta_and_finish_time() -> None:
    values = iter([0.0, 60.0])
    reporter = ProgressReporter(
        2,
        clock=lambda: next(values),
        total_weight=2,
        human=True,
        now=lambda: datetime(2026, 11, 4, 10, 0),
    )
    output = reporter.advance()
    assert "[  1/2 | 50.0 %]" in output
    assert "last 1 min" in output and "remaining ≈ 1 min" in output
    assert "done ≈ Wed 04.11. 10:01" in output
