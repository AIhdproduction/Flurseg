"""Plain-console progress and deterministic ETA coverage."""

from __future__ import annotations

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
