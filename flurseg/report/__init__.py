"""Quality and segmentation performance reporting."""

from __future__ import annotations

from flurseg.report.metrics import (
    ClassDistribution,
    ClassPrecisionItem,
    PrecisionMetrics,
    PromptPrecisionItem,
    ReportMetrics,
    SpotCheckErrorRate,
    StatusCounts,
    TimingMetrics,
    compute_metrics,
)
from flurseg.report.render import generate_report

__all__ = [
    "ClassDistribution",
    "ClassPrecisionItem",
    "PrecisionMetrics",
    "PromptPrecisionItem",
    "ReportMetrics",
    "SpotCheckErrorRate",
    "StatusCounts",
    "TimingMetrics",
    "compute_metrics",
    "generate_report",
]
