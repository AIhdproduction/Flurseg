"""Self-contained HTML report rendering with inline CSS and inline SVG charts."""

from __future__ import annotations

import importlib.resources
import logging
from datetime import UTC, datetime
from pathlib import Path

import jinja2

from flurseg.config import FlurConfig
from flurseg.paths import RepoPaths
from flurseg.report.metrics import ClassPrecisionItem, compute_metrics

logger = logging.getLogger(__name__)


def _escape_xml(text: str) -> str:
    """Escape text for safe inclusion in SVG content."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def render_status_chart(
    status_counts: dict[str, int],
    width: int = 700,
    bar_height: int = 24,
) -> str:
    """Generate an inline SVG horizontal bar chart for status distributions."""
    statuses = [
        ("ingested", "Ingested", "#94a3b8"),
        ("prelabeled", "Prelabeled", "#60a5fa"),
        ("auto_accepted", "Auto Accepted", "#34d399"),
        ("needs_review", "Needs Review", "#fbbf24"),
        ("reviewed", "Reviewed", "#a78bfa"),
        ("rejected", "Rejected", "#ef4444"),
    ]

    total = sum(status_counts.values())
    max_val = max(status_counts.values(), default=1)
    if max_val == 0:
        max_val = 1

    row_gap = 10
    label_width = 130
    value_width = 120
    chart_area_width = max(100, width - label_width - value_width)
    total_height = len(statuses) * (bar_height + row_gap) + 20

    lines: list[str] = [
        f'<svg viewBox="0 0 {width} {total_height}" width="100%" height="{total_height}">'
    ]

    for index, (key, label, color) in enumerate(statuses):
        count = status_counts.get(key, 0)
        percentage = (count / total * 100) if total > 0 else 0.0
        bar_w = max(2, int((count / max_val) * chart_area_width)) if count > 0 else 0
        y = 10 + index * (bar_height + row_gap)
        text_y = y + bar_height // 2 + 5

        lines.append(
            f'<text x="{label_width - 12}" y="{text_y}" text-anchor="end" fill="#ecedee" '
            f'font-family="system-ui, sans-serif" font-size="13">{_escape_xml(label)}</text>'
        )

        lines.append(
            f'<rect x="{label_width}" y="{y}" width="{chart_area_width}" height="{bar_height}" '
            f'rx="3" fill="#23262a" />'
        )

        if bar_w > 0:
            lines.append(
                f'<rect x="{label_width}" y="{y}" width="{bar_w}" height="{bar_height}" '
                f'rx="3" fill="{color}" />'
            )

        val_text = f"{count:,} ({percentage:.1f}%)"
        lines.append(
            f'<text x="{label_width + chart_area_width + 12}" y="{text_y}" text-anchor="start" fill="#a7acb2" '
            f'font-family="monospace" font-size="12">{_escape_xml(val_text)}</text>'
        )

    lines.append("</svg>")
    return "\n".join(lines)


def render_class_distribution_chart(
    cfg: FlurConfig,
    class_totals: dict[str, int],
    width: int = 700,
    bar_height: int = 22,
) -> str:
    """Generate an inline SVG horizontal bar chart for semantic class pixel shares."""
    classes = [c for c in cfg.classes.classes if c.id != 255]
    grand_total = sum(class_totals.get(str(c.id), 0) for c in classes)
    max_count = max((class_totals.get(str(c.id), 0) for c in classes), default=1)
    if max_count == 0:
        max_count = 1

    row_gap = 8
    label_width = 140
    value_width = 140
    chart_area_width = max(100, width - label_width - value_width)
    total_height = len(classes) * (bar_height + row_gap) + 20

    lines: list[str] = [
        f'<svg viewBox="0 0 {width} {total_height}" width="100%" height="{total_height}">'
    ]

    for index, c_def in enumerate(classes):
        cid_str = str(c_def.id)
        count = class_totals.get(cid_str, 0)
        percentage = (count / grand_total * 100) if grand_total > 0 else 0.0
        bar_w = max(2, int((count / max_count) * chart_area_width)) if count > 0 else 0
        y = 10 + index * (bar_height + row_gap)
        text_y = y + bar_height // 2 + 4

        lines.append(
            f'<circle cx="{label_width - 120}" cy="{y + bar_height // 2}" r="5" fill="{c_def.display_color}" />'
        )

        lines.append(
            f'<text x="{label_width - 108}" y="{text_y}" text-anchor="start" fill="#ecedee" '
            f'font-family="system-ui, sans-serif" font-size="12">{_escape_xml(c_def.name)}</text>'
        )

        lines.append(
            f'<rect x="{label_width}" y="{y}" width="{chart_area_width}" height="{bar_height}" '
            f'rx="3" fill="#23262a" />'
        )

        if bar_w > 0:
            lines.append(
                f'<rect x="{label_width}" y="{y}" width="{bar_w}" height="{bar_height}" '
                f'rx="3" fill="{c_def.display_color}" />'
            )

        val_text = f"{count:,} ({percentage:.1f}%)"
        lines.append(
            f'<text x="{label_width + chart_area_width + 12}" y="{text_y}" text-anchor="start" fill="#a7acb2" '
            f'font-family="monospace" font-size="11">{_escape_xml(val_text)}</text>'
        )

    lines.append("</svg>")
    return "\n".join(lines)


def render_spot_check_chart(
    per_class_error: dict[str, float],
    limit: float,
    cfg: FlurConfig,
    width: int = 700,
    bar_height: int = 22,
) -> str:
    """Generate an inline SVG horizontal bar chart of spot-check error rates with limit line."""
    classes = [c for c in cfg.classes.classes if c.id != 255]
    row_gap = 8
    label_width = 140
    value_width = 120
    chart_area_width = max(100, width - label_width - value_width)

    max_err = max([limit * 1.5, *per_class_error.values()])
    if max_err <= 0:
        max_err = max(0.05, limit * 1.5)

    total_height = len(classes) * (bar_height + row_gap) + 30
    limit_x = label_width + int((limit / max_err) * chart_area_width)

    lines: list[str] = [
        f'<svg viewBox="0 0 {width} {total_height}" width="100%" height="{total_height}">'
    ]

    lines.append(
        f'<line x1="{limit_x}" y1="5" x2="{limit_x}" y2="{total_height - 15}" '
        f'stroke="#ef4444" stroke-width="1.5" stroke-dasharray="4 3" />'
    )
    lines.append(
        f'<text x="{limit_x}" y="{total_height - 2}" text-anchor="middle" fill="#ef4444" '
        f'font-family="monospace" font-size="11">Limit: {limit * 100:.1f}%</text>'
    )

    for index, c_def in enumerate(classes):
        cid_str = str(c_def.id)
        rate = per_class_error.get(cid_str, 0.0)
        percentage = rate * 100
        bar_w = int((rate / max_err) * chart_area_width) if rate > 0 else 0
        y = 10 + index * (bar_height + row_gap)
        text_y = y + bar_height // 2 + 4

        lines.append(
            f'<text x="{label_width - 12}" y="{text_y}" text-anchor="end" fill="#ecedee" '
            f'font-family="system-ui, sans-serif" font-size="12">{_escape_xml(c_def.name)}</text>'
        )

        lines.append(
            f'<rect x="{label_width}" y="{y}" width="{chart_area_width}" height="{bar_height}" '
            f'rx="3" fill="#23262a" />'
        )

        bar_color = "#ef4444" if rate > limit else "#60a5fa"
        if bar_w > 0:
            lines.append(
                f'<rect x="{label_width}" y="{y}" width="{bar_w}" height="{bar_height}" '
                f'rx="3" fill="{bar_color}" />'
            )

        val_text = f"{percentage:.2f}%"
        val_color = "#ef4444" if rate > limit else "#a7acb2"
        lines.append(
            f'<text x="{label_width + chart_area_width + 12}" y="{text_y}" text-anchor="start" fill="{val_color}" '
            f'font-family="monospace" font-size="11">{_escape_xml(val_text)}</text>'
        )

    lines.append("</svg>")
    return "\n".join(lines)


def render_precision_chart(
    class_precisions: list[ClassPrecisionItem],
    width: int = 700,
    bar_height: int = 22,
) -> str:
    """Generate an inline SVG horizontal bar chart for per-class precision against target 90%."""
    row_gap = 8
    label_width = 140
    value_width = 160
    chart_area_width = max(100, width - label_width - value_width)
    total_height = len(class_precisions) * (bar_height + row_gap) + 30

    target_x = label_width + int(0.90 * chart_area_width)

    lines: list[str] = [
        f'<svg viewBox="0 0 {width} {total_height}" width="100%" height="{total_height}">'
    ]

    lines.append(
        f'<line x1="{target_x}" y1="5" x2="{target_x}" y2="{total_height - 15}" '
        f'stroke="#34d399" stroke-width="1.5" stroke-dasharray="4 3" />'
    )
    lines.append(
        f'<text x="{target_x}" y="{total_height - 2}" text-anchor="middle" fill="#34d399" '
        f'font-family="monospace" font-size="11">Target: 90%</text>'
    )

    for index, item in enumerate(class_precisions):
        percentage = item.precision * 100
        bar_w = int(item.precision * chart_area_width) if item.precision > 0 else 0
        y = 10 + index * (bar_height + row_gap)
        text_y = y + bar_height // 2 + 4

        lines.append(
            f'<circle cx="{label_width - 120}" cy="{y + bar_height // 2}" r="5" fill="{item.display_color}" />'
        )

        lines.append(
            f'<text x="{label_width - 108}" y="{text_y}" text-anchor="start" fill="#ecedee" '
            f'font-family="system-ui, sans-serif" font-size="12">{_escape_xml(item.class_name)}</text>'
        )

        lines.append(
            f'<rect x="{label_width}" y="{y}" width="{chart_area_width}" height="{bar_height}" '
            f'rx="3" fill="#23262a" />'
        )

        bar_color = item.display_color if item.won_pixels > 0 else "#4b5563"
        if bar_w > 0:
            lines.append(
                f'<rect x="{label_width}" y="{y}" width="{bar_w}" height="{bar_height}" '
                f'rx="3" fill="{bar_color}" />'
            )

        if item.won_pixels > 0:
            thresh_str = (
                f"thresh: {item.suggested_threshold:.2f}"
                if item.suggested_threshold is not None
                else "no thresh"
            )
            val_text = f"{percentage:.1f}% ({thresh_str})"
        else:
            val_text = "no data"

        lines.append(
            f'<text x="{label_width + chart_area_width + 12}" y="{text_y}" text-anchor="start" fill="#a7acb2" '
            f'font-family="monospace" font-size="11">{_escape_xml(val_text)}</text>'
        )

    lines.append("</svg>")
    return "\n".join(lines)


def _load_template() -> jinja2.Template:
    """Load the report Jinja2 template safely via importlib.resources or package directory."""
    try:
        content = (
            importlib.resources.files("flurseg.report")
            .joinpath("templates", "report.html.j2")
            .read_text(encoding="utf-8")
        )
        return jinja2.Template(content, autoescape=True)
    except (OSError, TypeError, ValueError, ModuleNotFoundError) as error:
        logger.warning(
            "Failed to load template via importlib.resources (%s); falling back to file path",
            error,
        )
        template_file = Path(__file__).resolve().parent / "templates" / "report.html.j2"
        content = template_file.read_text(encoding="utf-8")
        return jinja2.Template(content, autoescape=True)


def generate_report(paths: RepoPaths, cfg: FlurConfig) -> Path:
    """Compile metrics, render self-contained HTML report, and write timestamped and latest outputs."""
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_metrics(paths, cfg)

    status_svg = render_status_chart(metrics.status_counts.by_status)
    class_svg = render_class_distribution_chart(cfg, metrics.class_distribution.total)
    error_svg = render_spot_check_chart(
        metrics.spot_check.per_class_error_rate,
        metrics.spot_check_error_limit,
        cfg,
    )
    precision_svg = render_precision_chart(metrics.precision.per_class)

    template = _load_template()
    html_output = template.render(
        metrics=metrics,
        cfg=cfg,
        status_svg=status_svg,
        class_svg=class_svg,
        error_svg=error_svg,
        precision_svg=precision_svg,
    )

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    timestamped_path = paths.reports_dir / f"report_{timestamp}.html"
    latest_path = paths.reports_dir / "latest.html"

    timestamped_path.write_text(html_output, encoding="utf-8")
    latest_path.write_text(html_output, encoding="utf-8")

    return timestamped_path
