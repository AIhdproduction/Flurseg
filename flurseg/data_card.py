"""Generate a publication-safe training-data source card from local provenance."""

from __future__ import annotations

import json
import sqlite3
import urllib.parse
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from flurseg.io.atomic import atomic_write_bytes
from flurseg.paths import RepoPaths, ensure_inside

_LICENCE_LINKS = {
    "cc0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "cc-by": "https://creativecommons.org/licenses/by/4.0/",
    "etalab": "https://www.etalab.gouv.fr/licence-ouverte-open-licence/",
    "dl-de/zero": "https://www.govdata.de/dl-de/zero-2-0",
    "dl-de/by": "https://www.govdata.de/dl-de/by-2-0",
    "swisstopo": "https://www.swisstopo.admin.ch/en/geo-services/geo-services/free-geodata.html",
    "nlod": "https://data.norge.no/nlod/en/2.0",
}
_STATEMENTS = (
    "In addition the model was trained on proprietary drone, camera and phone photographs of the author; these images are not published.",
    "SAM 3 was used locally as a teacher to create pseudo-labels, reviewed by a human; SAM 3 weights are not distributed.",
    "Built with DINOv3. Weights are distributed under the DINOv3 License.",
    "Code is licensed under AGPL-3.0.",
)


def _read_lock(path: Path) -> list[dict[str, Any]]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return []
    entries = value.get("entries", []) if isinstance(value, dict) else []
    return [item for item in entries if isinstance(item, dict)]


def _source_group(record: dict[str, Any]) -> str:
    identifier = str(record.get("id") or "").lower()
    host = urllib.parse.urlsplit(str(record.get("url") or "")).netloc.lower()
    labels = (
        ("swisstopo", "swisstopo SWISSIMAGE"), ("ign", "IGN BD ORTHO"), ("pdok", "PDOK"),
        ("nrw", "NRW DOP"), ("berlin", "Berlin DOP"), ("pnoa", "PNOA"),
        ("geoportail", "LU geoportail"), ("panoramax", "Panoramax IGN"),
        ("amsterdam", "Amsterdam"), ("internet_archive", "Internet Archive"),
        ("rdd", "RDD2022"), ("zenodo", "Zenodo"), ("hf_", "Hugging Face"),
        ("vegbilder", "Statens vegvesen Vegbilder"), ("commons", "Wikimedia Commons"),
    )
    for needle, label in labels:
        if needle in identifier or needle in host:
            return label
    return host or str(record.get("id") or "Unspecified source")


def _region(record: dict[str, Any]) -> str:
    group = _source_group(record)
    if "Vegbilder" in group:
        return "Norway"
    if "swisstopo" in group:
        return "Switzerland"
    if "IGN" in group or "Panoramax" in group:
        return "France"
    if "PDOK" in group or "Amsterdam" in group:
        return "Netherlands"
    return "See source attribution"


def _manifest_rows(paths: RepoPaths) -> list[dict[str, Any]]:
    if not paths.database.is_file():
        return []
    try:
        uri = f"file:{paths.database.as_posix()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            connection.row_factory = sqlite3.Row
            return [dict(row) for row in connection.execute("SELECT hash, parent_hash, source, session, status, id FROM images WHERE status != 'rejected'")]
    except sqlite3.Error:
        return []


def _used_ids(paths: RepoPaths) -> tuple[set[str] | None, set[str]]:
    path = paths.dataset_dir / "dataset.json"
    if not path.is_file():
        return None, set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set(), set()
    ids: set[str] = set()
    for values in (data.get("source_image_ids_per_split", {}), data.get("image_ids_per_split", {})):
        if isinstance(values, dict):
            for split_ids in values.values():
                if isinstance(split_ids, list):
                    ids.update(str(item) for item in split_ids)
    excluded = data.get("excluded_licence", [])
    excluded_licences = {
        str(item.get("licence") if isinstance(item, dict) else item)
        for item in excluded
    }
    return ids, excluded_licences


def collect(paths: RepoPaths, *, only_used: bool = False) -> dict[str, Any]:
    """Collect safe aggregate provenance; no signed URLs or local paths are returned."""
    locks = _read_lock(paths.config_dir / "samples.lock.yaml") + _read_lock(paths.config_dir / "extra_sources.lock.yaml")
    by_hash = {str(item.get("sha256") or item.get("hash") or ""): item for item in locks}
    used_ids, excluded_licences = _used_ids(paths) if only_used else (None, set())
    groups: dict[tuple[str, str, str, str], int] = defaultdict(int)
    authors: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    own: dict[str, int] = defaultdict(int)
    for row in _manifest_rows(paths):
        if used_ids is not None and str(row["id"]) not in used_ids:
            continue
        lock = by_hash.get(str(row.get("parent_hash") or "")) or by_hash.get(str(row.get("hash") or ""))
        if lock is None:
            own[str(row.get("source") or "unknown")] += 1
            continue
        licence = str(lock.get("licence") or "Unknown")
        if licence in excluded_licences:
            continue
        group = _source_group(lock)
        key = (group, _region(lock), licence, str(lock.get("attribution") or ""))
        groups[key] += 1
        author = str(lock.get("author") or lock.get("artist") or "").strip()
        if author:
            authors[key].add(author)
    rows = [
        {
            "source": source,
            "country_region": region,
            "licence": licence,
            "attribution": attribution,
            "images_used": count,
            "authors": sorted(authors[(source, region, licence, attribution)]),
        }
        for (source, region, licence, attribution), count in sorted(groups.items())
    ]
    return {"sources": rows, "own_images": dict(sorted(own.items())), "statements": list(_STATEMENTS)}


def render_markdown(data: dict[str, Any]) -> str:
    """Render a paste-ready model-card section without exposing paths or URLs."""
    lines = ["# Training data", "", "| Source | Country/region | Licence | Attribution | Images used |", "|---|---|---|---|---:|"]
    for row in data["sources"]:
        lines.append(f"| {row['source']} | {row['country_region']} | {row['licence']} | {row['attribution']} | {row['images_used']} |")
    if not data["sources"]:
        lines.append("| No locked third-party images found | — | — | — | 0 |")
    lines.extend(["", "## Licences", ""])
    seen = sorted({str(row["licence"]) for row in data["sources"]})
    for licence in seen:
        normalised = licence.lower().replace(" ", "-")
        link = next((url for key, url in _LICENCE_LINKS.items() if key in normalised), None)
        lines.append(f"- [{licence}]({link})" if link else f"- {licence}")
    commons_authors = sorted({author for row in data["sources"] if row["source"] == "Wikimedia Commons" for author in row.get("authors", [])})
    if commons_authors:
        lines.extend(["", "<details><summary>Wikimedia Commons authors</summary>", "", *[f"- {author}" for author in commons_authors], "", "</details>"])
    lines.extend(["", "## Proprietary data", ""])
    for source, count in data["own_images"].items():
        lines.append(f"- {source}: {count} images")
    lines.extend(["", *data["statements"], ""])
    return "\n".join(lines)


def write_data_card(paths: RepoPaths, out: Path, *, only_used: bool = False, output_format: str = "md") -> Path:
    """Collect provenance and atomically write Markdown or JSON under the repository root."""
    target = out if out.is_absolute() else paths.root / out
    ensure_inside(target, paths.root)
    data = collect(paths, only_used=only_used)
    content = render_markdown(data) if output_format == "md" else json.dumps(data, indent=2, sort_keys=True) + "\n"
    atomic_write_bytes(target, content.encode("utf-8"))
    return target
