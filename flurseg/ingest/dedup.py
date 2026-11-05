"""Safe, reviewable near-duplicate detection for raw input images.

The module intentionally never deletes files. Applying a run only moves one image
at a time from ``input`` to its recorded quarantine location; undo performs the
opposite move after the same containment checks.
"""

from __future__ import annotations

import csv
import html
import json
import os
import re
import shutil
import sqlite3
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.fft import dctn

from flurseg.ingest.discover import Candidate, discover
from flurseg.io.atomic import atomic_save_jpg, atomic_save_png, atomic_write_json
from flurseg.io.hashing import sha256_file
from flurseg.paths import RepoPaths, ensure_inside, is_inside

_DJI_TIME = re.compile(r"DJI_(\d{14})", re.IGNORECASE)
_DJI_CAPTURE = re.compile(r"DJI_(\d{14})_(\d+)", re.IGNORECASE)
_XMP_VALUE = re.compile(r"(?:drone-dji:)?([A-Za-z]+)=[\"']([^\"']+)[\"']")
_GPS_VALUE = re.compile(r"(?:drone-dji:)?(GpsLatitude|GpsLongitude)=[\"']([^\"']+)[\"']")
_POPCOUNT = np.array([number.bit_count() for number in range(256)], dtype=np.uint8)


@dataclass(frozen=True, slots=True)
class DedupOptions:
    source: str = "all"
    session: str | None = None
    include_quality: bool = False
    drone_mode: str = "stride"
    drone_stride: int = 2
    drone_max_overlap: float = 0.5
    phash_threshold: int = 8
    burst_seconds: float = 3.0
    burst_phash_threshold: int = 16
    phone_stride: int = 1
    cross_session_days: int = 14
    max_drop_fraction: float = 0.8
    min_keep_per_session: int = 5


@dataclass(slots=True)
class Feature:
    width: int = 0
    height: int = 0
    phash: int = 0
    dhash: int = 0
    sharpness: float = 0.0
    exposed: bool = False
    captured: datetime | None = None
    latitude: float | None = None
    longitude: float | None = None
    altitude: float | None = None
    focal_px: float | None = None
    pitch: float | None = None
    yaw: float | None = None
    nominal_width: int | None = None
    nominal_height: int | None = None


@dataclass(slots=True)
class Decision:
    candidate: Candidate
    sha256: str
    feature: Feature
    decision: str = "keep"
    reason: str = "unique"
    kept_instead: str | None = None
    overlap: float | None = None
    quality_flag: str | None = None

    @property
    def key(self) -> str:
        return str(self.candidate.path)

    def metrics(self) -> dict[str, Any]:
        return {
            "phash": f"{self.feature.phash:016x}",
            "dhash": f"{self.feature.dhash:016x}",
            "overlap": self.overlap,
            "sharpness": self.feature.sharpness,
            "badly_exposed": self.feature.exposed,
            "quality_flag": self.quality_flag,
        }


@dataclass(slots=True)
class DedupResult:
    paths: RepoPaths
    options: DedupOptions
    run_id: str
    decisions: list[Decision]
    report_dir: Path
    can_apply: bool = True
    error: str | None = None
    exit_code: int = 0

    def print_summary(self) -> None:
        grouped: dict[tuple[str, str], list[Decision]] = defaultdict(list)
        for item in self.decisions:
            grouped[(item.candidate.source, item.candidate.session)].append(item)
        print(f"Drone mode: {self.options.drone_mode}")
        print(f"Phone stride: {self.options.phone_stride}")
        print("source  session  total  keep  drop (reason)")
        for (source, session), items in sorted(grouped.items()):
            reasons = Counter(item.reason for item in items if item.decision == "drop")
            dropped = sum(item.decision == "drop" for item in items)
            detail = ", ".join(f"{name}={count}" for name, count in sorted(reasons.items())) or "-"
            print(f"{source}  {session}  {len(items)}  {len(items) - dropped}  {dropped} ({detail})")
        print(f"Report: {self.report_dir}")


def _bits(value: int, other: int) -> int:
    return int((value ^ other).bit_count())


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value.replace(",", "."))
    except ValueError:
        return None


def _xmp_metadata(path: Path) -> dict[str, str]:
    try:
        with path.open("rb") as stream:
            text = stream.read(256 * 1024).decode("utf-8", errors="ignore")
    except OSError:
        return {}
    values = {name.lower(): value for name, value in _XMP_VALUE.findall(text)}
    values.update({name.lower(): value for name, value in _GPS_VALUE.findall(text)})
    return values


def _capture_time(path: Path, xmp: dict[str, str]) -> datetime | None:
    match = _DJI_TIME.search(path.name)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y%m%d%H%M%S")
        except ValueError:
            pass
    for key in ("datetimeoriginal", "createdate", "modifydate"):
        value = xmp.get(key)
        if value:
            for pattern in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
                try:
                    return datetime.strptime(value[:19], pattern)
                except ValueError:
                    continue
    try:
        with Image.open(path) as opened:
            exif = opened.getexif()
            sub_ifd = exif.get_ifd(0x8769)
            value = sub_ifd.get(36867) or exif.get(36867) or exif.get(306)
            subsec = sub_ifd.get(37521)
        if value:
            captured = datetime.strptime(str(value), "%Y:%m:%d %H:%M:%S")
            if subsec:
                digits = "".join(character for character in str(subsec) if character.isdigit())
                if digits:
                    captured = captured.replace(microsecond=int((digits + "000000")[:6]))
            return captured
    except (OSError, ValueError):
        pass
    return None


def extract_feature(path: Path) -> Feature:
    """Decode a bounded preview and extract perceptual, quality and DJI metadata."""
    xmp = _xmp_metadata(path)
    try:
        with Image.open(path) as opened:
            width, height = opened.size
            opened.draft("RGB", (1024, 1024))
            image = ImageOps.exif_transpose(opened).convert("L")
            image.thumbnail((1024, 1024))
            pixels = np.asarray(image, dtype=np.uint8).copy()
    except (OSError, ValueError):
        return Feature(captured=_capture_time(path, xmp))
    hash_image = Image.fromarray(pixels).resize((32, 32), Image.Resampling.LANCZOS)
    coefficients = dctn(np.asarray(hash_image, dtype=np.float32), type=2, norm="ortho")[:8, :9]
    values = coefficients.flatten()[1:65]
    median = float(np.median(values))
    phash = 0
    for value in values:
        phash = (phash << 1) | int(value >= median)
    dh_pixels = np.asarray(
        Image.fromarray(pixels).resize((9, 8), Image.Resampling.LANCZOS), dtype=np.uint8
    )
    dhash = 0
    for value in (dh_pixels[:, 1:] >= dh_pixels[:, :-1]).flat:
        dhash = (dhash << 1) | int(value)
    small = np.asarray(Image.fromarray(pixels).resize((1024, max(1, round(pixels.shape[0] * min(1, 1024 / pixels.shape[1]))))), dtype=np.uint8)
    sharpness = float(cv2.Laplacian(small, cv2.CV_64F).var())
    exposed = bool(np.mean((small <= 5) | (small >= 250)) > 0.40)
    return Feature(
        width=width,
        height=height,
        phash=phash,
        dhash=dhash,
        sharpness=sharpness,
        exposed=exposed,
        captured=_capture_time(path, xmp),
        latitude=_parse_float(xmp.get("gpslatitude")),
        longitude=_parse_float(xmp.get("gpslongitude")),
        altitude=_parse_float(xmp.get("relativealtitude")),
        focal_px=_parse_float(xmp.get("calibratedfocallength")),
        pitch=_parse_float(xmp.get("gimbalpitchdegree")),
        yaw=_parse_float(xmp.get("gimbalyawdegree")),
        nominal_width=int(float(xmp["imagewidth"])) if xmp.get("imagewidth", "").replace(".", "", 1).isdigit() else None,
        nominal_height=int(float(xmp["imageheight"])) if xmp.get("imageheight", "").replace(".", "", 1).isdigit() else None,
    )


def _db_hashes(paths: RepoPaths, hashes: set[str]) -> set[str]:
    if not paths.database.is_file() or not hashes:
        return set()
    try:
        connection = sqlite3.connect(f"file:{paths.database.as_posix()}?mode=ro", uri=True)
        try:
            found: set[str] = set()
            values = sorted(hashes)
            for start in range(0, len(values), 900):
                part = values[start : start + 900]
                query = "SELECT hash FROM images WHERE hash IN (" + ",".join("?" for _ in part) + ")"
                found.update(row[0] for row in connection.execute(query, part))
            return found
        finally:
            connection.close()
    except sqlite3.Error:
        return set()


def _archive_hashes(paths: RepoPaths, source: str) -> set[str]:
    root = paths.archive_dir / source
    if not root.is_dir():
        return set()
    allowed = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic", ".heif"}
    hashes: set[str] = set()
    for item in root.rglob("*"):
        if item.is_file() and item.suffix.lower() in allowed:
            try:
                hashes.add(sha256_file(item))
            except OSError:
                continue
    return hashes


def _footprint(feature: Feature) -> tuple[float, float, float, float, float] | None:
    if any(value is None for value in (feature.latitude, feature.longitude, feature.altitude, feature.focal_px, feature.pitch)):
        return None
    if feature.pitch is None or feature.pitch > -70 or feature.altitude is None or feature.altitude <= 0:
        return None
    width = feature.nominal_width or feature.width
    height = feature.nominal_height or feature.height
    if width <= 0 or height <= 0 or not feature.focal_px:
        return None
    scale = feature.altitude / feature.focal_px
    return (feature.latitude, feature.longitude, width * scale, height * scale, feature.yaw or 0.0)


def _coverage(current: tuple[float, float, float, float, float], previous: list[tuple[float, float, float, float, float]]) -> float:
    """Estimate prior footprint coverage using an adaptive occupancy grid in local ENU metres."""
    if not previous:
        return 0.0
    lat, lon, width, height, yaw = current
    cell = max(0.25, max(width, height) / 400)
    x = np.arange(-width / 2, width / 2, cell) + cell / 2
    y = np.arange(-height / 2, height / 2, cell) + cell / 2
    if not len(x) or not len(y):
        return 0.0
    xx, yy = np.meshgrid(x, y)
    covered = np.zeros(xx.shape, dtype=bool)
    radians = np.deg2rad(yaw)
    east = xx * np.cos(radians) - yy * np.sin(radians)
    north = xx * np.sin(radians) + yy * np.cos(radians)
    for old_lat, old_lon, old_width, old_height, old_yaw in previous:
        origin_east = (old_lon - lon) * 111_320 * np.cos(np.deg2rad(lat))
        origin_north = (old_lat - lat) * 110_540
        delta_east = east - origin_east
        delta_north = north - origin_north
        old_rad = np.deg2rad(old_yaw)
        local_x = delta_east * np.cos(old_rad) + delta_north * np.sin(old_rad)
        local_y = -delta_east * np.sin(old_rad) + delta_north * np.cos(old_rad)
        covered |= (np.abs(local_x) <= old_width / 2) & (np.abs(local_y) <= old_height / 2)
    return float(covered.mean())


def _winner(items: list[Decision]) -> Decision:
    return min(
        items,
        key=lambda item: (-item.feature.sharpness, -(item.feature.width * item.feature.height), item.key.lower()),
    )


def _drop(item: Decision, reason: str, winner: Decision | None = None) -> None:
    item.decision = "drop"
    item.reason = reason
    item.kept_instead = winner.key if winner else None


def _deduplicate_perceptual(decisions: list[Decision], threshold: int) -> None:
    eligible = [item for item in decisions if item.decision == "keep"]
    parent = list(range(len(eligible)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for index, item in enumerate(eligible):
        if not index:
            continue
        phashes = np.array([prior.feature.phash for prior in eligible[:index]], dtype=np.uint64)
        dhashes = np.array([prior.feature.dhash for prior in eligible[:index]], dtype=np.uint64)
        pcounts = _POPCOUNT[np.bitwise_xor(phashes, np.uint64(item.feature.phash)).view(np.uint8)].reshape(-1, 8).sum(axis=1)
        dcounts = _POPCOUNT[np.bitwise_xor(dhashes, np.uint64(item.feature.dhash)).view(np.uint8)].reshape(-1, 8).sum(axis=1)
        for prior in np.flatnonzero((pcounts <= threshold) & (dcounts <= threshold + 4)):
            left, right = find(int(prior)), find(index)
            if left != right:
                parent[right] = left
    groups: dict[int, list[Decision]] = defaultdict(list)
    for index, item in enumerate(eligible):
        groups[find(index)].append(item)
    for group in groups.values():
        if len(group) > 1:
            winner = _winner(group)
            for item in group:
                if item is not winner:
                    _drop(item, "perceptual", winner)


def _deduplicate_bursts(decisions: list[Decision], seconds: float, phash_threshold: int) -> None:
    for source_session, items in _group_sessions(decisions).items():
        if source_session[0] != "phone":
            continue
        pending = [item for item in items if item.decision == "keep" and item.feature.captured]
        pending.sort(key=lambda item: (item.feature.captured, item.key))
        for index, item in enumerate(pending):
            group = [item]
            for other in pending[index + 1 :]:
                if other.feature.captured is None or item.feature.captured is None:
                    continue
                if (other.feature.captured - item.feature.captured).total_seconds() > seconds:
                    break
                if _bits(item.feature.phash, other.feature.phash) <= phash_threshold:
                    group.append(other)
            if len(group) > 1:
                winner = _winner(group)
                for other in group:
                    if other is not winner:
                        _drop(other, "burst", winner)


def _group_sessions(decisions: list[Decision]) -> dict[tuple[str, str], list[Decision]]:
    groups: dict[tuple[str, str], list[Decision]] = defaultdict(list)
    for item in decisions:
        groups[(item.candidate.source, item.candidate.session)].append(item)
    return groups


def _drone_capture_key(item: Decision) -> tuple[int, datetime, int, str]:
    """Order mapping-flight images by DJI timestamp/sequence, then EXIF, then filename."""
    match = _DJI_CAPTURE.search(item.candidate.path.name)
    if match:
        try:
            return (0, datetime.strptime(match.group(1), "%Y%m%d%H%M%S"), int(match.group(2)), item.key.lower())
        except ValueError:
            pass
    if item.feature.captured is not None:
        return (1, item.feature.captured, 0, item.key.lower())
    return (2, datetime.max, 0, item.key.lower())


def _natural_key(value: str) -> tuple[tuple[int, int | str], ...]:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in re.split(r"(\d+)", value)
    )


def _phone_capture_key(item: Decision) -> tuple[int, datetime, tuple[tuple[int, int | str], ...]]:
    if item.feature.captured is not None:
        return (0, item.feature.captured, _natural_key(item.candidate.path.name))
    return (1, datetime.max, _natural_key(item.candidate.path.name))


def _deduplicate_drone_stride(decisions: list[Decision], stride: int) -> None:
    """Keep every Nth non-exact drone image within each session in capture order."""
    for (source, _session), items in _group_sessions(decisions).items():
        if source != "drone":
            continue
        kept: Decision | None = None
        eligible = sorted((item for item in items if item.decision == "keep"), key=_drone_capture_key)
        for index, item in enumerate(eligible):
            if index % stride == 0:
                kept = item
            else:
                _drop(item, "drone_stride", kept)


def _deduplicate_phone_stride(decisions: list[Decision], stride: int) -> None:
    """Keep every Nth non-exact phone image within each session in capture order."""
    for (source, _session), items in _group_sessions(decisions).items():
        if source != "phone":
            continue
        kept: Decision | None = None
        eligible = sorted((item for item in items if item.decision == "keep"), key=_phone_capture_key)
        for index, item in enumerate(eligible):
            if index % stride == 0:
                kept = item
            else:
                _drop(item, "phone_stride", kept)


def _deduplicate_drone_overlap(decisions: list[Decision], options: DedupOptions) -> None:
    """Apply the original footprint-coverage greedy selection across close drone sessions."""
    drone = [item for item in decisions if item.candidate.source == "drone" and item.decision == "keep"]
    drone.sort(key=_drone_capture_key)
    kept_footprints: list[tuple[datetime | None, tuple[float, float, float, float, float]]] = []
    for item in drone:
        footprint = _footprint(item.feature)
        if footprint is None:
            continue
        relevant = [
            old
            for old_time, old in kept_footprints
            if item.feature.captured is None
            or old_time is None
            or abs((item.feature.captured - old_time).total_seconds()) <= options.cross_session_days * 86_400
        ]
        item.overlap = _coverage(footprint, relevant)
        if item.overlap > options.drone_max_overlap:
            _drop(item, "drone_overlap")
        else:
            kept_footprints.append((item.feature.captured, footprint))


def _apply_minimum_keep(decisions: list[Decision], minimum: int) -> None:
    soft_reasons = {
        "drone_stride",
        "phone_stride",
        "drone_overlap",
        "perceptual",
        "burst",
        "blurry",
        "badly_exposed",
    }
    for items in _group_sessions(decisions).values():
        kept = [item for item in items if item.decision == "keep"]
        needed = max(0, min(minimum, len(items)) - len(kept))
        if needed:
            choices = sorted(
                (item for item in items if item.decision == "drop" and item.reason in soft_reasons),
                key=lambda item: (-item.feature.sharpness, -(item.feature.width * item.feature.height), item.key.lower()),
            )
            for item in choices[:needed]:
                item.decision, item.reason, item.kept_instead = "keep", "minimum_keep", None


def _quality(decisions: list[Decision], include_quality: bool) -> None:
    for items in _group_sessions(decisions).values():
        median = float(np.median([item.feature.sharpness for item in items])) if items else 0.0
        for item in items:
            if item.feature.exposed:
                item.quality_flag = "badly_exposed"
            elif median > 0 and item.feature.sharpness < median * 0.25:
                item.quality_flag = "blurry"
            if include_quality and item.decision == "keep" and item.quality_flag:
                _drop(item, item.quality_flag)


def _run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _selected(paths: RepoPaths, options: DedupOptions) -> list[Candidate]:
    allowed = {"drone", "phone"} if options.source == "all" else {options.source}
    return [item for item in discover(paths) if item.source in allowed and (not options.session or item.session == options.session)]


def _write_report(result: DedupResult) -> None:
    report = result.report_dir
    thumbs = report / "thumbs"
    report.mkdir(parents=True, exist_ok=True)
    thumbs.mkdir(parents=True, exist_ok=True)
    with (report / "decisions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "source", "session", "drone_mode", "phone_stride", "decision", "reason", "kept_instead", "phash", "overlap", "sharpness", "exposure_flags"])
        writer.writeheader()
        for item in result.decisions:
            writer.writerow({"path": item.key, "source": item.candidate.source, "session": item.candidate.session, "drone_mode": result.options.drone_mode if item.candidate.source == "drone" else "", "phone_stride": result.options.phone_stride if item.candidate.source == "phone" else "", "decision": item.decision, "reason": item.reason, "kept_instead": item.kept_instead or "", "phash": f"{item.feature.phash:016x}", "overlap": item.overlap if item.overlap is not None else "", "sharpness": f"{item.feature.sharpness:.3f}", "exposure_flags": item.quality_flag or ("badly_exposed" if item.feature.exposed else "")})
    grouped = _group_sessions(result.decisions)
    summary = {
        "run_id": result.run_id,
        "drone_mode": result.options.drone_mode,
        "phone_stride": result.options.phone_stride,
        "parameters": asdict(result.options),
        "sources": {},
    }
    for (source, session), items in grouped.items():
        summary["sources"].setdefault(source, {})[session] = {"total": len(items), "keep": sum(item.decision == "keep" for item in items), "drop": sum(item.decision == "drop" for item in items), "reasons": dict(Counter(item.reason for item in items if item.decision == "drop"))}
    atomic_write_json(report / "summary.json", summary)
    thumbnails: dict[str, str] = {}
    for index, item in enumerate(result.decisions):
        name = f"{index:05d}.jpg"
        try:
            with Image.open(item.candidate.path) as opened:
                preview = ImageOps.exif_transpose(opened).convert("RGB")
                preview.thumbnail((256, 256))
                atomic_save_jpg(thumbs / name, np.asarray(preview), quality=85)
        except OSError:
            name = ""
        thumbnails[item.key] = name
    _write_maps(report, grouped)
    sections: list[str] = []
    for (source, session), items in sorted(grouped.items()):
        groups: dict[str, list[Decision]] = defaultdict(list)
        for item in items:
            groups[item.kept_instead or item.key].append(item)
        cards: list[str] = []
        for kept_path, group in sorted(groups.items()):
            for item in sorted(group, key=lambda entry: (entry.decision != "keep", entry.key.lower())):
                thumbnail = thumbnails[item.key]
                image = f'<img src="thumbs/{html.escape(thumbnail)}" alt="thumbnail">' if thumbnail else ""
                cards.append(
                    "<article>"
                    + image
                    + f"<p>{html.escape(item.candidate.path.name)}<br>{item.decision}: {html.escape(item.reason)}"
                    + f"<br>kept: {html.escape(Path(kept_path).name)}</p></article>"
                )
        map_name = f"map-{source}-{session}.png"
        map_link = f'<img class="map" src="{html.escape(map_name)}" alt="footprint map">' if (report / map_name).is_file() else ""
        sections.append(f"<h2>{html.escape(source)}/{html.escape(session)}</h2>{map_link}<section>{''.join(cards)}</section>")
    style = "body{font-family:system-ui;margin:2rem}section{display:flex;flex-wrap:wrap;gap:1rem}article{width:270px;border:1px solid #ccc;padding:.5rem}img{max-width:256px;max-height:256px}.map{border:1px solid #ccc}p{word-break:break-word}"
    page = "<!doctype html><meta charset=\"utf-8\"><title>Flurseg dedup review</title><style>" + style + "</style><h1>Dedup review " + html.escape(result.run_id) + "</h1><p>Drone mode: " + html.escape(result.options.drone_mode) + "; phone stride: " + str(result.options.phone_stride) + "</p>" + "".join(sections)
    # HTML is deliberately self-contained except for local, relative thumbnail links.
    from flurseg.io.atomic import atomic_write_bytes
    atomic_write_bytes(report / "index.html", page.encode("utf-8"))


def _write_maps(report: Path, grouped: dict[tuple[str, str], list[Decision]]) -> None:
    for (source, session), items in grouped.items():
        drones = [item for item in items if source == "drone" and _footprint(item.feature)]
        if not drones:
            continue
        image = Image.new("RGB", (320, 240), "white")
        draw = ImageDraw.Draw(image)
        lats = [item.feature.latitude for item in drones if item.feature.latitude is not None]
        lons = [item.feature.longitude for item in drones if item.feature.longitude is not None]
        for item in drones:
            x = 160 if len(set(lons)) < 2 else 20 + 280 * (item.feature.longitude - min(lons)) / (max(lons) - min(lons))
            y = 120 if len(set(lats)) < 2 else 20 + 200 * (max(lats) - item.feature.latitude) / (max(lats) - min(lats))
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#238636" if item.decision == "keep" else "#cf222e")
        atomic_save_png(report / f"map-{source}-{session}.png", np.asarray(image))


def _manifest_data(result: DedupResult, entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {"run_id": result.run_id, "parameters": asdict(result.options), "files": entries}


def _apply(result: DedupResult) -> DedupResult:
    drops = [item for item in result.decisions if item.decision == "drop"]
    for source in {item.candidate.source for item in result.decisions}:
        source_items = [item for item in result.decisions if item.candidate.source == source]
        if source_items and len([item for item in source_items if item.decision == "drop"]) / len(source_items) > result.options.max_drop_fraction:
            result.can_apply = False
            result.error = f"planned drops for {source} exceed --max-drop-fraction"
            result.exit_code = 1
            return result
    if any(not is_inside(item.candidate.path, result.paths.input_dir) for item in drops):
        result.can_apply = False
        result.error = "a planned source path is outside input"
        result.exit_code = 1
        return result
    run_dir = result.paths.quarantine_dir / "dedup" / result.run_id
    manifest_path = run_dir / "manifest.json"
    entries: list[dict[str, Any]] = []
    atomic_write_json(manifest_path, _manifest_data(result, entries))
    for item in drops:
        original = ensure_inside(item.candidate.path, result.paths.input_dir)
        target_dir = run_dir / item.candidate.source / item.candidate.session
        target = target_dir / original.name
        number = 1
        while target.exists():
            target = target_dir / f"{original.stem}_{number}{original.suffix}"
            number += 1
        target = ensure_inside(target, result.paths.quarantine_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "status": "pending",
            "original_path": str(original),
            "quarantine_path": str(target),
            "sha256": item.sha256,
            "reason": item.reason,
            "kept_instead": item.kept_instead,
            "metrics": item.metrics(),
        }
        entries.append(entry)
        atomic_write_json(manifest_path, _manifest_data(result, entries))
        shutil.move(str(original), str(target))
        entry["status"] = "moved"
        atomic_write_json(manifest_path, _manifest_data(result, entries))
    return result


def _apply_guard(result: DedupResult) -> None:
    for source in {item.candidate.source for item in result.decisions}:
        source_items = [item for item in result.decisions if item.candidate.source == source]
        dropped = sum(item.decision == "drop" for item in source_items)
        if source_items and dropped / len(source_items) > result.options.max_drop_fraction:
            result.can_apply = False
            result.error = f"planned drops for {source} exceed --max-drop-fraction"
            result.exit_code = 1
            return
    if any(
        item.decision == "drop" and not is_inside(item.candidate.path, result.paths.input_dir)
        for item in result.decisions
    ):
        result.can_apply = False
        result.error = "a planned source path is outside input"
        result.exit_code = 1


def run_dedup(paths: RepoPaths, options: DedupOptions | None = None, *, apply: bool = False, run_id: str | None = None) -> DedupResult:
    """Analyse selected candidates, always write a report, and optionally quarantine drops."""
    options = options or DedupOptions()
    if not 0 <= options.drone_max_overlap <= 1 or not 0 <= options.max_drop_fraction <= 1:
        raise ValueError("overlap and drop fractions must be between 0 and 1")
    if options.drone_mode not in {"stride", "overlap"}:
        raise ValueError("drone_mode must be 'stride' or 'overlap'")
    if options.drone_stride < 1:
        raise ValueError("drone_stride must be at least 1")
    if options.phone_stride < 1:
        raise ValueError("phone_stride must be at least 1")
    if options.min_keep_per_session < 0:
        raise ValueError("min_keep_per_session must be non-negative")
    identifier = run_id or _run_id()
    candidates = _selected(paths, options)
    hashes_and_paths = [(item, sha256_file(item.path)) for item in candidates]
    if len(candidates) > 8:
        workers = min(8, os.cpu_count() or 1)
        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                features = list(executor.map(extract_feature, (item.path for item in candidates)))
        except OSError:
            # Restricted Windows sandboxes can deny named-pipe creation; decoding remains correct.
            features = [extract_feature(item.path) for item in candidates]
    else:
        features = [extract_feature(item.path) for item in candidates]
    decisions = [
        Decision(candidate=item, sha256=content_hash, feature=feature)
        for (item, content_hash), feature in zip(hashes_and_paths, features, strict=True)
    ]
    hashes = {item.sha256 for item in decisions}
    ingested = _db_hashes(paths, hashes)
    archive = {source: _archive_hashes(paths, source) for source in {item.candidate.source for item in decisions}}
    seen: dict[str, Decision] = {}
    for item in decisions:
        if item.sha256 in ingested:
            _drop(item, "already_ingested")
        elif item.sha256 in archive[item.candidate.source]:
            _drop(item, "already_archived")
        elif item.sha256 in seen:
            _drop(item, "exact", seen[item.sha256])
        else:
            seen[item.sha256] = item
    if options.drone_mode == "stride":
        _deduplicate_drone_stride(decisions, options.drone_stride)
    else:
        _deduplicate_drone_overlap(decisions, options)
    _deduplicate_phone_stride(decisions, options.phone_stride)
    for source in ("drone", "phone"):
        _deduplicate_perceptual([item for item in decisions if item.candidate.source == source], options.phash_threshold)
    _deduplicate_bursts(decisions, options.burst_seconds, options.burst_phash_threshold)
    _quality(decisions, options.include_quality)
    _apply_minimum_keep(decisions, options.min_keep_per_session)
    result = DedupResult(paths, options, identifier, decisions, paths.reports_dir / "dedup" / identifier)
    _apply_guard(result)
    _write_report(result)
    if apply:
        if not result.can_apply:
            return result
        return _apply(result)
    return result


def undo_dedup(paths: RepoPaths, run_id: str) -> int:
    """Restore a run without overwriting an existing original path; repeated calls are harmless."""
    run_dir = ensure_inside(paths.quarantine_dir / "dedup" / run_id, paths.quarantine_dir)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        print(f"ERROR: quarantine manifest not found: {manifest_path}")
        return 1
    try:
        entries = json.loads(manifest_path.read_text(encoding="utf-8")).get("files", [])
    except (OSError, json.JSONDecodeError) as error:
        print(f"ERROR: cannot read quarantine manifest: {error}")
        return 1
    restored = skipped = 0
    for entry in entries:
        source = ensure_inside(Path(entry["quarantine_path"]), paths.quarantine_dir)
        target = ensure_inside(Path(entry["original_path"]), paths.input_dir)
        if target.exists() or not source.is_file():
            skipped += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        restored += 1
    print(f"Undo {run_id}: restored {restored}, skipped {skipped}")
    return 0
