"""Tests for openly licensed street-level extra sources."""

from __future__ import annotations

import io
import json
import logging
import socket
from pathlib import Path
from typing import Any, Self

import cv2
import numpy as np
import pytest
import yaml
from PIL import Image

from flurseg.cli import _print_extra_table, main
from flurseg.samples.extra import commons_categories as commons
from flurseg.samples.extra import commons_video_frames as videos
from flurseg.samples.extra import panoramax
from flurseg.samples.extra.common import FileProduct
from flurseg.samples.extra.config import CommonsCategoriesEntry, CommonsVideoFramesEntry, PanoramaxEntry
from flurseg.samples.extra.runner import fetch_extra

REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeResponse:
    """Small context-manager response used by the no-network test doubles."""

    def __init__(self, data: bytes) -> None:
        self.data = data

    def read(self) -> bytes:
        return self.data

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        return None


@pytest.fixture(autouse=True)
def guard_sockets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent street-fetch tests from reaching the network."""
    blocked = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("network"))
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


def _jpg() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (1800, 1200), (40, 100, 160)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _pano_entry() -> PanoramaxEntry:
    return PanoramaxEntry.model_validate(
        {
            "id": "pano",
            "type": "panoramax",
            "source": "phone",
            "session": "ext_pano",
            "bbox": [8.5, 47.3, 8.6, 47.4],
            "max_per_entry": 8,
            "licence": "Etalab Licence Ouverte 2.0",
            "attribution": "Panoramax",
            "session_yaml": {"area_type": "street"},
        }
    )


def _item(identifier: str, longitude: float, licence: str = "CC0", *, asset: str = "sd") -> dict[str, Any]:
    assets = {asset: {"href": f"https://images.test/{identifier}-{asset}.jpg"}}
    return {
        "id": identifier,
        "geometry": {"coordinates": [longitude, 47.3]},
        "properties": {"providers": [{"name": "Survey"}], "heading": 10},
        "links": [{"rel": "license", "title": licence}],
        "assets": assets,
    }


def test_panoramax_filters_paginate_thins_and_selects_assets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Panoramax accepts permitted item and collection licences only once per nearby view."""
    first = {
        "features": [
            _item("sharealike", 8.50, "CC-BY-SA-4.0"),
            _item("first", 8.50, "Etalab-2.0"),
            _item("near", 8.50002, "CC0"),
            {**_item("sphere", 8.51), "properties": {"pers:interior_orientation": {"field_of_view": 360}}},
        ],
        "links": [{"rel": "next", "href": "https://api.test/next"}],
    }
    collection_item = _item("collection", 8.501, "")
    collection_item["links"] = [{"rel": "collection", "href": "https://api.test/collection"}]
    second = {"features": [collection_item, _item("far", 8.502, "CC0", asset="hd")]}
    monkeypatch.setattr(
        panoramax,
        "_json",
        lambda url: first if "bbox" in url else ({"license": "CC0"} if "collection" in url else second),
    )
    seen: list[str] = []
    monkeypatch.setattr(panoramax, "request_bytes", lambda url, **kwargs: seen.append(url) or _jpg())

    products = panoramax.products(_pano_entry())

    assert [product.title for product in products] == ["first", "collection", "far"]
    assert seen[-1].endswith("far-hd.jpg") and all("sharealike" not in url for url in seen)


def _photo_page(
    title: str, mime: str, width: int, licence: str, artist: str = "<b>Alice</b>"
) -> dict[str, Any]:
    return {
        "title": title,
        "imageinfo": [
            {
                "mime": mime,
                "width": width,
                "height": 1800,
                "thumburl": f"https://upload.wikimedia.org/{title}.jpg?width=2560",
                "extmetadata": {
                    "LicenseShortName": {"value": licence},
                    "Artist": {"value": artist},
                },
            }
        ],
    }


def _commons_entry() -> CommonsCategoriesEntry:
    return CommonsCategoriesEntry.model_validate(
        {
            "id": "commons",
            "type": "commons_categories",
            "source": "phone",
            "session": "ext_commons",
            "categories": ["Root"],
            "depth": 1,
            "min_width": 1600,
            "max_per_entry": 2,
            "licence": "CC BY 4.0",
            "attribution": "Commons",
            "session_yaml": {"area_type": "street"},
        }
    )


def test_commons_categories_recurses_filters_and_uses_thumbnails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Photo categories recurse to the cap and reject non-photo or incompatible records."""
    calls: list[dict[str, str]] = []

    def fake_api(parameters: dict[str, str]) -> dict[str, Any]:
        calls.append(parameters)
        if parameters.get("list") == "categorymembers":
            return {"query": {"categorymembers": [{"title": "Category:Child"}]}}
        title = parameters["gcmtitle"]
        pages = (
            [_photo_page("File:good", "image/jpeg", 2000, "CC BY 4.0")]
            if title.endswith("Root")
            else [
                _photo_page("File:small", "image/png", 100, "CC0"),
                _photo_page("File:nc", "image/jpeg", 2000, "CC BY-NC 4.0"),
            ]
        )
        return {"query": {"pages": {str(index): page for index, page in enumerate(pages)}}}

    monkeypatch.setattr(commons, "_api", fake_api)
    urls: list[str] = []
    monkeypatch.setattr(commons, "request_bytes", lambda url, **kwargs: urls.append(url) or _jpg())

    products = commons.products(_commons_entry())

    assert len(products) == 1 and products[0].attribution == "Alice, CC BY 4.0, via Wikimedia Commons"
    assert "width=2560" in urls[0] and any(call.get("list") == "categorymembers" for call in calls)


def _avi(path: Path) -> bytes:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 2, (80, 60))
    assert writer.isOpened()
    for index in range(8):
        frame = (
            np.zeros((60, 80, 3), dtype=np.uint8)
            if index == 7
            else np.indices((60, 80))[1].astype(np.uint8) * 3
        )
        if 4 <= index < 7:
            cv2.circle(frame, (10 + index * 8, 30), 8, (255, 255, 255), -1)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame)
    writer.release()
    return path.read_bytes()


def _video_entry() -> CommonsVideoFramesEntry:
    return CommonsVideoFramesEntry.model_validate(
        {
            "id": "video",
            "type": "commons_video_frames",
            "source": "phone",
            "session": "ext_video",
            "categories": ["Dashcam videos"],
            "exclude_categories": ["Dashcam footage of China"],
            "depth": 2,
            "max_videos": 1,
            "max_frames_per_video": 2,
            "frame_interval_s": 1.0,
            "licence": "CC BY 4.0",
            "attribution": "Commons",
            "session_yaml": {"area_type": "street"},
        }
    )


def test_commons_video_frames_cache_deduplicates_and_skips_bad_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Video frames stay outside input, reject static and dark frames, and tolerate bad codecs."""
    payload = _avi(tmp_path / "source.avi")
    page = {"title": "File:Drive.avi"}
    info = {
        "mime": "video/ogg",
        "url": "https://upload.wikimedia.org/drive.avi",
        "extmetadata": {"LicenseShortName": {"value": "CC0"}, "Artist": {"value": "Driver"}},
    }
    monkeypatch.setattr(videos, "_accepted", lambda entry: [(page, info, "CC0")])
    monkeypatch.setattr(videos, "request_bytes", lambda *args, **kwargs: payload)
    results = videos.products(_video_entry(), tmp_path / "work" / "cache" / "videos")
    assert 1 <= len(results) <= 2 and all("-seconds.jpg" in item.filename for item in results)
    assert list((tmp_path / "work" / "cache" / "videos").glob("*")) and not (tmp_path / "input").exists()

    caplog.set_level(logging.WARNING)
    assert videos._extract(_video_entry(), tmp_path / "missing.webm", page, info, "CC0") == []
    assert "Could not decode Commons video" in caplog.text


def test_commons_video_categories_honour_exclusions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Excluded Commons sub-categories are never queried for video files."""
    visited: list[str] = []
    monkeypatch.setattr(
        commons, "_children", lambda category: ["Category:Dashcam footage of China", "Category:France"]
    )
    monkeypatch.setattr(
        commons,
        "_pages",
        lambda category, thumb_width, **kwargs: (
            visited.append(category) or ([{"title": category}] if category.endswith("France") else [])
        ),
    )

    pages = list(
        commons.commons_files(
            ["Dashcam videos"],
            depth=1,
            cap=1,
            exclude_categories=["Dashcam footage of China"],
            thumb_width=None,
        )
    )

    assert len(pages) == 1 and all("China" not in category for category in visited)


def _write_config(root: Path, entry: dict[str, Any]) -> None:
    target = root / "config" / "extra_sources.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump({"sources": [entry]}, sort_keys=False), encoding="utf-8")


def test_runner_dry_run_lock_attribution_and_cli_type_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Street runner dry runs are side-effect-free and saved frames receive lock records."""
    entry = _commons_entry().model_dump()
    _write_config(tmp_path, entry)
    monkeypatch.setattr("flurseg.samples.extra.runner.plan_commons_categories", lambda entry: 1)
    assert fetch_extra(tmp_path, dry_run=True)[0]["planned_files"] == 1
    assert not (tmp_path / "input").exists()

    product = FileProduct(
        _jpg(), "ext_commons_photo.jpg", 1.0, "https://photo.test", "photo", "CC0", "Artist, CC0"
    )
    monkeypatch.setattr("flurseg.samples.extra.runner._products", lambda *args: [product])
    first = fetch_extra(tmp_path)
    second = fetch_extra(tmp_path)
    assert first[0]["status"] == "downloaded" and second[0]["status"] == "skipped"
    lock = yaml.safe_load((tmp_path / "config" / "extra_sources.lock.yaml").read_text(encoding="utf-8"))
    report = (tmp_path / "reports" / "attribution" / "extra_sources.md").read_text(encoding="utf-8")
    assert lock["entries"][0]["filename"].endswith(".jpg") and "Commons" in report

    _write_config(tmp_path, _video_entry().model_dump())
    frame = FileProduct(
        _jpg(), "ext_video_drive_0-seconds.jpg", 1.0, "https://video.test", "drive", "CC0", "Driver, CC0"
    )
    monkeypatch.setattr("flurseg.samples.extra.runner.commons_video_frame_products", lambda *args: [frame])
    assert fetch_extra(tmp_path)[0]["status"] == "downloaded"
    lock = yaml.safe_load((tmp_path / "config" / "extra_sources.lock.yaml").read_text(encoding="utf-8"))
    assert {item["session"] for item in lock["entries"]} == {"ext_commons", "ext_video"}

    monkeypatch.setattr("flurseg.samples.extra.runner.plan_panoramax", lambda entry: 1)
    assert main(["fetch-extra", "--dry-run", "--only-type", "panoramax", "--root", str(REPO_ROOT)]) == 0


def test_print_extra_table_tolerates_int_none_and_empty_rows(capsys: pytest.CaptureFixture[str]) -> None:
    """Table printer must not crash on empty records or cells containing int/None."""
    _print_extra_table([])
    out_empty = capsys.readouterr().out
    assert "ID" in out_empty and "Status" in out_empty

    records = [
        {
            "id": "src_1",
            "type": None,
            "source": 123,
            "session": None,
            "planned_files": 4,
            "estimate_size": 2048,
            "licence": None,
            "status": "planned",
        },
        {
            "id": None,
            "type": "panoramax",
            "source": "phone",
            "session": "ext_pano",
            "size": 1024,
            "status": None,
        },
    ]
    _print_extra_table(records)
    out_records = capsys.readouterr().out
    assert "src_1" in out_records
    assert "123" in out_records
    assert "panoramax" in out_records


def test_panoramax_dry_run_reports_counts_and_paginates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Panoramax dry run reports seen/accepted/rejected licence counts and paginates for planning."""
    entry = _pano_entry()
    page_1 = {
        "features": [
            _item("sa_1", 8.50, "CC-BY-SA-4.0"),
            {
                **_item("sphere_1", 8.51, "CC0"),
                "properties": {"pers:interior_orientation": {"field_of_view": 360}},
            },
            _item("ok_1", 8.52, "Etalab-2.0"),
        ],
        "links": [{"rel": "next", "href": "https://api.test/page2"}],
    }
    page_2 = {
        "features": [
            _item("ok_2", 8.53, "CC0"),
            _item("dup_near_ok_2", 8.53001, "CC0"),
        ],
        "links": [],
    }

    requested_urls: list[str] = []

    def mock_json(url: str) -> dict[str, Any]:
        requested_urls.append(url)
        if "page2" in url:
            return page_2
        return page_1

    monkeypatch.setattr(panoramax, "_json", mock_json)

    _write_config(tmp_path, entry.model_dump())

    caplog.set_level(logging.INFO)
    records = fetch_extra(tmp_path, dry_run=True)

    assert len(records) == 1
    rec = records[0]

    assert any("page2" in u for u in requested_urls)
    assert rec["seen"] == 5
    assert rec["accepted"] == 2
    assert rec["planned_files"] == 2
    assert rec["rejected_by_licence"] == {"cc-by-sa-4.0": 1}
    assert rec["skipped_360"] == 1
    assert rec["skipped_thinned"] == 1
    assert "Panoramax pano: seen 5, accepted 2" in caplog.text


def test_panoramax_collection_fallback_projection_aspect_and_sd_preference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Panoramax obtains collection licences, rejects two panorama markers, and prefers SD assets."""
    collection = _item("collection", 8.50, "")
    collection["links"] = [{"rel": "collection", "href": "https://api.test/collection"}]
    equirectangular = _item("projection", 8.51)
    equirectangular["properties"]["projection"] = "equirectangular"
    wide = _item("wide", 8.52)
    wide["assets"] = {"sd": {"href": "https://images.test/wide.jpg", "width": 2000, "height": 1000}}
    both = _item("both", 8.53)
    both["assets"] = {
        "sd": {"href": "https://images.test/both-sd.jpg"},
        "hd": {"href": "https://images.test/both-hd.jpg"},
    }
    page = {"features": [collection, equirectangular, wide, both], "links": []}
    monkeypatch.setattr(
        panoramax, "_json", lambda url: {"license": "Etalab-2.0"} if "collection" in url else page
    )
    downloaded: list[str] = []
    monkeypatch.setattr(panoramax, "request_bytes", lambda url, **kwargs: downloaded.append(url) or _jpg())
    products = panoramax.products(_pano_entry())
    assert [product.title for product in products] == ["collection", "both"]
    assert downloaded == ["https://images.test/collection-sd.jpg", "https://images.test/both-sd.jpg"]


def test_commons_cap_filters_and_requests_thumbnail_width(monkeypatch: pytest.MonkeyPatch) -> None:
    """Commons stops at the accepted cap, rejects SVG and BY-SA, and sends iiurlwidth to the API."""
    entry = _commons_entry().model_copy(update={"max_per_entry": 2})
    pages = {
        "1": _photo_page("File:one", "image/jpeg", 2000, "CC0"),
        "2": _photo_page("File:svg", "image/svg+xml", 2000, "CC0"),
        "3": _photo_page("File:sa", "image/jpeg", 2000, "CC BY-SA 4.0"),
        "4": _photo_page("File:two", "image/png", 2000, "CC BY 4.0"),
        "5": _photo_page("File:three", "image/jpeg", 2000, "CC BY 4.0"),
    }
    requested: list[str] = []

    def api(url: str, **kwargs: object) -> FakeResponse:
        requested.append(url)
        return FakeResponse(json.dumps({"query": {"pages": pages}}).encode("utf-8"))

    monkeypatch.setattr(commons, "open_url_with_retry", api)
    monkeypatch.setattr(commons, "request_bytes", lambda *args, **kwargs: _jpg())
    products = commons.products(entry)
    assert [product.title for product in products] == ["File:one", "File:three"]
    api_query = [url for url in requested if "generator=categorymembers" in url]
    assert len(api_query) == 1 and "iiurlwidth=2560" in api_query[0]


def test_video_frame_count_timestamps_dedup_dark_and_corrupt(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Video extraction returns capped non-duplicate bright frames with concrete timestamps."""
    path = tmp_path / "frames.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 2, (96, 64))
    assert writer.isOpened()
    rows, columns = np.indices((64, 96))
    checker = (((rows // 8) + (columns // 8)) % 2 * 190 + 30).astype(np.uint8)
    first = cv2.cvtColor(checker, cv2.COLOR_GRAY2BGR)
    second = np.random.default_rng(20261104).integers(20, 235, size=first.shape, dtype=np.uint8)
    assert (videos._hash(first) ^ videos._hash(second)).bit_count() > 10
    for frame in (first, first, second, np.zeros_like(first)):
        writer.write(frame)
        writer.write(frame)
    writer.release()
    entry = _video_entry().model_copy(update={"max_frames_per_video": 2, "frame_interval_s": 1.0})
    page = {"title": "File:Drive.avi"}
    info = {"url": "https://upload.wikimedia.org/drive.avi", "extmetadata": {"Artist": {"value": "Driver"}}}
    frames = videos._extract(entry, path, page, info, "CC0")
    assert [frame.filename for frame in frames] == [
        "ext_video_Drive_0-seconds.jpg",
        "ext_video_Drive_2-seconds.jpg",
    ]
    caplog.set_level(logging.WARNING)
    corrupt = tmp_path / "corrupt.webm"
    corrupt.write_bytes(b"not a video")
    assert videos._extract(entry, corrupt, page, info, "CC0") == []
    assert "Could not decode Commons video" in caplog.text
