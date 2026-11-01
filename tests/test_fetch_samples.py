"""Comprehensive test suite for sample dataset fetching, filtering, and lock handling."""

from __future__ import annotations

import email.message
import hashlib
import io
import json
import types
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Self

import numpy as np
import pytest
import rasterio
import yaml
from PIL import Image
from rasterio.transform import from_origin
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

from flurseg.cli import main
from flurseg.ingest.session import load_session
from flurseg.samples import fetch as fetch_module
from flurseg.samples.fetch import fetch_samples


@pytest.fixture(autouse=True)
def guard_sockets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Block all real socket connections during tests."""
    def block_connect(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Real network access is forbidden during tests")

    monkeypatch.setattr("socket.socket.connect", block_connect)


class FakeResponse:
    """Mock HTTP response returned by monkeypatched urllib.request.urlopen."""

    def __init__(self, data: bytes, headers: dict[str, str] | None = None, status: int = 200) -> None:
        self._data = data
        self._stream = io.BytesIO(data)
        self.status = status
        self.headers = email.message.Message()
        if headers:
            for key, value in headers.items():
                self.headers[key] = value
        if "Content-Length" not in self.headers and data:
            self.headers["Content-Length"] = str(len(data))

    def read(self, amt: int | None = None) -> bytes:
        if amt is None:
            return self._stream.read()
        return self._stream.read(amt)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: types.TracebackType | None,
    ) -> None:
        pass


def _create_sample_jpeg_bytes(color: tuple[int, int, int] = (120, 140, 160)) -> bytes:
    """Create valid in-memory RGB JPEG bytes."""
    buffer = io.BytesIO()
    Image.new("RGB", (20, 20), color=color).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_wikimedia_retry_honours_retry_after_and_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Commons 429 waits for Retry-After and retries with the policy user agent."""
    current_time = [100.0]
    waits: list[float] = []
    calls: list[urllib.request.Request] = []

    def fake_monotonic() -> float:
        return current_time[0]

    def fake_sleep(seconds: float) -> None:
        waits.append(seconds)
        current_time[0] += seconds

    def fake_urlopen(request: urllib.request.Request, **kwargs: Any) -> FakeResponse:
        calls.append(request)
        if len(calls) == 1:
            headers = email.message.Message()
            headers["Retry-After"] = "7"
            raise urllib.error.HTTPError(request.full_url, 429, "Too Many Requests", headers, None)
        return FakeResponse(b"ok")

    monkeypatch.setattr(fetch_module, "_last_wikimedia_request_at", None)
    monkeypatch.setattr(fetch_module.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(fetch_module.time, "sleep", fake_sleep)
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with fetch_module.open_url_with_retry("https://commons.wikimedia.org/w/api.php?action=query") as response:
        assert response.read() == b"ok"

    assert waits == [7.0]
    assert len(calls) == 2
    assert calls[0].get_header("User-agent") == fetch_module.USER_AGENT


def test_wikimedia_throttle_covers_api_and_upload_hosts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Consecutive Commons API and upload requests remain one second apart."""
    current_time = [50.0]
    waits: list[float] = []

    def fake_sleep(seconds: float) -> None:
        waits.append(seconds)
        current_time[0] += seconds

    monkeypatch.setattr(fetch_module, "_last_wikimedia_request_at", None)
    monkeypatch.setattr(fetch_module.time, "monotonic", lambda: current_time[0])
    monkeypatch.setattr(fetch_module.time, "sleep", fake_sleep)
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: FakeResponse(b"ok"))

    fetch_module.open_url_with_retry("https://commons.wikimedia.org/w/api.php?action=query")
    fetch_module.open_url_with_retry("https://upload.wikimedia.org/example.jpg")

    assert waits == [1.0]


def _write_samples_yaml(tmp_root: Path, samples: list[dict[str, Any]]) -> Path:
    """Write config/samples.yaml in the test root."""
    config_file = tmp_root / "config" / "samples.yaml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {"samples": samples}
    config_file.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_file


def test_github_dir_resolution_with_stride_and_count(
    tmp_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GitHub directory resolution picks every stride-th JPG file up to count."""
    sample_jpeg = _create_sample_jpeg_bytes()

    github_items = [
        {"name": "readme.txt", "type": "file", "size": 100},
        {"name": "DJI_0001.JPG", "type": "file", "size": len(sample_jpeg), "download_url": "https://raw.github.com/a/1.JPG"},
        {"name": "DJI_0002.JPG", "type": "file", "size": len(sample_jpeg), "download_url": "https://raw.github.com/a/2.JPG"},
        {"name": "DJI_0003.JPG", "type": "file", "size": len(sample_jpeg), "download_url": "https://raw.github.com/a/3.JPG"},
        {"name": "DJI_0004.JPG", "type": "file", "size": len(sample_jpeg), "download_url": "https://raw.github.com/a/4.JPG"},
        {"name": "DJI_0005.JPG", "type": "file", "size": len(sample_jpeg), "download_url": "https://raw.github.com/a/5.JPG"},
        {"name": "DJI_0006.JPG", "type": "file", "size": len(sample_jpeg), "download_url": "https://raw.github.com/a/6.JPG"},
        {"name": "metadata.json", "type": "file", "size": 50},
    ]

    def fake_urlopen(request: urllib.request.Request | str, **kwargs: Any) -> FakeResponse:
        url = request.full_url if isinstance(request, urllib.request.Request) else str(request)
        if "api.github.com" in url:
            return FakeResponse(json.dumps(github_items).encode("utf-8"), headers={"Content-Type": "application/json"})
        if url.endswith(".JPG"):
            return FakeResponse(sample_jpeg, headers={"Content-Type": "image/jpeg"})
        raise ValueError(f"Unexpected URL: {url}")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    samples_config = [
        {
            "source": "drone",
            "session": "odm_test_session",
            "type": "github_dir",
            "repo": "OpenDroneMap/test_repo",
            "path": "images",
            "count": 3,
            "stride": 2,
            "license": "CC0-1.0",
            "attribution": "OpenDroneMap",
        }
    ]
    _write_samples_yaml(tmp_root, samples_config)

    records = fetch_samples(tmp_root)
    # Stride 2 from DJI_0001, 0002, 0003, 0004, 0005, 0006 -> 0001 (idx 0), 0003 (idx 2), 0005 (idx 4)
    assert len(records) == 3
    filenames = [r["filename"] for r in records]
    assert filenames == ["DJI_0001.jpg", "DJI_0003.jpg", "DJI_0005.jpg"]

    session_dir = tmp_root / "input" / "drone" / "odm_test_session"
    for filename in filenames:
        dest_file = session_dir / filename
        assert dest_file.is_file()
        assert dest_file.read_bytes() == sample_jpeg


def test_commons_license_and_width_filtering(
    tmp_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Commons category resolver filters by acceptable license and minimum width."""
    sample_jpeg = _create_sample_jpeg_bytes()

    pages = {
        "1": {
            "title": "File:Accepted_CC_BY_SA.jpg",
            "imageinfo": [{
                "url": "https://upload.wikimedia.org/Accepted_CC_BY_SA.jpg",
                "width": 2500,
                "height": 1800,
                "size": len(sample_jpeg),
                "descriptionurl": "https://commons.wikimedia.org/wiki/File:Accepted_CC_BY_SA.jpg",
                "extmetadata": {
                    "License": {"value": "cc-by-sa-4.0"},
                    "LicenseShortName": {"value": "CC BY-SA 4.0"},
                    "Artist": {"value": "<b>Author One</b>"},
                },
            }],
        },
        "2": {
            "title": "File:Accepted_CC0.jpg",
            "imageinfo": [{
                "url": "https://upload.wikimedia.org/Accepted_CC0.jpg",
                "width": 2200,
                "height": 1600,
                "size": len(sample_jpeg),
                "descriptionurl": "https://commons.wikimedia.org/wiki/File:Accepted_CC0.jpg",
                "extmetadata": {
                    "License": {"value": "cc0"},
                    "LicenseShortName": {"value": "CC0 1.0"},
                    "Artist": {"value": "Author Two"},
                },
            }],
        },
        "3": {
            "title": "File:Rejected_Too_Small.jpg",
            "imageinfo": [{
                "url": "https://upload.wikimedia.org/Rejected_Too_Small.jpg",
                "width": 1200,
                "height": 900,
                "size": 5000,
                "extmetadata": {
                    "License": {"value": "cc-by-sa-4.0"},
                    "LicenseShortName": {"value": "CC BY-SA 4.0"},
                },
            }],
        },
        "4": {
            "title": "File:Rejected_NonCommercial.jpg",
            "imageinfo": [{
                "url": "https://upload.wikimedia.org/Rejected_NonCommercial.jpg",
                "width": 3000,
                "height": 2000,
                "size": 5000,
                "extmetadata": {
                    "License": {"value": "cc-by-nc-4.0"},
                    "LicenseShortName": {"value": "CC BY-NC 4.0"},
                },
            }],
        },
        "5": {
            "title": "File:Accepted_PublicDomain.jpg",
            "imageinfo": [{
                "url": "https://upload.wikimedia.org/Accepted_PublicDomain.jpg",
                "width": 2100,
                "height": 1500,
                "size": len(sample_jpeg),
                "descriptionurl": "https://commons.wikimedia.org/wiki/File:Accepted_PublicDomain.jpg",
                "extmetadata": {
                    "License": {"value": "pd"},
                    "LicenseShortName": {"value": "Public domain"},
                    "Artist": {"value": "Author Three"},
                },
            }],
        },
    }

    commons_payload = {"query": {"pages": pages}}

    def fake_urlopen(request: urllib.request.Request | str, **kwargs: Any) -> FakeResponse:
        url = request.full_url if isinstance(request, urllib.request.Request) else str(request)
        if "commons.wikimedia.org/w/api.php" in url:
            return FakeResponse(json.dumps(commons_payload).encode("utf-8"), headers={"Content-Type": "application/json"})
        if "upload.wikimedia.org" in url:
            return FakeResponse(sample_jpeg, headers={"Content-Type": "image/jpeg"})
        raise ValueError(f"Unexpected URL: {url}")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    samples_config = [
        {
            "source": "phone",
            "session": "streets_test",
            "type": "commons_category",
            "category": "Streets in Lucerne",
            "count": 5,
            "min_width": 2000,
            "session_yaml": {"area_type": "street"},
        }
    ]
    _write_samples_yaml(tmp_root, samples_config)

    records = fetch_samples(tmp_root)
    downloaded_names = {r["filename"] for r in records}

    assert "Accepted_CC_BY_SA.jpg" in downloaded_names
    assert "Accepted_CC0.jpg" in downloaded_names
    assert "Accepted_PublicDomain.jpg" in downloaded_names
    assert "Rejected_Too_Small.jpg" not in downloaded_names
    assert "Rejected_NonCommercial.jpg" not in downloaded_names


def test_existing_files_never_overwritten(
    tmp_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-existing files are skipped and never overwritten."""
    target_dir = tmp_root / "input" / "drone" / "existing_session"
    target_dir.mkdir(parents=True, exist_ok=True)
    existing_file = target_dir / "target.jpg"
    original_bytes = b"ORIGINAL_EXISTING_CONTENT"
    existing_file.write_bytes(original_bytes)

    new_download_bytes = b"NEW_DOWNLOAD_CONTENT_THAT_MUST_NOT_OVERWRITE"

    def fake_urlopen(request: urllib.request.Request | str, **kwargs: Any) -> FakeResponse:
        return FakeResponse(new_download_bytes, headers={"Content-Type": "image/jpeg"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    samples_config = [
        {
            "source": "drone",
            "session": "existing_session",
            "filename": "target.jpg",
            "url": "https://example.com/target.jpg",
            "license": "CC0-1.0",
            "attribution": "Tester",
        }
    ]
    _write_samples_yaml(tmp_root, samples_config)

    records = fetch_samples(tmp_root)
    assert len(records) == 1
    assert records[0]["status"] == "skipped"
    assert existing_file.read_bytes() == original_bytes


def test_sha_mismatch_is_recorded_and_leaves_no_partial_file(
    tmp_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SHA mismatch triggers an error and leaves no temporary or partial files."""
    download_bytes = b"CORRUPTED_OR_TAMPERED_CONTENT"

    def fake_urlopen(request: urllib.request.Request | str, **kwargs: Any) -> FakeResponse:
        return FakeResponse(download_bytes, headers={"Content-Type": "image/jpeg"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    target_dir = tmp_root / "input" / "drone" / "mismatch_session"
    wrong_expected_sha = "0000000000000000000000000000000000000000000000000000000000000000"

    samples_config = [
        {
            "source": "drone",
            "session": "mismatch_session",
            "filename": "broken.jpg",
            "url": "https://example.com/broken.jpg",
            "sha256": wrong_expected_sha,
            "license": "CC0-1.0",
            "attribution": "Tester",
        }
    ]
    _write_samples_yaml(tmp_root, samples_config)

    records = fetch_samples(tmp_root)

    assert records[0]["status"] == "failed"
    assert "SHA-256 mismatch" in records[0]["error"]
    assert not (target_dir / "broken.jpg").exists()
    tmp_files = list(target_dir.glob("*.tmp")) + list(target_dir.glob(".*.tmp"))
    assert tmp_files == []


def test_ortho_window_extraction_preserves_transform_and_crs(tmp_root: Path) -> None:
    """Ortho window extraction reads a subregion and preserves the transform and CRS."""
    synthetic_cog = tmp_root / "synthetic_source.tif"
    source_width, source_height = 200, 180
    base_transform = from_origin(2600000.0, 1200000.0, 0.1, 0.1)
    pixel_data = np.full((3, source_height, source_width), 160, dtype=np.uint8)

    with rasterio.open(
        synthetic_cog,
        "w",
        driver="GTiff",
        width=source_width,
        height=source_height,
        count=3,
        dtype="uint8",
        crs="EPSG:2056",
        transform=base_transform,
        compress="jpeg",
        photometric="ycbcr",
    ) as dataset:
        dataset.write(pixel_data)

    window_definition = [20, 30, 60, 50]  # col_off, row_off, width, height
    samples_config = [
        {
            "source": "ortho",
            "session": "test_ortho_session",
            "filename": "windowed.tif",
            "url": str(synthetic_cog),
            "window": window_definition,
            "license": "swisstopo OGD, free use with attribution",
            "attribution": "© swisstopo",
            "session_yaml": {"area_type": "urban", "gsd_m": 0.1},
        }
    ]
    _write_samples_yaml(tmp_root, samples_config)

    records = fetch_samples(tmp_root)
    assert len(records) == 1
    dest_path = records[0]["path"]
    assert dest_path.is_file()

    with rasterio.open(dest_path) as extracted:
        assert extracted.width == 60
        assert extracted.height == 50
        assert extracted.count == 3
        assert extracted.crs is not None
        assert extracted.crs.to_epsg() == 2056
        expected_window = Window(20, 30, 60, 50)
        expected_transform = window_transform(expected_window, base_transform)
        assert extracted.transform == expected_transform
        assert extracted.profile.get("compress") == "deflate"
        assert tuple(channel.name for channel in extracted.colorinterp) == ("red", "green", "blue")


def test_fetch_samples_continues_after_file_failure_and_cli_returns_one(
    tmp_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A failing file is reported while the remaining files are still written."""
    good_bytes = _create_sample_jpeg_bytes()

    def fake_urlopen(request: urllib.request.Request | str, **kwargs: Any) -> FakeResponse:
        url = request.full_url if isinstance(request, urllib.request.Request) else str(request)
        method = request.get_method() if isinstance(request, urllib.request.Request) else "GET"
        if method == "HEAD":
            return FakeResponse(b"", headers={"Content-Length": str(len(good_bytes))})
        if url.endswith("broken.jpg"):
            raise urllib.error.URLError("intentional test failure")
        return FakeResponse(good_bytes, headers={"Content-Type": "image/jpeg"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    _write_samples_yaml(tmp_root, [
        {"source": "drone", "session": "partial", "filename": "first.jpg", "url": "https://example.com/first.jpg"},
        {"source": "drone", "session": "partial", "filename": "broken.jpg", "url": "https://example.com/broken.jpg"},
        {"source": "drone", "session": "partial", "filename": "last.jpg", "url": "https://example.com/last.jpg"},
    ])

    assert main(["fetch-samples", "--yes", "--root", str(tmp_root)]) == 1
    output = capsys.readouterr().out
    session_dir = tmp_root / "input" / "drone" / "partial"
    assert (session_dir / "first.jpg").is_file()
    assert (session_dir / "last.jpg").is_file()
    assert not (session_dir / "broken.jpg").exists()
    assert "failed: <urlopen error intentional test failure>" in output
    assert "Downloaded 2, skipped 0, failed 1" in output

    lock_data = yaml.safe_load((tmp_root / "config" / "samples.lock.yaml").read_text(encoding="utf-8"))
    assert [entry["filename"] for entry in lock_data["entries"]] == ["first.jpg", "last.jpg"]


def test_session_yaml_notes_contain_license(
    tmp_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generated session.yaml contains the license and attribution in notes."""
    sample_jpeg = _create_sample_jpeg_bytes()

    def fake_urlopen(request: urllib.request.Request | str, **kwargs: Any) -> FakeResponse:
        return FakeResponse(sample_jpeg, headers={"Content-Type": "image/jpeg"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    samples_config = [
        {
            "source": "drone",
            "session": "notes_test_session",
            "filename": "sample_drone.jpg",
            "url": "https://example.com/sample_drone.jpg",
            "license": "CC0-1.0",
            "attribution": "OpenDroneMap",
            "session_yaml": {"area_type": "residential", "test": True},
        }
    ]
    _write_samples_yaml(tmp_root, samples_config)

    fetch_samples(tmp_root)
    session_file = tmp_root / "input" / "drone" / "notes_test_session" / "session.yaml"
    assert session_file.is_file()

    metadata = load_session(session_file)
    assert metadata.test is True
    assert metadata.area_type == "residential"
    assert "licence" in metadata.notes.lower()
    assert "cc0-1.0" in metadata.notes.lower()
    assert "opendronemap" in metadata.notes.lower()


def test_lock_file_written(
    tmp_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """config/samples.lock.yaml is written with resolved sample checksums."""
    sample_jpeg = _create_sample_jpeg_bytes()

    def fake_urlopen(request: urllib.request.Request | str, **kwargs: Any) -> FakeResponse:
        return FakeResponse(sample_jpeg, headers={"Content-Type": "image/jpeg"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    samples_config = [
        {
            "source": "drone",
            "session": "lock_test_session",
            "filename": "locked_image.jpg",
            "url": "https://example.com/locked_image.jpg",
            "license": "CC0-1.0",
            "attribution": "OpenDroneMap",
        }
    ]
    _write_samples_yaml(tmp_root, samples_config)

    fetch_samples(tmp_root)
    lock_file = tmp_root / "config" / "samples.lock.yaml"
    assert lock_file.is_file()

    with lock_file.open("r", encoding="utf-8") as stream:
        lock_data = yaml.safe_load(stream)

    assert "entries" in lock_data
    entries = lock_data["entries"]
    assert len(entries) >= 1
    target = next((e for e in entries if e["filename"] == "locked_image.jpg"), None)
    assert target is not None
    assert target["source"] == "drone"
    assert target["session"] == "lock_test_session"
    assert target["sha256"] == hashlib.sha256(sample_jpeg).hexdigest()
    assert target["size"] == len(sample_jpeg)


def test_dry_run_downloads_nothing(
    tmp_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dry run lists files and sizes without creating files on disk."""
    sample_jpeg = _create_sample_jpeg_bytes()

    def fake_urlopen(request: urllib.request.Request | str, **kwargs: Any) -> FakeResponse:
        method = request.get_method() if isinstance(request, urllib.request.Request) else "GET"
        if method == "HEAD":
            return FakeResponse(b"", headers={"Content-Length": str(len(sample_jpeg))})
        return FakeResponse(sample_jpeg, headers={"Content-Type": "image/jpeg"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    samples_config = [
        {
            "source": "drone",
            "session": "dry_session",
            "filename": "dry_image.jpg",
            "url": "https://example.com/dry_image.jpg",
            "license": "CC0-1.0",
            "attribution": "OpenDroneMap",
        }
    ]
    _write_samples_yaml(tmp_root, samples_config)

    records = fetch_samples(tmp_root, dry_run=True)
    assert len(records) == 1
    assert records[0]["status"] == "dry_run"
    assert records[0]["size"] == len(sample_jpeg)

    # Ensure no input files or directories were created
    input_dir = tmp_root / "input"
    if input_dir.exists():
        assert list(input_dir.rglob("*.jpg")) == []

    # Ensure lock file was not created
    lock_file = tmp_root / "config" / "samples.lock.yaml"
    assert not lock_file.exists()


def test_only_filter_restricts_sources(
    tmp_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The only argument restricts fetching exclusively to the requested source."""
    sample_jpeg = _create_sample_jpeg_bytes()

    def fake_urlopen(request: urllib.request.Request | str, **kwargs: Any) -> FakeResponse:
        return FakeResponse(sample_jpeg, headers={"Content-Type": "image/jpeg"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    samples_config = [
        {
            "source": "drone",
            "session": "drone_session",
            "filename": "drone.jpg",
            "url": "https://example.com/drone.jpg",
            "license": "CC0-1.0",
            "attribution": "Drone",
        },
        {
            "source": "phone",
            "session": "phone_session",
            "filename": "phone.jpg",
            "url": "https://example.com/phone.jpg",
            "license": "CC-BY-4.0",
            "attribution": "Phone",
        },
    ]
    _write_samples_yaml(tmp_root, samples_config)

    records = fetch_samples(tmp_root, only="drone")
    assert len(records) == 1
    assert records[0]["source"] == "drone"

    assert (tmp_root / "input" / "drone" / "drone_session" / "drone.jpg").is_file()
    assert not (tmp_root / "input" / "phone").exists()
