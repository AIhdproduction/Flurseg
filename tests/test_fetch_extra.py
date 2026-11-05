"""Tests for config-driven additional open imagery retrieval."""

from __future__ import annotations

import io
import logging
import math
import socket
import urllib.parse
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
import yaml
from PIL import Image
from rasterio.transform import from_origin
from rasterio.warp import transform

from flurseg.cli import main
from flurseg.ingest.session import load_session
from flurseg.samples.extra import arcgis, runner, stac_cog, wms, xyz
from flurseg.samples.extra.common import FileProduct, ImageProduct, is_low_information_image
from flurseg.samples.extra.config import ExtraSourcesConfig, WmsEntry, XyzEntry, load_extra_config
from flurseg.samples.extra.licences import is_acceptable_licence, normalise_licence
from flurseg.samples.extra.runner import fetch_extra

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def guard_sockets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent accidental real network access in every extra-fetch test."""
    blocked = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("network"))
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


def _jpeg(size: tuple[int, int] = (2048, 2048), color: tuple[int, int, int] = (40, 80, 120)) -> bytes:
    buffer = io.BytesIO()
    width, height = size
    rows, columns = np.indices((height, width))
    image = np.stack(
        (
            (color[0] + columns * 17 + rows * 3) % 256,
            (color[1] + columns * 5 + rows * 13) % 256,
            (color[2] + columns * 11 + rows * 7) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(image).save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


def _small_white_png() -> bytes:
    """Create a valid padded 854-byte blank tile like an outside-coverage response."""
    buffer = io.BytesIO()
    Image.new("RGB", (256, 256), "white").save(buffer, format="PNG")
    return buffer.getvalue().ljust(854, b" ")


def _base_entry(entry_type: str = "wms") -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": "test_source",
        "type": entry_type,
        "source": "ortho",
        "session": "ext_test_source",
        "licence": "CC BY 4.0",
        "attribution": "Test provider",
        "session_yaml": {"area_type": "urban"},
    }
    if entry_type == "wms":
        entry.update(
            {
                "url": "https://example.test/wms",
                "layer": "rgb",
                "crs": "EPSG:3857",
                "gsd_m": 0.2,
                "windows": [[8.5, 47.3]],
            }
        )
    return entry


def _write_config(root: Path, entries: list[dict[str, Any]]) -> None:
    path = root / "config" / "extra_sources.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"sources": entries}, sort_keys=False), encoding="utf-8")


def _product(color: int = 90) -> ImageProduct:
    return ImageProduct(
        data=np.full((2048, 2048, 3), color, dtype=np.uint8),
        transform=from_origin(0, 100, 0.2, 0.2),
        crs="EPSG:3857",
        gsd_m=0.2,
        url="https://example.test/image",
        title="Test image",
        licence="CC BY 4.0",
        attribution="Test provider",
    )


def _local_cog(path: Path) -> None:
    rows, columns = np.indices((2050, 2050))
    data = np.stack(
        ((columns * 3 + rows) % 256, (columns + rows * 5) % 256, (columns * 7 + rows * 2) % 256)
    ).astype(np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2050,
        height=2050,
        count=3,
        dtype="uint8",
        crs="EPSG:3857",
        transform=from_origin(0, 205, 0.1, 0.1),
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dataset:
        dataset.write(data)


def test_config_validation_and_real_config(tmp_path: Path) -> None:
    """The discriminated config validates real and representative source entries."""
    valid = {"sources": [_base_entry()]}
    assert ExtraSourcesConfig.model_validate(valid).sources[0].type == "wms"
    with pytest.raises(ValueError, match="union_tag_invalid"):
        ExtraSourcesConfig.model_validate({"sources": [_base_entry("missing")]})
    duplicate = _base_entry()
    with pytest.raises(ValueError, match="duplicate"):
        ExtraSourcesConfig.model_validate({"sources": [_base_entry(), duplicate]})
    invalid_area = _base_entry()
    invalid_area["session_yaml"] = {"area_type": "beach"}
    with pytest.raises(ValueError, match="area_type"):
        ExtraSourcesConfig.model_validate({"sources": [invalid_area]})
    missing_licence = _base_entry()
    missing_licence.pop("licence")
    with pytest.raises(ValueError, match="licence"):
        ExtraSourcesConfig.model_validate({"sources": [missing_licence]})
    _write_config(tmp_path, [_base_entry()])
    assert load_extra_config(tmp_path / "config" / "extra_sources.yaml").sources
    assert load_extra_config(REPO_ROOT / "config" / "extra_sources.yaml").sources


@pytest.mark.parametrize(
    ("value", "accepted", "expected_normalised"),
    [
        ("CC-BY-4.0", True, "cc-by-4.0"),
        ("CC BY 3.0", True, "cc-by-3.0"),
        ("CC BY-SA 4.0", False, "cc-by-sa-4.0"),
        ("cc-by-sa-4.0", False, "cc-by-sa-4.0"),
        ("Public domain", True, "public-domain"),
        ("CC0", True, "cc0"),
        ("etalab-2.0", True, "etalab-2.0"),
        ("dl-de/zero-2.0", True, "dl-de/zero-2.0"),
        ("dl-de/by-2.0", True, "dl-de/by-2.0"),
        ("CC-BY-SA-2.0", False, "cc-by-sa-2.0"),
        ("cc_by_sa", False, "cc-by-sa"),
        ("CC BY ShareAlike", False, "cc-by-sharealike"),
        ("swisstopo OGD, free use with attribution", True, "swisstopo-ogd"),
        ("CC BY-SA 3.0", False, "cc-by-sa-3.0"),
        ("CC-BY-SA-2.5", False, "cc-by-sa-2.5"),
        ("ODbL-1.0", False, "odbl-1.0"),
        ("CC-BY-NC-4.0", False, "cc-by-nc-4.0"),
        ("CC BY-ND 4.0", False, "cc-by-nd-4.0"),
        ("NLOD", True, "nlod-2.0"),
        ("NLOD 2.0", True, "nlod-2.0"),
        ("NLOD-2.0", True, "nlod-2.0"),
        ("Norwegian Licence for Open Government Data (NLOD) 2.0", True, "nlod-2.0"),
        ("Norsk lisens for offentlige data", True, "nlod-2.0"),
        ("nlod-1.0", False, "nlod-1.0"),
        ("OGL v3.0", False, "ogl-v3.0"),
        ("", False, ""),
        ("unknown", False, "unknown"),
    ],
)
def test_licence_allowlist_normalisation(value: str, accepted: bool, expected_normalised: str) -> None:
    """Only the documented compatible licences pass normalisation and policy checks."""
    assert is_acceptable_licence(value) is accepted
    assert normalise_licence(value) == expected_normalised


def test_wms_url_axis_stitch_and_session_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """WMS requests use native CRS bounds, axis order, stitched output, and actual GSD."""
    entry = WmsEntry.model_validate(
        {**_base_entry(), "crs": "EPSG:4326", "axis_order": "ne", "max_request_px": 1024, "gsd_m": 0.0001}
    )
    requests: list[str] = []
    monkeypatch.setattr(
        wms, "request_bytes", lambda url, **kwargs: requests.append(url) or _jpeg((1024, 1024))
    )
    product = wms.get_wms_window(entry, 8.5, 47.3)
    assert product is not None and product.data.shape == (2048, 2048, 3)
    assert len(requests) == 4
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(requests[0]).query)
    assert query["layers"] == ["rgb"] and query["crs"] == ["EPSG:4326"]
    bbox = [float(value) for value in query["bbox"][0].split(",")]
    assert bbox[0] < bbox[2] and bbox[1] < bbox[3]
    _write_config(tmp_path, [_base_entry()])
    monkeypatch.setattr("flurseg.samples.extra.runner._products", lambda entry, *args: [_product()])
    records = fetch_extra(tmp_path)
    session = load_session(tmp_path / "input" / "ortho" / "ext_test_source" / "session.yaml")
    assert records[0]["status"] == "downloaded" and session.gsd_m == 0.2


def test_xyz_stitching_gsd_and_empty_tile(monkeypatch: pytest.MonkeyPatch) -> None:
    """XYZ tile coverage is stitched, georeferenced, and blank tiny tiles are rejected."""
    entry = XyzEntry.model_validate(
        {
            **_base_entry("xyz"),
            "url": "https://tiles.test/{z}/{x}/{y}.jpg",
            "zoom": 18,
            "windows": [[8.5, 47.3]],
        }
    )
    calls: list[str] = []
    monkeypatch.setattr(xyz, "request_bytes", lambda url, **kwargs: calls.append(url) or _jpeg((256, 256)))
    product = xyz.get_xyz_window(entry, 8.5, 47.3)
    assert product is not None and product.crs == "EPSG:3857" and 0 < product.gsd_m < 1
    assert calls and product.data.shape == (2048, 2048, 3)
    monkeypatch.setattr(xyz, "request_bytes", lambda *args, **kwargs: _small_white_png())
    assert xyz.get_xyz_window(entry, 8.5, 47.3) is None


def test_arcgis_export_parameters_and_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """ArcGIS export requests include projected bbox and image parameters."""
    entry = arcgis.ArcgisEntry.model_validate(
        {
            **_base_entry("arcgis_export"),
            "url": "https://arc.test/export",
            "gsd_m": 0.2,
            "windows": [[8.5, 47.3]],
        }
    )
    seen: list[str] = []
    monkeypatch.setattr(arcgis, "request_bytes", lambda url, **kwargs: seen.append(url) or _jpeg())
    product = arcgis.get_arcgis_window(entry, 8.5, 47.3)
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(seen[0]).query)
    assert product is not None and query["size"] == ["2048,2048"] and query["f"] == ["image"]


def test_stac_static_ignores_non_intersecting_items(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Static STAC processing ignores outside items and reads a local COG window."""
    cog = tmp_path / "cog.tif"
    _local_cog(cog)
    entry = stac_cog.StacStaticEntry.model_validate(
        {
            **_base_entry("stac_static"),
            "url": "https://stac.test/collection.json",
            "bboxes": [[0, 0, 1, 1]],
            "target_gsd_m": 0.1,
        }
    )
    collection = {"links": [{"rel": "item", "href": "inside.json"}, {"rel": "item", "href": "outside.json"}]}
    inside = {
        "id": "inside",
        "bbox": [0, 0, 1, 1],
        "properties": {"license": "CC BY 4.0"},
        "assets": {"visual": {"href": str(cog)}},
    }
    outside = {"id": "outside", "bbox": [10, 10, 11, 11], "assets": {"visual": {"href": str(cog)}}}
    monkeypatch.setattr(
        stac_cog,
        "_json",
        lambda url, *args, **kwargs: (
            collection if url.endswith("collection.json") else (inside if "inside" in url else outside)
        ),
    )
    products = stac_cog.static_products(entry)
    assert len(products) == 1 and products[0].data.shape == (2048, 2048, 3)


def test_stac_search_filters_licence_gsd_and_builds_attribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """STAC search skips incompatible items and records the OAM provider attribution."""
    cog = tmp_path / "cog.tif"
    _local_cog(cog)
    entry = stac_cog.StacSearchEntry.model_validate(
        {
            **_base_entry("stac_search"),
            "url": "https://stac.test/search",
            "bboxes": [[0, 0, 1, 1]],
            "max_gsd_m": 0.2,
            "windows_per_item": 1,
        }
    )
    response = {
        "features": [
            {
                "id": "bad",
                "properties": {"license": "CC-BY-SA-4.0", "gsd": 0.1},
                "assets": {"visual": {"href": str(cog)}},
            },
            {
                "id": "good",
                "properties": {"license": "CC-BY-4.0", "gsd": 0.1, "providers": [{"name": "Survey"}]},
                "assets": {"visual": {"href": str(cog)}},
            },
        ]
    }
    monkeypatch.setattr(stac_cog, "_json", lambda *args, **kwargs: response)
    products = stac_cog.search_products(entry)
    assert len(products) == 1 and products[0].attribution == "© Survey via OpenAerialMap, CC BY 4.0"


def test_idempotency_lock_attribution_and_suffix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Identical products are skipped, changed content is suffixed, and records are written."""
    _write_config(tmp_path, [_base_entry()])
    current = [_product(90)]
    monkeypatch.setattr("flurseg.samples.extra.runner._products", lambda entry, *args: current)
    assert fetch_extra(tmp_path)[0]["status"] == "downloaded"
    assert fetch_extra(tmp_path)[0]["status"] == "skipped"
    current[:] = [_product(100)]
    changed = fetch_extra(tmp_path)[0]
    assert changed["filename"] == "ext_test_source_0_1.tif"
    lock = yaml.safe_load((tmp_path / "config" / "extra_sources.lock.yaml").read_text(encoding="utf-8"))
    report = (tmp_path / "reports" / "attribution" / "extra_sources.md").read_text(encoding="utf-8")
    assert lock["entries"][0]["licence"] == "CC BY 4.0" and "Test provider" in report


def test_dry_run_and_cli_write_no_input(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Dry runs create no input data and expose the fetch-extra parser options."""
    _write_config(tmp_path, [_base_entry()])
    records = fetch_extra(tmp_path, dry_run=True)
    assert records[0]["planned_files"] == 1 and not (tmp_path / "input").exists()
    assert main(["fetch-extra", "--dry-run", "--only-type", "wms", "--root", str(tmp_path)]) == 0
    assert "test_source" in capsys.readouterr().out


def test_max_per_entry_caps_writes_and_dry_run_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The --max-per-entry CLI parameter caps planned files in dry run and written files in real run."""
    entries = [
        {
            **_base_entry("wms"),
            "id": "src_wms",
            "session": "ext_src_wms",
            "count": 5,
            "max_per_entry": 5,
            "windows": [[8.5 + 0.01 * i, 47.3] for i in range(5)],
        },
        {
            **_base_entry("xyz"),
            "id": "src_xyz",
            "session": "ext_src_xyz",
            "url": "https://tiles.test/{z}/{x}/{y}.jpg",
            "zoom": 18,
            "count": 5,
            "max_per_entry": 5,
            "windows": [[8.5 + 0.01 * i, 47.3] for i in range(5)],
        },
        {
            **_base_entry("stac_search"),
            "id": "src_stac_search",
            "session": "ext_src_stac_search",
            "url": "https://stac.test/search",
            "bboxes": [[0, 0, 1, 1]],
            "max_items": 5,
            "windows_per_item": 2,
            "max_per_entry": 5,
        },
        {
            "id": "src_commons",
            "type": "commons_categories",
            "source": "phone",
            "session": "ext_src_commons",
            "categories": ["Test Category"],
            "depth": 1,
            "min_width": 1600,
            "max_per_entry": 5,
            "licence": "CC BY 4.0",
            "attribution": "Test provider",
            "session_yaml": {"area_type": "street"},
        },
    ]
    _write_config(tmp_path, entries)

    monkeypatch.setattr(
        runner, "search_products", lambda entry, *args, **kwargs: [_product() for _ in range(5)]
    )
    monkeypatch.setattr(
        runner,
        "plan_commons_categories",
        lambda entry, limit=None: (
            min(entry.max_per_entry, limit) if limit is not None else entry.max_per_entry
        ),
    )
    monkeypatch.setattr(
        runner,
        "commons_category_products",
        lambda entry, limit=None: [
            FileProduct(
                _jpeg((800, 600)),
                f"ext_src_commons_{i}.jpg",
                1.0,
                "https://test",
                "title",
                "CC0",
                "Attribution",
            )
            for i in range(min(entry.max_per_entry, limit) if limit is not None else entry.max_per_entry)
        ],
    )
    monkeypatch.setattr(runner, "get_wms_window", lambda entry, lon, lat, *args, **kwargs: _product())
    monkeypatch.setattr(runner, "get_xyz_window", lambda entry, lon, lat, *args, **kwargs: _product())

    plan = fetch_extra(tmp_path, dry_run=True, max_per_entry=2)
    assert len(plan) == 4
    for record in plan:
        assert record["planned_files"] == 2

    results = fetch_extra(tmp_path, max_per_entry=2)
    counts: dict[str, int] = {}
    for record in results:
        if record.get("status") == "downloaded":
            counts[record["id"]] = counts.get(record["id"], 0) + 1

    assert counts == {
        "src_wms": 2,
        "src_xyz": 2,
        "src_stac_search": 2,
        "src_commons": 2,
    }


def test_low_information_window_rejection_and_jitter_retry(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Low-information images (uniform water, low std) are rejected and jitter retry yields valid data."""
    rng = np.random.default_rng(12345)
    water_image = np.full((2048, 2048, 3), (30, 80, 160), dtype=np.uint8)
    low_std_image = np.clip(128 + rng.normal(0, 2, (2048, 2048, 3)), 0, 255).astype(np.uint8)
    textured_image = rng.integers(20, 230, size=(2048, 2048, 3), dtype=np.uint8)

    assert is_low_information_image(water_image) is True
    assert is_low_information_image(low_std_image) is True
    assert is_low_information_image(textured_image) is False

    entry = WmsEntry.model_validate(
        {
            **_base_entry("wms"),
            "windows": [[8.5, 47.3]],
        }
    )

    def _encode_jpeg(arr: np.ndarray) -> bytes:
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    call_index = 0

    def mock_request(url: str, **kwargs: Any) -> bytes:
        nonlocal call_index
        call_index += 1
        if call_index <= 1:
            return _encode_jpeg(water_image)
        return _encode_jpeg(textured_image)

    monkeypatch.setattr(wms, "request_bytes", mock_request)

    caplog.set_level(logging.INFO)
    product = wms.get_wms_window(entry, 8.5, 47.3)

    assert product is not None
    assert is_low_information_image(product.data) is False
    assert "skipped_low_information" in caplog.text


def test_wms_axis_order_transform_and_real_runner_gsd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WMS axis ordering, stitched georeferencing, and runner session GSD are numeric."""
    entry_data = {
        **_base_entry(),
        "crs": "EPSG:4326",
        "axis_order": "ne",
        "gsd_m": 0.0001,
        "max_request_px": 2048,
    }
    entry = WmsEntry.model_validate(entry_data)
    requested: list[str] = []
    monkeypatch.setattr(wms, "request_bytes", lambda url, **kwargs: requested.append(url) or _jpeg())
    product = wms.get_wms_window(entry, 8.5, 47.3)
    assert product is not None
    bbox = urllib.parse.parse_qs(urllib.parse.urlsplit(requested[0]).query)["bbox"][0]
    assert bbox == "47.1976000000,8.3976000000,47.4024000000,8.6024000000"
    assert product.transform.a == pytest.approx(0.0001)
    assert product.transform.e == pytest.approx(-0.0001)
    assert product.transform.c == pytest.approx(8.3976)
    assert product.transform.f == pytest.approx(47.4024)

    _write_config(tmp_path, [entry_data])
    records = fetch_extra(tmp_path)
    session = load_session(tmp_path / "input" / "ortho" / "ext_test_source" / "session.yaml")
    assert records[0]["status"] == "downloaded"
    assert session.gsd_m == pytest.approx(0.0001)


def test_xyz_range_crop_ground_gsd_and_transform(monkeypatch: pytest.MonkeyPatch) -> None:
    """XYZ uses exact tile bounds, crop origin, Web Mercator transform, and ground GSD."""
    lon, lat, zoom = 6.16, 49.6, 19
    entry = XyzEntry.model_validate(
        {
            **_base_entry("xyz"),
            "url": "https://tiles.test/{z}/{x}/{y}.png",
            "zoom": zoom,
            "windows": [[lon, lat]],
        }
    )
    min_x, min_y, max_x, max_y, left, top = xyz.tile_range(lon, lat, zoom)
    calls: list[tuple[int, int]] = []

    def tile(url: str, **kwargs: Any) -> bytes:
        parts = url.removesuffix(".png").split("/")
        tile_x, tile_y = int(parts[-2]), int(parts[-1])
        calls.append((tile_x, tile_y))
        rows, columns = np.indices((256, 256))
        image = np.stack(
            ((columns + tile_x * 256) % 256, (rows + tile_y * 256) % 256, (columns + rows) % 256), axis=-1
        ).astype(np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="PNG")
        return buffer.getvalue() + b"x" * 1024

    monkeypatch.setattr(xyz, "request_bytes", tile)
    product = xyz.get_xyz_window(entry, lon, lat)
    assert product is not None
    assert calls == [
        (tile_x, tile_y) for tile_y in range(min_y, max_y + 1) for tile_x in range(min_x, max_x + 1)
    ]
    assert tuple(product.data[0, 0]) == (left % 256, top % 256, (left + top) % 256)
    resolution = xyz.WEB_MERCATOR_PIXEL / 2**zoom
    assert product.gsd_m == pytest.approx(resolution * math.cos(math.radians(lat)), abs=1e-9)
    assert product.transform.a == pytest.approx(resolution)
    assert product.transform.c == pytest.approx(-xyz.WEB_MERCATOR_HALF + left * resolution)
    assert product.transform.f == pytest.approx(xyz.WEB_MERCATOR_HALF - top * resolution)


@pytest.mark.parametrize("colour", [(255, 255, 255), (0, 0, 0)])
def test_wms_large_blank_tiles_are_skipped(
    monkeypatch: pytest.MonkeyPatch, colour: tuple[int, int, int]
) -> None:
    """Large white and black WMS coverage responses are rejected independently of byte size."""
    entry = WmsEntry.model_validate(_base_entry())
    payload = io.BytesIO()
    Image.new("RGB", (2048, 2048), colour).save(payload, format="JPEG")
    monkeypatch.setattr(wms, "request_bytes", lambda *args, **kwargs: payload.getvalue())
    assert wms.get_wms_window(entry, 8.5, 47.3) is None


def test_arcgis_parameters_and_geotiff_transform(monkeypatch: pytest.MonkeyPatch) -> None:
    """ArcGIS export sends spatial parameters and returns the matching GeoTIFF grid."""
    entry = arcgis.ArcgisEntry.model_validate(
        {
            **_base_entry("arcgis_export"),
            "url": "https://arc.test/export",
            "crs": "EPSG:3857",
            "gsd_m": 0.2,
            "windows": [[8.5, 47.3]],
        }
    )
    urls: list[str] = []
    monkeypatch.setattr(arcgis, "request_bytes", lambda url, **kwargs: urls.append(url) or _jpeg())
    product = arcgis.get_arcgis_window(entry, 8.5, 47.3)
    assert product is not None
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(urls[0]).query)
    assert query["bboxSR"] == ["3857"] and query["imageSR"] == ["3857"]
    assert query["size"] == ["2048,2048"] and query["format"] == ["jpg"] and query["f"] == ["image"]
    x, y = transform("EPSG:4326", "EPSG:3857", [8.5], [47.3])
    half = 2048 * 0.2 / 2
    assert product.crs == "EPSG:3857"
    assert product.transform.c == pytest.approx(x[0] - half)
    assert product.transform.f == pytest.approx(y[0] + half)
    assert product.transform.a == pytest.approx(0.2)


def test_stac_output_resolution_and_search_filters(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """COG output is resampled to target GSD and STAC search filters licence and GSD metadata."""
    cog = tmp_path / "fine.tif"
    rows, columns = np.indices((4096, 4096))
    data = np.stack(((columns + rows) % 256, (columns * 3) % 256, (rows * 5) % 256)).astype(np.uint8)
    with rasterio.open(
        cog,
        "w",
        driver="GTiff",
        width=4096,
        height=4096,
        count=3,
        dtype="uint8",
        crs="EPSG:3857",
        transform=from_origin(0, 205, 0.05, 0.05),
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dataset:
        dataset.write(data)
    static_entry = stac_cog.StacStaticEntry.model_validate(
        {
            **_base_entry("stac_static"),
            "url": "https://stac.test/collection.json",
            "bboxes": [[0, 0, 1, 1]],
            "target_gsd_m": 0.1,
        }
    )
    collection = {"links": [{"rel": "item", "href": "item.json"}]}
    item = {
        "id": "fine",
        "bbox": [0, 0, 1, 1],
        "properties": {"license": "CC BY 4.0"},
        "assets": {"visual": {"href": str(cog)}},
    }
    monkeypatch.setattr(
        stac_cog,
        "_json",
        lambda url, *args, **kwargs: collection if url.endswith("collection.json") else item,
    )
    product = stac_cog.static_products(static_entry)[0]
    assert product.transform.a == pytest.approx(0.1) and abs(product.transform.e) == pytest.approx(0.1)

    search_entry = stac_cog.StacSearchEntry.model_validate(
        {
            **_base_entry("stac_search"),
            "url": "https://stac.test/search",
            "bboxes": [[0, 0, 1, 1]],
            "max_gsd_m": 0.2,
            "windows_per_item": 1,
        }
    )
    response = {
        "features": [
            {
                "id": "nc",
                "properties": {"license": "CC-BY-NC-4.0", "gsd": 0.1},
                "assets": {"visual": {"href": str(cog)}},
            },
            {"id": "missing", "properties": {"gsd": 0.1}, "assets": {"visual": {"href": str(cog)}}},
            {
                "id": "coarse",
                "properties": {"license": "CC-BY-4.0", "gsd": 0.3},
                "assets": {"visual": {"href": str(cog)}},
            },
            {
                "id": "good",
                "properties": {"license": "CC-BY-4.0", "gsd": 0.1},
                "assets": {"visual": {"href": str(cog)}},
            },
        ]
    }
    monkeypatch.setattr(stac_cog, "_json", lambda *args, **kwargs: response)
    products = stac_cog.search_products(search_entry)
    assert [item.title for item in products] == ["good"]


def test_attribution_report_and_cli_filters(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Attribution retains each product licence and CLI forwards every extra-source filter."""
    alpha = {
        **_base_entry(),
        "id": "alpha",
        "session": "ext_alpha",
        "licence": "CC0",
        "attribution": "Alpha provider",
    }
    beta = {
        **_base_entry(),
        "id": "beta",
        "session": "ext_beta",
        "licence": "Etalab Licence Ouverte 2.0",
        "attribution": "Beta provider",
    }
    _write_config(tmp_path, [alpha, beta])
    monkeypatch.setattr(runner, "get_wms_window", lambda entry, lon, lat: _product())
    records = fetch_extra(tmp_path)
    report = (tmp_path / "reports" / "attribution" / "extra_sources.md").read_text(encoding="utf-8")
    assert {record["id"] for record in records} == {"alpha", "beta"}
    assert "Licence: CC0" in report and "Alpha provider" in report
    assert "Licence: Etalab Licence Ouverte 2.0" in report and "Beta provider" in report

    calls: list[dict[str, object]] = []

    def fake_fetch(*args: object, **kwargs: object) -> list[dict[str, object]]:
        calls.append(kwargs)
        return [
            {
                "id": "alpha",
                "type": "wms",
                "source": "ortho",
                "session": "ext_alpha",
                "planned_files": 1,
                "status": "dry_run",
            }
        ]

    monkeypatch.setattr("flurseg.samples.extra.runner.fetch_extra", fake_fetch)
    assert (
        main(
            [
                "fetch-extra",
                "--dry-run",
                "--only-id",
                "alpha",
                "--only-source",
                "ortho",
                "--max-per-entry",
                "1",
                "-y",
                "--root",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert calls == [
        {
            "dry_run": True,
            "only_types": None,
            "only_ids": {"alpha"},
            "only_source": "ortho",
            "max_per_entry": 1,
        }
    ]


def test_real_config_types_and_panoramax_cli_filter(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The checked-in config covers every supported type and the CLI shows only Panoramax entries."""
    config = load_extra_config(REPO_ROOT / "config" / "extra_sources.yaml")
    types = {entry.type for entry in config.sources}
    assert {
        "panoramax",
        "commons_categories",
        "commons_video_frames",
        "wms",
        "xyz",
        "arcgis_export",
        "stac_static",
        "stac_search",
    } <= types
    monkeypatch.setattr(
        "flurseg.samples.extra.runner.plan_panoramax",
        lambda entry, limit=None: min(entry.max_per_entry, limit or entry.max_per_entry),
    )
    assert main(["fetch-extra", "--dry-run", "--only-type", "panoramax", "--root", str(REPO_ROOT)]) == 0
    output = capsys.readouterr().out
    panorama_ids = [entry.id for entry in config.sources if entry.type == "panoramax"]
    non_panorama_ids = [entry.id for entry in config.sources if entry.type != "panoramax"]
    assert all(identifier in output for identifier in panorama_ids)
    assert all(identifier not in output for identifier in non_panorama_ids)
