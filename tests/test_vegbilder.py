"""Contract tests for NLOD Vegbilder WFS planning and safe download provenance."""

from __future__ import annotations

import io
import random
import urllib.parse
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from flurseg.paths import RepoPaths
from flurseg.samples.extra import runner, vegbilder
from flurseg.samples.extra.common import FileProduct
from flurseg.samples.extra.config import VegbilderWfsEntry, load_extra_config
from flurseg.samples.extra.vegbilder import _selected, feature_url, hit_count, layer_name, products

REPO_ROOT = Path(__file__).resolve().parents[1]


def _entry() -> VegbilderWfsEntry:
    return VegbilderWfsEntry.model_validate({
        "id": "ext_vegbilder_test", "type": "vegbilder_wfs", "source": "phone", "session": "ext_vegbilder_test",
        "url": "https://example.test/ows", "layer_year": [2024], "bboxes": [{"name": "Test", "bbox": [10, 60, 11, 61]}],
        "per_bbox": 2, "min_spacing_m": 80, "max_per_entry": 2, "licence": "NLOD-2.0",
        "attribution": "© Statens vegvesen (Vegbilder), NLOD 2.0", "session_yaml": {"area_type": "street"},
    })


def _feature(identifier: str, lon: float, lat: float, url: str = "https://images.test/a.jpg?sig=secret") -> dict[str, object]:
    return {"id": identifier, "geometry": {"coordinates": [lon, lat]}, "properties": {"URL": url, "TIDSPUNKT": "2024-01-01", "RETNING": 90}}


def _jpeg(value: int = 100) -> bytes:
    stream = io.BytesIO()
    rows, cols = np.indices((64, 64))
    Image.fromarray(np.stack(((rows + value) % 255, (cols + value) % 255, (rows * cols + value) % 255), axis=-1).astype(np.uint8)).save(stream, format="JPEG")
    return stream.getvalue()


def test_wfs_request_and_hits_use_planar_year_layer(monkeypatch) -> None:
    entry = _entry()
    url = feature_url(entry, 2024, entry.bboxes[0], start_index=1000, count=2000)
    assert url == "https://example.test/ows?service=WFS&version=2.0.0&request=GetFeature&typeNames=vegbilder_1_0:Vegbilder_2024&outputFormat=application%2Fjson&srsName=EPSG:4326&bbox=10.0,60.0,11.0,61.0,EPSG:4326&count=1000&startIndex=1000"
    calls: list[str] = []
    monkeypatch.setattr(vegbilder, "request_bytes", lambda url, **kwargs: calls.append(url) or b'<FeatureCollection numberMatched="17"/>')
    assert hit_count(entry, 2024, entry.bboxes[0]) == 17
    assert "resultType=hits" in calls[0] and layer_name(2024) in calls[0] and "360" not in calls[0]


def test_selection_is_seeded_and_thins_nearby_features() -> None:
    entry = _entry()
    close = [_feature("one", 10, 60), _feature("two", 10.0001, 60), _feature("three", 10.004, 60)]
    first = _selected(entry, close, 3)
    second = _selected(entry, close, 3)
    assert [item["id"] for item in first] == [item["id"] for item in second]
    assert len(first) == 2 and {item["id"] for item in first} <= {"one", "two", "three"}


def test_download_retries_expired_url_and_strips_signed_query(monkeypatch) -> None:
    entry = _entry()
    feature = _feature("fid", 10, 60)
    calls: list[str] = []
    monkeypatch.setattr("flurseg.samples.extra.vegbilder._candidates", lambda *args: [feature])
    monkeypatch.setattr("flurseg.samples.extra.vegbilder._refresh_feature", lambda *args: _feature("fid", 10, 60, "https://images.test/b.jpg?st=token"))

    def download(url: str, **kwargs: object) -> bytes:
        calls.append(url)
        if len(calls) == 1:
            raise OSError("403")
        return _jpeg()

    monkeypatch.setattr("flurseg.samples.extra.vegbilder.request_bytes", download)
    result = products(entry)
    assert len(result) == 1 and len(calls) == 2
    assert result[0].url == "https://images.test/b.jpg" and result[0].metadata["heading"] == "90"


def test_real_config_has_bounded_nlod_vegbilder_plan() -> None:
    config = load_extra_config(REPO_ROOT / "config" / "extra_sources.yaml")
    entries = [entry for entry in config.sources if entry.type == "vegbilder_wfs"]
    assert len(entries) == 12 and all(entry.licence == "NLOD-2.0" for entry in entries)
    assert {entry.per_bbox for entry in entries} == {50} and {entry.max_per_entry for entry in entries} == {125}
    assert 1200 <= sum(entry.max_per_entry for entry in entries) <= 1800


def test_hits_response_parses_realistic_wfs_xml_and_unknown_is_zero(monkeypatch) -> None:
    entry = _entry()
    payload = b'<wfs:FeatureCollection xmlns:wfs="http://www.opengis.net/wfs/2.0" numberMatched="44279" numberReturned="0" next="https://example.test/next"/>'
    monkeypatch.setattr(vegbilder, "request_bytes", lambda *args, **kwargs: payload)
    assert hit_count(entry, 2024, entry.bboxes[0]) == 44279
    assert vegbilder._hit_count_from_xml(b'<wfs:FeatureCollection numberMatched="unknown"/>') == 0


def test_bbox_uses_lon_lat_epsg_suffix_without_urn_axis_order() -> None:
    raw = _entry().model_dump()
    raw["bboxes"] = [{"name": "Oslo", "bbox": [10.62, 59.88, 10.90, 59.98]}]
    entry = VegbilderWfsEntry.model_validate(raw)
    query = feature_url(entry, 2024, entry.bboxes[0])
    assert "bbox=10.62,59.88,10.9,59.98,EPSG:4326" in query
    assert "urn:ogc:def:crs:EPSG::4326" not in query


def test_offsets_are_spread_deterministic_and_request_count_is_bounded() -> None:
    entry = _entry().model_copy(update={"per_bbox": 40})
    first, page_size = vegbilder._sample_offsets(entry, 2024, entry.bboxes[0], 44_279)
    second, _ = vegbilder._sample_offsets(entry, 2024, entry.bboxes[0], 44_279)
    assert first == second and first[0] < 1_000 and first[-1] > 40_000
    assert page_size <= 5 and len(first) <= 100


def test_seeded_selection_changes_with_entry_seed() -> None:
    features = [_feature(f"id-{index}", 10 + index * 0.01, 60) for index in range(10)]
    first = _selected(_entry().model_copy(update={"per_bbox": 3}), features, 3)
    other = _selected(_entry().model_copy(update={"id": "ext_vegbilder_other", "per_bbox": 3}), features, 3)
    assert [item["id"] for item in first] != [item["id"] for item in other]


def test_spacing_keeps_features_two_hundred_metres_apart() -> None:
    entry = _entry()
    chosen = _selected(entry, [_feature("west", 10, 60), _feature("east", 10.0036, 60)], 2)
    assert {item["id"] for item in chosen} == {"west", "east"}


def test_spacing_thins_ten_metres_but_keeps_two_hundred_metres() -> None:
    entry = _entry()
    close = _selected(entry, [_feature("first", 10, 60), _feature("second", 10.0001, 60)], 2)
    far = _selected(entry, [_feature("third", 10, 60), _feature("fourth", 10.0036, 60)], 2)
    assert len(close) == 1 and len(far) == 2


def test_layer_name_uses_requested_year_and_never_uses_360_layer(monkeypatch) -> None:
    entry = _entry()
    urls: list[str] = []
    monkeypatch.setattr(vegbilder, "hit_count", lambda *args: 1)
    monkeypatch.setattr(vegbilder, "_sample_offsets", lambda *args: ([0], 1))
    monkeypatch.setattr(vegbilder, "_json", lambda url: urls.append(url) or {"features": []})
    assert vegbilder._candidates(entry, 2024, entry.bboxes[0]) == []
    assert layer_name(2024).endswith("Vegbilder_2024") and all("360" not in url for url in urls)


def test_image_request_follows_candidate_fetch_immediately(monkeypatch) -> None:
    entry = _entry()
    events: list[str] = []
    monkeypatch.setattr(vegbilder, "_candidates", lambda *args: events.append("feature") or [_feature("fid", 10, 60)])
    monkeypatch.setattr(vegbilder, "request_bytes", lambda *args, **kwargs: events.append("image") or _jpeg())
    assert len(products(entry)) == 1 and events[:2] == ["feature", "image"]


def test_second_expired_url_is_logged_and_other_feature_continues(monkeypatch, caplog) -> None:
    entry = _entry()
    first, second = _feature("bad", 10, 60), _feature("good", 10.01, 60, "https://images.test/good.jpg?sig=x")
    monkeypatch.setattr(vegbilder, "_candidates", lambda *args: [first, second])
    monkeypatch.setattr(vegbilder, "_refresh_feature", lambda *args: _feature("bad", 10, 60, "https://images.test/retry.jpg?sig=x"))
    attempts: list[str] = []

    def download(url: str, **kwargs: object) -> bytes:
        attempts.append(url)
        if "good" not in url:
            raise OSError("403")
        return _jpeg(180)

    monkeypatch.setattr(vegbilder, "request_bytes", download)
    assert [item.title for item in products(entry)] == ["good"]
    assert len(attempts) == 3 and "remained unavailable" in caplog.text


def test_dark_image_is_skipped(monkeypatch) -> None:
    entry = _entry()
    dark = io.BytesIO()
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(dark, format="JPEG")
    monkeypatch.setattr(vegbilder, "_candidates", lambda *args: [_feature("dark", 10, 60)])
    monkeypatch.setattr(vegbilder, "request_bytes", lambda *args, **kwargs: dark.getvalue())
    assert products(entry) == []


def test_near_duplicate_phash_is_skipped(monkeypatch) -> None:
    entry = _entry()
    monkeypatch.setattr(vegbilder, "_candidates", lambda *args: [_feature("one", 10, 60), _feature("two", 10.01, 60)])
    monkeypatch.setattr(vegbilder, "request_bytes", lambda *args, **kwargs: _jpeg(120))
    assert len(products(entry)) == 1


def test_max_per_entry_limits_kept_products(monkeypatch) -> None:
    entry = _entry().model_copy(update={"max_per_entry": 1})
    monkeypatch.setattr(vegbilder, "_candidates", lambda *args: [_feature("one", 10, 60), _feature("two", 10.01, 60, "https://images.test/b.jpg?sig=x")])
    monkeypatch.setattr(vegbilder, "request_bytes", lambda *args, **kwargs: _jpeg(90))
    assert len(products(entry)) == 1


def test_runner_writes_safe_vegbilder_file_and_is_idempotent(tmp_root: Path, monkeypatch) -> None:
    paths = RepoPaths.from_root(tmp_root)
    entry = _entry().model_dump()
    (paths.config_dir / "extra_sources.yaml").write_text(yaml.safe_dump({"sources": [entry]}), encoding="utf-8")
    product = FileProduct(_jpeg(), "ext_vegbilder_test_2024_fid.jpg", 1.0, "https://images.test/a.jpg", "fid", "NLOD-2.0", _entry().attribution)
    download_calls = 0
    received_existing_ids: list[set[str]] = []

    def source(*args, existing_feature_ids: set[str] | None = None, **kwargs) -> list[FileProduct]:
        nonlocal download_calls
        received_existing_ids.append(existing_feature_ids or set())
        if product.title in (existing_feature_ids or set()):
            return []
        download_calls += 1
        return [product]

    monkeypatch.setattr(runner, "vegbilder_products", source)
    first, second = runner.fetch_extra(paths), runner.fetch_extra(paths)
    session = paths.phone_input_dir / "ext_vegbilder_test"
    text = (paths.config_dir / "extra_sources.lock.yaml").read_text(encoding="utf-8") + (session / "session.yaml").read_text(encoding="utf-8") + (paths.reports_dir / "attribution" / "extra_sources.md").read_text(encoding="utf-8")
    assert first[0]["filename"] == "ext_vegbilder_test_2024_fid.jpg"
    assert second == []
    assert download_calls == 1
    assert received_existing_ids == [set(), {"fid"}]
    assert len(list(session.glob("*.jpg"))) == 1
    assert not list(session.glob("*_1.jpg"))
    assert all(token not in text for token in ("st=", "sig=", "X-Amz", "Signature")) and "NLOD-2.0" in text and "Statens vegvesen" in text


def test_small_wfs_pages_use_count_at_most_five_and_bounded_requests(monkeypatch) -> None:
    entry = _entry().model_copy(update={"per_bbox": 50})
    urls: list[str] = []
    monkeypatch.setattr(vegbilder, "hit_count", lambda *args: 100_000)
    monkeypatch.setattr(vegbilder, "_json", lambda url: urls.append(url) or {"features": []})
    assert vegbilder._candidates(entry, 2024, entry.bboxes[0]) == []
    counts = [int(urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)["count"][0]) for url in urls]
    assert len(urls) <= 100 and len(urls) >= 50 and max(counts) <= 5


def test_small_page_offsets_remain_deterministic_and_cover_result_range() -> None:
    entry = _entry().model_copy(update={"per_bbox": 50})
    offsets, size = vegbilder._sample_offsets(entry, 2024, entry.bboxes[0], 100_000)
    repeated, _ = vegbilder._sample_offsets(entry, 2024, entry.bboxes[0], 100_000)
    assert offsets == repeated and offsets[0] == 0 and offsets[-1] == 99_996
    assert size == 4 and len(offsets) == 75


def test_candidate_pool_has_six_times_per_bbox_before_spatial_thinning(monkeypatch) -> None:
    entry = _entry().model_copy(update={"per_bbox": 50})
    monkeypatch.setattr(vegbilder, "hit_count", lambda *args: 100_000)

    def page(url: str) -> dict[str, object]:
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        start, count = int(query["startIndex"][0]), int(query["count"][0])
        return {"features": [_feature(f"f-{start + index}", 10 + (start + index) * 0.00001, 60) for index in range(count)]}

    monkeypatch.setattr(vegbilder, "_json", page)
    assert len(vegbilder._candidates(entry, 2024, entry.bboxes[0])) >= entry.per_bbox * 6


def test_existing_feature_id_is_skipped_before_image_download(monkeypatch) -> None:
    entry = _entry()
    monkeypatch.setattr(vegbilder, "_candidates", lambda *args: [_feature("already", 10, 60), _feature("new", 10.01, 60, "https://images.test/new.jpg?sig=x")])
    downloads: list[str] = []
    monkeypatch.setattr(vegbilder, "request_bytes", lambda url, **kwargs: downloads.append(url) or _jpeg(170))
    kept = products(entry, existing_feature_ids={"already"})
    assert [item.title for item in kept] == ["new"] and downloads == ["https://images.test/new.jpg?sig=x"]


def test_runner_cap_counts_existing_and_only_requests_remaining_images(tmp_root: Path, monkeypatch) -> None:
    paths = RepoPaths.from_root(tmp_root)
    raw = _entry().model_dump()
    raw["max_per_entry"] = 3
    (paths.config_dir / "extra_sources.yaml").write_text(yaml.safe_dump({"sources": [raw]}), encoding="utf-8")
    session = paths.phone_input_dir / "ext_vegbilder_test"
    session.mkdir(parents=True)
    (session / "ext_vegbilder_test_2023_old-a.jpg").write_bytes(_jpeg(70))
    (session / "ext_vegbilder_test_2024_old-b.jpg").write_bytes(_jpeg(80))
    limits: list[int] = []

    def source(entry, *, limit, existing_feature_ids):
        limits.append(limit)
        return [FileProduct(_jpeg(180), "ext_vegbilder_test_2025_new.jpg", 1.0, "https://images.test/new", "new", "NLOD-2.0", entry.attribution)]

    monkeypatch.setattr(runner, "vegbilder_products", source)
    records = runner.fetch_extra(paths)
    assert limits == [1] and [record["filename"] for record in records] == ["ext_vegbilder_test_2025_new.jpg"]


def test_year_quotas_share_entry_cap_without_starving_latest_year(monkeypatch) -> None:
    raw = _entry().model_dump()
    raw.update({"layer_year": [2023, 2024, 2025], "max_per_entry": 5, "per_bbox": 1})
    entry = VegbilderWfsEntry.model_validate(raw)
    monkeypatch.setattr(vegbilder, "_candidates", lambda entry, year, bbox: [_feature(f"{year}-a", 10 + year / 1_000_000, 60, f"https://images.test/{year}.jpg?sig=x"), _feature(f"{year}-b", 10 + year / 1_000_000 + 0.01, 60, f"https://images.test/{year}b.jpg?sig=x")])
    counter = iter(range(100, 200))
    monkeypatch.setattr(vegbilder, "request_bytes", lambda *args, **kwargs: _jpeg(next(counter)))
    monkeypatch.setattr(vegbilder, "_hash", lambda frame: random.Random(next(counter)).getrandbits(64))
    years = [int(item.filename.split("_")[-2]) for item in products(entry)]
    assert years.count(2023) == 2 and years.count(2024) == 2 and years.count(2025) == 1


def test_small_pages_improve_spacing_yield_over_consecutive_page_strategy(monkeypatch) -> None:
    entry = _entry().model_copy(update={"per_bbox": 10})
    old_page = [_feature(f"old-{index}", 10 + index * 0.00005, 60) for index in range(25)]
    old_kept = _selected(entry, old_page, 60)
    monkeypatch.setattr(vegbilder, "hit_count", lambda *args: 100_000)

    def spread_pages(url: str) -> dict[str, object]:
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        start, count = int(query["startIndex"][0]), int(query["count"][0])
        return {"features": [_feature(f"new-{start + index}", 10 + start * 0.00001 + index * 0.00005, 60) for index in range(count)]}

    monkeypatch.setattr(vegbilder, "_json", spread_pages)
    new_kept = _selected(entry, vegbilder._candidates(entry, 2024, entry.bboxes[0]), 60)
    assert len(new_kept) > len(old_kept) * 10
