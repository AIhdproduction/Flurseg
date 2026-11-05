"""Part C source configuration smoke tests."""

from __future__ import annotations

import bz2
import gzip
import io
import json
import logging
import lzma
import tarfile
import urllib.parse
import zipfile
from pathlib import Path
from typing import Self
from urllib.error import HTTPError

import cv2
import numpy as np
import pytest
import yaml
from PIL import Image

from flurseg.cli import main
from flurseg.samples.extra import amsterdam, archive_dataset, internet_archive, panoramax, runner
from flurseg.samples.extra.common import FileProduct, request_bytes
from flurseg.samples.extra.config import (
    AmsterdamPanoramaEntry,
    ArchiveDatasetEntry,
    InternetArchiveEntry,
    PanoramaxEntry,
    load_extra_config,
)
from flurseg.samples.extra.licences import is_acceptable_licence, normalise_licence
from flurseg.samples.extra.runner import fetch_extra
from flurseg.samples.fetch import USER_AGENT

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_part_c_types_and_disabled_legacy_panoramax_entries_are_present() -> None:
    """The checked-in source catalogue documents all Part C source families."""
    config = load_extra_config(REPO_ROOT / "config" / "extra_sources.yaml")
    by_id = {entry.id: entry for entry in config.sources}
    assert {"amsterdam_panorama", "internet_archive", "archive_dataset"} <= {entry.type for entry in config.sources}
    assert by_id["ext_panoramax_paris_pleyel"].enabled is False
    assert by_id["ext_ia_dashcam_cc_by_4"].max_videos == 25
    assert by_id["ext_rdd2022"].max_download_gb == 3
    assert all(by_id[identifier].enabled is False for identifier in ("ext_rdd2022", "ext_zenodo_plitterstreet", "ext_zenodo_indian_dashcam"))
    assert all(by_id[identifier].enabled is False for identifier in ("ext_panoramax_lyon", "ext_panoramax_bordeaux", "ext_panoramax_nantes", "ext_panoramax_toulouse", "ext_panoramax_lille", "ext_panoramax_marseille", "ext_panoramax_brussels", "ext_panoramax_rennes"))
    assert all(by_id[f"ext_zenodo_{identifier}"].enabled is False for identifier in ("22301900", "3764792", "18327797", "21370736"))


def _pano(**changes: object) -> PanoramaxEntry:
    values = {"id": "pano_c", "type": "panoramax", "source": "phone", "session": "ext_pano_c",
              "bbox": [4.8, 52.3, 4.9, 52.4], "max_per_entry": 20, "licence": "CC0",
              "attribution": "Test", "session_yaml": {"area_type": "street"}, "mode": "collections_prefilter"}
    values.update(changes)
    return PanoramaxEntry.model_validate(values)


def _item(identifier: str, licence: str = "CC0", *, panorama_view: bool = False, lon: float = 4.85) -> dict[str, object]:
    assets: dict[str, object] = {"sd": {"href": f"https://images.test/{identifier}-sd.jpg"}}
    properties: dict[str, object] = {"license": licence, "field_of_view": 80}
    if panorama_view:
        assets["hd"] = {"href": f"https://images.test/{identifier}-hd.jpg", "width": 400, "height": 200}
        properties["field_of_view"] = 360
    return {"id": identifier, "geometry": {"coordinates": [lon, 52.35]}, "properties": properties, "assets": assets}


def _jpg(width: int = 120, height: int = 80, seed: int = 7) -> bytes:
    image = Image.fromarray(np.random.default_rng(seed).integers(10, 240, size=(height, width, 3), dtype=np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _pano_jpg() -> bytes:
    image = np.zeros((200, 400, 3), dtype=np.uint8)
    image[..., 0] = np.arange(400, dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_collection_prefilter_paginates_caps_and_keeps_only_allowed_licences(monkeypatch: pytest.MonkeyPatch) -> None:
    entry = _pano(max_collection_pages=1)
    calls: list[str] = []
    monkeypatch.setattr(panoramax, "_json", lambda url: calls.append(url) or {"collections": [
        {"id": "sa", "license": "CC-BY-SA-4.0"}, {"id": "ok", "license": "Etalab-2.0"},
        {"id": "zero", "license": "CC0-1.0"}], "links": [{"rel": "next", "href": "https://next.test"}]})
    assert panoramax._allowed_collections(entry, None) == ["ok", "zero"]
    assert calls == ["https://api.panoramax.xyz/api/collections?limit=1000"]


def test_collection_cache_is_reused_then_refreshed_after_ttl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    entry = _pano()
    calls: list[str] = []
    monkeypatch.setattr(panoramax, "_json", lambda url: calls.append(url) or {"collections": [{"id": "one", "license": "CC0"}]})
    assert panoramax._allowed_collections(entry, tmp_path / "work" / "cache") == ["one"]
    assert panoramax._allowed_collections(entry, tmp_path / "work" / "cache") == ["one"] and len(calls) == 1
    cache = panoramax._collection_cache_path(entry, tmp_path / "work" / "cache")
    cache.write_text(json.dumps({"created": 0, "ids": ["old"]}), encoding="utf-8")
    assert panoramax._allowed_collections(entry, tmp_path / "work" / "cache") == ["one"] and len(calls) == 2


def test_collection_ids_are_chunked_to_fifty(monkeypatch: pytest.MonkeyPatch) -> None:
    entry = _pano()
    monkeypatch.setattr(panoramax, "_allowed_collections", lambda *_: [f"c{index}" for index in range(101)])
    urls: list[str] = []
    monkeypatch.setattr(panoramax, "_json", lambda url: urls.append(url) or {"features": []})
    assert list(panoramax._items(entry)) == []
    assert [len(url.split("collections=")[1].split("&")[0].split(",")) for url in urls] == [50, 50, 1]


def test_panorama_filter_url_is_encoded_and_400_falls_back(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    entry = _pano(mode="search", flat_only=True)
    requested: list[str] = []
    error = HTTPError("https://api.test", 400, "bad", {}, io.BytesIO(b"filter rejected"))
    def fake_json(url: str) -> dict[str, object]:
        requested.append(url)
        if "filter=" in url:
            raise error
        return {"features": [_item("flat")]}
    monkeypatch.setattr(panoramax, "_json", fake_json)
    assert [item["id"] for item in panoramax._items(entry)] == ["flat"]
    assert "filter=field_of_view%3C360" in requested[0] and "retrying without filter" in caplog.text


def test_per_item_sharealike_is_rejected_inside_an_allowed_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    entry = _pano(mode="search")
    monkeypatch.setattr(panoramax, "_items", lambda *_args, **_kwargs: iter([_item("bad", "CC-BY-SA-4.0"), _item("good")]))
    accepted, stats = panoramax.accepted_items_with_stats(entry)
    assert [item["id"] for item, _ in accepted] == ["good"] and stats["rejected_by_licence"] == {"cc-by-sa-4.0": 1}


@pytest.mark.parametrize(
    ("title", "href", "accepted"),
    [
        ("License for this object (etalab-2.0)", "https://www.etalab.gouv.fr/licence-ouverte-open-licence/", True),
        ("License for this object (CC0-1.0)", "https://creativecommons.org/publicdomain/zero/1.0/", True),
        ("License for this object (CC-BY-4.0)", "https://creativecommons.org/licenses/by/4.0/", True),
        ("License for this object (CC-BY-SA-4.0)", "https://creativecommons.org/licenses/by-sa/4.0/", False),
        ("License for this object (CC-BY-NC-4.0)", "https://creativecommons.org/licenses/by-nc/4.0/", False),
        ("License for this object (CC-BY-ND-4.0)", "https://creativecommons.org/licenses/by-nd/4.0/", False),
        ("", "https://www.etalab.gouv.fr/licence-ouverte-open-licence/", True),
        ("", "https://creativecommons.org/licenses/by-sa/4.0/", False),
        ("", "", False),
    ],
)
def test_panoramax_license_link_titles_are_normalised_for_collections_and_items(
    title: str, href: str, accepted: bool
) -> None:
    links = [{"rel": "license", "title": title, "href": href}] if title or href else []
    collection = {"links": links}
    item = {"links": links, "properties": {}}
    value = panoramax._collection_licence(collection)
    assert is_acceptable_licence(value) is accepted
    assert is_acceptable_licence(panoramax._licence(item, {})) is accepted
    if title:
        assert normalise_licence(title).startswith(normalise_licence(value).split("-")[0])


def test_360_picture_uses_hd_and_emits_four_named_crops(monkeypatch: pytest.MonkeyPatch) -> None:
    entry = _pano(mode="search", views=4, crop_size=(80, 60))
    monkeypatch.setattr(panoramax, "accepted_items", lambda *_args, **_kwargs: [(_item("sphere", panorama_view=True), "CC0")])
    seen: list[str] = []
    monkeypatch.setattr(panoramax, "request_bytes", lambda url, **_kwargs: seen.append(url) or _pano_jpg())
    products = panoramax.products(entry)
    assert [product.filename for product in products] == [f"ext_pano_c_sphere_yaw{yaw}.jpg" for yaw in (0, 90, 180, 270)]
    assert seen == ["https://images.test/sphere-hd.jpg"]


def test_flat_picture_keeps_sd_bytes_and_crop_cap_counts_individually(monkeypatch: pytest.MonkeyPatch) -> None:
    entry = _pano(mode="search", max_per_entry=2, views=4, crop_size=(80, 60))
    flat, sphere = _item("flat"), _item("sphere", panorama_view=True)
    monkeypatch.setattr(panoramax, "accepted_items", lambda *_args, **_kwargs: [(flat, "CC0"), (sphere, "CC0")])
    monkeypatch.setattr(panoramax, "request_bytes", lambda url, **_kwargs: _jpg() if "flat" in url else _pano_jpg())
    products = panoramax.products(entry, limit=2)
    assert products[0].data == _jpg() or products[0].filename == "ext_pano_c_flat.jpg"
    assert len(products) == 2 and products[1].filename.endswith("yaw0.jpg")


def test_request_bytes_follows_redirects_with_the_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def read(self) -> bytes:
            return b"redirected"
        def __enter__(self) -> Self:
            return self
        def __exit__(self, *_args: object) -> None:
            return None
    headers: list[dict[str, str] | None] = []
    monkeypatch.setattr("flurseg.samples.extra.common.open_url_with_retry", lambda _url, **kwargs: headers.append(kwargs.get("headers")) or Response())
    assert request_bytes("https://redirect.test/image.jpg") == b"redirected"
    assert headers == [{"User-Agent": USER_AGENT}]


def test_360_spatial_thinning_respects_minimum_spacing(monkeypatch: pytest.MonkeyPatch) -> None:
    entry = _pano(mode="search", min_spacing_m=30)
    monkeypatch.setattr(panoramax, "_items", lambda *_args, **_kwargs: iter([_item("one", panorama_view=True), _item("two", panorama_view=True, lon=4.85001)]))
    accepted, stats = panoramax.accepted_items_with_stats(entry)
    assert len(accepted) == 1 and stats["skipped_thinned"] == 1


def _amsterdam(**changes: object) -> AmsterdamPanoramaEntry:
    values = {"id": "ams", "type": "amsterdam_panorama", "source": "phone", "session": "ext_ams",
              "near": [4.87, 52.34], "max_per_entry": 4, "licence": "CC BY 4.0", "attribution": "Amsterdam",
              "session_yaml": {"area_type": "urban"}, "crop_size": [80, 60]}
    values.update(changes)
    return AmsterdamPanoramaEntry.model_validate(values)


def test_amsterdam_parses_hal_records_and_ignores_missing_image_data(monkeypatch: pytest.MonkeyPatch) -> None:
    entry = _amsterdam()
    payload = {"_embedded": {"panoramas": [{"id": "one", "date": "2025-01-02", "geometry": {"coordinates": [4.87, 52.34]}}, {"id": "bad"}]}}
    monkeypatch.setattr(amsterdam, "_json", lambda _url: payload)
    monkeypatch.setattr(amsterdam, "request_bytes", lambda *_args, **_kwargs: _pano_jpg())
    products = amsterdam.products(entry)
    assert len(products) == 4 and all(product.attribution == "© Gemeente Amsterdam, CC BY 4.0" for product in products)


def test_amsterdam_crops_are_capped_and_spatially_thinned(monkeypatch: pytest.MonkeyPatch) -> None:
    entry = _amsterdam(max_per_entry=4, min_spacing_m=25)
    payload = {"panoramas": [{"id": "one", "date": "2025-01-02", "geometry": {"coordinates": [4.87, 52.34]}}, {"id": "near", "date": "2025-01-02", "geometry": {"coordinates": [4.87001, 52.34]}}]}
    monkeypatch.setattr(amsterdam, "_json", lambda _url: payload)
    monkeypatch.setattr(amsterdam, "request_bytes", lambda *_args, **_kwargs: _pano_jpg())
    assert len(amsterdam.products(entry, limit=4)) == 4


def test_amsterdam_uses_real_equirectangular_medium_record_and_follows_next_page(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = _amsterdam(max_per_entry=4)
    record = {
        "pano_id": "TMX7316010203-002918_pano_0013_000370",
        "timestamp": "2023-01-05T11:52:15.168330Z",
        "geometry": {"type": "Point", "coordinates": [4.8700064965015, 52.3396994058252, 47.99]},
        "_links": {
            "equirectangular_full": {"href": "https://images.test/panorama_8000.jpg"},
            "equirectangular_medium": {"href": "https://images.test/panorama_4000.jpg"},
            "equirectangular_small": {"href": "https://images.test/panorama_2000.jpg"},
        },
    }
    pages = [
        {"_embedded": {"panoramas": [{"pano_id": "missing"}]}, "_links": {"next": {"href": "https://api.test/next"}}},
        {"_embedded": {"panoramas": [record]}},
    ]
    urls: list[str] = []
    monkeypatch.setattr(amsterdam, "_json", lambda url: urls.append(url) or pages.pop(0))
    requested: list[str] = []
    monkeypatch.setattr(amsterdam, "request_bytes", lambda url, **_kwargs: requested.append(url) or _pano_jpg())
    products = amsterdam.products(entry, limit=4)
    assert [product.filename for product in products] == [
        f"ext_ams_TMX7316010203-002918_pano_0013_000370_yaw{yaw}.jpg" for yaw in (0, 90, 180, 270)
    ]
    assert requested == ["https://images.test/panorama_4000.jpg"]
    assert "page_size=1" in urls[0] and urls[1] == "https://api.test/next"


def test_amsterdam_attribution_is_written_to_lock_and_session_notes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    entry = _amsterdam().model_dump()
    config_path = tmp_path / "config" / "extra_sources.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump({"sources": [entry]}, sort_keys=False), encoding="utf-8")
    product = FileProduct(_jpg(), "ext_ams_one_yaw0.jpg", 1.0, "https://amsterdam.test/pano.jpg", "one", "CC BY 4.0", "© Gemeente Amsterdam, CC BY 4.0")
    monkeypatch.setattr("flurseg.samples.extra.runner.amsterdam_products", lambda *_args, **_kwargs: [product])
    assert fetch_extra(tmp_path)[0]["status"] == "downloaded"
    lock = yaml.safe_load((tmp_path / "config" / "extra_sources.lock.yaml").read_text(encoding="utf-8"))
    notes = yaml.safe_load((tmp_path / "input" / "phone" / "ext_ams" / "session.yaml").read_text(encoding="utf-8"))["notes"]
    assert lock["entries"][0]["attribution"] == "© Gemeente Amsterdam, CC BY 4.0" and "© Gemeente Amsterdam" in notes


def _ia(**changes: object) -> InternetArchiveEntry:
    values = {"id": "ia", "type": "internet_archive", "source": "phone", "session": "ext_ia", "search_query": "dashcam",
              "max_videos": 2, "max_frames_per_video": 2, "licence": "CC BY 4.0", "attribution": "IA", "session_yaml": {"area_type": "street"}}
    values.update(changes)
    return InternetArchiveEntry.model_validate(values)


@pytest.mark.parametrize(("url", "allowed"), [("https://creativecommons.org/licenses/by/4.0/", True), ("https://creativecommons.org/publicdomain/zero/1.0/", True), ("https://creativecommons.org/licenses/by-sa/4.0/", False), ("https://creativecommons.org/licenses/by-nd/4.0/", False), ("https://creativecommons.org/licenses/by-nc/4.0/", False), ("", False)])
def test_internet_archive_licence_mapping_enforces_the_allowlist(url: str, allowed: bool) -> None:
    from flurseg.samples.extra.licences import is_acceptable_licence
    assert is_acceptable_licence(internet_archive.licence_from_url(url)) is allowed


def test_internet_archive_creator_and_identifier_glob_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    entry = _ia(creator="Road Agency", identifier_glob="gov.*")
    urls: list[str] = []
    monkeypatch.setattr(internet_archive, "_json", lambda url: urls.append(url) or {"response": {"docs": [{"identifier": "gov.fhwa"}, {"identifier": "other"}]}})
    assert internet_archive._identifiers(entry) == ["gov.fhwa"] and "creator%3A%22Road+Agency%22" in urls[0]


def test_internet_archive_search_error_logs_and_retries_without_licenseurl_filter(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    entry = _ia(search_query='(dashcam OR "dash cam") AND mediatype:movies AND licenseurl:*creativecommons.org/licenses/by/4.0*')
    queries: list[str] = []
    urls: list[str] = []

    def search(url: str) -> dict[str, object]:
        urls.append(url)
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)["q"][0]
        queries.append(query)
        if "licenseurl:" in query:
            return {"error": "[BACKEND_ERROR] Invalid or no response from Elasticsearch"}
        return {"response": {"docs": [{"identifier": "z"}, {"identifier": "a"}]}}

    monkeypatch.setattr(internet_archive, "_json", search)
    assert internet_archive._identifiers(entry) == ["a", "z"]
    assert queries == [entry.search_query, '(dashcam OR "dash cam") AND mediatype:movies']
    parameters = urllib.parse.parse_qs(urllib.parse.urlsplit(urls[0]).query)
    assert parameters["rows"] == ["200"] and parameters["sort[]"] == ["identifier asc"]
    assert "BACKEND_ERROR" in caplog.text and "retrying without licenseurl filter" in caplog.text


def test_internet_archive_download_uses_part_then_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def read(self) -> bytes:
            return b"movie"
        def __enter__(self) -> Self:
            return self
        def __exit__(self, *_args: object) -> None:
            return None
    monkeypatch.setattr(internet_archive, "open_url_with_retry", lambda *_args, **_kwargs: Response())
    target = internet_archive._download("https://archive.test/movie.mp4", tmp_path / "work" / "cache" / "videos" / "movie.mp4")
    assert target.read_bytes() == b"movie" and not target.with_suffix(".mp4.part").exists() and not (tmp_path / "input").exists()


def test_internet_archive_respects_frame_and_video_caps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    entry = _ia(max_videos=1, max_frames_per_video=2, max_total_gb=0.001)
    monkeypatch.setattr(internet_archive, "_identifiers", lambda _entry: ["one", "two"])
    monkeypatch.setattr(internet_archive, "_json", lambda url: {"metadata": {"licenseurl": "https://creativecommons.org/licenses/by/4.0/", "title": url}, "files": [{"name": "x.mp4", "size": "10", "format": "h.264", "width": "800"}]})
    monkeypatch.setattr(internet_archive, "_download", lambda _url, target, *_args: target)
    monkeypatch.setattr(
        internet_archive,
        "extract_video_frames",
        lambda _path, **kwargs: [
            FileProduct(b"one", "one.jpg", 1.0, "url", "one", "CC BY 4.0", "IA"),
            FileProduct(b"two", "two.jpg", 1.0, "url", "one", "CC BY 4.0", "IA"),
        ],
    )
    assert len(internet_archive.products(entry, tmp_path / "work" / "cache" / "videos")) == 2


def test_internet_archive_extracts_capped_frames_from_a_synthetic_video(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 2, (80, 60))
    assert writer.isOpened()
    for value in (30, 30, 150, 210):
        texture = np.random.default_rng(value).integers(45, 220, size=(60, 80), dtype=np.uint8)
        frame = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)
        cv2.circle(frame, (value % 65 + 8, 30), 8, (255, 240, 50), -1)
        writer.write(frame)
    writer.release()
    entry = _ia(max_videos=1, max_frames_per_video=2, frame_interval_s=1.0)
    monkeypatch.setattr(internet_archive, "_identifiers", lambda _entry: ["video"])
    monkeypatch.setattr(internet_archive, "_json", lambda _url: {"metadata": {"licenseurl": "https://creativecommons.org/licenses/by/4.0/"}, "files": [{"name": "x.avi", "size": "1", "format": "h.264", "width": "800"}]})
    monkeypatch.setattr(internet_archive, "_download", lambda _url, _target, *_args: video)
    products = internet_archive.products(entry, tmp_path / "work" / "cache" / "videos")
    assert 1 <= len(products) <= 2 and all(product.filename.endswith("-seconds.jpg") for product in products)


def _archive(**changes: object) -> ArchiveDatasetEntry:
    values = {"id": "archive", "type": "archive_dataset", "source": "phone", "session": "ext_archive", "url": "https://datasets.test/data.zip",
              "expected_licence": "CC BY 4.0", "max_images": 10, "licence": "CC BY 4.0", "attribution": "Dataset", "session_yaml": {"area_type": "street"}}
    values.update(changes)
    return ArchiveDatasetEntry.model_validate(values)


def _zip(images: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in images.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def test_archive_dataset_rejects_licence_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(archive_dataset, "_resolved", lambda _entry: ([], "CC0"))
    with pytest.raises(ValueError, match="dataset licence mismatch"):
        archive_dataset.products(_archive(), tmp_path / "work" / "cache" / "datasets")


def test_archive_dataset_subset_groups_phash_and_safe_zip_members(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    photo = _jpg(96, 64)
    payload = _zip({"de/train/a.jpg": photo, "de/train/b.jpg": photo, "fr/train/c.jpg": _jpg(97, 64, seed=19), "../evil.jpg": _jpg(seed=23)})
    monkeypatch.setattr(
        archive_dataset,
        "request_bytes_with_content_length",
        lambda *_args, **_kwargs: (payload, None),
    )
    entry = _archive(subset=".*/train/.*", max_per_group=1)
    products = archive_dataset.products(entry, tmp_path / "work" / "cache" / "datasets")
    titles = {product.title for product in products}
    assert len(titles & {"de/train/a.jpg", "de/train/b.jpg"}) == 1 and "fr/train/c.jpg" in titles
    assert all(".." not in product.url for product in products)


def test_archive_dataset_allows_missing_subset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payload = _zip({"images/one.jpg": _jpg(96, 64)})
    monkeypatch.setattr(archive_dataset, "request_bytes_with_content_length", lambda *_args, **_kwargs: (payload, None))
    products = archive_dataset.products(_archive(subset=None), tmp_path / "work" / "cache" / "datasets")
    assert [product.title for product in products] == ["images/one.jpg"]


def test_archive_dataset_complete_cache_is_reused_but_partial_file_is_redownloaded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "cache" / "data.zip"
    target.parent.mkdir(parents=True)
    complete = _zip({"image.jpg": b"x" * 2048})
    target.write_bytes(complete)
    calls: list[str] = []
    fresh = _zip({"fresh.jpg": b"y" * 2048})
    monkeypatch.setattr(
        archive_dataset,
        "request_bytes_with_content_length",
        lambda url, **_kwargs: ((calls.append(url), fresh)[1], None),
    )
    assert archive_dataset._download("https://datasets.test/data.zip", target, len(complete), 3) == target and not calls
    partial_target = target.with_name("partial.zip")
    partial_target.with_suffix(".zip.part").write_bytes(b"partial")
    archive_dataset._download("https://datasets.test/partial.zip", partial_target, len(fresh), 3)
    assert partial_target.read_bytes() == fresh and len(calls) == 1


def test_archive_rejects_html_and_promotes_part_only_after_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    target = tmp_path / "cache" / "dataset.zip"
    target.parent.mkdir()
    payloads = [b"<html><body>download denied</body></html>", _zip({"image.jpg": b"x" * 2048})]
    monkeypatch.setattr(
        archive_dataset,
        "request_bytes_with_content_length",
        lambda *_args, **_kwargs: (payloads.pop(0), None),
    )
    with pytest.raises(archive_dataset.ArchiveDownloadError, match="invalid archive"):
        archive_dataset._download("https://datasets.test/dataset.zip", target, None, 3)
    assert not target.exists() and target.with_suffix(".zip.part").exists()
    assert archive_dataset._download("https://datasets.test/dataset.zip", target, None, 3) == target
    assert target.is_file() and not target.with_suffix(".zip.part").exists()
    assert "download denied" in caplog.text or "invalid archive" in caplog.text


def test_archive_validation_accepts_supported_zip_and_tar_compressions(tmp_path: Path) -> None:
    payload = np.random.default_rng(5).integers(0, 256, size=2048, dtype=np.uint8).tobytes()
    zip_payload = _zip({"images/sample.jpg": payload})
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w") as archive:
        info = tarfile.TarInfo("images/sample.jpg")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    tar_payload = tar_buffer.getvalue()
    archives = {
        "sample.zip": (zip_payload, "zip"),
        "sample.tar.gz": (gzip.compress(tar_payload), "gzip"),
        "sample.tar.bz2": (bz2.compress(tar_payload), "bzip2"),
        "sample.tar.xz": (lzma.compress(tar_payload), "xz"),
    }
    for filename, (data, expected_kind) in archives.items():
        path = tmp_path / filename
        path.write_bytes(data)
        assert archive_dataset._validate_archive(path) == expected_kind


def test_internet_archive_skips_response_larger_than_download_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Response:
        def __init__(self) -> None:
            self.headers = {"Content-Length": str(4 * 1024**3)}

        def read(self) -> bytes:
            raise AssertionError("oversized response must not be read")

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(internet_archive, "open_url_with_retry", lambda *_args, **_kwargs: Response())
    target = tmp_path / "cache" / "video.mp4"
    with pytest.raises(ValueError, match="max_download_gb"):
        internet_archive._download("https://archive.test/video.mp4", target, max_download_gb=3)
    assert not target.exists()


def test_empty_external_sources_log_a_reason(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    caplog.set_level(logging.WARNING)
    monkeypatch.setattr(panoramax, "accepted_items", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(amsterdam, "_json", lambda *_args, **_kwargs: {"results": []})
    monkeypatch.setattr(internet_archive, "_identifiers", lambda *_args, **_kwargs: [])
    assert panoramax.products(_pano(mode="search")) == []
    assert amsterdam.products(_amsterdam()) == []
    assert internet_archive.products(_ia(), tmp_path / "cache") == []
    assert "Panoramax pano_c yielded 0 items" in caplog.text
    assert "Amsterdam panorama ams yielded 0 items" in caplog.text
    assert "Internet Archive ia yielded 0 items" in caplog.text


def test_runner_continues_after_one_source_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    failed = _archive(id="bad_archive", session="ext_bad_archive").model_dump()
    successful = _archive(id="good_archive", session="ext_good_archive").model_dump()
    config_path = tmp_path / "config" / "extra_sources.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump({"sources": [failed, successful]}, sort_keys=False), encoding="utf-8")
    product = FileProduct(_jpg(), "good_archive.jpg", 1.0, "https://datasets.test/good.zip", "good", "CC BY 4.0", "Dataset")

    def source(entry: ArchiveDatasetEntry, *_args: object, **_kwargs: object) -> list[FileProduct]:
        if entry.id == "bad_archive":
            raise archive_dataset.ArchiveDownloadError("invalid archive cache")
        return [product]

    monkeypatch.setattr(runner, "archive_dataset_products", source)
    records = fetch_extra(tmp_path)
    assert [(record["id"], record["status"]) for record in records] == [
        ("bad_archive", "failed"),
        ("good_archive", "downloaded"),
    ]
    assert (tmp_path / "input" / "phone" / "ext_good_archive" / "good_archive.jpg").is_file()


def test_runner_records_download_limit_as_skipped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    entry = _archive(id="large_archive", session="ext_large_archive").model_dump()
    config_path = tmp_path / "config" / "extra_sources.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump({"sources": [entry]}, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "archive_dataset_products",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(archive_dataset.ArchiveDownloadSkipped("archive is too large")),
    )
    records = fetch_extra(tmp_path)
    assert records == [
        {
            "id": "large_archive",
            "type": "archive_dataset",
            "source": "phone",
            "session": "ext_large_archive",
            "status": "skipped",
            "licence": "CC BY 4.0",
            "error": "archive is too large",
        }
    ]


def test_cli_dry_run_lists_only_internet_archive_entries(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr("flurseg.samples.extra.runner.plan_internet_archive", lambda entry, limit=None: min(1, limit or entry.max_per_entry))
    assert main(["fetch-extra", "--dry-run", "--only-type", "internet_archive", "--root", str(REPO_ROOT)]) == 0
    output = capsys.readouterr().out
    assert "ext_ia_dashcam_cc_by_4" in output and "ext_ia_fhwa_cc0" in output and "ext_amsterdam_zuidas" not in output
