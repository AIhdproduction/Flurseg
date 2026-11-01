from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from flurseg.config import load_config
from flurseg.review.app import create_app
from flurseg.segment.fake_backend import FakeBackend
from tests.test_review_service import _record


def test_review_api_endpoints(tmp_root) -> None:
    from flurseg.paths import RepoPaths

    paths = RepoPaths.from_root(tmp_root)
    _record(paths, "review")
    _record(paths, "pool", status="auto_accepted")
    client = TestClient(create_app(paths, load_config(paths), FakeBackend))
    assert client.get("/api/config").status_code == 200
    assert client.get("/api/queue").json()[0]["id"] == "review"
    assert client.get("/api/stats").json()["open"] == 1
    meta = client.get("/api/images/review").json()
    assert meta["size"] == {"width": 300, "height": 300}
    assert client.get("/api/images/review/tiles/image/0/0/0").headers["content-type"] == "image/jpeg"
    assert client.get("/api/images/review/tiles/raw_mask/0/0/0").headers["content-type"] == "image/png"
    mask_response = client.get("/api/images/review/mask")
    assert len(mask_response.content) == 300 * 300
    draft = np.full(300 * 300, 2, dtype=np.uint8).tobytes()
    assert client.put("/api/images/review/mask", content=draft).status_code == 204
    assert client.get("/api/images/review/superpixels").status_code == 200
    assert client.get("/api/images/review/proposal-classes").status_code == 200
    similar = client.post("/api/images/review/find-similar", json={"class_id": 1, "boxes": [[0, 0, 20, 20]]})
    assert similar.status_code == 200 and len(similar.content) == 300 * 300
    assert similar.headers["x-window"] == "0,0,300,300"
    assert client.post("/api/pool/pool/to-review").status_code == 200
    assert client.post("/api/images/review/accept", json={"review_seconds": 1}).status_code == 200
    assert client.post("/api/images/pool/skip", json={}).status_code == 200
    assert client.post("/api/images/pool/reject", json={}).status_code == 200


def test_api_responses_are_not_cached_except_tiles(tmp_root) -> None:
    from flurseg.paths import RepoPaths

    paths = RepoPaths.from_root(tmp_root)
    _record(paths, "review")
    client = TestClient(create_app(paths, load_config(paths)))
    assert client.get("/api/config").headers["cache-control"] == "no-store"
    assert "cache-control" not in client.get("/api/images/review/tiles/image/0/0/0").headers


def test_find_similar_without_backend_is_unavailable(tmp_root) -> None:
    from flurseg.paths import RepoPaths

    paths = RepoPaths.from_root(tmp_root)
    _record(paths, "review")
    client = TestClient(create_app(paths, load_config(paths)))
    response = client.post("/api/images/review/find-similar", json={"class_id": 1, "boxes": [[0, 0, 20, 20]]})
    assert response.status_code == 503


def test_queue_filters_and_pool_listing(tmp_root) -> None:
    """Queue filters retain only matching manifest rows and pool is separate."""
    from flurseg.paths import RepoPaths

    paths = RepoPaths.from_root(tmp_root)
    _record(paths, "uncertain", uncertainty=0.9)
    _record(paths, "pool", status="auto_accepted")
    client = TestClient(create_app(paths, load_config(paths)))
    assert [item["id"] for item in client.get("/api/queue?source=drone&session=s1").json()] == ["uncertain"]
    assert client.get("/api/queue?reason=missing").json() == []
    assert [item["id"] for item in client.get("/api/pool").json()] == ["pool"]


def test_mask_put_validation_and_round_trip(tmp_root) -> None:
    """Draft mask writes reject invalid buffers and preserve valid bytes."""
    from flurseg.paths import RepoPaths

    paths = RepoPaths.from_root(tmp_root)
    _record(paths, "review")
    client = TestClient(create_app(paths, load_config(paths)))
    assert client.put("/api/images/review/mask", content=b"bad").status_code == 400
    invalid = bytes([99]) * (300 * 300)
    assert client.put("/api/images/review/mask", content=invalid).status_code == 400
    valid = bytes([3]) * (300 * 300)
    assert client.put("/api/images/review/mask", content=valid).status_code == 204
    assert client.get("/api/images/review/mask").content == valid


def test_tile_edge_and_action_transitions(tmp_root) -> None:
    """Tile edges remain valid while review actions expose their expected statuses."""
    from flurseg.paths import RepoPaths

    paths = RepoPaths.from_root(tmp_root)
    _record(paths, "review")
    _record(paths, "other")
    client = TestClient(create_app(paths, load_config(paths)))
    assert client.get("/api/images/review/tiles/image/0/2/1").status_code == 404
    assert client.post("/api/images/review/reject", json={"review_seconds": 3}).json()["status"] == "rejected"
    assert client.post("/api/images/other/skip", json={"review_seconds": 2}).json()["status"] == "needs_review"


def test_config_contains_classes_rules_and_shortcuts(tmp_root) -> None:
    """Config exposes the editor's class palette, rules, and key help."""
    from flurseg.paths import RepoPaths

    paths = RepoPaths.from_root(tmp_root)
    config = TestClient(create_app(paths, load_config(paths))).get("/api/config").json()
    assert config["classes"] and len(config["labeling_rules"]) == 5 and config["shortcuts"]
    assert config["superpixel_levels"] == ["fine", "coarse"]


def test_queue_is_sorted_by_uncertainty(tmp_root) -> None:
    """Higher-uncertainty queue entries are listed first."""
    from flurseg.paths import RepoPaths

    paths = RepoPaths.from_root(tmp_root)
    _record(paths, "low", uncertainty=0.1)
    _record(paths, "high", uncertainty=0.9)
    queue = TestClient(create_app(paths, load_config(paths))).get("/api/queue").json()
    assert [item["id"] for item in queue] == ["high", "low"]


def test_stats_reports_open_image_count(tmp_root) -> None:
    """Stats count review-queue records as open."""
    from flurseg.paths import RepoPaths

    paths = RepoPaths.from_root(tmp_root)
    _record(paths, "one")
    assert TestClient(create_app(paths, load_config(paths))).get("/api/stats").json()["open"] == 1


def test_image_meta_reports_dimensions(tmp_root) -> None:
    """Image metadata reports the original full-resolution dimensions."""
    from flurseg.paths import RepoPaths

    paths = RepoPaths.from_root(tmp_root)
    _record(paths, "image")
    meta = TestClient(create_app(paths, load_config(paths))).get("/api/images/image").json()
    assert meta["width"] == 300 and meta["height"] == 300


def test_superpixels_and_proposal_classes_are_binary_resources(tmp_root) -> None:
    """Both editor array resources are served as binary content."""
    from flurseg.paths import RepoPaths

    paths = RepoPaths.from_root(tmp_root)
    _record(paths, "image")
    client = TestClient(create_app(paths, load_config(paths)))
    assert len(client.get("/api/images/image/superpixels").content) == 300 * 300 * 4
    assert len(client.get("/api/images/image/proposal-classes").content) == 300 * 300


def test_accept_updates_manifest_and_pool(tmp_root) -> None:
    """Accept writes the reviewed image into the train pool."""
    from flurseg.paths import RepoPaths

    paths = RepoPaths.from_root(tmp_root)
    _record(paths, "image")
    client = TestClient(create_app(paths, load_config(paths)))
    response = client.post("/api/images/image/accept", json={"review_seconds": 2})
    assert response.json()["status"] == "reviewed"
    assert (paths.pool_images_dir / "image.jpg").is_file()


def test_pool_to_review_moves_a_pool_image_back_to_queue(tmp_root) -> None:
    """Pool recall returns an automatic acceptance to the review queue."""
    from flurseg.paths import RepoPaths

    paths = RepoPaths.from_root(tmp_root)
    _record(paths, "image", status="auto_accepted")
    client = TestClient(create_app(paths, load_config(paths)))
    assert client.post("/api/pool/image/to-review").json()["status"] == "needs_review"


def test_superpixels_endpoint_levels_and_errors(tmp_root) -> None:
    """Superpixel endpoint supports fine/coarse levels with 404 for missing coarse."""
    from flurseg.artifacts import ArtifactStore
    from flurseg.paths import RepoPaths

    paths = RepoPaths.from_root(tmp_root)
    _record(paths, "image")
    client = TestClient(create_app(paths, load_config(paths)))

    # Missing coarse superpixels returns 404 with detail message
    missing_response = client.get("/api/images/image/superpixels?level=coarse")
    assert missing_response.status_code == 404
    assert missing_response.json()["detail"] == "coarse superpixels not available"

    # Default level is fine, returns correct byte length (300*300*4 bytes)
    fine_response = client.get("/api/images/image/superpixels")
    assert fine_response.status_code == 200
    assert len(fine_response.content) == 300 * 300 * 4

    # Explicit fine level
    fine_explicit = client.get("/api/images/image/superpixels?level=fine")
    assert fine_explicit.status_code == 200
    assert len(fine_explicit.content) == 300 * 300 * 4

    # Invalid level returns 400
    invalid_level = client.get("/api/images/image/superpixels?level=invalid")
    assert invalid_level.status_code == 400

    # Write coarse superpixels and verify 200 and byte length
    store = ArtifactStore(paths.work_dir / "image")
    coarse_labels = np.arange(300 * 300, dtype=np.int32).reshape((300, 300))
    store.write_superpixels_coarse(coarse_labels)

    coarse_response = client.get("/api/images/image/superpixels?level=coarse")
    assert coarse_response.status_code == 200
    assert len(coarse_response.content) == 300 * 300 * 4
    restored = np.frombuffer(coarse_response.content, dtype="<u4").reshape((300, 300))
    assert np.array_equal(restored, coarse_labels.astype(np.uint32))
