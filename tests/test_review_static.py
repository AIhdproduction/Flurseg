from __future__ import annotations

import re
import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from flurseg.config import load_config
from flurseg.paths import RepoPaths
from flurseg.review.app import create_app


def _versioned_assets(html: str) -> tuple[str, set[str]]:
    """Return the asset version and HTML asset names from the index document."""
    matches = re.findall(r'(?:src|href)="/static/([^/"]+)/([^"?]+)', html)
    versions = {version for version, _ in matches}
    assert len(versions) == 1
    return versions.pop(), {asset for _, asset in matches}


def test_static_assets_are_offline_and_shortcuts_are_exposed(tmp_root) -> None:
    paths = RepoPaths.from_root(tmp_root)
    client = TestClient(create_app(paths, load_config(paths)))
    html = client.get("/").text
    assert "http://" not in html and "https://" not in html
    version, assets = _versioned_assets(html)
    for asset in assets:
        assert client.get(f"/static/{version}/{asset}").status_code == 200
    config = client.get("/api/config").json()
    keys = " ".join(item["keys"] for item in config["shortcuts"])
    for expected in (
        "0", "9", "-", "I", "[", "]", "E", "P", "F", "G", "A", "B", "Y", "N",
        "Shift+1", "Shift+6", "O", "Ctrl+Z", "Ctrl+Y", "Enter", "R", "S", "←", "→",
        "Home", "?", "F1", "Esc",
    ):
        assert expected in keys
    home_shortcut = next(item for item in config["shortcuts"] if item["keys"] == "Home")
    assert "fit" in home_shortcut["action"].lower()
    assert len(config["labeling_rules"]) == 5


def test_static_assets_and_module_imports_are_served_with_types(tmp_root) -> None:
    """Serve every local HTML asset and ES module with a browser-safe content type."""
    paths = RepoPaths.from_root(tmp_root)
    client = TestClient(create_app(paths, load_config(paths)))
    html = client.get("/").text
    version, assets = _versioned_assets(html)
    static_dir = Path(__file__).parents[1] / "flurseg" / "review" / "static"
    for module in static_dir.glob("*.js"):
        assets.add(module.name)
        assert 'from "/static/' not in module.read_text(encoding="utf-8")
    for asset in assets:
        response = client.get(f"/static/{version}/{asset}")
        assert response.status_code == 200
        assert "javascript" in response.headers["content-type"] or asset.endswith(".css")


def test_static_assets_are_versioned_and_entry_module_is_immutable(tmp_root) -> None:
    paths = RepoPaths.from_root(tmp_root)
    client = TestClient(create_app(paths, load_config(paths)))
    index_response = client.get("/")
    assert index_response.headers["cache-control"] == "no-cache"
    html = index_response.text
    version, _ = _versioned_assets(html)
    assert re.search(rf'<script type="module" src="/static/{re.escape(version)}/main\.js">', html)
    for asset in ("main.js", "render.js"):
        response = client.get(f"/static/{version}/{asset}")
        assert response.headers["cache-control"] == "public, max-age=31536000, immutable"
    assert client.get("/static/main.js").headers["cache-control"] == "no-cache"


def test_entry_module_version_changes_with_static_content(tmp_root, tmp_path, monkeypatch) -> None:
    source_dir = Path(__file__).parents[1] / "flurseg" / "review" / "static"
    static_dir = tmp_path / "static"
    shutil.copytree(source_dir, static_dir)
    monkeypatch.setattr("flurseg.review.app.STATIC_DIR", static_dir)
    paths = RepoPaths.from_root(tmp_root)
    first = TestClient(create_app(paths, load_config(paths))).get("/").text
    main_js = static_dir / "main.js"
    main_js.write_text(main_js.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    second = TestClient(create_app(paths, load_config(paths))).get("/").text
    first_version, _ = _versioned_assets(first)
    second_version, _ = _versioned_assets(second)
    assert first_version != second_version


def test_static_javascript_is_readable_and_offline(tmp_root) -> None:
    """Keep static modules readable and ensure no external URL can reach the UI."""
    static_dir = Path(__file__).parents[1] / "flurseg" / "review" / "static"
    for module in static_dir.glob("*.js"):
        source = module.read_text(encoding="utf-8")
        assert "http://" not in source and "https://" not in source
        assert all(len(line) <= 140 for line in source.splitlines())


def test_static_module_imports_are_relative() -> None:
    """Keep module imports within the versioned static asset directory."""
    static_dir = Path(__file__).parents[1] / "flurseg" / "review" / "static"
    for module in static_dir.glob("*.js"):
        imports = re.findall(r'\bfrom\s+["\']([^"\']+)["\']|\bimport\s+["\']([^"\']+)["\']', module.read_text(encoding="utf-8"))
        assert all((from_path or direct_path).startswith("./") for from_path, direct_path in imports)


def test_proposal_threshold_is_shared_by_rendering_and_acceptance() -> None:
    """Proposal display and acceptance import one threshold constant."""
    static_dir = Path(__file__).parents[1] / "flurseg" / "review" / "static"
    state = (static_dir / "state.js").read_text(encoding="utf-8")
    render = (static_dir / "render.js").read_text(encoding="utf-8")
    main = (static_dir / "main.js").read_text(encoding="utf-8")
    assert "export const PROPOSAL_THRESHOLD = 128" in state
    assert "PROPOSAL_THRESHOLD" in render and 'from "./state.js"' in render
    assert "acceptProposalPixels" in main and "PROPOSAL_THRESHOLD" in main


def test_stale_draft_banner_texts_are_present() -> None:
    """The review UI explains stale drafts and offers both recovery actions."""
    html = (Path(__file__).parents[1] / "flurseg" / "review" / "static" / "index.html").read_text(encoding="utf-8")
    assert "The automatic proposal was improved after your draft was saved." in html
    assert "Use new proposal" in html
    assert "Keep my draft" in html
    assert "mask-source" in html
