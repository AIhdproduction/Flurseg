"""Training-data card aggregation remains safe and deterministic."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from flurseg.cli import main
from flurseg.data_card import collect, render_markdown, write_data_card
from flurseg.manifest.db import Database
from flurseg.manifest.repo import ImageRecord, ManifestRepository
from flurseg.paths import RepoPaths

REPO_ROOT = Path(__file__).resolve().parents[1]


def _record(repository: ManifestRepository, identifier: str, content_hash: str, source: str, status: str = "auto_accepted") -> None:
    repository.insert(ImageRecord(id=identifier, hash=content_hash, source=source, session="s", gsd_method="session", original_path="hidden.jpg", image_path="hidden.jpg", work_dir="hidden", width=1, height=1, status=status))


def test_data_card_counts_locked_and_own_images_without_paths(tmp_root: Path) -> None:
    paths = RepoPaths.from_root(tmp_root)
    (paths.config_dir / "extra_sources.lock.yaml").write_text(yaml.safe_dump({"entries": [{"sha256": "locked", "id": "ext_vegbilder_oslo", "licence": "NLOD-2.0", "attribution": "© Statens vegvesen", "url": "https://example.test/a?sig=secret"}]}), encoding="utf-8")
    with Database(paths.database) as connection:
        repository = ManifestRepository(connection)
        _record(repository, "locked-id", "locked", "phone")
        _record(repository, "own-id", "own", "drone")
        _record(repository, "rejected", "locked-rejected", "phone", "rejected")
    markdown = render_markdown(collect(paths))
    assert "Statens vegvesen Vegbilder" in markdown and "| 1 |" in markdown
    assert "- drone: 1 images" in markdown and "sig=secret" not in markdown and "hidden.jpg" not in markdown
    assert "not published" in markdown and "Built with DINOv3" in markdown and "AGPL-3.0" in markdown


def test_data_card_only_used_excludes_licences_and_writes_json(tmp_root: Path) -> None:
    paths = RepoPaths.from_root(tmp_root)
    (paths.config_dir / "samples.lock.yaml").write_text(yaml.safe_dump({"entries": [{"sha256": "cc", "id": "commons", "licence": "CC BY 4.0", "attribution": "Author"}]}), encoding="utf-8")
    with Database(paths.database) as connection:
        _record(ManifestRepository(connection), "image-id", "cc", "phone")
    paths.dataset_dir.mkdir(parents=True)
    (paths.dataset_dir / "dataset.json").write_text(json.dumps({"image_ids_per_split": {"train": ["image-id"]}, "excluded_licence": ["CC BY 4.0"]}), encoding="utf-8")
    output = write_data_card(paths, Path("reports/card.json"), only_used=True, output_format="json")
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["sources"] == [] and output == paths.root / "reports" / "card.json"


def test_data_card_groups_counts_by_source_and_licence(tmp_root: Path) -> None:
    paths = RepoPaths.from_root(tmp_root)
    (paths.config_dir / "samples.lock.yaml").write_text(yaml.safe_dump({"entries": [{"sha256": "a", "id": "commons", "licence": "CC BY 4.0", "attribution": "Author"}, {"sha256": "b", "id": "ext_vegbilder_oslo", "licence": "NLOD-2.0", "attribution": "Vegvesen"}]}), encoding="utf-8")
    with Database(paths.database) as connection:
        repo = ManifestRepository(connection)
        _record(repo, "a1", "a", "phone")
        _record(repo, "a2", "a2", "phone")
        _record(repo, "b1", "b", "phone")
    data = collect(paths)
    assert [(row["source"], row["images_used"]) for row in data["sources"]] == [("Statens vegvesen Vegbilder", 1), ("Wikimedia Commons", 1)]


def test_data_card_counts_unlocked_drone_phone_and_ortho_without_names(tmp_root: Path) -> None:
    paths = RepoPaths.from_root(tmp_root)
    with Database(paths.database) as connection:
        repo = ManifestRepository(connection)
        for source in ("drone", "phone", "ortho"):
            _record(repo, f"{source}-id", f"{source}-hash", source)
    markdown = render_markdown(collect(paths))
    assert all(f"- {source}: 1 images" in markdown for source in ("drone", "phone", "ortho"))
    assert "-id" not in markdown and "-hash" not in markdown


def test_data_card_ignores_rejected_manifest_rows(tmp_root: Path) -> None:
    paths = RepoPaths.from_root(tmp_root)
    with Database(paths.database) as connection:
        _record(ManifestRepository(connection), "rejected", "unique-rejected", "phone", "rejected")
    assert collect(paths)["own_images"] == {}


def test_data_card_only_used_counts_only_dataset_images(tmp_root: Path) -> None:
    paths = RepoPaths.from_root(tmp_root)
    with Database(paths.database) as connection:
        repo = ManifestRepository(connection)
        _record(repo, "included", "one", "phone")
        _record(repo, "excluded", "two", "phone")
    paths.dataset_dir.mkdir(parents=True)
    (paths.dataset_dir / "dataset.json").write_text(json.dumps({"source_image_ids_per_split": {"train": ["included"]}}), encoding="utf-8")
    assert collect(paths, only_used=True)["own_images"] == {"phone": 1}


def test_data_card_json_and_markdown_have_equal_source_counts(tmp_root: Path) -> None:
    paths = RepoPaths.from_root(tmp_root)
    (paths.config_dir / "samples.lock.yaml").write_text(yaml.safe_dump({"entries": [{"sha256": "locked", "id": "commons", "licence": "CC BY 4.0", "attribution": "A"}]}), encoding="utf-8")
    with Database(paths.database) as connection:
        _record(ManifestRepository(connection), "item", "locked", "phone")
    data = collect(paths)
    assert data["sources"][0]["images_used"] == 1 and "| 1 |" in render_markdown(data)
    rendered_json = json.loads(write_data_card(paths, Path("reports/card.json"), output_format="json").read_text(encoding="utf-8"))
    assert rendered_json["sources"][0]["images_used"] == data["sources"][0]["images_used"]


def test_data_card_contains_all_fixed_publication_statements(tmp_root: Path) -> None:
    markdown = render_markdown(collect(RepoPaths.from_root(tmp_root)))
    assert all(text in markdown for text in ("not published", "SAM 3 was used locally as a teacher", "Built with DINOv3", "DINOv3 License", "AGPL-3.0"))


def test_data_card_lists_licence_links_for_nlod_and_cc_by(tmp_root: Path) -> None:
    paths = RepoPaths.from_root(tmp_root)
    (paths.config_dir / "samples.lock.yaml").write_text(yaml.safe_dump({"entries": [{"sha256": "n", "id": "ext_vegbilder_oslo", "licence": "NLOD 2.0", "attribution": "A"}, {"sha256": "c", "id": "commons", "licence": "CC BY 4.0", "attribution": "B"}]}), encoding="utf-8")
    with Database(paths.database) as connection:
        repo = ManifestRepository(connection)
        _record(repo, "n", "n", "phone")
        _record(repo, "c", "c", "phone")
    markdown = render_markdown(collect(paths))
    assert "[NLOD 2.0](https://data.norge.no/nlod/en/2.0)" in markdown
    assert "[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)" in markdown


def test_data_card_never_emits_signed_urls_or_absolute_paths(tmp_root: Path) -> None:
    paths = RepoPaths.from_root(tmp_root)
    (paths.config_dir / "samples.lock.yaml").write_text(yaml.safe_dump({"entries": [{"sha256": "x", "id": "commons", "licence": "CC0", "attribution": "A", "url": "https://x.test/a?st=one&sig=two"}]}), encoding="utf-8")
    with Database(paths.database) as connection:
        _record(ManifestRepository(connection), "x", "x", "phone")
    output = render_markdown(collect(paths))
    assert all(token not in output for token in ("st=", "sig=", "D:\\", "/home", str(tmp_root)))


def test_data_card_handles_missing_lock_files(tmp_root: Path) -> None:
    paths = RepoPaths.from_root(tmp_root)
    with Database(paths.database) as connection:
        _record(ManifestRepository(connection), "own", "own-hash", "phone")
    assert collect(paths)["own_images"] == {"phone": 1}


def test_data_card_handles_missing_database(tmp_root: Path) -> None:
    paths = RepoPaths.from_root(tmp_root)
    (paths.config_dir / "samples.lock.yaml").write_text(yaml.safe_dump({"entries": [{"sha256": "unused", "licence": "CC0"}]}), encoding="utf-8")
    assert collect(paths)["sources"] == [] and collect(paths)["own_images"] == {}


def test_data_card_cli_writes_under_root_and_help_works(tmp_root: Path, capsys) -> None:
    assert main(["data-card", "--root", str(tmp_root), "--out", "reports/out.md"]) == 0
    assert (tmp_root / "reports" / "out.md").is_file()
    with pytest.raises(SystemExit) as result:
        main(["data-card", "--help"])
    assert result.value.code == 0
    assert "--only-used" in capsys.readouterr().out


def test_data_card_lists_commons_authors_in_details(tmp_root: Path) -> None:
    paths = RepoPaths.from_root(tmp_root)
    (paths.config_dir / "samples.lock.yaml").write_text(yaml.safe_dump({"entries": [{"sha256": "commons", "id": "commons_photo", "licence": "CC BY 4.0", "attribution": "Wikimedia", "author": "Jane Author"}]}), encoding="utf-8")
    with Database(paths.database) as connection:
        _record(ManifestRepository(connection), "commons", "commons", "phone")
    markdown = render_markdown(collect(paths))
    assert "<details><summary>Wikimedia Commons authors</summary>" in markdown and "Jane Author" in markdown


def test_data_card_labels_vegbilder_as_statens_vegvesen(tmp_root: Path) -> None:
    paths = RepoPaths.from_root(tmp_root)
    (paths.config_dir / "extra_sources.lock.yaml").write_text(yaml.safe_dump({"entries": [{"sha256": "veg", "id": "ext_vegbilder_oslo", "licence": "NLOD-2.0", "attribution": "Vegvesen"}]}), encoding="utf-8")
    with Database(paths.database) as connection:
        _record(ManifestRepository(connection), "veg", "veg", "phone")
    assert "Statens vegvesen Vegbilder" in render_markdown(collect(paths))
