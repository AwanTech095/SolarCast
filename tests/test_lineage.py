"""Unit tests for lineage tracking and cryptographic provenance helper (src/models/lineage.py)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.models.lineage import (
    compute_file_sha256,
    get_feature_schema_info,
    get_git_commit,
    get_git_dirty,
    get_lineage_tags,
    log_lineage,
)


def test_declared_dvc_snapshot_has_pointer_identity(tmp_path: Path) -> None:
    """Tag the actual DVC pointer bytes without requiring the restored dataset."""
    pointer = tmp_path / "data/processed.dvc"
    pointer.parent.mkdir()
    pointer.write_bytes(b"outs: []\n")
    tags = get_lineage_tags(tmp_path)
    assert tags["dvc_pointer_processed_sha256"] == hashlib.sha256(pointer.read_bytes()).hexdigest()
    assert "data_hash_hourly" not in tags


def test_compute_file_sha256(tmp_path: Path) -> None:
    """Test SHA-256 calculation for existing and nonexistent files."""
    test_file = tmp_path / "sample.txt"
    content = b"SolarCast test content for sha256"
    test_file.write_bytes(content)

    expected_hash = hashlib.sha256(content).hexdigest()
    actual_hash = compute_file_sha256(test_file)
    assert actual_hash == expected_hash

    missing_file = tmp_path / "nonexistent.txt"
    assert compute_file_sha256(missing_file) == ""


def test_get_git_commit_and_dirty() -> None:
    """Test retrieval of git commit hash and dirty status."""
    commit = get_git_commit()
    assert isinstance(commit, str)
    assert len(commit) > 0

    dirty = get_git_dirty()
    assert isinstance(dirty, bool)


def test_get_feature_schema_info(tmp_path: Path) -> None:
    """Test feature schema JSON inspection."""
    schema_file = tmp_path / "schema.json"
    schema_file.write_text(json.dumps({"schema_version": "1.2.3"}), encoding="utf-8")

    ver, sha = get_feature_schema_info(schema_file.name, repo_dir=tmp_path)
    assert ver == "1.2.3"
    assert len(sha) == 64

    missing_ver, missing_sha = get_feature_schema_info("nonexistent.json", repo_dir=tmp_path)
    assert missing_ver == "unknown"
    assert missing_sha == ""


def test_get_lineage_tags() -> None:
    """Test that get_lineage_tags collects expected provenance tags."""
    tags = get_lineage_tags()
    assert isinstance(tags, dict)

    required_keys = [
        "git_commit",
        "git_dirty",
        "python_version",
        "mlflow_version",
        "data_manifest_hash",
        "feature_schema_version",
        "feature_schema_hash",
        "config_hash_evaluation",
        "config_hash_model_params",
    ]
    for key in required_keys:
        assert key in tags, f"Missing required tag: {key}"
        assert isinstance(tags[key], str)


def test_lineage_is_stdlib_only() -> None:
    """Verify that src.models.lineage imports no heavy data-science dependencies."""
    import src.models.lineage as mod

    # Inspect imported module attributes
    heavy_modules = ["pandas", "numpy", "scipy", "sklearn", "feast"]
    for mod_name in heavy_modules:
        assert not hasattr(mod, mod_name), f"lineage.py should not import {mod_name}"


def test_log_lineage_with_mock_mlflow() -> None:
    """Test log_lineage sets tags on active MLflow run."""
    mock_run = MagicMock()
    mock_run.info.run_id = "test_run_123"

    with patch("mlflow.active_run", return_value=mock_run), patch("mlflow.set_tags") as mock_set:
        tags = log_lineage(run=mock_run, extra_tags={"custom_key": "custom_val"})
        assert tags["custom_key"] == "custom_val"
        mock_set.assert_called_once_with(tags)


def test_log_lineage_inactive_run_falls_back_to_client() -> None:
    """Test log_lineage uses MlflowClient if run is passed but no active run."""
    mock_run = MagicMock()
    mock_run.info.run_id = "test_inactive_run"

    mock_client = MagicMock()
    with patch("mlflow.active_run", return_value=None), patch(
        "mlflow.tracking.MlflowClient", return_value=mock_client
    ):
        tags = log_lineage(run=mock_run)
        assert mock_client.set_tag.call_count == len(tags)
