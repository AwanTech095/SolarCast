"""Lineage tracking and cryptographic provenance helper for SolarCast MLflow runs.

Extracts immutable Git commit hashes, dirty working-tree state, SHA-256 hashes of
processed datasets, configuration files, feature schema, model bundles, and runtime
environment versions.

STDLIB-ONLY: This module contains no third-party dependencies (no pandas, numpy,
scipy) so it can safely be imported across isolated environments (.venv, .venv-feast).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def compute_file_sha256(file_path: Path | str) -> str:
    """Compute SHA-256 hash of a file or return empty string if missing."""
    p = Path(file_path)
    if not p.is_file():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit(repo_dir: Path | str | None = None) -> str:
    """Retrieve current Git commit SHA."""
    cwd = str(repo_dir) if repo_dir else None
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown_commit"


def get_git_dirty(repo_dir: Path | str | None = None) -> bool:
    """Check if the working tree has uncommitted modifications or untracked changes."""
    cwd = str(repo_dir) if repo_dir else None
    try:
        res = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(res.stdout.strip())
    except Exception:
        return False


def get_feature_schema_info(
    schema_path: Path | str = "configs/feature_schema.json",
    repo_dir: Path | str | None = None,
) -> tuple[str, str]:
    """Return (schema_version, sha256_hash) for feature schema JSON."""
    root = Path(repo_dir) if repo_dir else Path.cwd()
    p = root / schema_path
    if not p.is_file():
        return "unknown", ""
    sha = compute_file_sha256(p)
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        version = str(data.get("schema_version", "1.0.0"))
    except Exception:
        version = "unknown"
    return version, sha


def get_lineage_tags(repo_dir: Path | str | None = None) -> dict[str, str]:
    """Collect full lineage provenance dictionary for MLflow logging.

    Args:
        repo_dir: Optional root path of the repository checkout (defaults to cwd).

    Returns:
        Dictionary mapping tag names to string values.
    """
    root = Path(repo_dir) if repo_dir else Path.cwd()

    git_commit = get_git_commit(root)
    git_dirty = get_git_dirty(root)

    tags: dict[str, str] = {
        "git_commit": git_commit,
        "git_dirty": "true" if git_dirty else "false",
        "python_version": sys.version.split()[0],
    }
    # These describe the declared DVC snapshot. Data hashes below describe local bytes.
    for name, pointer in {
        "raw": "data/raw.dvc",
        "processed": "data/processed.dvc",
        "model": "models/solarcast_f1_bundle.joblib.dvc",
    }.items():
        digest = compute_file_sha256(root / pointer)
        if digest:
            tags[f"dvc_pointer_{name}_sha256"] = digest

    # MLflow version (if available in runtime)
    mlflow_mod = sys.modules.get("mlflow")
    if mlflow_mod is not None:
        tags["mlflow_version"] = getattr(mlflow_mod, "__version__", "unknown")
    else:
        try:
            import mlflow

            tags["mlflow_version"] = getattr(mlflow, "__version__", "unknown")
        except ImportError:
            tags["mlflow_version"] = "not_installed"

    # Datasets
    hourly_p = root / "data" / "processed" / "hourly.parquet"
    if hourly_p.is_file():
        tags["data_hash_hourly"] = compute_file_sha256(hourly_p)

    hrrr_p = root / "data" / "processed" / "hrrr_forecast_features.parquet"
    if hrrr_p.is_file():
        tags["data_hash_hrrr"] = compute_file_sha256(hrrr_p)

    qh_p = root / "data" / "processed" / "quarter_hourly.parquet"
    if qh_p.is_file():
        tags["data_hash_qh"] = compute_file_sha256(qh_p)

    feast_p = root / "data" / "processed" / "feast" / "features_set_b.parquet"
    if feast_p.is_file():
        tags["data_hash_feast"] = compute_file_sha256(feast_p)

    # Configs and Manifests
    manifest_p = root / "configs" / "data_manifest.json"
    if manifest_p.is_file():
        tags["data_manifest_hash"] = compute_file_sha256(manifest_p)

    eval_cfg_p = root / "configs" / "evaluation.yaml"
    if eval_cfg_p.is_file():
        tags["config_hash_evaluation"] = compute_file_sha256(eval_cfg_p)

    params_cfg_p = root / "configs" / "model_params.yaml"
    if params_cfg_p.is_file():
        tags["config_hash_model_params"] = compute_file_sha256(params_cfg_p)

    # Feature schema version & hash
    schema_ver, schema_sha = get_feature_schema_info("configs/feature_schema.json", root)
    if schema_ver != "unknown":
        tags["feature_schema_version"] = schema_ver
    if schema_sha:
        tags["feature_schema_hash"] = schema_sha

    # Model bundle (if present on disk)
    bundle_p = root / "models" / "solarcast_f1_bundle.joblib"
    if bundle_p.is_file():
        tags["bundle_sha256"] = compute_file_sha256(bundle_p)

    return tags


def log_lineage(
    run: Any | None = None,
    repo_dir: Path | str | None = None,
    extra_tags: dict[str, str] | None = None,
) -> dict[str, str]:
    """Compute lineage metadata and log tags to the active MLflow run.

    Args:
        run: Optional MLflow ActiveRun object or None (uses active run).
        repo_dir: Root path of repository checkout.
        extra_tags: Optional dictionary of additional tags to set.

    Returns:
        The dictionary of logged lineage tags.
    """
    tags = get_lineage_tags(repo_dir=repo_dir)
    if extra_tags:
        tags.update(extra_tags)

    try:
        import mlflow

        if mlflow.active_run():
            mlflow.set_tags(tags)
        elif run is not None and hasattr(run, "info") and hasattr(run.info, "run_id"):
            client = mlflow.tracking.MlflowClient()
            for k, v in tags.items():
                client.set_tag(run.info.run_id, k, str(v))
    except Exception as e:
        print(f"[lineage] Notice: MLflow tag logging skipped: {e}", file=sys.stderr)

    return tags
