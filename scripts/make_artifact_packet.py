"""Artifact Packet Generator for SolarCast (Member 1 to Member 2 Handoff).

Creates a standalone, clean artifact packet outside the repository containing
all raw and processed datasets, model bundles, manifests, and evaluation audits
required by Member 2 to reproduce pipelines, configure DVC, and run CI/CD.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("make_artifact_packet")

# Processed paths specifically excluded per specification
PROCESSED_EXCLUSIONS = {
    "hrrr_parts": (
        "Intermediate monthly chunk files prior to consolidation; "
        "superseded by hrrr_forecast_features.parquet"
    ),
    "june_week_extracted.parquet": (
        "Exploratory sample from early M1.1b feasibility check, "
        "not used in documented pipeline"
    ),
    "mixed_week_extracted.parquet": (
        "Exploratory sample from early M1.1b feasibility check, "
        "not used in documented pipeline"
    ),
    ".gitkeep": "Git placeholder file",
}

# Convenience files to include
CONVENIENCE_FILES = [
    Path("reports/evaluation.json"),
    Path("reports/final_test_audit.json"),
    Path("configs/data_manifest.json"),
]

MODEL_FILES = [
    Path("models/solarcast_f1_bundle.joblib"),
]

# Pipeline commands producing each artifact (from docs/handoff_member2.md)
PIPELINE_COMMANDS = {
    "data/raw/2107_meter_15m_data.csv": (
        "python -m src.data.ingestion --config configs/data.yaml"
    ),
    "data/raw/manifest.json": (
        "python -m src.data.ingestion --config configs/data.yaml"
    ),
    "data/raw/metadata/2105_system_metadata.json": (
        "python -m src.data.ingestion --config configs/data.yaml"
    ),
    "data/raw/metadata/2107_system_metadata.json": (
        "python -m src.data.ingestion --config configs/data.yaml"
    ),
    "data/raw/metadata/9068_system_metadata.json": (
        "python -m src.data.ingestion --config configs/data.yaml"
    ),
    "data/raw/samples/2105_environment_1_data.csv.head.csv": (
        "python -m src.data.ingestion --config configs/data.yaml"
    ),
    "data/raw/samples/2105_inv01_data.csv.head.csv": (
        "python -m src.data.ingestion --config configs/data.yaml"
    ),
    "data/raw/samples/2107_electrical_data.csv.head.csv": (
        "python -m src.data.ingestion --config configs/data.yaml"
    ),
    "data/raw/samples/2107_environment_data.csv.head.csv": (
        "python -m src.data.ingestion --config configs/data.yaml"
    ),
    "data/raw/samples/2107_irradiance_data.csv.head.csv": (
        "python -m src.data.ingestion --config configs/data.yaml"
    ),
    "data/raw/samples/9068_ac_power_data.csv.head.csv": (
        "python -m src.data.ingestion --config configs/data.yaml"
    ),
    "data/raw/samples/9068_environment_data.csv.head.csv": (
        "python -m src.data.ingestion --config configs/data.yaml"
    ),
    "data/raw/samples/nasa_power_2107_sample.json": (
        "python -m src.data.ingestion --config configs/data.yaml"
    ),
    "data/raw/weather/nasa_power_2017.json": (
        "python -m src.data.ingestion --config configs/data.yaml"
    ),
    "data/raw/weather/nasa_power_2018.json": (
        "python -m src.data.ingestion --config configs/data.yaml"
    ),
    "data/raw/weather/nasa_power_2019.json": (
        "python -m src.data.ingestion --config configs/data.yaml"
    ),
    "data/raw/weather/nasa_power_2020.json": (
        "python -m src.data.ingestion --config configs/data.yaml"
    ),
    "data/raw/weather/nasa_power_2021.json": (
        "python -m src.data.ingestion --config configs/data.yaml"
    ),
    "data/raw/weather/nasa_power_2022.json": (
        "python -m src.data.ingestion --config configs/data.yaml"
    ),
    "data/raw/weather/nasa_power_2023.json": (
        "python -m src.data.ingestion --config configs/data.yaml"
    ),
    "data/processed/quarantine.parquet": (
        "python -m src.data.aggregation --config configs/data.yaml"
    ),
    "data/processed/hourly.parquet": (
        "python -m src.data.aggregation --config configs/data.yaml"
    ),
    "data/processed/quarter_hourly.parquet": (
        "python -m src.data.aggregation --config configs/data.yaml"
    ),
    "data/processed/hrrr_coverage_scan.parquet": (
        "python scripts/step2_coverage_scan.py"
    ),
    "data/processed/hrrr_forecast_features.parquet": (
        "python scripts/hrrr_extract.py --merge --parts-dir data/processed/hrrr_parts "
        "--merged-file data/processed/hrrr_forecast_features.parquet"
    ),
    "data/processed/feast/features_set_b.parquet": (
        "python -m src.features.build_feature_table --hourly data/processed/hourly.parquet "
        "--quarter-hourly data/processed/quarter_hourly.parquet "
        "--weather data/processed/hrrr_forecast_features.parquet "
        "--output data/processed/feast/features_set_b.parquet "
        '--start "2018-07-12 00:00:00+00:00" --end "2023-11-09 23:00:00+00:00"'
    ),
    "data/processed/validation_residuals.parquet": (
        "python -m src.models.train --eval-config configs/evaluation.yaml "
        "--params-config configs/model_params.yaml --hourly data/processed/hourly.parquet "
        "--quarter-hourly data/processed/quarter_hourly.parquet "
        "--weather data/processed/hrrr_forecast_features.parquet "
        "--reports-dir reports --models-dir models"
    ),
    "models/solarcast_f1_bundle.joblib": (
        "python -m src.models.train --eval-config configs/evaluation.yaml "
        "--params-config configs/model_params.yaml --hourly data/processed/hourly.parquet "
        "--quarter-hourly data/processed/quarter_hourly.parquet "
        "--weather data/processed/hrrr_forecast_features.parquet "
        "--reports-dir reports --models-dir models"
    ),
    "reports/evaluation.json": (
        "python -m src.models.train --eval-config configs/evaluation.yaml "
        "--params-config configs/model_params.yaml --hourly data/processed/hourly.parquet "
        "--quarter-hourly data/processed/quarter_hourly.parquet "
        "--weather data/processed/hrrr_forecast_features.parquet "
        "--reports-dir reports --models-dir models"
    ),
    "reports/final_test_audit.json": (
        "python -m src.models.evaluate --final-test --force "
        '--reason "one-time final evaluation after model selection"'
    ),
    "configs/data_manifest.json": (
        "python -m src.data.ingestion --config configs/data.yaml"
    ),
}

KEY_PACKAGES = [
    "pandas",
    "numpy",
    "pyarrow",
    "scikit-learn",
    "scipy",
    "pandera",
    "pyyaml",
    "requests",
    "boto3",
    "s3fs",
    "joblib",
    "matplotlib",
    "mlflow",
    "pytest",
    "ruff",
    "pvlib",
    "zarr",
    "numcodecs",
    "xarray",
    "pyproj",
]


def sha256_of_file(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def collect_packet_files(repo_dir: Path) -> tuple[list[Path], dict[str, str]]:
    """Identify files to include in the artifact packet and track excluded paths.

    Args:
        repo_dir: Root directory of the SolarCast repository.

    Returns:
        A tuple of (included_files, exclusions_dict).
    """
    included: list[Path] = []
    exclusions: dict[str, str] = {}

    # 1. Everything under data/raw/ (except .gitkeep)
    raw_dir = repo_dir / "data" / "raw"
    if raw_dir.exists():
        for path in sorted(raw_dir.rglob("*")):
            if path.is_file():
                if path.name == ".gitkeep":
                    exclusions[str(path.relative_to(repo_dir)).replace("\\", "/")] = (
                        "Git placeholder file"
                    )
                else:
                    included.append(path)

    # 2. Everything under data/processed/ with specific exclusions
    processed_dir = repo_dir / "data" / "processed"
    if processed_dir.exists():
        for path in sorted(processed_dir.rglob("*")):
            if not path.is_file():
                continue
            rel_proc = path.relative_to(processed_dir).as_posix()
            parts = rel_proc.split("/")
            top_part = parts[0]

            if top_part in PROCESSED_EXCLUSIONS:
                rel_repo = str(path.relative_to(repo_dir)).replace("\\", "/")
                exclusions[rel_repo] = PROCESSED_EXCLUSIONS[top_part]
                continue
            if rel_proc in PROCESSED_EXCLUSIONS:
                rel_repo = str(path.relative_to(repo_dir)).replace("\\", "/")
                exclusions[rel_repo] = PROCESSED_EXCLUSIONS[rel_proc]
                continue
            if path.name == ".gitkeep":
                exclusions[str(path.relative_to(repo_dir)).replace("\\", "/")] = (
                    "Git placeholder file"
                )
                continue

            included.append(path)

    # 3. Model bundle
    for model_path in MODEL_FILES:
        full_path = repo_dir / model_path
        if full_path.exists():
            included.append(full_path)
        else:
            logger.warning("Expected model file not found: %s", full_path)

    # 4. Reports and config convenience files
    for conv_path in CONVENIENCE_FILES:
        full_path = repo_dir / conv_path
        if full_path.exists():
            included.append(full_path)
        else:
            logger.warning("Expected convenience file not found: %s", full_path)

    return sorted(included), exclusions


def copy_packet_files(repo_dir: Path, out_dir: Path, files: list[Path]) -> None:
    """Copy collected files to the destination directory preserving relative paths.

    Args:
        repo_dir: Root directory of the source repository.
        out_dir: Destination root directory for the packet.
        files: List of file paths to copy.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for src_path in files:
        rel_path = src_path.relative_to(repo_dir)
        dest_path = out_dir / rel_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dest_path)


def read_mlflow_run(run_dir: Path) -> dict[str, Any]:
    """Read run metadata, tags, params, metrics, and artifacts directly from disk."""
    meta_file = run_dir / "meta.yaml"
    meta = {}
    if meta_file.exists():
        with open(meta_file, encoding="utf-8") as f:
            meta = yaml.safe_load(f) or {}

    tags: dict[str, str] = {}
    tags_dir = run_dir / "tags"
    if tags_dir.exists():
        for t in tags_dir.iterdir():
            if t.is_file():
                tags[t.name] = t.read_text(encoding="utf-8").strip()

    params: dict[str, str] = {}
    params_dir = run_dir / "params"
    if params_dir.exists():
        for p in params_dir.iterdir():
            if p.is_file():
                params[p.name] = p.read_text(encoding="utf-8").strip()

    metrics: dict[str, float] = {}
    metrics_dir = run_dir / "metrics"
    if metrics_dir.exists():
        for m in metrics_dir.iterdir():
            if m.is_file():
                lines = m.read_text(encoding="utf-8").strip().splitlines()
                if lines:
                    last_line = lines[-1].split()
                    if len(last_line) >= 2:
                        try:
                            metrics[m.name] = float(last_line[1])
                        except ValueError:
                            pass

    artifacts: list[str] = []
    artifacts_dir = run_dir / "artifacts"
    if artifacts_dir.exists():
        for a in artifacts_dir.rglob("*"):
            if a.is_file():
                artifacts.append(str(a.relative_to(artifacts_dir)).replace("\\", "/"))

    return {
        "run_id": meta.get("run_id", run_dir.name),
        "run_name": tags.get("mlflow.runName", meta.get("run_name", "")),
        "status": meta.get("status"),
        "tags": tags,
        "params": params,
        "metrics": metrics,
        "artifacts": sorted(artifacts),
    }


def audit_mlflow(repo_dir: Path) -> dict[str, Any]:
    """Extract MLflow facts and audit lineage across local mlruns store."""
    mlruns_dir = repo_dir / "mlruns"
    exp_training_id = "754448501154260354"
    exp_solarcast_id = "938233506511185438"

    # Known key run IDs
    champion_run_id = "f7fef3f7bdf241c5a53dc50d8e4713f9"
    final_test_run_id = "620faa341a8a4c14824f31e40b157fd6"
    deviation_run_id = "cce94db3448e405ab0fee01415e82da1"
    monitoring_run_id = "16ffd7429be642928941e95f932fe8b3"
    feast_run_id = "56f349ec19f64a62917396b79a8e9fa9"

    # Count tuning runs in solarcast-model-training
    tuning_runs = []
    exp_training_dir = mlruns_dir / exp_training_id
    if exp_training_dir.exists():
        for r_dir in sorted(exp_training_dir.iterdir()):
            if not r_dir.is_dir() or r_dir.name.startswith("."):
                continue
            r_info = read_mlflow_run(r_dir)
            name = r_info["run_name"]
            if name.startswith("HistGradientBoosting_") or name.startswith("Ridge_"):
                tuning_runs.append({
                    "run_id": r_info["run_id"],
                    "run_name": name,
                    "is_best": r_info["tags"].get("is_best_candidate", "False"),
                })

    # Read registered model solarcast-f1 versions
    models_dir = mlruns_dir / "models" / "solarcast-f1"
    model_versions = []
    if models_dir.exists():
        for v_dir in sorted(models_dir.iterdir()):
            if not v_dir.is_dir() or not v_dir.name.startswith("version-"):
                continue
            meta_f = v_dir / "meta.yaml"
            if meta_f.exists():
                with open(meta_f, encoding="utf-8") as f:
                    y = yaml.safe_load(f) or {}
                ver_num = int(y.get("version", 0))
                model_versions.append({
                    "version": ver_num,
                    "run_id": y.get("run_id"),
                    "current_stage": y.get("current_stage"),
                    "aliases": y.get("aliases", []),
                    "tags": y.get("tags"),
                    "description": y.get("description"),
                    "status": y.get("status"),
                    "is_champion": (ver_num == 4),
                    "audit": {
                        "alias": "LOGGED" if y.get("aliases") else "MISSING",
                        "stage": (
                            "LOGGED"
                            if y.get("current_stage") and y.get("current_stage") != "None"
                            else "MISSING"
                        ),
                        "tags": "LOGGED" if y.get("tags") else "MISSING",
                        "description": "LOGGED" if y.get("description") else "MISSING",
                    },
                })

    runs_to_audit = [
        ("champion_run", exp_training_id, champion_run_id, "solarcast-model-training"),
        ("final_test_evaluation", exp_training_id, final_test_run_id, "solarcast-model-training"),
        ("deviation_detector", exp_solarcast_id, deviation_run_id, "solarcast"),
        ("monitoring_replay", exp_solarcast_id, monitoring_run_id, "solarcast"),
        ("feature_store_materialization", exp_solarcast_id, feast_run_id, "solarcast"),
    ]

    runs_audit = []
    for label, exp_id, run_id, exp_name in runs_to_audit:
        r_dir = mlruns_dir / exp_id / run_id
        if not r_dir.exists():
            logger.warning("Run dir not found: %s", r_dir)
            continue
        data = read_mlflow_run(r_dir)
        tags = data["tags"]

        # Git commit
        commit = (
            tags.get("git_commit")
            or tags.get("mlflow.source.git.commit")
            or tags.get("feast_repo_git_commit")
        )
        git_audit = {
            "status": "LOGGED" if commit else "MISSING",
            "value": commit if commit else None,
        }

        # Data file hashes or manifest hash
        data_hashes = {
            k: v for k, v in tags.items()
            if "data" in k or "hash" in k or "manifest" in k
        }
        if data_hashes:
            data_audit = {
                "status": "PARTIALLY_LOGGED" if "manifest_hash" not in tags else "LOGGED",
                "detail": (
                    f"Data hashes logged ({', '.join(data_hashes.keys())}); "
                    f"data_manifest.json hash {'LOGGED' if 'manifest_hash' in tags else 'MISSING'}"
                ),
                "logged_keys": list(data_hashes.keys()),
            }
        else:
            data_audit = {
                "status": "MISSING",
                "detail": "No data file hashes or manifest hash recorded in run tags",
                "logged_keys": [],
            }

        # Feature schema version and hash
        schema_ver = tags.get("feature_schema_version") or tags.get("schema_version")
        schema_hash = tags.get("feature_schema_hash") or tags.get("schema_hash")
        if schema_ver and schema_hash:
            schema_audit = {
                "status": "LOGGED",
                "detail": f"Version {schema_ver}, hash {schema_hash}",
            }
        elif schema_ver:
            schema_audit = {
                "status": "PARTIALLY_LOGGED",
                "detail": f"Feature schema version logged ({schema_ver}); schema hash MISSING",
            }
        else:
            schema_audit = {
                "status": "MISSING",
                "detail": "Feature schema version and hash MISSING",
            }

        # Bundle sha256
        bundle_sha = tags.get("bundle_sha256") or tags.get("bundle_hash")
        has_bundle_artifact = any("solarcast_f1_bundle.joblib" in a for a in data["artifacts"])
        if bundle_sha:
            bundle_audit = {"status": "LOGGED", "value": bundle_sha}
        elif has_bundle_artifact:
            bundle_audit = {
                "status": "MISSING",
                "detail": "Bundle logged as artifact in run, but bundle_sha256 tag MISSING",
            }
        else:
            bundle_audit = {
                "status": "MISSING",
                "detail": "Bundle not logged in run tags or artifacts",
            }

        # evaluation.json as artifact
        has_eval_json = any(
            "evaluation.json" in a for a in data["artifacts"]
        )
        eval_audit = {
            "status": "LOGGED" if has_eval_json else "MISSING",
            "artifact_path": "evaluation/evaluation.json" if has_eval_json else None,
        }

        runs_audit.append({
            "entity": label,
            "run_id": run_id,
            "run_name": data["run_name"],
            "experiment_name": exp_name,
            "git_commit_hash": git_audit,
            "data_hashes_or_manifest_hash": data_audit,
            "feature_schema_version_and_hash": schema_audit,
            "bundle_sha256": bundle_audit,
            "params": {
                "status": "LOGGED" if data["params"] else "MISSING",
                "count": len(data["params"]),
            },
            "metrics": {
                "status": "LOGGED" if data["metrics"] else "MISSING",
                "count": len(data["metrics"]),
            },
            "evaluation_json_artifact": eval_audit,
        })

    return {
        "tuning_runs_count": len(tuning_runs),
        "tuning_runs": tuning_runs,
        "registered_model_solarcast_f1": {
            "model_name": "solarcast-f1",
            "champion_version": 4,
            "versions": model_versions,
            "registry_audit_summary": (
                "Registered model solarcast-f1 has 4 versions in local mlruns file store. "
                "Version 4 is the champion (HistGradientBoosting Set B, run "
                "f7fef3f7bdf241c5a53dc50d8e4713f9). In local file store, aliases ([]), "
                "current_stage (None), tags (None), and description (None) are all "
                "UNSET/MISSING across all versions because MLflow Model Registry stages/aliases "
                "require either MLflow Client transition calls or a centralized SQL-backed store."
            ),
        },
        "runs_audit": runs_audit,
    }


def print_lineage_audit_table(audit_data: dict[str, Any]) -> None:
    """Print readable audit table of MLflow lineage facts to stdout."""
    print("\n" + "=" * 100)
    print("MLFLOW LINEAGE AUDIT TABLE (READ-ONLY AUDIT; NOTHING MODIFIED)")
    print("=" * 100)

    # Registered models table
    print("\n[1] REGISTERED MODEL: solarcast-f1")
    header_reg = (
        f"{'Version':<9} {'Run ID':<34} {'Stage':<10} {'Aliases':<10} "
        f"{'Tags':<10} {'Desc':<10} {'Champion?'}"
    )
    print(header_reg)
    print("-" * 100)
    for v in audit_data["registered_model_solarcast_f1"]["versions"]:
        is_champ = "CHAMPION" if v["is_champion"] else "-"
        stage = str(v["current_stage"])
        aliases = str(v["aliases"]) if v["aliases"] else "MISSING"
        tags = str(v["tags"]) if v["tags"] else "MISSING"
        desc = str(v["description"]) if v["description"] else "MISSING"
        print(
            f"v{v['version']:<8} {v['run_id']:<34} {stage:<10} "
            f"{aliases:<10} {tags:<10} {desc:<10} {is_champ}"
        )
    print("Summary:", audit_data["registered_model_solarcast_f1"]["registry_audit_summary"])

    # Runs table
    print(f"\n[2] KEY PIPELINE RUNS AUDIT (Tuning Runs Count: {audit_data['tuning_runs_count']})")
    print("-" * 100)
    for r in audit_data["runs_audit"]:
        print(f"\nEntity: {r['entity']} | Run ID: {r['run_id']} | Name: {r['run_name']}")
        print(f"  - Git Commit Hash:           {r['git_commit_hash']['status']:<16} "
              f"({r['git_commit_hash'].get('value')})")
        print(f"  - Data / Manifest Hashes:    {r['data_hashes_or_manifest_hash']['status']:<16} "
              f"({r['data_hashes_or_manifest_hash']['detail']})")
        schema_stat = r['feature_schema_version_and_hash']['status']
        print(f"  - Feature Schema Ver & Hash: {schema_stat:<16} "
              f"({r['feature_schema_version_and_hash']['detail']})")
        b_info = r['bundle_sha256'].get('detail', r['bundle_sha256'].get('value'))
        print(f"  - Bundle SHA-256 Tag:        {r['bundle_sha256']['status']:<16} "
              f"({b_info})")
        print(f"  - Parameters Logged:         {r['params']['status']:<16} "
              f"({r['params']['count']} parameters)")
        print(f"  - Metrics Logged:            {r['metrics']['status']:<16} "
              f"({r['metrics']['count']} metrics)")
        eval_p = r['evaluation_json_artifact'].get('artifact_path')
        print(f"  - evaluation.json Artifact:  {r['evaluation_json_artifact']['status']:<16} "
              f"({eval_p})")
    print("=" * 100)


def build_and_write_manifest(
    repo_dir: Path,
    out_dir: Path,
    files: list[Path],
) -> None:
    """Generate CHECKSUMS.sha256 and PACKET_MANIFEST.json inside the packet directory."""
    logger.info("Computing SHA-256 checksums for %d packet files...", len(files))

    checksums_lines: list[str] = []
    file_records: list[dict[str, Any]] = []

    for src_path in files:
        rel_posix = src_path.relative_to(repo_dir).as_posix()
        dest_path = out_dir / src_path.relative_to(repo_dir)
        sha = sha256_of_file(dest_path)
        sz = dest_path.stat().st_size
        checksums_lines.append(f"{sha}  {rel_posix}\n")

        pipeline_cmd = PIPELINE_COMMANDS.get(
            rel_posix,
            "Documented in docs/handoff_member2.md"
        )

        file_records.append({
            "path": rel_posix,
            "bytes": sz,
            "sha256": sha,
            "pipeline_command": pipeline_cmd,
        })

    # Sort checksum lines and file records
    checksums_lines.sort(key=lambda line: line.split()[1])
    file_records.sort(key=lambda r: r["path"])

    # Git details
    git_commit = "unknown"
    git_branch = "unknown"
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True
        ).strip()
        git_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_dir, text=True
        ).strip()
    except Exception as e:
        logger.warning("Could not read git details: %s", e)

    # Package versions
    pkg_versions = {}
    for pkg in KEY_PACKAGES:
        try:
            pkg_versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            pkg_versions[pkg] = "not installed"

    # MLflow audit
    mlflow_audit = audit_mlflow(repo_dir)

    manifest_data = {
        "manifest_version": "1.0",
        "packet_name": "SolarCast Artifact Packet (Member 1 to Member 2 Handoff)",
        "created_for": "Member 2 (DVC, Airflow/KFP pipelines, CI/CD orchestration)",
        "source_repo": "SolarCast",
        "git": {
            "commit_hash": git_commit,
            "branch": git_branch,
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "key_packages": pkg_versions,
        },
        "mlflow_facts": {
            "tracking_uri": "file:./mlruns",
            "tracking_address": "none",
            "shared_server_configured": False,
            "note": (
                "No shared MLflow server is configured yet (tracking address: none). "
                "Local store contains absolute Windows file URIs and must not be ported "
                "directly across machines; all runs will be re-logged to the shared server "
                "by executing python -m src.models.train once the central tracking URI exists."
            ),
            "experiments": {
                "solarcast-model-training": "754448501154260354",
                "solarcast": "938233506511185438",
                "Default": "0",
            },
            "champion_run_id": "f7fef3f7bdf241c5a53dc50d8e4713f9",
            "champion_model_name": "HistGradientBoosting_Set_B_15mDelay",
            "registered_model": "solarcast-f1",
            "registered_model_version": 4,
            "run_ids": {
                "champion_training_run": "f7fef3f7bdf241c5a53dc50d8e4713f9",
                "final_test_evaluation": "620faa341a8a4c14824f31e40b157fd6",
                "deviation_detector": "cce94db3448e405ab0fee01415e82da1",
                "monitoring_replay": "16ffd7429be642928941e95f932fe8b3",
                "feature_store_materialization": "56f349ec19f64a62917396b79a8e9fa9",
            },
        },
        "mlflow_lineage_audit": mlflow_audit,
        "total_files": len(file_records),
        "total_bytes": sum(r["bytes"] for r in file_records),
        "files": file_records,
    }

    # Write CHECKSUMS.sha256
    checksums_path = out_dir / "CHECKSUMS.sha256"
    checksums_path.write_text("".join(checksums_lines), encoding="utf-8")
    logger.info("Wrote %s (%d files)", checksums_path, len(checksums_lines))

    # Write PACKET_MANIFEST.json
    manifest_path = out_dir / "PACKET_MANIFEST.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    logger.info("Wrote %s", manifest_path)

    # Print lineage audit table
    print_lineage_audit_table(mlflow_audit)


def create_zip_archive(packet_dir: Path, zip_path: Path) -> tuple[int, str]:
    """Create a zip archive of the packet folder and report facts."""
    logger.info("Creating ZIP archive %s from %s ...", zip_path, packet_dir)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(packet_dir.rglob("*")):
            if p.is_file():
                arcname = p.relative_to(packet_dir).as_posix()
                zf.write(p, arcname)

    # Compute SHA-256 of zip
    zip_sha = sha256_of_file(zip_path)
    zip_bytes = zip_path.stat().st_size
    zip_mb = zip_bytes / (1024 * 1024)

    total_uncompressed = 0
    file_count = 0
    print("\n" + "=" * 80)
    print(f"SOLARCAST ARTIFACT PACKET ZIP ARCHIVE: {zip_path}")
    print("=" * 80)
    print(f"Archive Size:    {zip_bytes:,} bytes ({zip_mb:.2f} MB)")
    print(f"Archive SHA-256: {zip_sha}")
    print("=" * 80)

    print("\nFiles in ZIP archive:")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            file_count += 1
            total_uncompressed += info.file_size
            print(
                f"  {info.filename:<55} {info.file_size:>10,d} bytes "
                f"(deflated: {info.compress_size:>10,d})"
            )

    print("-" * 80)
    print(f"Total files in archive:       {file_count}")
    print(
        f"Total uncompressed size:      {total_uncompressed:,} bytes "
        f"({total_uncompressed / (1024 * 1024):.2f} MB)"
    )
    print(f"Compressed archive size:      {zip_bytes:,} bytes ({zip_mb:.2f} MB)")
    ratio = (1 - zip_bytes / total_uncompressed) * 100
    print(f"Compression ratio:            {ratio:.2f}% reduction")
    print("=" * 80)

    return zip_bytes, zip_sha


def main() -> None:
    """Main CLI entrypoint for packet generation."""
    parser = argparse.ArgumentParser(description="Create SolarCast artifact packet for Member 2")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("D:/SolarCast_packet"),
        help="Destination directory for artifact packet (default: D:/SolarCast_packet)",
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="SolarCast repository root directory",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Also package packet folder into <out>.zip archive",
    )
    args = parser.parse_args()

    repo_dir: Path = args.repo_dir.resolve()
    out_dir: Path = args.out.resolve()

    logger.info("Source repository: %s", repo_dir)
    logger.info("Destination packet: %s", out_dir)

    # Safety checks
    if str(out_dir).startswith(str(repo_dir)):
        logger.error("Destination must be OUTSIDE the repository directory: %s", out_dir)
        sys.exit(1)

    included_files, exclusions = collect_packet_files(repo_dir)

    print("\n" + "=" * 80)
    print("SOLARCAST ARTIFACT PACKET INCLUSION REPORT")
    print("=" * 80)
    total_bytes = 0
    for p in included_files:
        rel = p.relative_to(repo_dir).as_posix()
        sz = p.stat().st_size
        total_bytes += sz
        print(f"  [INCLUDE] {rel:<55} ({sz:>10,d} bytes)")

    print("\n" + "=" * 80)
    print("SOLARCAST ARTIFACT PACKET EXCLUSION REPORT (WITH REASONS)")
    print("=" * 80)
    for path_str, reason in sorted(exclusions.items()):
        print(f"  [EXCLUDE] {path_str:<55} -> Reason: {reason}")

    print("\n" + "=" * 80)
    print(f"Total files to copy: {len(included_files)}")
    print(f"Total size: {total_bytes:,} bytes ({total_bytes / (1024 * 1024):.2f} MB)")
    print("=" * 80)

    logger.info("Copying files to %s ...", out_dir)
    copy_packet_files(repo_dir, out_dir, included_files)
    logger.info("Step 1 complete: %d files copied", len(included_files))

    logger.info("Generating CHECKSUMS.sha256 and PACKET_MANIFEST.json (Step 2)...")
    build_and_write_manifest(repo_dir, out_dir, included_files)
    logger.info("Step 2 complete!")

    if args.zip:
        zip_path = out_dir.with_suffix(".zip")
        create_zip_archive(out_dir, zip_path)
        logger.info("Step 4 complete: Zip archive created at %s", zip_path)


if __name__ == "__main__":
    main()
