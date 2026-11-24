"""Resolve an MLflow run or 'champion' to exact code, data, schema, and artifact hashes.

Verifies each recorded cryptographic hash against current local workspace files,
reporting [OK], [MISMATCH], or [NOT RECORDED].
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

from mlflow.tracking import MlflowClient

TRACKING_URI = "file:./mlruns"
MODEL_NAME = "solarcast-f1"


def compute_sha256(path: Path | str) -> str:
    """Compute SHA-256 hash of a file or empty string if not found."""
    p = Path(path)
    if not p.is_file():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def check_hash(
    item_name: str,
    recorded_hash: str | None,
    file_path: Path | str,
) -> tuple[str, str, str]:
    """Verify recorded hash against local file on disk.

    Returns:
        (status_str, recorded_display, local_display)
    """
    local_p = Path(file_path)
    local_hash = compute_sha256(local_p) if local_p.is_file() else ""

    if not recorded_hash or recorded_hash.strip().lower() in ("unverified", "none", ""):
        status = "[NOT RECORDED]"
        rec_disp = "unverified / not recorded"
        loc_disp = local_hash[:16] + "..." if local_hash else "(file missing)"
    else:
        rec_disp = recorded_hash[:16] + "..."
        if not local_hash:
            status = "[FILE MISSING]"
            loc_disp = "(file missing on disk)"
        elif recorded_hash.strip() == local_hash.strip():
            status = "[OK]"
            loc_disp = local_hash[:16] + "..."
        else:
            status = "[MISMATCH]"
            loc_disp = local_hash[:16] + "..."

    return status, rec_disp, loc_disp


def resolve_run(identifier: str) -> int:
    client = MlflowClient(tracking_uri=TRACKING_URI)

    target_run_id = identifier.strip()
    registry_info: dict[str, Any] = {}

    if target_run_id.lower() == "champion":
        try:
            mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
            target_run_id = mv.run_id
            registry_info["model_name"] = MODEL_NAME
            registry_info["alias"] = "champion"
            registry_info["version"] = mv.version
            registry_info["stage"] = mv.current_stage
        except Exception as e:
            print(
                f"[ERROR] Could not resolve alias 'champion' via MLflow Registry: {e}",
                file=sys.stderr,
            )
            return 1

    try:
        run = client.get_run(target_run_id)
    except Exception as e:
        print(f"[ERROR] Failed to load MLflow run '{target_run_id}': {e}", file=sys.stderr)
        return 1

    tags = run.data.tags
    params = run.data.params
    metrics = run.data.metrics

    print("=" * 80)
    print(f" SOLARCAST RUN LINEAGE RESOLVER: {target_run_id}")
    print("=" * 80)
    print(f"Run Name       : {run.info.run_name}")
    print(f"Experiment ID  : {run.info.experiment_id}")
    print(f"Status         : {run.info.status}")
    print(f"Start Time     : {run.info.start_time}")

    if registry_info:
        v_num = registry_info["version"]
        print(f"Registry Model : {registry_info['model_name']} (version {v_num})")
        print(f"Registry Alias : {registry_info['alias']}")

    # 1. Code Lineage & Git Status
    print("\n--- 1. CODE PROVENANCE & GIT REPOSITORY STATE ---")
    git_commit = tags.get("git_commit") or tags.get("mlflow.source.git.commit", "NOT RECORDED")
    git_dirty = tags.get("git_dirty") or tags.get("lineage_added_later.git_dirty", "NOT RECORDED")
    git_tree_note = tags.get("lineage_added_later.git_working_tree")

    print(f"  Git Commit SHA       : {git_commit}")
    print(f"  Working Tree Dirty   : {git_dirty}")
    if git_tree_note:
        print(f"  Provenance Note      : {git_tree_note}")

    # 2. Cryptographic Data Hashes
    print("\n--- 2. DATASET CRYPTOGRAPHIC INTEGRITY ---")
    data_checks = [
        (
            "Hourly Processed (hourly.parquet)",
            tags.get("data_hash_hourly") or tags.get("lineage_added_later.data_hash_hourly"),
            "data/processed/hourly.parquet",
        ),
        (
            "HRRR Weather Features (hrrr_forecast_features.parquet)",
            tags.get("data_hash_hrrr") or tags.get("lineage_added_later.data_hash_hrrr"),
            "data/processed/hrrr_forecast_features.parquet",
        ),
        (
            "Quarter-Hourly Power (quarter_hourly.parquet)",
            tags.get("data_hash_qh") or tags.get("lineage_added_later.data_hash_qh"),
            "data/processed/quarter_hourly.parquet",
        ),
        (
            "Feast Feature View Store (features_set_b.parquet)",
            tags.get("data_hash_feast") or tags.get("lineage_added_later.data_hash_feast"),
            "data/processed/feast/features_set_b.parquet",
        ),
        (
            "Data Pipeline Manifest (configs/data_manifest.json)",
            tags.get("data_manifest_hash") or tags.get("lineage_added_later.data_manifest_hash"),
            "configs/data_manifest.json",
        ),
    ]

    for label, rec_h, path_str in data_checks:
        status, rec_d, loc_d = check_hash(label, rec_h, path_str)
        print(f"  {status:<16} {label}")
        print(f"                   Logged : {rec_d}")
        print(f"                   Disk   : {loc_d}")

    # 3. Schema & Configuration Provenance
    print("\n--- 3. FEATURE SCHEMA & CONFIGURATION INTEGRITY ---")
    schema_ver = tags.get("feature_schema_version") or tags.get("schema_version", "NOT RECORDED")
    schema_hash = tags.get("feature_schema_hash") or tags.get(
        "lineage_added_later.feature_schema_hash"
    )

    s_status, s_rec, s_loc = check_hash(
        "feature_schema.json", schema_hash, "configs/feature_schema.json"
    )
    print(f"  Feature Schema Version : {schema_ver}")
    print(f"  {s_status:<16} configs/feature_schema.json")
    print(f"                   Logged : {s_rec}")
    print(f"                   Disk   : {s_loc}")

    # 4. Model Artifact / Bundle Provenance
    print("\n--- 4. MODEL BUNDLE CRYPTOGRAPHIC PROVENANCE ---")
    bundle_hash = tags.get("bundle_sha256") or tags.get("lineage_added_later.bundle_sha256")
    b_status, b_rec, b_loc = check_hash(
        "solarcast_f1_bundle.joblib", bundle_hash, "models/solarcast_f1_bundle.joblib"
    )
    print(f"  {b_status:<16} models/solarcast_f1_bundle.joblib")
    print(f"                   Logged : {b_rec}")
    print(f"                   Disk   : {b_loc}")

    # 5. Parameters
    print("\n--- 5. LOGGED PARAMETERS ---")
    if params:
        for k in sorted(params.keys()):
            print(f"  {k:<32}: {params[k]}")
    else:
        print("  (No parameters recorded)")

    # 6. Key Metrics
    print("\n--- 6. LOGGED METRICS ---")
    if metrics:
        for k in sorted(metrics.keys()):
            print(f"  {k:<32}: {metrics[k]}")
    else:
        print("  (No metrics recorded)")

    print("=" * 80)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve MLflow run or champion alias to cryptographic lineage."
    )
    parser.add_argument(
        "run_id",
        help="MLflow Run ID or 'champion' to resolve the registered champion model.",
    )
    args = parser.parse_args()
    sys.exit(resolve_run(args.run_id))


if __name__ == "__main__":
    main()
