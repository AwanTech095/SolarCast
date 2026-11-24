"""Retroactively tag existing MLflow runs with verified lineage tags and explicit unverified flags.

Following Task Step 2:
- Champion run: compare sha256 of its logged bundle artifact with models/solarcast_f1_bundle.joblib;
  tag bundle_sha256 only if they match.
- Tag current data/schema/manifest hashes only where the run's own logged hashes equal
  the file's current hash.
- Every added tag gets the prefix "lineage_added_later." and a tag lineage_added_later_at (UTC ISO).
- Where a value cannot be verified write "unverified" and leave the real value out.
- For the champion run, document that its git commit was logged from an uncommitted working tree.
"""

from __future__ import annotations

import datetime
import hashlib
from pathlib import Path

from mlflow.tracking import MlflowClient

TRACKING_URI = "file:./mlruns"

CHAMPION_RUN_ID = "f7fef3f7bdf241c5a53dc50d8e4713f9"
FINAL_TEST_RUN_ID = "620faa341a8a4c14824f31e40b157fd6"
DEVIATION_RUN_ID = "cce94db3448e405ab0fee01415e82da1"
MONITORING_RUN_ID = "16ffd7429be642928941e95f932fe8b3"
MATERIALIZE_RUN_ID = "56f349ec19f64a62917396b79a8e9fa9"


def compute_sha256(path: Path | str) -> str:
    p = Path(path)
    if not p.is_file():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    client = MlflowClient(tracking_uri=TRACKING_URI)
    now_utc = datetime.datetime.now(datetime.UTC).isoformat()

    # 1. Compute current disk hashes
    disk_bundle_sha = compute_sha256("models/solarcast_f1_bundle.joblib")
    hourly_sha = compute_sha256("data/processed/hourly.parquet")
    hrrr_sha = compute_sha256("data/processed/hrrr_forecast_features.parquet")
    qh_sha = compute_sha256("data/processed/quarter_hourly.parquet")

    # 2. Champion Run
    champ_bundle_p = Path(
        "mlruns/754448501154260354/f7fef3f7bdf241c5a53dc50d8e4713f9/artifacts/bundle/solarcast_f1_bundle.joblib"
    )
    champ_artifact_bundle_sha = compute_sha256(champ_bundle_p)

    champ_tags: dict[str, str] = {
        "lineage_added_later_at": now_utc,
    }

    if champ_artifact_bundle_sha and champ_artifact_bundle_sha == disk_bundle_sha:
        champ_tags["lineage_added_later.bundle_sha256"] = champ_artifact_bundle_sha
    else:
        champ_tags["lineage_added_later.bundle_sha256"] = "unverified"

    # Verify run's logged data hashes against current disk hashes
    # Run f7fef3 logged data_hash_hourly = f9157a..., data_hash_hrrr = fbc02d...
    if hourly_sha == "f9157a5dd5c4378823880c43da6e586c046d09debab99d99be20cf9041105cb6":
        champ_tags["lineage_added_later.data_hash_hourly"] = hourly_sha
    else:
        champ_tags["lineage_added_later.data_hash_hourly"] = "unverified"

    if hrrr_sha == "fbc02d8bbbbdb9de818fbdd607cf9e54d7c7177e69d4ebb6d04a91e66360024f":
        champ_tags["lineage_added_later.data_hash_hrrr"] = hrrr_sha
    else:
        champ_tags["lineage_added_later.data_hash_hrrr"] = "unverified"

    if qh_sha == "7c488a111f2eb7b5502f33d3c377bd5f4414b9e06753171efac7771f8c6cb57a":
        champ_tags["lineage_added_later.data_hash_qh"] = qh_sha
    else:
        champ_tags["lineage_added_later.data_hash_qh"] = "unverified"

    # Manifest and schema hashes were not logged by the run at execution time
    champ_tags["lineage_added_later.data_manifest_hash"] = "unverified"
    champ_tags["lineage_added_later.feature_schema_hash"] = "unverified"
    champ_tags["lineage_added_later.config_hashes"] = "unverified"

    # Working tree status at run time (commit 4eb8276 had uncommitted changes committed in 0afc5c6)
    champ_tags["lineage_added_later.git_dirty"] = "true"
    champ_tags["lineage_added_later.git_working_tree"] = (
        "uncommitted_working_tree_at_execution; changes subsequent to 4eb8276 committed in 0afc5c6"
    )

    print(f"Tagging Champion Run ({CHAMPION_RUN_ID})...")
    for k, v in champ_tags.items():
        client.set_tag(CHAMPION_RUN_ID, k, v)
        print(f"  {k} = {v}")

    # 3. Final Test Evaluation Run
    final_test_tags: dict[str, str] = {
        "lineage_added_later_at": now_utc,
        "lineage_added_later.bundle_sha256": "unverified",
        "lineage_added_later.data_hashes": "unverified",
        "lineage_added_later.data_manifest_hash": "unverified",
        "lineage_added_later.feature_schema_hash": "unverified",
        "lineage_added_later.git_dirty": "unverified",
    }
    print(f"\nTagging Final Test Run ({FINAL_TEST_RUN_ID})...")
    for k, v in final_test_tags.items():
        client.set_tag(FINAL_TEST_RUN_ID, k, v)
        print(f"  {k} = {v}")

    # 4. Deviation Detector Run
    deviation_tags: dict[str, str] = {
        "lineage_added_later_at": now_utc,
        "lineage_added_later.bundle_sha256": "unverified",
        "lineage_added_later.data_hashes": "unverified",
        "lineage_added_later.data_manifest_hash": "unverified",
        "lineage_added_later.feature_schema_hash": "unverified",
        "lineage_added_later.git_commit": "unverified",
        "lineage_added_later.git_dirty": "unverified",
    }
    print(f"\nTagging Deviation Detector Run ({DEVIATION_RUN_ID})...")
    for k, v in deviation_tags.items():
        client.set_tag(DEVIATION_RUN_ID, k, v)
        print(f"  {k} = {v}")

    # 5. Monitoring Replay Run
    monitoring_tags: dict[str, str] = {
        "lineage_added_later_at": now_utc,
        "lineage_added_later.bundle_sha256": "unverified",
        "lineage_added_later.data_hashes": "unverified",
        "lineage_added_later.data_manifest_hash": "unverified",
        "lineage_added_later.feature_schema_hash": "unverified",
        "lineage_added_later.git_commit": "unverified",
        "lineage_added_later.git_dirty": "unverified",
    }
    print(f"\nTagging Monitoring Replay Run ({MONITORING_RUN_ID})...")
    for k, v in monitoring_tags.items():
        client.set_tag(MONITORING_RUN_ID, k, v)
        print(f"  {k} = {v}")

    # 6. Feature Store Materialization Run
    materialize_tags: dict[str, str] = {
        "lineage_added_later_at": now_utc,
        "lineage_added_later.bundle_sha256": "unverified",
        "lineage_added_later.data_hashes": "unverified",
        "lineage_added_later.data_manifest_hash": "unverified",
        "lineage_added_later.feature_schema_hash": "unverified",
        "lineage_added_later.git_dirty": "unverified",
    }
    print(f"\nTagging Feature Store Materialization Run ({MATERIALIZE_RUN_ID})...")
    for k, v in materialize_tags.items():
        client.set_tag(MATERIALIZE_RUN_ID, k, v)
        print(f"  {k} = {v}")

    print("\nAll 5 runs successfully tagged with verified/unverified lineage provenance.")


if __name__ == "__main__":
    main()
