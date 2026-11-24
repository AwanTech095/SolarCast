"""Configure MLflow Model Registry alias, tags, and descriptions for solarcast-f1.

Following Task Step 3:
- Tests set_registered_model_alias, tags and descriptions support in mlflow 2.17.2 local file store.
- Alias 'champion' on solarcast-f1 version 4.
- Version tags on v4: mode=F1, feature_set=B, schema_version=1.1.0, gate_eligible=true,
  final_test_opened=true, bundle_sha256, source_run_id.
- Tag v1-v3 with status=superseded.
- Descriptions for v1-v4 based strictly on run records.
- Outputs snapshot to reports/model_registry_status.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from mlflow.tracking import MlflowClient

TRACKING_URI = "file:./mlruns"
MODEL_NAME = "solarcast-f1"
CHAMPION_BUNDLE_SHA = "f72de152727fd406ac35a08cfa83289b6a61b031e0b3f4eb52008288e3980619"

VERSION_DESCRIPTIONS = {
    1: (
        "HistGradientBoosting Set A candidate (23 features). Cross-validation daylight MAE: "
        "32.24 kWh. Initial benchmark against ClearSkyPersistence baseline (+57.36% improvement)."
    ),
    2: (
        "HistGradientBoosting Set A candidate (23 features). Cross-validation daylight MAE: "
        "32.24 kWh. Evaluated against fair 15-minute Clear-Sky Scaled Persistence champion "
        "baseline (+23.56% improvement)."
    ),
    3: (
        "HistGradientBoosting Set B candidate (29 features) incorporating NOAA HRRR forecast "
        "weather features (TMP, TCDC). Cross-validation daylight MAE: 31.46 kWh "
        "(+25.25% vs baseline)."
    ),
    4: (
        "HistGradientBoosting Set B champion with sanitized weather range bounds (29 features). "
        "Cross-validation daylight MAE: 31.43 kWh (+25.33% vs champion baseline, +2.52% vs Set A). "
        "Final test evaluation daylight MAE: 32.69 kWh. "
        "Bundled as models/solarcast_f1_bundle.joblib."
    ),
}


def main() -> None:
    client = MlflowClient(tracking_uri=TRACKING_URI)
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    status_file = reports_dir / "model_registry_status.json"

    registry_report: dict = {
        "model_name": MODEL_NAME,
        "tracking_uri": TRACKING_URI,
        "backend": "local_file_store",
        "operations": {},
        "versions": {},
    }

    # 1. Alias 'champion' on version 4
    try:
        client.set_registered_model_alias(MODEL_NAME, "champion", "4")
        registry_report["operations"]["set_registered_model_alias"] = "SUCCESS"
        print("Successfully set alias 'champion' -> solarcast-f1 v4")
    except Exception as e:
        registry_report["operations"]["set_registered_model_alias"] = f"FAILED: {e}"
        print(f"Notice: set_registered_model_alias failed: {e}")

    # 2. Set tags on version 4 (champion)
    v4_tags = {
        "mode": "F1",
        "feature_set": "B",
        "schema_version": "1.1.0",
        "gate_eligible": "true",
        "final_test_opened": "true",
        "bundle_sha256": CHAMPION_BUNDLE_SHA,
        "source_run_id": "f7fef3f7bdf241c5a53dc50d8e4713f9",
        "status": "champion",
    }
    for k, v in v4_tags.items():
        client.set_model_version_tag(MODEL_NAME, "4", k, v)
    print("Successfully set version tags on solarcast-f1 v4")

    # 3. Set tags on versions 1-3 (superseded)
    for v in ["1", "2", "3"]:
        client.set_model_version_tag(MODEL_NAME, v, "status", "superseded")
    print("Successfully set status=superseded tags on solarcast-f1 v1-v3")

    # 4. Set descriptions for versions 1-4
    for v_num, desc in VERSION_DESCRIPTIONS.items():
        client.update_model_version(MODEL_NAME, str(v_num), description=desc)
        print(f"Successfully updated description for solarcast-f1 v{v_num}")

    # 5. Read back status from registry
    for v_num in [1, 2, 3, 4]:
        mv = client.get_model_version(MODEL_NAME, str(v_num))
        registry_report["versions"][f"v{v_num}"] = {
            "version": mv.version,
            "run_id": mv.run_id,
            "current_stage": mv.current_stage,
            "aliases": getattr(mv, "aliases", []),
            "tags": dict(mv.tags) if mv.tags else {},
            "description": mv.description,
        }

    status_file.write_text(json.dumps(registry_report, indent=2), encoding="utf-8")
    print(f"\nModel registry status saved to {status_file}")


if __name__ == "__main__":
    main()
