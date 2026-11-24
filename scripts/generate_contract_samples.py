"""Generate real sample contract files for Member 3 and CI test suites (Step 3).

Generates:
1. docs/samples/predict_requests.json (5 real examples, including missing weather and stale)
2. docs/samples/predict_responses.json (5 matching responses with predictions and HTTP statuses)
3. tests/fixtures/features_sample.csv (48 rows of features_set_b under 200 KB with attribution)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.preprocessing import build_features  # noqa: E402
from src.models.bundle import load_bundle  # noqa: E402


def main() -> int:
    bundle = load_bundle("models/solarcast_f1_bundle.joblib")
    features_df = pd.read_parquet("data/processed/feast/features_set_b.parquet")
    hourly = pd.read_parquet("data/processed/hourly.parquet")
    qh = pd.read_parquet("data/processed/quarter_hourly.parquet")
    weather = pd.read_parquet("data/processed/hrrr_forecast_features.parquet")

    # Ensure output directories exist
    samples_dir = Path("docs/samples")
    samples_dir.mkdir(parents=True, exist_ok=True)
    fixtures_dir = Path("tests/fixtures")
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    # Example 1: Morning clear sky (2023-06-15 15:00 UTC)
    row1 = features_df[features_df["event_timestamp"] == "2023-06-15 15:00:00+00:00"].iloc[0]

    # Example 2: Peak solar noon clear sky (2023-06-15 19:00 UTC)
    row2 = features_df[features_df["event_timestamp"] == "2023-06-15 19:00:00+00:00"].iloc[0]

    # Example 3: Nighttime hour (2023-06-15 06:00 UTC, solar elevation < 0)
    night_times = pd.DatetimeIndex(["2023-06-15 06:00:00+00:00"])
    row3_df = build_features(night_times, hourly, qh, weather_df=weather, feature_set="B")
    row3 = row3_df.iloc[0]

    # Example 4: Daylight with missing weather (2023-08-16 16:00 UTC)
    row4 = features_df[features_df["event_timestamp"] == "2023-08-16 16:00:00+00:00"].iloc[0]

    # Example 5: Stale request (2023-06-15 20:00 UTC against 15:00 UTC features, age=5.0h)
    stale_issue_time = "2023-06-15T20:00:00+00:00"

    requests_data = []
    responses_data = []

    def extract_features_dict(row: pd.Series) -> dict[str, float | int | None]:
        feat_dict = {}
        for col in bundle.feature_names:
            val = row[col]
            if pd.isna(val):
                feat_dict[col] = None
            elif isinstance(val, np.integer | int):
                feat_dict[col] = int(val)
            else:
                feat_dict[col] = float(val)
        return feat_dict

    # Case 1
    feats1 = extract_features_dict(row1)
    df1 = pd.DataFrame([feats1])
    pred1 = float(bundle.predict(df1)[0])
    req1 = {
        "example_id": "example_1_clear_morning",
        "description": "Morning daylight hour under clear sky conditions",
        "site_id": "2107",
        "issue_time_utc": "2023-06-15T15:00:00+00:00",
        "target_interval_start_utc": "2023-06-15T15:00:00+00:00",
        "target_interval_end_utc": "2023-06-15T16:00:00+00:00",
        "features": feats1,
    }
    resp1 = {
        "example_id": "example_1_clear_morning",
        "status_code": 200,
        "site_id": "2107",
        "issue_time_utc": "2023-06-15T15:00:00+00:00",
        "interval_start_utc": "2023-06-15T15:00:00+00:00",
        "interval_end_utc": "2023-06-15T16:00:00+00:00",
        "predicted_energy_kwh": pred1,
        "model_name": bundle.model_name,
        "model_version": bundle.feature_schema_version,
        "feature_schema_version": bundle.feature_schema_version,
        "feature_staleness_hours": 0.0,
        "warnings": [],
    }
    requests_data.append(req1)
    responses_data.append(resp1)

    # Case 2
    feats2 = extract_features_dict(row2)
    df2 = pd.DataFrame([feats2])
    pred2 = float(bundle.predict(df2)[0])
    req2 = {
        "example_id": "example_2_clear_solar_noon",
        "description": "Solar noon peak irradiance under clear sky conditions",
        "site_id": "2107",
        "issue_time_utc": "2023-06-15T19:00:00+00:00",
        "target_interval_start_utc": "2023-06-15T19:00:00+00:00",
        "target_interval_end_utc": "2023-06-15T20:00:00+00:00",
        "features": feats2,
    }
    resp2 = {
        "example_id": "example_2_clear_solar_noon",
        "status_code": 200,
        "site_id": "2107",
        "issue_time_utc": "2023-06-15T19:00:00+00:00",
        "interval_start_utc": "2023-06-15T19:00:00+00:00",
        "interval_end_utc": "2023-06-15T20:00:00+00:00",
        "predicted_energy_kwh": pred2,
        "model_name": bundle.model_name,
        "model_version": bundle.feature_schema_version,
        "feature_schema_version": bundle.feature_schema_version,
        "feature_staleness_hours": 0.0,
        "warnings": [],
    }
    requests_data.append(req2)
    responses_data.append(resp2)

    # Case 3
    feats3 = extract_features_dict(row3)
    df3 = pd.DataFrame([feats3])
    pred3 = float(bundle.predict(df3)[0])
    req3 = {
        "example_id": "example_3_nighttime",
        "description": "Nighttime hour with solar elevation below horizon (elevation <= 0)",
        "site_id": "2107",
        "issue_time_utc": "2023-06-15T06:00:00+00:00",
        "target_interval_start_utc": "2023-06-15T06:00:00+00:00",
        "target_interval_end_utc": "2023-06-15T07:00:00+00:00",
        "features": feats3,
    }
    resp3 = {
        "example_id": "example_3_nighttime",
        "status_code": 200,
        "site_id": "2107",
        "issue_time_utc": "2023-06-15T06:00:00+00:00",
        "interval_start_utc": "2023-06-15T06:00:00+00:00",
        "interval_end_utc": "2023-06-15T07:00:00+00:00",
        "predicted_energy_kwh": pred3,
        "model_name": bundle.model_name,
        "model_version": bundle.feature_schema_version,
        "feature_schema_version": bundle.feature_schema_version,
        "feature_staleness_hours": 0.0,
        "warnings": ["night_forced_zero"],
    }
    requests_data.append(req3)
    responses_data.append(resp3)

    # Case 4
    feats4 = extract_features_dict(row4)
    df4 = pd.DataFrame([feats4])
    pred4 = float(bundle.predict(df4)[0])
    req4 = {
        "example_id": "example_4_weather_missing",
        "description": "Daylight hour with missing NOAA HRRR forecast weather",
        "site_id": "2107",
        "issue_time_utc": "2023-08-16T16:00:00+00:00",
        "target_interval_start_utc": "2023-08-16T16:00:00+00:00",
        "target_interval_end_utc": "2023-08-16T17:00:00+00:00",
        "features": feats4,
    }
    resp4 = {
        "example_id": "example_4_weather_missing",
        "status_code": 200,
        "site_id": "2107",
        "issue_time_utc": "2023-08-16T16:00:00+00:00",
        "interval_start_utc": "2023-08-16T16:00:00+00:00",
        "interval_end_utc": "2023-08-16T17:00:00+00:00",
        "predicted_energy_kwh": pred4,
        "model_name": bundle.model_name,
        "model_version": bundle.feature_schema_version,
        "feature_schema_version": bundle.feature_schema_version,
        "feature_staleness_hours": 0.0,
        "warnings": ["weather_missing"],
    }
    requests_data.append(req4)
    responses_data.append(resp4)

    # Case 5
    msg5 = (
        "Online store features are stale for issue_time_utc=2023-06-15T20:00:00+00:00. "
        "Feature event_timestamp=2023-06-15T15:00:00+00:00 is 5.00 hours old "
        "(exceeds max_staleness_hours=2.0h)."
    )
    req5 = {
        "example_id": "example_5_stale_rejection",
        "description": "Inference request rejected because online store features are stale",
        "site_id": "2107",
        "issue_time_utc": stale_issue_time,
        "target_interval_start_utc": "2023-06-15T20:00:00+00:00",
        "target_interval_end_utc": "2023-06-15T21:00:00+00:00",
        "max_staleness_hours": 2.0,
    }
    resp5 = {
        "example_id": "example_5_stale_rejection",
        "status_code": 503,
        "site_id": "2107",
        "issue_time_utc": stale_issue_time,
        "interval_start_utc": "2023-06-15T20:00:00+00:00",
        "interval_end_utc": "2023-06-15T21:00:00+00:00",
        "error": "StaleFeatureError",
        "message": msg5,
        "feature_staleness_hours": 5.0,
        "warnings": ["stale_features"],
    }
    requests_data.append(req5)
    responses_data.append(resp5)

    req_path = samples_dir / "predict_requests.json"
    with open(req_path, "w", encoding="utf-8") as f:
        json.dump(requests_data, f, indent=2)
    print(f"Saved {len(requests_data)} request samples -> {req_path}")

    resp_path = samples_dir / "predict_responses.json"
    with open(resp_path, "w", encoding="utf-8") as f:
        json.dump(responses_data, f, indent=2)
    print(f"Saved {len(responses_data)} response samples -> {resp_path}")

    # Generate tests/fixtures/features_sample.csv (48 rows)
    time_cutoff = "2023-06-13 13:00:00+00:00"
    sample_48 = features_df[features_df["event_timestamp"] >= time_cutoff].head(48).copy()
    attribution_str = (
        "PVDAQ Site 2107 (NREL DOI 10.25984/1846021) & NOAA HRRR MesoWest AWS Zarr"
    )
    sample_48["attribution"] = attribution_str

    csv_path = fixtures_dir / "features_sample.csv"
    sample_48.to_csv(csv_path, index=False)
    file_bytes = csv_path.stat().st_size
    print(f"Saved 48 rows fixture ({file_bytes} bytes, {file_bytes / 1024:.2f} KB) -> {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
