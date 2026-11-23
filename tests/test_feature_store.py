"""Tests for Feast feature store integration (Task Step 5).

Verifies:
1. Feature table schema and data types match configs/feature_schema.json.
2. Entity key integrity (site_id == "2107").
3. Monotonicity of event timestamps per site.
4. Operational timing audit columns <= event_timestamp (no lookahead leakage).
5. Weather NaN values are preserved correctly.
6. Offline point-in-time retrieval correctness.
7. Online retrieval after materialization.
8. Offline/online parity (zero training-serving skew).
9. Stale-feature rejection.
10. Future-feature rejection.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Skip this entire test module if Feast is not installed (e.g. running in .venv)
pytest.importorskip("feast")

from feature_store.materialize import materialize_features  # noqa: E402
from src.features.feast_provider import (  # noqa: E402
    FutureFeatureError,
    StaleFeatureError,
    get_historical_features,
    get_online_features,
)
from src.features.preprocessing import get_feature_names  # noqa: E402

FEATURE_PARQUET_PATH = Path("data/processed/feast/features_set_b.parquet")
SCHEMA_JSON_PATH = Path("configs/feature_schema.json")
REPO_PATH = Path("feature_store/feature_repo")


@pytest.fixture(scope="module")
def feature_table() -> pd.DataFrame:
    """Load the Feast feature table Parquet file."""
    assert FEATURE_PARQUET_PATH.exists(), f"Missing {FEATURE_PARQUET_PATH}"
    return pd.read_parquet(FEATURE_PARQUET_PATH)


@pytest.fixture(scope="module")
def schema_spec() -> dict:
    """Load feature schema JSON specification."""
    assert SCHEMA_JSON_PATH.exists(), f"Missing {SCHEMA_JSON_PATH}"
    with open(SCHEMA_JSON_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_schema_and_dtypes(feature_table: pd.DataFrame, schema_spec: dict):
    """Verify that schema and dtypes match configs/feature_schema.json."""
    set_b_spec = schema_spec["feature_sets"]["B"]["features"]
    expected_features = {f["name"]: f["dtype"] for f in set_b_spec}

    # Verify all 29 features exist in the parquet file
    for name, expected_dtype in expected_features.items():
        assert name in feature_table.columns, f"Feature {name} missing from Parquet table"
        actual_dtype = str(feature_table[name].dtype)
        if expected_dtype == "float64":
            assert actual_dtype in ("float64", "float32", "Float64"), (
                f"{name} dtype mismatch: expected {expected_dtype}, got {actual_dtype}"
            )
        elif expected_dtype == "int64":
            assert actual_dtype in ("int64", "int32", "Int64"), (
                f"{name} dtype mismatch: expected {expected_dtype}, got {actual_dtype}"
            )

    # Verify primary key and audit metadata columns
    assert "site_id" in feature_table.columns
    assert "event_timestamp" in feature_table.columns
    assert "created_timestamp" in feature_table.columns
    assert "production_cutoff_utc" in feature_table.columns
    assert "weather_available_at_utc" in feature_table.columns
    assert "source_event_time_max_utc" in feature_table.columns


def test_entity_key(feature_table: pd.DataFrame):
    """Verify entity key consistency (all rows have site_id == '2107')."""
    assert (feature_table["site_id"] == "2107").all()
    dtype_str = str(feature_table["site_id"].dtype)
    assert feature_table["site_id"].dtype == object or dtype_str == "string"


def test_event_timestamps_monotonic(feature_table: pd.DataFrame):
    """Verify event timestamps are strictly monotonically increasing."""
    ts = pd.to_datetime(feature_table["event_timestamp"])
    assert ts.is_monotonic_increasing, "event_timestamp is not monotonically increasing"
    assert ts.dt.tz is not None, "event_timestamp must be timezone-aware (UTC)"


def test_availability_columns_lte_event_timestamp(feature_table: pd.DataFrame):
    """Verify no-lookahead condition: all audit timestamps <= event_timestamp."""
    event_ts = pd.to_datetime(feature_table["event_timestamp"])
    prod_cutoff = pd.to_datetime(feature_table["production_cutoff_utc"])
    source_max = pd.to_datetime(feature_table["source_event_time_max_utc"])

    # Production cutoff must always be <= event_timestamp
    assert (prod_cutoff <= event_ts).all(), "Leakage: production_cutoff_utc > event_timestamp"

    # Source event time max must always be <= event_timestamp
    assert (source_max <= event_ts).all(), "Leakage: source_event_time_max_utc > event_timestamp"

    # Weather available timestamp for valid weather rows must be <= event_timestamp
    weather_avail = pd.to_datetime(feature_table["weather_available_at_utc"])
    valid_weather = weather_avail.notna()
    assert (
        weather_avail[valid_weather] <= event_ts[valid_weather]
    ).all(), "Leakage: weather_available_at_utc > event_timestamp"


def test_nan_weather_preserved(feature_table: pd.DataFrame):
    """Verify NaN weather values are preserved and flagged when weather is missing or invalid."""
    # Completely missing weather (unobtainable HRRR runs, 962 rows)
    completely_missing = feature_table["weather_available_at_utc"].isna()
    assert completely_missing.sum() == 962
    assert feature_table.loc[completely_missing, "tcdc_pct"].isna().all()
    assert feature_table.loc[completely_missing, "tmp_c"].isna().all()
    assert feature_table.loc[completely_missing, "clearsky_ghi_cloud_adj_wm2"].isna().all()

    # Out-of-bounds tcdc fill value fix (-10000 on 2020-07-29 21:00 UTC) must be preserved as NaN
    fill_row = feature_table[feature_table["event_timestamp"] == "2020-07-29T21:00:00+00:00"]
    assert len(fill_row) == 1
    assert np.isnan(fill_row["tcdc_pct"].values[0]), "tcdc_pct was not preserved as NaN"
    assert fill_row["weather_missing"].values[0] == 1, "weather_missing flag not set"

    # In total, 963 rows have NaN for tcdc_pct
    assert feature_table["tcdc_pct"].isna().sum() == 963


def test_offline_retrieval_correct(feature_table: pd.DataFrame):
    """Verify point-in-time correctness of get_historical_features vs offline Parquet table."""
    sample_df = (
        feature_table.sample(20, random_state=123)
        .sort_values("event_timestamp")
        .reset_index(drop=True)
    )
    entity_df = pd.DataFrame({
        "site_id": sample_df["site_id"],
        "event_timestamp": pd.to_datetime(sample_df["event_timestamp"]),
    })

    feature_names = get_feature_names("B")
    hist_df = get_historical_features(entity_df=entity_df, repo_path=REPO_PATH)
    hist_df = hist_df.sort_values("event_timestamp").reset_index(drop=True)

    for col in feature_names:
        raw_vals = sample_df[col].values
        hist_vals = hist_df[col].values

        nan_raw = np.isnan(raw_vals)
        nan_hist = np.isnan(hist_vals)
        assert (nan_raw == nan_hist).all(), f"NaN discrepancy in {col}"

        valid = ~nan_raw
        if valid.any():
            diff = np.abs(raw_vals[valid] - hist_vals[valid])
            max_d = float(np.max(diff))
            assert max_d < 1e-9, f"Feature {col} exceeded 1e-9 tolerance (max diff {max_d})"


def test_online_retrieval_and_parity(feature_table: pd.DataFrame):
    """Verify online materialization, online retrieval, and offline/online parity."""
    # Materialize a test window around 2023-06-15 15:00 UTC
    target_time_str = "2023-06-15T15:00:00+00:00"
    target_dt = datetime.datetime.fromisoformat(target_time_str)
    start_dt = target_dt - datetime.timedelta(days=2)

    materialize_features(
        start_date=start_dt,
        end_date=target_dt,
        repo_path=REPO_PATH,
    )

    # Query online features
    online_res = get_online_features(
        site_id="2107",
        issue_time_utc=target_dt,
        max_staleness_hours=2.0,
        repo_path=REPO_PATH,
    )

    assert online_res["site_id"] == "2107"
    assert online_res["event_timestamp"] == target_dt
    assert online_res["staleness_hours"] == 0.0

    # Verify parity against offline parquet table for this exact hour
    target_row = feature_table[feature_table["event_timestamp"] == target_time_str].iloc[0]
    feature_names = get_feature_names("B")

    for feat in feature_names:
        offline_val = target_row[feat]
        online_val = online_res[feat]
        if np.isnan(offline_val):
            assert online_val is None or np.isnan(online_val)
        else:
            diff = abs(float(offline_val) - float(online_val))
            assert diff < 1e-9, f"Parity mismatch for {feat}: {offline_val} vs {online_val}"


def test_stale_feature_rejection():
    """Verify get_online_features rejects queries where stored features are stale."""
    # Last materialized record is 2023-06-15 15:00 UTC
    query_time = datetime.datetime.fromisoformat("2023-06-15T20:00:00+00:00")  # 5 hours later

    with pytest.raises(StaleFeatureError) as exc_info:
        get_online_features(
            site_id="2107",
            issue_time_utc=query_time,
            max_staleness_hours=2.0,
            repo_path=REPO_PATH,
        )
    assert "exceeds max_staleness_hours=2.0h" in str(exc_info.value)


def test_future_feature_rejection():
    """Verify get_online_features rejects queries when stored record is from the future."""
    # Stored record is 2023-06-15 15:00 UTC, query asks for 11:00 UTC
    query_time = datetime.datetime.fromisoformat("2023-06-15T11:00:00+00:00")

    with pytest.raises(FutureFeatureError) as exc_info:
        get_online_features(
            site_id="2107",
            issue_time_utc=query_time,
            max_staleness_hours=24.0,
            repo_path=REPO_PATH,
        )
    assert "is later than requested issue_time_utc" in str(exc_info.value)
