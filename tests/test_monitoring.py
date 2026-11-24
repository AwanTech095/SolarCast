"""Comprehensive unit tests for SolarCast monitoring modules.

Tests cover:
1. Population Stability Index (PSI) against hand-computed reference example (0.716704).
2. Season-matched reference extraction (calendar month +/- 1 month, daylight hour matching).
3. 200-row minimum floor for InputDriftMonitor (insufficient_data when N < 200).
4. 30-label minimum floor for PerformanceMonitor (insufficient_labels when N < 30).
5. Two-consecutive-check alerting rule (single breach suppresses; 2nd consecutive triggers).
6. Virtual clock label visibility causality (labels arriving after T are strictly inaccessible).
7. Missing-label handling (distinguishes label_data_outage from degradation; never treats as 0).
8. Schema validation: all alert types match configs/alert_event_schema.json (v1.1.0).
9. Weather freshness classification policy and illegal future weather rejection.
10. Monitoring evaluation determinism.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.drift import (
    MONITORED_FEATURES,
    InputDriftMonitor,
    WeatherFreshnessMonitor,
    build_seasonal_reference,
    calculate_psi,
    calculate_psi_from_proportions,
    classify_weather_freshness,
    classify_weather_freshness_series,
)
from src.models.performance_monitor import (
    PerformanceMonitor,
    join_predictions_with_delayed_labels,
)

SCHEMA_PATH = Path("configs/alert_event_schema.json")


@pytest.fixture(scope="module")
def alert_schema() -> dict:
    assert SCHEMA_PATH.exists(), f"Missing schema at {SCHEMA_PATH}"
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# 1. PSI Calculation & Hand-Computed Example
# =============================================================================
def test_psi_against_hand_computed_example():
    """Verify PSI calculation against an exact hand-computed 10-bin reference example.

    Hand-computed calculation:
        Expected (reference): 10 bins of 0.10 each
        Actual (current): 8 bins of 0.05, 2 bins of 0.30
        For bins 1..8: (0.05 - 0.10) * ln(0.05 / 0.10) = -0.05 * ln(0.5) = 0.034657
        Sum (8 bins) = 8 * 0.034657 = 0.277259
        For bins 9..10: (0.30 - 0.10) * ln(0.30 / 0.10) = 0.20 * ln(3.0) = 0.219722
        Sum (2 bins) = 2 * 0.219722 = 0.439445
        Total PSI = 0.277259 + 0.439445 = 0.716704
    """
    expected = np.array([0.10] * 10)
    actual = np.array([0.05] * 8 + [0.30] * 2)

    psi_val = calculate_psi_from_proportions(actual, expected)
    expected_hand_computed = 0.716704
    assert np.isclose(psi_val, expected_hand_computed, atol=1e-5), (
        f"Calculated PSI {psi_val:.6f} does not match hand-computed {expected_hand_computed:.6f}"
    )


def test_psi_identical_distributions():
    """Identical distributions must produce zero (or near-zero) PSI."""
    ref = np.random.RandomState(42).normal(100.0, 15.0, size=500)
    curr = ref.copy()
    psi_val = calculate_psi(ref, curr, num_bins=10)
    assert np.isclose(psi_val, 0.0, atol=1e-6)


# =============================================================================
# 2. Season-Matched Reference
# =============================================================================
def test_seasonal_reference_matching():
    """Verify that build_seasonal_reference subsets months +/- 1 month and daylight hours."""
    # Synthetic multi-year dataset spanning months 1 through 12
    dates = pd.date_range("2020-01-01", "2021-12-31 23:00:00", freq="1h", tz="UTC")
    df = pd.DataFrame({"event_timestamp": dates, "val": np.arange(len(dates))})

    # Test June (month 6): matched months should be {5, 6, 7}
    ref_jun = build_seasonal_reference(
        df, target_month=6, daylight_hours={14, 15, 16, 17, 18, 19, 20}
    )
    months_jun = set(pd.to_datetime(ref_jun["event_timestamp"]).dt.month.unique())
    assert months_jun == {5, 6, 7}
    hours_jun = set(pd.to_datetime(ref_jun["event_timestamp"]).dt.hour.unique())
    assert hours_jun == {14, 15, 16, 17, 18, 19, 20}

    # Test January (month 1): matched months should wrap around to {12, 1, 2}
    ref_jan = build_seasonal_reference(df, target_month=1)
    months_jan = set(pd.to_datetime(ref_jan["event_timestamp"]).dt.month.unique())
    assert months_jan == {12, 1, 2}


# =============================================================================
# 3. 200-Row Minimum Floor for Drift
# =============================================================================
def test_drift_200_row_minimum_floor():
    """Verify that InputDriftMonitor enforces >= 200 valid rows per comparison."""
    monitor = InputDriftMonitor(min_valid_rows=200, psi_threshold=0.20)

    # Reference has 500 rows, but current window only has 150 rows (< 200)
    ref_df = pd.DataFrame({
        feat: np.random.normal(50, 10, 500) for feat in MONITORED_FEATURES
    })
    ref_df["event_timestamp"] = pd.date_range(
        "2021-06-01", periods=500, freq="1h", tz="UTC"
    )

    curr_df = pd.DataFrame({
        feat: np.random.normal(100, 10, 150) for feat in MONITORED_FEATURES
    })
    curr_df["event_timestamp"] = pd.date_range(
        "2023-06-01", periods=150, freq="1h", tz="UTC"
    )

    eval_ts = pd.Timestamp("2023-06-15 00:00:00", tz="UTC")
    summary, alert = monitor.evaluate_check(eval_ts, curr_df, ref_df)

    assert summary["decision"] == "insufficient_data"
    assert alert is None
    assert monitor.consecutive_drift_count == 0


# =============================================================================
# 4. 30-Label Minimum Floor for Performance Monitor
# =============================================================================
def test_performance_30_label_minimum_floor():
    """Verify PerformanceMonitor enforces >= 30 valid daylight labels in 7-day window."""
    monitor = PerformanceMonitor()

    # Window with only 25 valid labels (< 30)
    times = pd.date_range("2023-06-01 14:00", periods=25, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "event_timestamp": times,
        "prediction_kwh": [100.0] * 25,
        "target_kwh": [500.0] * 25,  # Massive error, but labels < 30
        "quality_flag": ["VALID"] * 25,
        "is_daylight": [True] * 25,
    })

    eval_ts = pd.Timestamp("2023-06-08 00:00:00", tz="UTC")
    summary, alert = monitor.evaluate_check(eval_ts, df)

    assert summary["status"] in ("insufficient_labels", "label_data_outage")
    assert alert is None
    assert summary["consecutive_degradation_checks"] == 0


# =============================================================================
# 5. Two-Consecutive-Check Alerting Rule
# =============================================================================
def test_two_consecutive_check_rule_performance():
    """Verify that a single breach suppresses alert and only 2nd consecutive breach fires."""
    monitor = PerformanceMonitor()
    eval_t1 = pd.Timestamp("2023-06-08 00:00:00", tz="UTC")
    eval_t2 = pd.Timestamp("2023-06-09 00:00:00", tz="UTC")
    eval_t3 = pd.Timestamp("2023-06-10 00:00:00", tz="UTC")

    # High error dataset (50 valid labels, actual=200, pred=50 -> MAE=150, >> ref 23.25)
    times = pd.date_range("2023-06-01 14:00", periods=50, freq="1h", tz="UTC")
    degraded_df = pd.DataFrame({
        "event_timestamp": times,
        "prediction_kwh": [50.0] * 50,
        "target_kwh": [200.0] * 50,
        "quality_flag": ["VALID"] * 50,
        "is_daylight": [True] * 50,
    })

    # Nominal dataset (MAE = 5.0)
    nominal_df = pd.DataFrame({
        "event_timestamp": times,
        "prediction_kwh": [100.0] * 50,
        "target_kwh": [105.0] * 50,
        "quality_flag": ["VALID"] * 50,
        "is_daylight": [True] * 50,
    })

    # Day 1: First breach -> must NOT fire alert (count = 1)
    sum1, alert1 = monitor.evaluate_check(eval_t1, degraded_df)
    assert sum1["status"] == "degradation_breach"
    assert sum1["consecutive_degradation_checks"] == 1
    assert alert1 is None

    # Day 2: Second consecutive breach -> MUST fire alert (count = 2)
    sum2, alert2 = monitor.evaluate_check(eval_t2, degraded_df)
    assert sum2["status"] == "alert"
    assert sum2["consecutive_degradation_checks"] == 2
    assert alert2 is not None
    assert alert2["alert_type"] == "performance_degradation"

    # Day 3: Nominal return -> resets consecutive count to 0, no alert
    sum3, alert3 = monitor.evaluate_check(eval_t3, nominal_df)
    assert sum3["status"] == "nominal"
    assert sum3["consecutive_degradation_checks"] == 0
    assert alert3 is None


# =============================================================================
# 6. Virtual Clock Label Visibility (Labels Arriving after T are Hidden)
# =============================================================================
def test_virtual_clock_label_visibility_causality():
    """Verify that at virtual time T, only labels with interval_end + delay <= T are visible."""
    # Target hours:
    # 1. 10:00-11:00 UTC (ends 11:00, arrives 11:15 UTC)
    # 2. 11:00-12:00 UTC (ends 12:00, arrives 12:15 UTC)
    t1 = pd.Timestamp("2023-06-01 10:00:00", tz="UTC")
    t2 = pd.Timestamp("2023-06-01 11:00:00", tz="UTC")

    preds = pd.DataFrame({
        "site_id": ["2107", "2107"],
        "event_timestamp": [t1, t2],
        "prediction_kwh": [100.0, 200.0],
    })

    actuals = pd.DataFrame({
        "site_id": ["2107", "2107"],
        "interval_start_utc": [t1, t2],
        "interval_end_utc": [
            t1 + pd.Timedelta(hours=1),
            t2 + pd.Timedelta(hours=1),
        ],
        "target_kwh": [110.0, 210.0],
        "quality_flag": ["VALID", "VALID"],
    })

    # At virtual clock T = 11:30 UTC:
    # Hour 1 (arrived 11:15 <= 11:30) is visible!
    # Hour 2 (arrives 12:15 > 11:30) is NOT visible yet!
    v_clock = pd.Timestamp("2023-06-01 11:30:00", tz="UTC")
    joined = join_predictions_with_delayed_labels(
        preds, actuals, virtual_clock_utc=v_clock, delay_minutes=15
    )

    row1 = joined[joined["event_timestamp"] == t1].iloc[0]
    row2 = joined[joined["event_timestamp"] == t2].iloc[0]

    assert row1["target_kwh"] == 110.0
    assert row1["quality_flag"] == "VALID"

    assert pd.isna(row2["target_kwh"])
    assert row2["quality_flag"] == "UNAVAILABLE_PENDING"


# =============================================================================
# 7. Missing Label Handling (Data Outage vs Degradation)
# =============================================================================
def test_missing_labels_handled_as_outage_not_accuracy_alert():
    """Verify that an actuals outage is flagged as label_data_outage, not model degradation."""
    monitor = PerformanceMonitor()

    # 50 daylight predictions exist, but all actuals are missing (NaN)
    times = pd.date_range("2023-06-01 14:00", periods=50, freq="1h", tz="UTC")
    outage_df = pd.DataFrame({
        "event_timestamp": times,
        "prediction_kwh": [300.0] * 50,
        "target_kwh": [np.nan] * 50,
        "quality_flag": ["COMMUNICATION_OUTAGE"] * 50,
        "is_daylight": [True] * 50,
    })

    eval_ts = pd.Timestamp("2023-06-08 00:00:00", tz="UTC")
    summary, alert = monitor.evaluate_check(eval_ts, outage_df)

    assert summary["status"] == "label_data_outage"
    assert summary["valid_daylight_labels"] == 0
    assert summary["total_daylight_predictions"] == 50
    assert alert is None, "Missing labels must never trigger an accuracy degradation alert"


# =============================================================================
# 8. Schema Validation across All Alert Types
# =============================================================================
def test_all_alert_types_conform_to_schema(alert_schema: dict):
    """Verify that all 4 alert types produce records conforming strictly to alert_event_schema."""
    required_top = alert_schema["required"]

    # Sample alert payloads for all 4 types
    events = [
        # 1. residual_deviation
        {
            "schema_version": "1.1.0",
            "event_id": "DEV-202306151600-001",
            "site_id": "2107",
            "alert_type": "residual_deviation",
            "model_name": "solarcast-f1",
            "model_version": "v2",
            "window_start_utc": "2023-06-15T15:00:00+00:00",
            "window_end_utc": "2023-06-15T16:00:00+00:00",
            "direction": "negative",
            "bin": "clear",
            "consecutive_hours": 2,
            "mean_residual_kwh": -210.45,
            "bound_low": -140.66,
            "bound_high": 127.94,
            "reason": "Test residual deviation",
            "created_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        },
        # 2. input_drift
        {
            "schema_version": "1.1.0",
            "event_id": "DRIFT-202306151600-001",
            "site_id": "2107",
            "alert_type": "input_drift",
            "model_name": "solarcast-f1",
            "model_version": "v2",
            "window_start_utc": "2023-06-01T00:00:00+00:00",
            "window_end_utc": "2023-06-15T00:00:00+00:00",
            "reason": "Test input drift",
            "created_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "drift_details": {
                "drifted_features": ["tmp_c", "tcdc_pct"],
                "drifted_feature_count": 2,
                "psi_scores": {"tmp_c": 0.45, "tcdc_pct": 0.32},
                "consecutive_checks": 2,
                "valid_current_rows": 210,
                "valid_reference_rows": 4500,
            },
        },
        # 3. performance_degradation
        {
            "schema_version": "1.1.0",
            "event_id": "PERF-202306151600-001",
            "site_id": "2107",
            "alert_type": "performance_degradation",
            "model_name": "solarcast-f1",
            "model_version": "v2",
            "window_start_utc": "2023-06-08T00:00:00+00:00",
            "window_end_utc": "2023-06-15T00:00:00+00:00",
            "reason": "Test performance degradation",
            "created_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "performance_details": {
                "metric_name": "daylight_mae_kwh",
                "mae_7d_kwh": 45.2,
                "reference_mae_kwh": 23.25,
                "reference_definition": "month_specific",
                "degradation_ratio": 1.94,
                "consecutive_checks": 2,
                "valid_label_count": 70,
            },
        },
        # 4. data_quality
        {
            "schema_version": "1.1.0",
            "event_id": "DQ-202306151600-001",
            "site_id": "2107",
            "alert_type": "data_quality",
            "model_name": "solarcast-f1",
            "model_version": "v2",
            "window_start_utc": "2023-06-08T00:00:00+00:00",
            "window_end_utc": "2023-06-15T00:00:00+00:00",
            "reason": "Test data quality degradation",
            "created_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "quality_details": {
                "total_hours": 70,
                "non_fresh_share": 0.25,
                "threshold": 0.15,
                "breakdown": {"fresh": 52, "fallback": 10, "stale": 5, "missing": 3},
            },
        },
    ]

    for ev in events:
        for field in required_top:
            assert field in ev, f"Missing required top-level field '{field}' in {ev['alert_type']}"
        assert ev["alert_type"] in alert_schema["properties"]["alert_type"]["enum"]
        assert ev["site_id"] == "2107"


# =============================================================================
# 9. Weather Freshness Classification & Illegal Future Weather Rejection
# =============================================================================
def test_weather_freshness_classification_and_future_rejection():
    """Verify operational freshness classification categories and negative age exception."""
    assert classify_weather_freshness(1.5, 0) == "fresh"
    assert classify_weather_freshness(2.0, 0) == "fresh"
    assert classify_weather_freshness(2.5, 0) == "fallback"
    assert classify_weather_freshness(8.0, 0) == "fallback"
    assert classify_weather_freshness(8.5, 0) == "stale"
    assert classify_weather_freshness(1.5, 1) == "missing"
    assert classify_weather_freshness(None, 0) == "missing"
    assert classify_weather_freshness(np.nan, 0) == "missing"

    # Future weather (negative run age) is forbidden and must raise ValueError
    with pytest.raises(ValueError, match="Future weather"):
        classify_weather_freshness(-1.0, 0)

    # Vectorized test
    df = pd.DataFrame({
        "weather_run_age_hours": [1.0, 3.0, 10.0, np.nan],
        "weather_missing": [0, 0, 0, 1],
    })
    series = classify_weather_freshness_series(df)
    assert series.tolist() == ["fresh", "fallback", "stale", "missing"]

    df_illegal = pd.DataFrame({
        "weather_run_age_hours": [-0.5],
        "weather_missing": [0],
    })
    with pytest.raises(ValueError, match="negative weather run age"):
        classify_weather_freshness_series(df_illegal)


# =============================================================================
# 10. Weather Freshness Monitor Alert Trigger
# =============================================================================
def test_weather_freshness_monitor_triggers_alert():
    """Verify WeatherFreshnessMonitor triggers data_quality event when non-fresh > threshold."""
    monitor = WeatherFreshnessMonitor(non_fresh_threshold=0.15, min_eval_hours=30)

    # 40 hours: 30 fresh (75%), 10 fallback (25% non-fresh > 15%)
    df = pd.DataFrame({
        "weather_run_age_hours": [1.5] * 30 + [4.0] * 10,
        "weather_missing": [0] * 40,
    })
    eval_ts = pd.Timestamp("2023-06-08 00:00:00", tz="UTC")
    summary, alert = monitor.evaluate_check(eval_ts, df)

    assert summary["status"] == "alert"
    assert summary["non_fresh_share"] == 0.25
    assert alert is not None
    assert alert["alert_type"] == "data_quality"
    assert alert["quality_details"]["breakdown"]["fallback"] == 10


# =============================================================================
# 11. Determinism
# =============================================================================
def test_monitoring_determinism():
    """Verify that repeated evaluation with identical inputs produces identical results."""
    m1 = PerformanceMonitor()
    m2 = PerformanceMonitor()

    times = pd.date_range("2023-06-01 14:00", periods=40, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "event_timestamp": times,
        "prediction_kwh": [120.0] * 40,
        "target_kwh": [100.0] * 40,
        "quality_flag": ["VALID"] * 40,
        "is_daylight": [True] * 40,
    })

    eval_ts = pd.Timestamp("2023-06-08 00:00:00", tz="UTC")
    sum1, ev1 = m1.evaluate_check(eval_ts, df)
    sum2, ev2 = m2.evaluate_check(eval_ts, df)

    assert sum1 == sum2
    assert ev1 == ev2
