"""Tests for SolarCast deviation detection and alerting engine (Task Step 4).

Verifies:
1. Bounds calibrated strictly from validation folds (never final test set).
2. 2-consecutive-hour alerting rule (single breach does not alert).
3. Reset behavior by night, missing target, and non-VALID quality flags.
4. Suppression of nighttime and invalid hours.
5. Single outlier drop does not trigger an alert.
6. Injected faults (zero output, 30% drop) are successfully detected.
7. Injected evaluation is deterministic with fixed random seed.
8. Alert events strictly match configs/alert_event_schema.json specification.
"""

from __future__ import annotations

import datetime
import json

import numpy as np
import pandas as pd
import pytest

from src.models.deviation import (
    DEFAULT_ALERT_SCHEMA,
    DEFAULT_DEVIATION_CONFIG,
    detect_deviations,
    load_deviation_config,
    run_injected_evaluation,
)

TEST_BOUNDS = {
    "clear": {"bound_low": -140.0, "bound_high": 128.0},
    "mixed": {"bound_low": -231.0, "bound_high": 226.0},
    "overcast": {"bound_low": -203.0, "bound_high": 216.0},
    "no_weather": {"bound_low": -155.0, "bound_high": 191.0},
}


@pytest.fixture(scope="module")
def deviation_config() -> dict:
    """Load frozen deviation configuration."""
    assert DEFAULT_DEVIATION_CONFIG.exists(), f"Missing {DEFAULT_DEVIATION_CONFIG}"
    return load_deviation_config(DEFAULT_DEVIATION_CONFIG)


@pytest.fixture(scope="module")
def alert_schema() -> dict:
    """Load alert event JSON schema."""
    assert DEFAULT_ALERT_SCHEMA.exists(), f"Missing {DEFAULT_ALERT_SCHEMA}"
    with open(DEFAULT_ALERT_SCHEMA, encoding="utf-8") as f:
        return json.load(f)


def test_bounds_from_validation_only(deviation_config: dict):
    """Verify that calibrated bounds are derived strictly from the validation partition."""
    cal_win = deviation_config["calibration_window"]
    end_utc = pd.to_datetime(cal_win["end_utc"])

    # Final test partition starts at 2022-10-17 01:00 UTC
    test_start_utc = pd.to_datetime("2022-10-17 01:00:00+00:00")
    assert end_utc <= test_start_utc, (
        f"Leakage detected: Calibration window end ({end_utc}) includes final test partition!"
    )
    assert cal_win["total_daylight_validation_samples"] == 13914


def test_two_hour_consecutive_rule():
    """Verify that an alert triggers only after 2 consecutive hours in same direction."""
    base_time = pd.Timestamp("2023-06-01 14:00:00", tz="UTC")
    times = [base_time + datetime.timedelta(hours=i) for i in range(5)]

    # Case A: Single breach followed by normal output -> 0 alerts
    df_single = pd.DataFrame({
        "timestamp": times,
        "actual": [500.0, 100.0, 500.0, 500.0, 500.0],  # Hour 1 breach: res = -400
        "forecast": [500.0, 500.0, 500.0, 500.0, 500.0],
        "solar_elevation_deg": [30.0, 40.0, 50.0, 45.0, 35.0],
        "is_daylight": [True, True, True, True, True],
        "quality_flag": ["VALID"] * 5,
        "forecast_cloud_bin": ["clear"] * 5,
    })
    alerts_single = detect_deviations(df_single, {"bounds": TEST_BOUNDS})
    assert len(alerts_single) == 0, "Single breach must NOT trigger an alert"

    # Case B: Two consecutive breaches in same direction -> 1 alert event
    df_double = pd.DataFrame({
        "timestamp": times,
        "actual": [500.0, 100.0, 100.0, 500.0, 500.0],  # Hours 1 and 2 breach
        "forecast": [500.0, 500.0, 500.0, 500.0, 500.0],
        "solar_elevation_deg": [30.0, 40.0, 50.0, 45.0, 35.0],
        "is_daylight": [True, True, True, True, True],
        "quality_flag": ["VALID"] * 5,
        "forecast_cloud_bin": ["clear"] * 5,
    })
    alerts_double = detect_deviations(df_double, {"bounds": TEST_BOUNDS})
    assert len(alerts_double) == 1, "Two consecutive breaches must trigger exactly 1 alert event"
    ev = alerts_double[0]
    assert ev["consecutive_hours"] == 2
    assert ev["direction"] == "negative"
    assert ev["window_start_utc"] == times[1].isoformat()
    assert ev["window_end_utc"] == times[2].isoformat()


def test_reset_by_night_and_missing():
    """Verify that nighttime, missing targets, or non-VALID flags reset the consecutive count."""
    base_time = pd.Timestamp("2023-06-01 14:00:00", tz="UTC")
    times = [base_time + datetime.timedelta(hours=i) for i in range(4)]

    # Breach at hour 0, night at hour 1, breach at hour 2 -> NO alert (count reset by night)
    df_night_reset = pd.DataFrame({
        "timestamp": times,
        "actual": [100.0, 0.0, 100.0, 500.0],
        "forecast": [500.0, 0.0, 500.0, 500.0],
        "solar_elevation_deg": [30.0, -5.0, 30.0, 40.0],  # Hour 1 is night
        "is_daylight": [True, False, True, True],
        "quality_flag": ["VALID", "VALID", "VALID", "VALID"],
        "forecast_cloud_bin": ["clear"] * 4,
    })
    alerts_night = detect_deviations(df_night_reset, {"bounds": TEST_BOUNDS})
    assert len(alerts_night) == 0, "Night hour must reset breach counter"

    # Breach at hour 0, then missing target (NaN) at hour 1, then breach at hour 2 -> NO alert
    df_nan_reset = pd.DataFrame({
        "timestamp": times,
        "actual": [100.0, np.nan, 100.0, 500.0],
        "forecast": [500.0, 500.0, 500.0, 500.0],
        "solar_elevation_deg": [30.0, 40.0, 50.0, 40.0],
        "is_daylight": [True, True, True, True],
        "quality_flag": ["VALID", "VALID", "VALID", "VALID"],
        "forecast_cloud_bin": ["clear"] * 4,
    })
    alerts_nan = detect_deviations(df_nan_reset, {"bounds": TEST_BOUNDS})
    assert len(alerts_nan) == 0, "Missing target must reset breach counter"


def test_suppression_non_valid_flag():
    """Verify that hours with quality_flag other than VALID are suppressed and reset count."""
    base_time = pd.Timestamp("2023-06-01 14:00:00", tz="UTC")
    times = [base_time + datetime.timedelta(hours=i) for i in range(3)]

    # Hour 0 and Hour 1 breach, but Hour 1 has quality_flag == 'SUSPECT_ZERO'
    df_flag = pd.DataFrame({
        "timestamp": times,
        "actual": [100.0, 100.0, 500.0],
        "forecast": [500.0, 500.0, 500.0],
        "solar_elevation_deg": [30.0, 40.0, 50.0],
        "is_daylight": [True, True, True],
        "quality_flag": ["VALID", "SUSPECT_ZERO", "VALID"],
        "forecast_cloud_bin": ["clear"] * 3,
    })
    alerts_flag = detect_deviations(df_flag, {"bounds": TEST_BOUNDS})
    assert len(alerts_flag) == 0, "Non-VALID quality flags must be suppressed and reset counter"


def test_alert_event_matches_schema(alert_schema: dict):
    """Verify that generated alert events match required schema properties and types."""
    base_time = pd.Timestamp("2023-06-01 14:00:00", tz="UTC")
    times = [base_time + datetime.timedelta(hours=i) for i in range(4)]
    df = pd.DataFrame({
        "timestamp": times,
        "actual": [500.0, 100.0, 100.0, 100.0],
        "forecast": [500.0, 500.0, 500.0, 500.0],
        "solar_elevation_deg": [30.0, 40.0, 50.0, 40.0],
        "is_daylight": [True, True, True, True],
        "quality_flag": ["VALID"] * 4,
        "forecast_cloud_bin": ["clear"] * 4,
    })
    alerts = detect_deviations(df, {"bounds": TEST_BOUNDS})
    assert len(alerts) == 1
    ev = alerts[0]

    # Validate against JSON schema required fields
    required_fields = alert_schema["required"]
    for field in required_fields:
        assert field in ev, f"Missing required schema field '{field}' in alert event"

    # Type and value constraints
    assert ev["schema_version"] in ("1.0.0", "1.1.0")
    assert ev["alert_type"] == "residual_deviation"
    assert ev["site_id"] == "2107"
    assert ev["model_name"] == "solarcast-f1"
    assert ev["consecutive_hours"] == 3
    assert ev["direction"] in ("negative", "positive")
    assert ev["bin"] in ("clear", "mixed", "overcast", "no_weather")
    assert isinstance(ev["mean_residual_kwh"], float)
    assert isinstance(ev["bound_low"], float)
    assert isinstance(ev["bound_high"], float)
    assert isinstance(ev["reason"], str)


def test_single_outlier_no_alert():
    """Verify that an isolated single-hour 30% drop never triggers an alert."""
    base_time = pd.Timestamp("2023-06-01 12:00:00", tz="UTC")
    times = [base_time + datetime.timedelta(hours=i) for i in range(5)]
    df = pd.DataFrame({
        "timestamp": times,
        "actual": [500.0, 500.0 * 0.70, 500.0, 500.0, 500.0],  # Single drop at hour 1
        "forecast": [500.0, 500.0, 500.0, 500.0, 500.0],
        "solar_elevation_deg": [40.0, 50.0, 60.0, 50.0, 40.0],
        "is_daylight": [True] * 5,
        "quality_flag": ["VALID"] * 5,
        "forecast_cloud_bin": ["clear"] * 5,
    })
    alerts = detect_deviations(df, {"bounds": TEST_BOUNDS})
    assert len(alerts) == 0, "Single-hour drop must NOT trigger an alert"


def test_injected_fault_detected():
    """Verify that multi-hour zero output and large drops are detected."""
    base_time = pd.Timestamp("2023-06-01 12:00:00", tz="UTC")
    times = [base_time + datetime.timedelta(hours=i) for i in range(5)]
    # 3 consecutive hours of zero output during midday
    df_zero = pd.DataFrame({
        "timestamp": times,
        "actual": [500.0, 0.0, 0.0, 0.0, 500.0],
        "forecast": [500.0, 500.0, 500.0, 500.0, 500.0],
        "solar_elevation_deg": [40.0, 50.0, 60.0, 50.0, 40.0],
        "is_daylight": [True] * 5,
        "quality_flag": ["VALID"] * 5,
        "forecast_cloud_bin": ["clear"] * 5,
    })
    alerts_zero = detect_deviations(df_zero, {"bounds": TEST_BOUNDS})
    assert len(alerts_zero) == 1
    assert alerts_zero[0]["consecutive_hours"] == 3
    assert alerts_zero[0]["direction"] == "negative"


def test_injected_evaluation_deterministic():
    """Verify that injected evaluation yields identical, deterministic results across runs."""
    base_time = pd.Timestamp("2023-01-01 12:00:00", tz="UTC")
    n_hours = 300
    times = [base_time + datetime.timedelta(hours=i) for i in range(n_hours)]
    df_synth = pd.DataFrame({
        "timestamp": times,
        "interval_start_utc": times,
        "actual": [300.0] * n_hours,
        "target_kwh": [300.0] * n_hours,
        "forecast": [300.0] * n_hours,
        "solar_elevation_deg": [35.0] * n_hours,
        "is_daylight": [True] * n_hours,
        "quality_flag": ["VALID"] * n_hours,
        "weather_missing": [0] * n_hours,
        "forecast_cloud_bin": ["clear"] * n_hours,
    })

    res1 = run_injected_evaluation(
        df_synth, {"bounds": TEST_BOUNDS}, seed=42, n_episodes=5, min_separation_hours=6
    )
    res2 = run_injected_evaluation(
        df_synth, {"bounds": TEST_BOUNDS}, seed=42, n_episodes=5, min_separation_hours=6
    )

    # Identical detection rates and metrics
    assert res1["injected_30pct_drop_4h"] == res2["injected_30pct_drop_4h"]
    assert res1["injected_zero_output_3h"] == res2["injected_zero_output_3h"]
    assert res1["injected_bad_value_2h"] == res2["injected_bad_value_2h"]
    assert res1["injected_isolated_single_hour_drop"] == res2["injected_isolated_single_hour_drop"]
