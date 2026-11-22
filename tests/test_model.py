"""Unit tests for baseline models and evaluation metrics (tests/test_model.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.baselines import (
    ClearSkyPersistence,
    LatestHourPersistence,
    SameHourPreviousDayPersistence,
    compute_metrics,
    compute_monthly_daylight_mae,
)


@pytest.fixture
def sample_feature_df() -> pd.DataFrame:
    """Create a minimal synthetic feature DataFrame for baseline testing."""
    return pd.DataFrame({
        "solar_elevation_target_deg": [35.0, 50.0, -10.0, 20.0, 40.0],
        "energy_kwh_latest_avail_hour": [300.0, np.nan, 250.0, 100.0, np.nan],
        "latest_hour_missing": [0, 1, 0, 0, 1],
        "energy_kwh_prev_day_same_hour": [320.0, 400.0, 260.0, np.nan, np.nan],
        "prev_day_missing": [0, 0, 0, 1, 1],
        "energy_kwh_prev_week_same_hour": [310.0, 390.0, 270.0, 150.0, np.nan],
        "prev_week_missing": [0, 0, 0, 0, 1],
        "clearsky_energy_target_kwh": [500.0, 600.0, 0.0, 400.0, 500.0],
        "clearsky_index_latest_avail": [0.8, 0.9, 0.7, 0.5, np.nan],
    })


class TestBaselineModels:
    def test_latest_hour_persistence(self, sample_feature_df: pd.DataFrame) -> None:
        model = LatestHourPersistence()
        preds = model.predict(sample_feature_df)

        # Row 0: daylight (35 deg), latest available is 300.0 -> 300.0
        assert preds[0] == pytest.approx(300.0)

        # Row 1: daylight (50 deg), latest is NaN / missing=1 -> fallback to prev day (400.0)
        assert preds[1] == pytest.approx(400.0)

        # Row 2: nighttime (-10 deg) -> strictly 0.0 regardless of available lag
        assert preds[2] == 0.0

        # Row 3: daylight (20 deg), latest is 100.0 -> 100.0
        assert preds[3] == pytest.approx(100.0)

        # Row 4: daylight (40 deg), all lags NaN -> fallback to 0.0
        assert preds[4] == 0.0

    def test_same_hour_prev_day_persistence(self, sample_feature_df: pd.DataFrame) -> None:
        model = SameHourPreviousDayPersistence()
        preds = model.predict(sample_feature_df)

        # Row 0: daylight, prev day is 320.0 -> 320.0
        assert preds[0] == pytest.approx(320.0)

        # Row 1: daylight, prev day is 400.0 -> 400.0
        assert preds[1] == pytest.approx(400.0)

        # Row 2: nighttime (-10 deg) -> strictly 0.0
        assert preds[2] == 0.0

        # Row 3: daylight, prev day is NaN / missing=1 -> fallback to prev week (150.0)
        assert preds[3] == pytest.approx(150.0)

        # Row 4: daylight, all missing -> 0.0
        assert preds[4] == 0.0

    def test_clearsky_persistence(self, sample_feature_df: pd.DataFrame) -> None:
        model = ClearSkyPersistence()
        preds = model.predict(sample_feature_df)

        # Row 0: daylight, kt=0.8, cs=500.0 -> 400.0
        assert preds[0] == pytest.approx(400.0)

        # Row 1: daylight, kt=0.9, cs=600.0 -> 540.0
        assert preds[1] == pytest.approx(540.0)

        # Row 2: nighttime -> 0.0
        assert preds[2] == 0.0

        # Row 3: daylight, kt=0.5, cs=400.0 -> 200.0
        assert preds[3] == pytest.approx(200.0)

        # Row 4: daylight, kt is NaN -> 0.0
        assert preds[4] == 0.0


class TestEvaluationMetrics:
    def test_compute_metrics_known_values(self) -> None:
        # y_true = [100, 200, 0, 50]
        # y_pred = [120, 180, 0, 40]
        # is_day = [True, True, False, True]
        # daylight errors: |100-120|=20, |200-180|=20, |50-40|=10 -> mean = 50 / 3 = 16.6667
        # daylight squared: 400 + 400 + 100 = 900 -> rmse = sqrt(300) = 17.3205
        # all-hour errors: 20, 20, 0, 10 -> mean = 50 / 4 = 12.5
        # all-hour squared: 900 / 4 = 225 -> rmse = 15.0
        y_true = np.array([100.0, 200.0, 0.0, 50.0])
        y_pred = np.array([120.0, 180.0, 0.0, 40.0])
        is_day = np.array([True, True, False, True])

        metrics = compute_metrics(y_true, y_pred, is_day, capacity_denominator_kwh=1000.0)

        assert metrics["daylight_mae_kwh"] == pytest.approx(50.0 / 3.0)
        assert metrics["daylight_rmse_kwh"] == pytest.approx(np.sqrt(300.0))
        assert metrics["capacity_norm_daylight_mae"] == pytest.approx((50.0 / 3.0) / 1000.0)
        assert metrics["all_hour_mae_kwh"] == pytest.approx(12.5)
        assert metrics["all_hour_rmse_kwh"] == pytest.approx(15.0)

    def test_compute_monthly_daylight_mae(self) -> None:
        y_true = np.array([100.0, 200.0, 0.0, 300.0])
        y_pred = np.array([120.0, 150.0, 0.0, 310.0])
        is_day = np.array([True, True, False, True])
        times = pd.DatetimeIndex([
            "2020-01-01 10:00:00+00:00",
            "2020-01-01 11:00:00+00:00",
            "2020-01-01 22:00:00+00:00",
            "2020-02-01 12:00:00+00:00",
        ])

        monthly = compute_monthly_daylight_mae(y_true, y_pred, times, is_day)
        assert len(monthly) == 2
        jan = monthly[monthly["month"] == "2020-01"].iloc[0]
        feb = monthly[monthly["month"] == "2020-02"].iloc[0]

        assert jan["daylight_hours"] == 2
        assert jan["daylight_mae"] == pytest.approx((20.0 + 50.0) / 2.0)

        assert feb["daylight_hours"] == 1
        assert feb["daylight_mae"] == pytest.approx(10.0)
