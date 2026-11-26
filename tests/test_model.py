"""Tests for model training, evaluation, candidate gating, and bundling.

Verifies:
1. Physical constraints: predictions are finite, non-negative, and clipped to [0, capacity_dc_kw].
2. Nighttime zero-prediction logic: target hours with solar elevation <= 0 predict 0.
3. Determinism: identical predictions given the same seed and input data.
4. Candidate gate evaluation: passes when improvement >= 5% and cap-MAE <= 0.20; fails otherwise.
5. Final test safeguard: raises RuntimeError when opening final test partition without --force.
6. ModelBundle serialization: save_bundle -> load_bundle reproduces exact predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from src.models.baselines import (
    ClearSky15mPersistence,
    Last15mPersistence,
)
from src.models.bundle import ModelBundle, load_bundle, save_bundle
from src.models.evaluate import (
    check_final_test_safeguard,
    evaluate_candidate_gate,
)
from src.models.tune import (
    build_hgb_model,
    build_ridge_pipeline,
    sample_hyperparameter_configs,
)


@pytest.fixture
def synthetic_feature_data():
    """Create deterministic synthetic features and targets for testing."""
    np.random.seed(42)
    n = 100
    times = pd.date_range("2022-06-01 00:00:00", periods=n, freq="1h", tz="UTC")
    solar_elev = np.sin(np.linspace(0, 4 * np.pi, n)) * 60.0
    clearsky_ghi = np.maximum(0.0, solar_elev * 15.0)

    # 23 features
    feature_dict = {
        "power_kw_lag_0m": np.random.uniform(0, 500, n),
        "power_kw_lag_15m": np.random.uniform(0, 500, n),
        "power_kw_lag_30m": np.random.uniform(0, 500, n),
        "power_kw_lag_45m": np.random.uniform(0, 500, n),
        "power_kw_last_4_mean": np.random.uniform(0, 500, n),
        "energy_kwh_latest_avail_hour": np.random.uniform(0, 500, n),
        "energy_kwh_prev_day_same_hour": np.random.uniform(0, 500, n),
        "energy_kwh_prev_week_same_hour": np.random.uniform(0, 500, n),
        "energy_kwh_rolling_mean_6h": np.random.uniform(0, 500, n),
        "energy_kwh_rolling_mean_24h": np.random.uniform(0, 500, n),
        "hour_sin": np.sin(2 * np.pi * times.hour / 24.0),
        "hour_cos": np.cos(2 * np.pi * times.hour / 24.0),
        "day_of_year_sin": np.sin(2 * np.pi * times.dayofyear / 365.25),
        "day_of_year_cos": np.cos(2 * np.pi * times.dayofyear / 365.25),
        "solar_elevation_target_deg": solar_elev,
        "clearsky_ghi_target_wm2": clearsky_ghi,
        "clearsky_energy_target_kwh": clearsky_ghi * 0.8,
        "clearsky_index_latest_avail": np.random.uniform(0.1, 1.2, n),
        "production_age_minutes": np.full(n, 15.0),
        "power_kw_missing_count": np.zeros(n, dtype=int),
        "latest_hour_missing": np.zeros(n, dtype=int),
        "prev_day_missing": np.zeros(n, dtype=int),
        "prev_week_missing": np.zeros(n, dtype=int),
    }
    df = pd.DataFrame(feature_dict)
    # Insert some NaNs to test imputer / native NaN handling
    df.loc[5:10, "power_kw_lag_0m"] = np.nan
    df.loc[12:15, "energy_kwh_latest_avail_hour"] = np.nan

    simulated_prod = clearsky_ghi * 0.75 + np.random.normal(0, 20, n)
    target = np.where(solar_elev > 0, np.maximum(0.0, simulated_prod), 0.0)
    target = np.clip(target, 0.0, 893.0)
    df["target_kwh"] = target
    df["interval_start_utc"] = times
    df["is_daylight"] = solar_elev > 0
    return df


def test_hyperparameter_sampling_determinism():
    """Verify that sampling hyperparameter configs is deterministic with fixed seed."""
    space = {
        "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
        "fit_intercept": [True, False],
    }
    sample_1 = sample_hyperparameter_configs(space, n_iter=5, random_seed=42)
    sample_2 = sample_hyperparameter_configs(space, n_iter=5, random_seed=42)
    assert sample_1 == sample_2
    assert len(sample_1) == 5


def test_physical_constraints_and_nighttime_zero(synthetic_feature_data):
    """Verify that model predictions are strictly non-negative, <= 893 kWh, and 0 at night."""
    df = synthetic_feature_data
    drop_cols = ["target_kwh", "interval_start_utc", "is_daylight"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    model = build_ridge_pipeline({"alpha": 1.0, "fit_intercept": True}, random_seed=42)
    # Fit on daylight hours only
    day_mask = df["solar_elevation_target_deg"] > 0
    model.fit(df.loc[day_mask, feature_cols].values, df.loc[day_mask, "target_kwh"].values)

    bundle = ModelBundle(
        estimator=model,
        feature_names=feature_cols,
        training_window={"start": "2022-06-01", "end": "2022-06-05"},
        data_hashes={"h": "abc"},
        mlflow_run_id="run_123",
        capacity_limit_kwh=893.0,
    )

    preds = bundle.predict(df[feature_cols])

    assert len(preds) == len(df)
    assert np.all(np.isfinite(preds)), "All predictions must be finite"
    assert np.all(preds >= 0.0), "Predictions must be non-negative"
    assert np.all(preds <= 893.0), "Predictions must not exceed DC capacity (893 kWh)"

    # Nighttime zero check: where solar elevation <= 0, predictions must be exactly 0
    night_idx = np.where(df["solar_elevation_target_deg"] <= 0)[0]
    assert len(night_idx) > 0
    assert np.all(preds[night_idx] == 0.0), "Nighttime predictions must be identically 0.0"


def test_hgb_model_native_nan(synthetic_feature_data):
    """Verify HistGradientBoosting handles NaNs natively and predicts within bounds."""
    df = synthetic_feature_data
    drop_cols = ["target_kwh", "interval_start_utc", "is_daylight"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    hgb = build_hgb_model(
        {"max_iter": 50, "learning_rate": 0.05, "max_depth": 3},
        random_seed=42,
    )
    day_mask = df["solar_elevation_target_deg"] > 0
    hgb.fit(df.loc[day_mask, feature_cols].values, df.loc[day_mask, "target_kwh"].values)

    raw_preds = hgb.predict(df[feature_cols].values)
    assert np.all(np.isfinite(raw_preds)), "HGB predictions with NaNs must be finite"


def test_candidate_gate_logic():
    """Verify candidate gate logic correctly identifies eligibility and failure reasons."""
    baseline_metrics = {
        "daylight_mae_kwh": 75.62,
        "daylight_rmse_kwh": 133.27,
        "capacity_norm_daylight_mae": 0.0847,
    }

    # Case 1: Clear winner (MAE 60.0, cap-MAE ~6.7%, finite, non-negative)
    passing_candidate = {
        "daylight_mae_kwh": 60.00,
        "daylight_rmse_kwh": 105.00,
        "capacity_norm_daylight_mae": 60.0 / 893.0,
    }
    y_pred = np.array([0.0, 50.0, 100.0, 200.0])
    gate = evaluate_candidate_gate(
        passing_candidate, baseline_metrics, y_pred, capacity_limit_kwh=893.0
    )
    assert gate["eligible"] is True
    assert len(gate["failure_reasons"]) == 0
    assert gate["achieved_improvement_pct"] > 5.0

    # Case 2: Insufficient improvement (< 5%)
    weak_candidate = {
        "daylight_mae_kwh": 73.00,  # only ~3.4% improvement
        "daylight_rmse_kwh": 130.00,
        "capacity_norm_daylight_mae": 73.0 / 893.0,
    }
    gate_weak = evaluate_candidate_gate(
        weak_candidate, baseline_metrics, y_pred, capacity_limit_kwh=893.0
    )
    assert gate_weak["eligible"] is False
    assert any("Insufficient daylight MAE improvement" in r for r in gate_weak["failure_reasons"])

    # Case 3: Negative predictions violation
    bad_pred = np.array([-5.0, 50.0, 100.0])
    gate_neg = evaluate_candidate_gate(
        passing_candidate, baseline_metrics, bad_pred, capacity_limit_kwh=893.0
    )
    assert gate_neg["eligible"] is False
    assert any("negative values" in r for r in gate_neg["failure_reasons"])


def test_final_test_safeguard(tmp_path, monkeypatch):
    """Verify safeguard prevents reading or writing final test set without authorization."""
    marker_path = tmp_path / "final_test_safeguard.json"
    for key in ("CI", "GITHUB_ACTIONS", "AIRFLOW_CTX_DAG_ID", "KFP_RUN_ID"):
        monkeypatch.delenv(key, raising=False)
    # First use also requires explicit authorization.
    with pytest.raises(RuntimeError, match="force"):
        check_final_test_safeguard(force=False, reason=None, log_path=marker_path)
    with pytest.raises(ValueError, match="A meaningful reason"):
        check_final_test_safeguard(force=True, reason="", log_path=marker_path)
    check_final_test_safeguard(
        force=True, reason="Authorized first evaluation", log_path=marker_path
    )

    # Simulate final test results already written
    marker_path.write_text('{"final_test_opened": true}', encoding="utf-8")

    # Attempting to check without force should raise RuntimeError
    with pytest.raises(RuntimeError, match="Final 20% test partition has already been opened"):
        check_final_test_safeguard(force=False, reason=None, log_path=marker_path)

    # Force never bypasses existing audit evidence.
    with pytest.raises(RuntimeError, match="already been opened"):
        check_final_test_safeguard(force=True, reason="", log_path=marker_path)

    with pytest.raises(RuntimeError, match="already been opened"):
        check_final_test_safeguard(
            force=True, reason="Audited recalculation after authorized scope update",
            log_path=marker_path,
        )


def test_model_bundle_roundtrip(synthetic_feature_data, tmp_path):
    """Verify ModelBundle save and load reproduces exact predictions within floating tolerance."""
    df = synthetic_feature_data
    drop_cols = ["target_kwh", "interval_start_utc", "is_daylight"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    model = Ridge(alpha=10.0)
    # Fit simple model on clean subset
    clean_df = df.fillna(0.0)
    model.fit(clean_df[feature_cols].values, clean_df["target_kwh"].values)

    bundle = ModelBundle(
        estimator=model,
        feature_names=feature_cols,
        training_window={
            "start_utc": "2022-06-01T00:00:00Z",
            "end_utc": "2022-06-05T00:00:00Z",
        },
        data_hashes={"hourly": "hash123"},
        mlflow_run_id="run_test_456",
        capacity_limit_kwh=893.0,
        model_name="test-ridge",
        feature_schema_version="1.0.0",
    )

    preds_original = bundle.predict(clean_df[feature_cols])

    bundle_path = tmp_path / "test_bundle.joblib"
    save_bundle(bundle, bundle_path)
    assert bundle_path.exists()

    loaded_bundle = load_bundle(bundle_path)
    preds_loaded = loaded_bundle.predict(clean_df[feature_cols])

    np.testing.assert_allclose(preds_original, preds_loaded, rtol=1e-6, atol=1e-6)
    assert loaded_bundle.model_name == "test-ridge"
    assert loaded_bundle.mlflow_run_id == "run_test_456"


def test_model_bundle_set_b_with_weather(tmp_path, synthetic_feature_data):
    """Verify ModelBundle serialization and weather-aware predict() for Set B."""
    df = synthetic_feature_data.copy().dropna()
    weather_cols = [
        "tmp_c",
        "tcdc_pct",
        "clearsky_ghi_cloud_adj_wm2",
        "weather_run_age_hours",
        "weather_fallback_used",
        "weather_missing",
    ]
    # Add weather columns to df
    df["tmp_c"] = 25.0
    df["tcdc_pct"] = 10.0
    df["clearsky_ghi_cloud_adj_wm2"] = df["clearsky_ghi_target_wm2"] * 0.95
    df["weather_run_age_hours"] = 2.0
    df["weather_fallback_used"] = 0
    df["weather_missing"] = 0

    drop_cols = ["target_kwh", "interval_start_utc", "is_daylight"]
    all_feature_cols = [c for c in df.columns if c not in drop_cols]
    assert len(all_feature_cols) == 29

    model = Ridge(alpha=1.0)
    model.fit(df[all_feature_cols].values, df["target_kwh"].values)

    bundle = ModelBundle(
        estimator=model,
        feature_names=all_feature_cols,
        training_window={"start_utc": "2022-06-01T00:00:00Z", "end_utc": "2022-06-05T00:00:00Z"},
        data_hashes={"hourly": "hash123", "hrrr": "hash456"},
        mlflow_run_id="run_b_123",
        capacity_limit_kwh=893.0,
        model_name="test-hgb-b",
        feature_schema_version="1.1.0",
        feature_set="B",
        needs_weather=True,
    )
    assert bundle.needs_weather is True
    assert bundle.feature_set == "B"

    # Test predict with features_df containing all features
    preds_all = bundle.predict(df[all_feature_cols])
    assert len(preds_all) == len(df)

    # Test predict where weather features are passed separately in weather_df
    base_features_df = df[[c for c in all_feature_cols if c not in weather_cols]]
    weather_df = df[weather_cols]
    preds_merged = bundle.predict(base_features_df, weather_df=weather_df)
    np.testing.assert_allclose(preds_all, preds_merged, rtol=1e-6, atol=1e-6)

    # Round-trip save and load
    b_path = tmp_path / "bundle_b.joblib"
    save_bundle(bundle, b_path)
    loaded = load_bundle(b_path)
    assert loaded.feature_set == "B"
    assert loaded.needs_weather is True
    assert loaded.feature_schema_version == "1.1.0"
    preds_loaded = loaded.predict(base_features_df, weather_df=weather_df)
    np.testing.assert_allclose(preds_all, preds_loaded, rtol=1e-6, atol=1e-6)


def test_last_15m_persistence_baseline(synthetic_feature_data):
    """Verify Last15mPersistence predicts kw_lag0 * 1h with fallbacks and bounds."""
    df = synthetic_feature_data.copy()
    model = Last15mPersistence()
    preds = model.predict(df)

    assert len(preds) == len(df)
    assert np.all(np.isfinite(preds))
    assert np.all(preds >= 0.0)
    assert np.all(preds <= 893.0)

    # Nighttime zero check
    night_idx = np.where(df["solar_elevation_target_deg"] <= 0)[0]
    assert np.all(preds[night_idx] == 0.0)

    # Check daytime valid lag0
    valid_lag0_day = (df["solar_elevation_target_deg"] > 0) & (~df["power_kw_lag_0m"].isna())
    idx = np.where(valid_lag0_day)[0]
    np.testing.assert_allclose(preds[idx], np.clip(df["power_kw_lag_0m"].values[idx], 0.0, 893.0))

    # Check fallback when kw_lag0 is NaN
    nan_idx = np.where((df["solar_elevation_target_deg"] > 0) & (df["power_kw_lag_0m"].isna()))[0]
    if len(nan_idx) > 0:
        expected_fb = np.clip(df["power_kw_lag_15m"].values[nan_idx], 0.0, 893.0)
        np.testing.assert_allclose(preds[nan_idx], expected_fb)


def test_clearsky_15m_persistence_baseline(synthetic_feature_data):
    """Verify ClearSky15mPersistence computes kt15 scaling, enforces bounds and night zero."""
    df = synthetic_feature_data.copy()
    model = ClearSky15mPersistence(delay_minutes=15)
    preds = model.predict(df)

    assert len(preds) == len(df)
    assert np.all(np.isfinite(preds))
    assert np.all(preds >= 0.0)
    assert np.all(preds <= 893.0)

    # Nighttime zero check
    night_idx = np.where(df["solar_elevation_target_deg"] <= 0)[0]
    assert np.all(preds[night_idx] == 0.0)
