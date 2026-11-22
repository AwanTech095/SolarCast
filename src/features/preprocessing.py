"""Feature engineering and preprocessing pipeline for SolarCast (Task M1.3 / Step 2).

Provides a single shared feature extraction function `build_features(issue_times_utc, ...)`
used symmetrically by offline model training, backtesting folds, and online serving.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pvlib
import yaml

DEFAULT_CONFIG_PATH = Path("configs/features.yaml")
DEFAULT_SCHEMA_PATH = Path("configs/feature_schema.json")
DEFAULT_HOURLY_PATH = Path("data/processed/hourly.parquet")
DEFAULT_QH_PATH = Path("data/processed/quarter_hourly.parquet")


def load_feature_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load feature engineering configuration from YAML."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {
        "schema_version": "1.0.0",
        "site_id": "2107",
        "latitude": 38.996306,
        "longitude": -122.134111,
        "elevation_m": 10.0,
        "capacity_dc_kw": 893.0,
        "p95_clearsky_ratio": 1.330944,
        "operational_timing": {
            "production_available_delay_minutes": 15,
        },
    }


def build_features(
    issue_times_utc: pd.DatetimeIndex | list[pd.Timestamp] | pd.Series,
    hourly_df: pd.DataFrame | None = None,
    quarter_hourly_df: pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
    hourly_path: Path | str | None = None,
    quarter_hourly_path: Path | str | None = None,
) -> pd.DataFrame:
    """Build feature matrix for given prediction issue times.

    This function is strictly deterministic, past-only, and leak-free:
    at issue time H (UTC) predicting interval [H, H+1), no information
    at or after (H - production_available_delay_minutes) is ever accessed.

    Args:
        issue_times_utc: Prediction issue times in UTC.
        hourly_df: Optional preloaded hourly DataFrame.
        quarter_hourly_df: Optional preloaded quarter-hourly DataFrame.
        config: Optional feature configuration dictionary.
        hourly_path: Optional path to hourly.parquet.
        quarter_hourly_path: Optional path to quarter_hourly.parquet.

    Returns:
        DataFrame with feature columns matching configs/feature_schema.json,
        indexed by issue_time_utc.
    """
    if config is None:
        config = load_feature_config()

    lat = float(config.get("latitude", 38.996306))
    lon = float(config.get("longitude", -122.134111))
    elev = float(config.get("elevation_m", 10.0))
    p95 = float(config.get("p95_clearsky_ratio", 1.330944))
    op_timing = config.get("operational_timing", {})
    delay_min = int(op_timing.get("production_available_delay_minutes", 15))

    # Normalize issue_times to UTC DatetimeIndex
    if isinstance(issue_times_utc, pd.DatetimeIndex):
        dt_index = issue_times_utc
    elif isinstance(issue_times_utc, pd.Series):
        dt_index = pd.DatetimeIndex(pd.to_datetime(issue_times_utc, utc=True))
    else:
        dt_index = pd.DatetimeIndex([pd.to_datetime(t, utc=True) for t in issue_times_utc])

    if dt_index.tz is None:
        dt_index = dt_index.tz_localize("UTC")
    else:
        dt_index = dt_index.tz_convert("UTC")

    # Load datasets if not provided
    if hourly_df is None:
        h_path = Path(hourly_path) if hourly_path else DEFAULT_HOURLY_PATH
        hourly_df = pd.read_parquet(h_path)
    if quarter_hourly_df is None:
        qh_path = Path(quarter_hourly_path) if quarter_hourly_path else DEFAULT_QH_PATH
        quarter_hourly_df = pd.read_parquet(qh_path)

    # Ensure UTC datetime indexing for fast lookups
    h_work = hourly_df.copy()
    if not isinstance(h_work.index, pd.DatetimeIndex):
        h_work["interval_start_utc"] = pd.to_datetime(h_work["interval_start_utc"], utc=True)
        h_work.set_index("interval_start_utc", inplace=True)
    elif h_work.index.tz is None:
        h_work.index = h_work.index.tz_localize("UTC")

    qh_work = quarter_hourly_df.copy()
    if not isinstance(qh_work.index, pd.DatetimeIndex):
        qh_work["interval_start_utc"] = pd.to_datetime(qh_work["interval_start_utc"], utc=True)
        qh_work.set_index("interval_start_utc", inplace=True)
    elif qh_work.index.tz is None:
        qh_work.index = qh_work.index.tz_localize("UTC")

    # 1. 15-minute lags before cutoff (cutoff = H - delay_min)
    # With delay_min=15, cutoff is H - 15m.
    # Latest interval ending at or before H-15m starts at H - 30m.
    t_lag0 = dt_index - pd.Timedelta(minutes=delay_min + 15)   # H - 30m
    t_lag15 = dt_index - pd.Timedelta(minutes=delay_min + 30)  # H - 45m
    t_lag30 = dt_index - pd.Timedelta(minutes=delay_min + 45)  # H - 60m
    t_lag45 = dt_index - pd.Timedelta(minutes=delay_min + 60)  # H - 75m

    kw_lag0 = qh_work["power_kw"].reindex(t_lag0).values
    kw_lag15 = qh_work["power_kw"].reindex(t_lag15).values
    kw_lag30 = qh_work["power_kw"].reindex(t_lag30).values
    kw_lag45 = qh_work["power_kw"].reindex(t_lag45).values

    # Mean of last 4 available intervals (safely computed without empty slice warnings)
    lags_matrix = np.vstack([kw_lag0, kw_lag15, kw_lag30, kw_lag45])
    valid_count_last_4 = (~np.isnan(lags_matrix)).sum(axis=0)
    sum_kw_last_4 = np.nansum(lags_matrix, axis=0)
    kw_last_4_mean = np.full_like(sum_kw_last_4, np.nan, dtype=float)
    np.divide(
        sum_kw_last_4,
        valid_count_last_4,
        out=kw_last_4_mean,
        where=valid_count_last_4 > 0,
    )

    # Missing count in last 4 intervals
    missing_count_last_4 = (4 - valid_count_last_4).astype(int)

    # Age in minutes of latest available production reading
    # If lag0 is valid, age is delay_min (15m). If lag0 missing but lag15 valid, 30m, etc.
    age_min = np.where(
        ~np.isnan(kw_lag0),
        delay_min,
        np.where(
            ~np.isnan(kw_lag15),
            delay_min + 15,
            np.where(
                ~np.isnan(kw_lag30),
                delay_min + 30,
                np.where(~np.isnan(kw_lag45), delay_min + 45, delay_min + 60),
            ),
        ),
    ).astype(float)

    # 2. Hourly kWh features
    # Latest fully available hour: [H - 2h, H - 1h)
    t_h_latest = dt_index - pd.Timedelta(hours=2)
    t_h_prev_day = dt_index - pd.Timedelta(hours=24)
    t_h_prev_week = dt_index - pd.Timedelta(hours=168)

    kwh_latest = h_work["target_kwh"].reindex(t_h_latest).values
    qf_latest = h_work["quality_flag"].reindex(t_h_latest).values

    kwh_prev_day = h_work["target_kwh"].reindex(t_h_prev_day).values
    qf_prev_day = h_work["quality_flag"].reindex(t_h_prev_day).values

    kwh_prev_week = h_work["target_kwh"].reindex(t_h_prev_week).values
    qf_prev_week = h_work["quality_flag"].reindex(t_h_prev_week).values

    # Flags for missing hourly history
    latest_hour_missing = (np.isnan(kwh_latest) | (qf_latest != "VALID")).astype(int)
    prev_day_missing = (np.isnan(kwh_prev_day) | (qf_prev_day != "VALID")).astype(int)
    prev_week_missing = (np.isnan(kwh_prev_week) | (qf_prev_week != "VALID")).astype(int)

    # 3. Past-only rolling means of hourly energy (shifted so they never touch target hour H)
    # The rolling series is computed on hourly dataset, shifted by 2 hours so that
    # at timestamp H, the rolling mean uses timestamps <= H - 2h.
    h_kwh_series = h_work["target_kwh"]
    roll_6h_series = h_kwh_series.shift(2).rolling(window=6, min_periods=1).mean()
    roll_24h_series = h_kwh_series.shift(2).rolling(window=24, min_periods=1).mean()

    roll_6h = roll_6h_series.reindex(dt_index).values
    roll_24h = roll_24h_series.reindex(dt_index).values

    # 4. Cyclical calendar encodings (hour and day of year) for target hour H
    hours = dt_index.hour.values
    doys = dt_index.dayofyear.values

    hour_sin = np.sin(2.0 * np.pi * hours / 24.0)
    hour_cos = np.cos(2.0 * np.pi * hours / 24.0)
    doy_sin = np.sin(2.0 * np.pi * doys / 365.25)
    doy_cos = np.cos(2.0 * np.pi * doys / 365.25)

    # 5. Solar elevation and clear-sky energy for target hour H (at interval midpoint H + 30m)
    location = pvlib.location.Location(latitude=lat, longitude=lon, altitude=elev)
    midpoint_target = dt_index + pd.Timedelta(minutes=30)
    cs_target = location.get_clearsky(midpoint_target)
    sol_target = pvlib.solarposition.get_solarposition(midpoint_target, lat, lon)

    solar_elevation_target = sol_target["elevation"].values
    clearsky_ghi_target = cs_target["ghi"].values
    clearsky_energy_target = np.maximum(0.0, clearsky_ghi_target * p95)

    # 6. Clear-sky index of the latest fully available hour [H - 2h, H - 1h)
    midpoint_latest = t_h_latest + pd.Timedelta(minutes=30)
    cs_latest = location.get_clearsky(midpoint_latest)
    clearsky_ghi_latest = cs_latest["ghi"].values
    clearsky_energy_latest = np.maximum(0.0, clearsky_ghi_latest * p95)

    # Safe clear-sky index calculation with fallback
    # If daytime (clearsky_energy > 10 kWh) and kwh_latest valid: ratio
    # Else fallback to 1.0 (clear-sky assumption)
    valid_kt_mask = (clearsky_energy_latest > 10.0) & (~np.isnan(kwh_latest))
    clearsky_index_latest = np.ones_like(kwh_latest, dtype=float)
    np.divide(
        kwh_latest,
        clearsky_energy_latest,
        out=clearsky_index_latest,
        where=valid_kt_mask,
    )
    clearsky_index_latest = np.clip(clearsky_index_latest, 0.0, 2.0)

    # Assemble feature DataFrame matching ordered schema
    data = {
        "power_kw_lag_0m": kw_lag0,
        "power_kw_lag_15m": kw_lag15,
        "power_kw_lag_30m": kw_lag30,
        "power_kw_lag_45m": kw_lag45,
        "power_kw_last_4_mean": kw_last_4_mean,
        "energy_kwh_latest_avail_hour": kwh_latest,
        "energy_kwh_prev_day_same_hour": kwh_prev_day,
        "energy_kwh_prev_week_same_hour": kwh_prev_week,
        "energy_kwh_rolling_mean_6h": roll_6h,
        "energy_kwh_rolling_mean_24h": roll_24h,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "day_of_year_sin": doy_sin,
        "day_of_year_cos": doy_cos,
        "solar_elevation_target_deg": solar_elevation_target,
        "clearsky_ghi_target_wm2": clearsky_ghi_target,
        "clearsky_energy_target_kwh": clearsky_energy_target,
        "clearsky_index_latest_avail": clearsky_index_latest,
        "production_age_minutes": age_min,
        "power_kw_missing_count": missing_count_last_4,
        "latest_hour_missing": latest_hour_missing,
        "prev_day_missing": prev_day_missing,
        "prev_week_missing": prev_week_missing,
    }

    feat_df = pd.DataFrame(data, index=dt_index)
    feat_df.index.name = "issue_time_utc"

    return feat_df
