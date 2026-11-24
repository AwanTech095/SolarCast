"""Input drift detection and weather freshness monitoring for SolarCast Mode F1.

This module implements:
1. Population Stability Index (PSI) with 10 quantile bins derived from reference.
2. Season-matched reference extraction (calendar month +/- 1 month, matching daylight hours)
   to prevent seasonal solar geometry and weather changes from triggering false drift alarms.
3. InputDriftMonitor evaluating rolling 14-day daylight valid windows daily, enforcing:
   - PSI > 0.20 on >= 2 monitored features on two consecutive daily checks.
   - >= 200 valid daylight rows per comparison (otherwise 'insufficient_data').
   - Per-feature NaN handling, reporting missing weather as a data quality signal rather than drift.
4. Weather freshness classification (fresh / fallback / stale / missing) based on
   configs/data.yaml operational availability policies, rejecting future weather substitutions.
5. WeatherFreshnessMonitor evaluating rolling 7-day non-fresh weather share.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# Monitored features: 8 numeric features selected from configs/feature_schema.json (Feature Set B).
# Rationale:
# 1. power_kw_lag_0m: Top permutation importance; captures immediate pre-cutoff generation.
# 2. energy_kwh_prev_day_same_hour: Diurnal lag reference capturing day-to-day persistence.
# 3. energy_kwh_rolling_mean_24h: 24-hour moving energy baseline tracking synoptic weather trends.
# 4. clearsky_ghi_target_wm2: Clear-sky solar irradiance; checks astronomical geometry validity.
# 5. clearsky_index_latest_avail: Sky clearness ratio [0, 2]; monitors atmospheric transparency.
# 6. tmp_c: NOAA HRRR 2m temperature; impacts PV cell thermal efficiency (-0.4%/degC).
# 7. tcdc_pct: NOAA HRRR total cloud cover percentage; primary physical driver of cloud attenuation.
# 8. clearsky_ghi_cloud_adj_wm2: Cloud-adjusted irradiance combining geometry & clouds.
MONITORED_FEATURES: list[str] = [
    "power_kw_lag_0m",
    "energy_kwh_prev_day_same_hour",
    "energy_kwh_rolling_mean_24h",
    "clearsky_ghi_target_wm2",
    "clearsky_index_latest_avail",
    "tmp_c",
    "tcdc_pct",
    "clearsky_ghi_cloud_adj_wm2",
]

DEFAULT_DATA_CONFIG = Path("configs/data.yaml")


def load_operational_lags(config_path: Path = DEFAULT_DATA_CONFIG) -> dict[str, Any]:
    """Load operational weather lag and meter delay policies from configs/data.yaml."""
    if not config_path.exists():
        return {
            "weather_availability_lag_hours": 2.0,
            "weather_lag_sensitivity_hours": 3.0,
            "production_available_delay_minutes": 15,
        }
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {
        "weather_availability_lag_hours": float(
            cfg.get("weather_availability_lag_hours", 2.0)
        ),
        "weather_lag_sensitivity_hours": float(
            cfg.get("weather_lag_sensitivity_hours", 3.0)
        ),
        "production_available_delay_minutes": int(
            cfg.get("production_available_delay_minutes", 15)
        ),
    }


def calculate_psi_from_proportions(
    actual_pct: np.ndarray,
    expected_pct: np.ndarray,
    epsilon: float = 1e-4,
) -> float:
    """Calculate Population Stability Index (PSI) directly from bin proportions.

    Formula:
        PSI = sum((A_i - E_i) * ln(A_i / E_i))

    Args:
        actual_pct: Proportions in current/actual distribution across bins.
        expected_pct: Proportions in reference/expected distribution across bins.
        epsilon: Epsilon floor to prevent division by zero or log(0).

    Returns:
        Float PSI value.
    """
    act = np.clip(np.asarray(actual_pct, dtype=float), epsilon, None)
    exp = np.clip(np.asarray(expected_pct, dtype=float), epsilon, None)

    # Re-normalize to sum to 1.0 after clipping
    act = act / np.sum(act)
    exp = exp / np.sum(exp)

    psi_val = np.sum((act - exp) * np.log(act / exp))
    return float(psi_val)


def calculate_psi(
    reference: np.ndarray | pd.Series,
    current: np.ndarray | pd.Series,
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Calculate 10-quantile Population Stability Index between reference and current samples.

    Bins are derived from the reference distribution using quantiles. Outer bounds are
    expanded to [-inf, inf] so all current values fall within the binned domain.

    Args:
        reference: Reference values (NaNs must already be removed).
        current: Current values (NaNs must already be removed).
        num_bins: Target number of quantile bins (default 10).
        epsilon: Floor for zero-probability bins to avoid log(0) and div-by-zero.

    Returns:
        Float PSI score (>= 0.0).
    """
    ref = np.asarray(reference, dtype=float)
    curr = np.asarray(current, dtype=float)

    if len(ref) == 0 or len(curr) == 0:
        return 0.0

    # Derive quantile edges from reference
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(ref, quantiles)
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) < 2:
        # All values in reference are identical; cannot form bins
        return 0.0

    # Expand outermost edges to infinity to catch all current observations
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    curr_counts, _ = np.histogram(curr, bins=bin_edges)

    ref_pct = ref_counts / len(ref)
    curr_pct = curr_counts / len(curr)

    return calculate_psi_from_proportions(curr_pct, ref_pct, epsilon=epsilon)


def build_seasonal_reference(
    reference_df: pd.DataFrame,
    target_month: int,
    daylight_hours: set[int] | list[int] | None = None,
    timestamp_col: str = "event_timestamp",
) -> pd.DataFrame:
    """Extract a season-matched daylight reference slice from the Train/Dev partition.

    Matches calendar months (target_month +/- 1 month, wrapping Dec/Jan) and daylight
    hours to prevent seasonal astronomical cycles from appearing as distribution drift.

    Args:
        reference_df: Historical Train/Dev features dataframe.
        target_month: Month of current evaluation (1 to 12).
        daylight_hours: Optional set/list of hours of day to include.
        timestamp_col: Column name containing UTC timestamps.

    Returns:
        Filtered DataFrame representing the season-matched reference.
    """
    ts = pd.to_datetime(reference_df[timestamp_col], utc=True)
    m = int(target_month)
    prev_m = 12 if m == 1 else m - 1
    next_m = 1 if m == 12 else m + 1
    target_months = {prev_m, m, next_m}

    mask = ts.dt.month.isin(target_months)
    if daylight_hours is not None:
        mask = mask & ts.dt.hour.isin(set(daylight_hours))

    return reference_df[mask].copy()


def classify_weather_freshness(
    weather_run_age_hours: float | None,
    weather_missing: int | float | None = 0,
    nominal_lag_hours: float = 2.0,
    max_fallback_hours: float = 8.0,
) -> str:
    """Classify the freshness status of weather for a single forecast issue hour.

    Operational categories:
    - 'missing': Weather was unavailable / unobtainable (weather_missing == 1 or NaN age).
    - 'fresh': Run age <= nominal operational lag (nominally 2.0 hours).
    - 'fallback': Run age between nominal lag and max allowed fallback (nominally 2h < age <= 8h).
    - 'stale': Run age > max allowed fallback (> 8.0 hours).

    Future weather runs (run_age < 0) are strictly forbidden and raise ValueError.

    Args:
        weather_run_age_hours: Age of forecast initialization relative to issue time H.
        weather_missing: Missing flag indicator (1 if missing, 0 otherwise).
        nominal_lag_hours: Maximum age for fresh classification (default 2.0).
        max_fallback_hours: Maximum allowable age for fallback (default 8.0).

    Returns:
        Status string: 'fresh', 'fallback', 'stale', or 'missing'.
    """
    if weather_missing == 1 or weather_run_age_hours is None or pd.isna(weather_run_age_hours):
        return "missing"

    age = float(weather_run_age_hours)
    if age < 0:
        raise ValueError(
            f"Negative weather run age ({age:.2f}h) detected. "
            "Future weather substitution is forbidden."
        )

    if age <= nominal_lag_hours:
        return "fresh"
    if age <= max_fallback_hours:
        return "fallback"
    return "stale"


def classify_weather_freshness_series(
    df: pd.DataFrame,
    age_col: str = "weather_run_age_hours",
    missing_col: str = "weather_missing",
    nominal_lag_hours: float = 2.0,
    max_fallback_hours: float = 8.0,
) -> pd.Series:
    """Vectorized classification of weather freshness for a dataframe."""
    has_age = age_col in df.columns
    has_missing = missing_col in df.columns

    if not has_age and not has_missing:
        return pd.Series("missing", index=df.index)

    ages = df[age_col] if has_age else pd.Series(np.nan, index=df.index)
    missing = df[missing_col] if has_missing else pd.Series(0, index=df.index)

    # Check for illegal negative ages
    valid_ages = ages.dropna()
    if (valid_ages < 0).any():
        neg_count = int((valid_ages < 0).sum())
        raise ValueError(
            f"Found {neg_count} records with negative weather run age. "
            "Future weather is strictly prohibited."
        )

    cond_missing = (missing == 1) | ages.isna()
    cond_fresh = ~cond_missing & (ages <= nominal_lag_hours)
    cond_fallback = ~cond_missing & (ages > nominal_lag_hours) & (ages <= max_fallback_hours)
    cond_stale = ~cond_missing & (ages > max_fallback_hours)

    res = pd.Series("missing", index=df.index, dtype=object)
    res[cond_fresh] = "fresh"
    res[cond_fallback] = "fallback"
    res[cond_stale] = "stale"
    return res


class InputDriftMonitor:
    """Monitors 8 key numeric features for population drift against seasonal baseline.

    Operational Rule:
    - Minimum sample count: >= min_valid_rows (default 200) valid rows per comparison.
    - If valid rows < min_valid_rows, status is 'insufficient_data' (no drift decision).
    - Drift condition: PSI > psi_threshold (default 0.20) in >= min_drifted_features (2).
    - Alert trigger: Drift condition satisfied on consecutive_evals_required (2) checks.
    - Missing weather: Excluded per-feature; reported as data quality metric rather than drift.
    """

    def __init__(
        self,
        monitored_features: list[str] | None = None,
        psi_threshold: float = 0.20,
        min_drifted_features: int = 2,
        consecutive_evals_required: int = 2,
        min_valid_rows: int = 200,
        num_bins: int = 10,
        site_id: str = "2107",
        model_name: str = "solarcast-f1",
        model_version: str = "v2",
        schema_version: str = "1.1.0",
    ) -> None:
        self.monitored_features = monitored_features or list(MONITORED_FEATURES)
        self.psi_threshold = psi_threshold
        self.min_drifted_features = min_drifted_features
        self.consecutive_evals_required = consecutive_evals_required
        self.min_valid_rows = min_valid_rows
        self.num_bins = num_bins
        self.site_id = site_id
        self.model_name = model_name
        self.model_version = model_version
        self.schema_version = schema_version

        # State tracking
        self.consecutive_drift_count: int = 0
        self.event_counter: int = 1
        self.history: list[dict[str, Any]] = []

    def evaluate_check(
        self,
        eval_timestamp: pd.Timestamp,
        current_window_df: pd.DataFrame,
        reference_train_dev_df: pd.DataFrame,
        timestamp_col: str = "event_timestamp",
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Execute a single daily evaluation of input feature drift.

        Args:
            eval_timestamp: Virtual clock timestamp T of the daily check.
            current_window_df: Daylight valid records from trailing 14-day window [T - 14d, T).
            reference_train_dev_df: Historical Train/Dev partition features dataframe.
            timestamp_col: Column name containing timestamps.

        Returns:
            Tuple of (evaluation_summary_dict, alert_event_or_None).
        """
        eval_dt = pd.to_datetime(eval_timestamp, utc=True)
        target_month = eval_dt.month

        # Determine daylight hours present in current window
        if len(current_window_df) > 0:
            window_ts = pd.to_datetime(current_window_df[timestamp_col], utc=True)
            daylight_hours = list(window_ts.dt.hour.unique())
        else:
            daylight_hours = None

        # Build season-matched reference
        ref_matched = build_seasonal_reference(
            reference_train_dev_df,
            target_month=target_month,
            daylight_hours=daylight_hours,
            timestamp_col=timestamp_col,
        )

        n_curr_total = len(current_window_df)
        n_ref_total = len(ref_matched)

        # Evaluate PSI per feature
        feature_psi: dict[str, float] = {}
        feature_drifted: dict[str, bool] = {}
        feature_valid_counts: dict[str, dict[str, int]] = {}
        missing_weather_counts: dict[str, int] = {}

        drifted_features_list: list[str] = []
        has_insufficient_data = False

        for feat in self.monitored_features:
            if feat not in current_window_df.columns or feat not in ref_matched.columns:
                has_insufficient_data = True
                continue

            curr_series = current_window_df[feat].dropna()
            ref_series = ref_matched[feat].dropna()

            n_valid_curr = int(len(curr_series))
            n_valid_ref = int(len(ref_series))
            nan_curr = n_curr_total - n_valid_curr

            feature_valid_counts[feat] = {
                "valid_current": n_valid_curr,
                "nan_current": nan_curr,
                "valid_reference": n_valid_ref,
            }

            if feat in ("tmp_c", "tcdc_pct", "clearsky_ghi_cloud_adj_wm2"):
                missing_weather_counts[feat] = nan_curr

            # Enforce 200 valid row threshold per comparison
            if n_valid_curr < self.min_valid_rows or n_valid_ref < self.min_valid_rows:
                has_insufficient_data = True
                feature_psi[feat] = 0.0
                feature_drifted[feat] = False
                continue

            psi_val = calculate_psi(
                reference=ref_series.values,
                current=curr_series.values,
                num_bins=self.num_bins,
            )
            feature_psi[feat] = round(psi_val, 4)
            is_drift = psi_val > self.psi_threshold
            feature_drifted[feat] = is_drift
            if is_drift:
                drifted_features_list.append(feat)

        # Decision logic
        if has_insufficient_data:
            decision = "insufficient_data"
            self.consecutive_drift_count = 0
            alert_event = None
        else:
            drift_detected_today = len(drifted_features_list) >= self.min_drifted_features
            if drift_detected_today:
                self.consecutive_drift_count += 1
                decision = "drift_detected"
            else:
                self.consecutive_drift_count = 0
                decision = "stable"

            alert_event = None
            if self.consecutive_drift_count >= self.consecutive_evals_required:
                # Trigger alert event conforming to alert_event_schema v1.1.0
                start_w = eval_dt - pd.Timedelta(days=14)
                ev_id = f"DRIFT-{eval_dt.strftime('%Y%m%d%H%M')}-{self.event_counter:03d}"
                self.event_counter += 1

                reason = (
                    f"Input feature drift detected: {len(drifted_features_list)} features "
                    f"exceeded PSI {self.psi_threshold:.2f} over "
                    f"{self.consecutive_evals_required} consecutive daily checks "
                    f"(drifted: {', '.join(drifted_features_list)})."
                )

                alert_event = {
                    "schema_version": self.schema_version,
                    "event_id": ev_id,
                    "site_id": self.site_id,
                    "alert_type": "input_drift",
                    "model_name": self.model_name,
                    "model_version": self.model_version,
                    "window_start_utc": start_w.isoformat(),
                    "window_end_utc": eval_dt.isoformat(),
                    "reason": reason,
                    "created_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
                    "drift_details": {
                        "drifted_features": drifted_features_list,
                        "drifted_feature_count": len(drifted_features_list),
                        "psi_scores": {f: feature_psi[f] for f in drifted_features_list},
                        "consecutive_checks": self.consecutive_drift_count,
                        "valid_current_rows": n_curr_total,
                        "valid_reference_rows": n_ref_total,
                    },
                }

        summary = {
            "eval_timestamp_utc": eval_dt.isoformat(),
            "decision": decision,
            "consecutive_drift_checks": self.consecutive_drift_count,
            "drifted_feature_count": len(drifted_features_list),
            "drifted_features": drifted_features_list,
            "psi_scores": feature_psi,
            "current_window_rows": n_curr_total,
            "reference_matched_rows": n_ref_total,
            "missing_weather_counts": missing_weather_counts,
            "alert_triggered": alert_event is not None,
        }
        self.history.append(summary)
        return summary, alert_event


class WeatherFreshnessMonitor:
    """Monitors NWP forecast weather freshness over rolling 7-day windows.

    Operational Rule:
    - Evaluates daylight issue hours over trailing 7 days.
    - Classifies each hour as 'fresh' (<= 2h), 'fallback' (2h-8h), 'stale' (> 8h), or 'missing'.
    - If non-fresh share (fallback + stale + missing) exceeds non_fresh_threshold (default 0.15)
      with >= min_eval_hours (default 30) valid daylight hours, emits a 'data_quality' alert event.
    """

    def __init__(
        self,
        nominal_lag_hours: float = 2.0,
        max_fallback_hours: float = 8.0,
        non_fresh_threshold: float = 0.15,
        min_eval_hours: int = 30,
        site_id: str = "2107",
        model_name: str = "solarcast-f1",
        model_version: str = "v2",
        schema_version: str = "1.1.0",
    ) -> None:
        self.nominal_lag_hours = nominal_lag_hours
        self.max_fallback_hours = max_fallback_hours
        self.non_fresh_threshold = non_fresh_threshold
        self.min_eval_hours = min_eval_hours
        self.site_id = site_id
        self.model_name = model_name
        self.model_version = model_version
        self.schema_version = schema_version
        self.event_counter: int = 1
        self.history: list[dict[str, Any]] = []

    def evaluate_check(
        self,
        eval_timestamp: pd.Timestamp,
        window_7d_df: pd.DataFrame,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Evaluate weather freshness for trailing 7-day daylight window [T - 7d, T)."""
        eval_dt = pd.to_datetime(eval_timestamp, utc=True)
        start_w = eval_dt - pd.Timedelta(days=7)

        n_total = len(window_7d_df)
        if n_total < self.min_eval_hours:
            summary = {
                "eval_timestamp_utc": eval_dt.isoformat(),
                "status": "insufficient_data",
                "total_hours": n_total,
                "min_required_hours": self.min_eval_hours,
                "alert_triggered": False,
            }
            self.history.append(summary)
            return summary, None

        classes = classify_weather_freshness_series(
            window_7d_df,
            nominal_lag_hours=self.nominal_lag_hours,
            max_fallback_hours=self.max_fallback_hours,
        )

        counts = {
            "fresh": int((classes == "fresh").sum()),
            "fallback": int((classes == "fallback").sum()),
            "stale": int((classes == "stale").sum()),
            "missing": int((classes == "missing").sum()),
        }
        non_fresh_count = counts["fallback"] + counts["stale"] + counts["missing"]
        non_fresh_share = round(non_fresh_count / n_total, 4)

        alert_event = None
        if non_fresh_share > self.non_fresh_threshold:
            ev_id = f"DQ-{eval_dt.strftime('%Y%m%d%H%M')}-{self.event_counter:03d}"
            self.event_counter += 1

            reason = (
                f"Degraded weather freshness: {non_fresh_share * 100:.1f}% non-fresh hours "
                f"in trailing 7-day window (threshold: {self.non_fresh_threshold * 100:.1f}%, "
                f"fresh: {counts['fresh']}, fallback: {counts['fallback']}, "
                f"stale: {counts['stale']}, missing: {counts['missing']})."
            )

            alert_event = {
                "schema_version": self.schema_version,
                "event_id": ev_id,
                "site_id": self.site_id,
                "alert_type": "data_quality",
                "model_name": self.model_name,
                "model_version": self.model_version,
                "window_start_utc": start_w.isoformat(),
                "window_end_utc": eval_dt.isoformat(),
                "reason": reason,
                "created_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
                "quality_details": {
                    "total_hours": n_total,
                    "non_fresh_share": non_fresh_share,
                    "threshold": self.non_fresh_threshold,
                    "breakdown": counts,
                },
            }

        summary = {
            "eval_timestamp_utc": eval_dt.isoformat(),
            "status": "alert" if alert_event else "nominal",
            "total_hours": n_total,
            "non_fresh_share": non_fresh_share,
            "breakdown": counts,
            "alert_triggered": alert_event is not None,
        }
        self.history.append(summary)
        return summary, alert_event
