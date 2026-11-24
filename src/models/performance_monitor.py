"""Delayed-label performance monitoring for SolarCast Mode F1.

This module implements:
1. Virtual clock label visibility join: At virtual clock time T, ground-truth production
   actuals are ONLY accessible if interval_end_utc + production_available_delay_minutes <= T.
   Labels arriving after T are strictly inaccessible.
2. 7-day rolling daylight MAE evaluation:
   - Enforces >= 30 valid daylight labels in the trailing 7-day window.
   - If valid labels < 30, states 'insufficient_labels' (or 'label_data_outage'
     if predictions exist without actuals), never triggering an accuracy alert
     or treating missing labels as zero error.
3. Degradation Alerting Rule:
   - 7-day daylight MAE > 1.25 * reference_mae on two consecutive daily checks.
   - Denominator floor applied to prevent division by near-zero.
   - Month-specific reference MAE (frozen in configs/monitoring.yaml) selected based on validation
     evidence over fixed reference, cutting false alerts by >50% and eliminating winter distortion.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

DEFAULT_MONITORING_CONFIG = Path("configs/monitoring.yaml")
DEFAULT_DATA_CONFIG = Path("configs/data.yaml")


def load_monitoring_config(config_path: Path = DEFAULT_MONITORING_CONFIG) -> dict[str, Any]:
    """Load monitoring configuration from YAML."""
    if not config_path.exists():
        return {
            "schema_version": "1.1.0",
            "site_id": "2107",
            "model_name": "solarcast-f1",
            "model_version": "v2",
            "performance": {
                "rolling_window_days": 7,
                "metric_name": "daylight_mae_kwh",
                "degradation_threshold_ratio": 1.25,
                "consecutive_checks_required": 2,
                "min_valid_labels": 30,
                "denominator_floor_kwh": 1.0,
                "label_delay_minutes": 15,
                "reference_definition": "month_specific",
                "fixed_reference_mae_kwh": 31.43,
                "month_specific_reference_mae_kwh": {
                    1: 50.64,
                    2: 39.06,
                    3: 44.01,
                    4: 37.65,
                    5: 29.91,
                    6: 23.25,
                    7: 16.67,
                    8: 21.84,
                    9: 28.51,
                    10: 29.62,
                    11: 36.66,
                    12: 39.56,
                },
            },
        }
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def join_predictions_with_delayed_labels(
    predictions_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    virtual_clock_utc: pd.Timestamp,
    delay_minutes: int = 15,
    pred_time_col: str = "interval_start_utc",
    actual_time_col: str = "interval_start_utc",
    site_col: str = "site_id",
) -> pd.DataFrame:
    """Join forecast predictions with delayed ground-truth actuals using virtual clock visibility.

    Enforces temporal causality: an actual label is visible at virtual clock time T if and only if
    actual_available_at_utc = interval_end_utc + delay_minutes <= T.
    If actual_available_at_utc > T, actual values are masked as NaN and
    flagged as 'UNAVAILABLE_PENDING'.

    Args:
        predictions_df: Predictions dataframe containing forecast values and timestamps.
        actuals_df: Ground truth dataframe containing target_kwh and quality_flag.
        virtual_clock_utc: Current simulated evaluation timestamp T (UTC).
        delay_minutes: Telemetry transmission delay after interval close (nominally 15 min).
        pred_time_col: Timestamp column in predictions_df.
        actual_time_col: Timestamp column in actuals_df.
        site_col: Site identifier column name.

    Returns:
        Merged DataFrame with causality-safe actual_kwh and quality_flag columns.
    """
    v_clock = pd.to_datetime(virtual_clock_utc, utc=True)

    preds = predictions_df.copy()
    if pred_time_col not in preds.columns and "event_timestamp" in preds.columns:
        pred_time_col = "event_timestamp"
    preds[pred_time_col] = pd.to_datetime(preds[pred_time_col], utc=True)

    acts = actuals_df.copy()
    acts[actual_time_col] = pd.to_datetime(acts[actual_time_col], utc=True)
    if "interval_end_utc" not in acts.columns:
        acts["interval_end_utc"] = acts[actual_time_col] + pd.Timedelta(hours=1)
    else:
        acts["interval_end_utc"] = pd.to_datetime(acts["interval_end_utc"], utc=True)

    # Compute label arrival timestamp
    acts["actual_available_at_utc"] = acts["interval_end_utc"] + pd.Timedelta(
        minutes=delay_minutes
    )

    # Filter acts by virtual clock visibility: records where available_at <= T
    # Records arriving after T are treated as unobserved
    is_visible = acts["actual_available_at_utc"] <= v_clock
    acts_visible = acts.copy()
    acts_visible.loc[~is_visible, "target_kwh"] = np.nan
    acts_visible.loc[~is_visible, "quality_flag"] = "UNAVAILABLE_PENDING"

    # Merge predictions with visible actuals
    left_on = [pred_time_col]
    right_on = [actual_time_col]
    if site_col in preds.columns and site_col in acts.columns:
        left_on.append(site_col)
        right_on.append(site_col)

    actual_keep = right_on + ["target_kwh", "quality_flag", "actual_available_at_utc"]
    if "is_daylight" in acts_visible.columns and "is_daylight" not in preds.columns:
        actual_keep.append("is_daylight")

    joined = pd.merge(
        preds,
        acts_visible[actual_keep],
        left_on=left_on,
        right_on=right_on,
        how="left",
    )
    if "interval_start_utc" not in joined.columns and pred_time_col in joined.columns:
        joined["interval_start_utc"] = joined[pred_time_col]
    return joined


class PerformanceMonitor:
    """Monitors model forecast performance over rolling 7-day daylight windows with delayed labels.

    Operational Rule:
    - Trailing window: 7 days [T - 7d, T).
    - Minimum sample count: >= min_valid_labels (default 30) valid daylight labels.
    - If valid labels < min_valid_labels:
      - If daylight predictions exist: status is 'label_data_outage'.
      - Else: status is 'insufficient_labels'.
      - Never triggers an accuracy alert; never treats missing labels as zero error.
    - Accuracy degradation: 7-day daylight MAE > degradation_threshold_ratio * reference_mae.
    - Alert trigger: Degradation condition breached on consecutive_checks_required (2) daily checks.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        reference_definition: str | None = None,
        site_id: str | None = None,
        model_name: str | None = None,
        model_version: str | None = None,
        schema_version: str | None = None,
    ) -> None:
        cfg = config or load_monitoring_config()
        perf_cfg = cfg.get("performance", {})

        self.site_id = site_id or cfg.get("site_id", "2107")
        self.model_name = model_name or cfg.get("model_name", "solarcast-f1")
        self.model_version = model_version or cfg.get("model_version", "v2")
        self.schema_version = schema_version or cfg.get("schema_version", "1.1.0")

        self.rolling_window_days: int = int(perf_cfg.get("rolling_window_days", 7))
        self.degradation_ratio: float = float(perf_cfg.get("degradation_threshold_ratio", 1.25))
        self.consecutive_checks_required: int = int(
            perf_cfg.get("consecutive_checks_required", 2)
        )
        self.min_valid_labels: int = int(perf_cfg.get("min_valid_labels", 30))
        self.denominator_floor: float = float(perf_cfg.get("denominator_floor_kwh", 1.0))
        self.label_delay_minutes: int = int(perf_cfg.get("label_delay_minutes", 15))

        self.reference_definition = reference_definition or perf_cfg.get(
            "reference_definition", "month_specific"
        )
        self.fixed_reference_mae = float(perf_cfg.get("fixed_reference_mae_kwh", 31.43))
        raw_month_ref = perf_cfg.get("month_specific_reference_mae_kwh", {})
        self.month_specific_ref = {int(k): float(v) for k, v in raw_month_ref.items()}

        # State tracking
        self.consecutive_degradation_count: int = 0
        self.event_counter: int = 1
        self.history: list[dict[str, Any]] = []

    def get_reference_mae(self, month: int) -> float:
        """Retrieve the frozen reference MAE according to the selected reference definition."""
        if self.reference_definition == "month_specific":
            base_ref = self.month_specific_ref.get(month, self.fixed_reference_mae)
        else:
            base_ref = self.fixed_reference_mae
        return max(base_ref, self.denominator_floor)

    def evaluate_check(
        self,
        eval_timestamp: pd.Timestamp,
        joined_window_df: pd.DataFrame,
        actual_col: str = "target_kwh",
        pred_col: str = "prediction_kwh",
        flag_col: str = "quality_flag",
        daylight_col: str = "is_daylight",
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Execute a daily performance monitor evaluation for trailing 7 days [T - 7d, T).

        Args:
            eval_timestamp: Virtual clock evaluation timestamp T (UTC).
            joined_window_df: Causality-joined dataframe for the 7-day window.
            actual_col: Target/actual kWh column name.
            pred_col: Prediction kWh column name.
            flag_col: Quality flag column name.
            daylight_col: Boolean flag for daylight hours.

        Returns:
            Tuple of (summary_dict, alert_event_or_None).
        """
        eval_dt = pd.to_datetime(eval_timestamp, utc=True)
        start_w = eval_dt - pd.Timedelta(days=self.rolling_window_days)

        # Filter for daylight issue hours
        if daylight_col in joined_window_df.columns:
            daylight_df = joined_window_df[joined_window_df[daylight_col] == True]  # noqa: E712
        elif "solar_elevation_target_deg" in joined_window_df.columns:
            daylight_df = joined_window_df[joined_window_df["solar_elevation_target_deg"] > 0]
        else:
            daylight_df = joined_window_df

        total_daylight_predictions = len(daylight_df)

        # Filter for valid joined ground truth labels
        valid_labels_mask = (
            daylight_df[actual_col].notna()
            & daylight_df[pred_col].notna()
            & (daylight_df[flag_col] == "VALID")
        )
        valid_df = daylight_df[valid_labels_mask]
        n_valid = len(valid_df)

        # Check sample count floor
        if n_valid < self.min_valid_labels:
            if total_daylight_predictions >= self.min_valid_labels:
                status = "label_data_outage"
            else:
                status = "insufficient_labels"

            self.consecutive_degradation_count = 0
            summary = {
                "eval_timestamp_utc": eval_dt.isoformat(),
                "status": status,
                "total_daylight_predictions": total_daylight_predictions,
                "valid_daylight_labels": n_valid,
                "min_required_labels": self.min_valid_labels,
                "mae_7d_kwh": None,
                "reference_mae_kwh": None,
                "consecutive_degradation_checks": 0,
                "alert_triggered": False,
            }
            self.history.append(summary)
            return summary, None

        # Compute 7-day daylight MAE
        residuals = np.abs(valid_df[actual_col].values - valid_df[pred_col].values)
        mae_7d = float(np.mean(residuals))

        month = eval_dt.month
        ref_mae = self.get_reference_mae(month)
        threshold_mae = self.degradation_ratio * ref_mae
        ratio = round(mae_7d / ref_mae, 4)

        is_degraded = mae_7d > threshold_mae
        if is_degraded:
            self.consecutive_degradation_count += 1
            status = "degradation_breach"
        else:
            self.consecutive_degradation_count = 0
            status = "nominal"

        alert_event = None
        if self.consecutive_degradation_count >= self.consecutive_checks_required:
            ev_id = f"PERF-{eval_dt.strftime('%Y%m%d%H%M')}-{self.event_counter:03d}"
            self.event_counter += 1

            reason = (
                f"Model performance degradation: 7-day daylight MAE ({mae_7d:.2f} kWh) "
                f"exceeded {self.degradation_ratio:.2f}x reference "
                f"({ref_mae:.2f} kWh, threshold {threshold_mae:.2f} kWh) "
                f"on {self.consecutive_checks_required} consecutive daily checks "
                f"(ratio: {ratio:.2f})."
            )

            alert_event = {
                "schema_version": self.schema_version,
                "event_id": ev_id,
                "site_id": self.site_id,
                "alert_type": "performance_degradation",
                "model_name": self.model_name,
                "model_version": self.model_version,
                "window_start_utc": start_w.isoformat(),
                "window_end_utc": eval_dt.isoformat(),
                "reason": reason,
                "created_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
                "performance_details": {
                    "metric_name": "daylight_mae_kwh",
                    "mae_7d_kwh": round(mae_7d, 2),
                    "reference_mae_kwh": round(ref_mae, 2),
                    "reference_definition": self.reference_definition,
                    "degradation_ratio": ratio,
                    "consecutive_checks": self.consecutive_degradation_count,
                    "valid_label_count": n_valid,
                },
            }

        summary = {
            "eval_timestamp_utc": eval_dt.isoformat(),
            "status": "alert" if alert_event else status,
            "total_daylight_predictions": total_daylight_predictions,
            "valid_daylight_labels": n_valid,
            "mae_7d_kwh": round(mae_7d, 2),
            "reference_mae_kwh": round(ref_mae, 2),
            "ratio": ratio,
            "consecutive_degradation_checks": self.consecutive_degradation_count,
            "alert_triggered": alert_event is not None,
        }
        self.history.append(summary)
        return summary, alert_event
