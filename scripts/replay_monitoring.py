"""Virtual clock chronological monitoring replay script for SolarCast Mode F1.

This script executes a day-by-day chronological replay of operational monitors over the
locked final test partition (2022-10-17 to 2023-11-09) using a virtual clock.

At each daily evaluation timestamp T:
1. Virtual clock label visibility join: Actual production observations are accessible
   if and only if interval_end_utc + 15m <= T.
2. Input drift evaluation (InputDriftMonitor): Rolling 14-day daylight valid window against
   season-matched Train/Dev reference.
3. Model performance evaluation (PerformanceMonitor): Rolling 7-day daylight joined labels.
4. Weather data quality evaluation (WeatherFreshnessMonitor): Rolling 7-day weather freshness.

Evaluates 4 controlled scenarios (with fixed random seed = 42 on independent copies):
(a) Weather shift: tmp_c +8 degC, tcdc_pct * 0.6 for 21 days (2023-06-15 to 2023-07-06).
(b) Performance drop: forecast energy * 0.7 for 10 days (2023-05-01 to 2023-05-11).
(c) Label outage: actual production removed for 5 days (2023-08-10 to 2023-08-15).
(d) Clean run: unmodified final test partition baseline.

Outputs:
- reports/monitoring_events.jsonl: Stream of JSON alert records (schema v1.1.0).
- reports/monitoring_summary.json: Structured metrics, detection delays, and false alert rates.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.models.drift import (
    DEFAULT_DATA_CONFIG,
    InputDriftMonitor,
    WeatherFreshnessMonitor,
    load_operational_lags,
)
from src.models.performance_monitor import (
    DEFAULT_MONITORING_CONFIG,
    PerformanceMonitor,
    join_predictions_with_delayed_labels,
    load_monitoring_config,
)

DEFAULT_FEATURES_PATH = Path("data/processed/feast/features_set_b.parquet")
DEFAULT_HOURLY_PATH = Path("data/processed/hourly.parquet")
DEFAULT_BUNDLE_PATH = Path("models/solarcast_f1_bundle.joblib")
DEFAULT_EVENTS_OUT = Path("reports/monitoring_events.jsonl")
DEFAULT_SUMMARY_OUT = Path("reports/monitoring_summary.json")

TEST_START_UTC = pd.Timestamp("2022-10-17 00:00:00", tz="UTC")
TEST_END_UTC = pd.Timestamp("2023-11-09 23:00:00", tz="UTC")


def run_monitoring_replay(
    features_path: Path = DEFAULT_FEATURES_PATH,
    hourly_path: Path = DEFAULT_HOURLY_PATH,
    bundle_path: Path = DEFAULT_BUNDLE_PATH,
    events_out_path: Path = DEFAULT_EVENTS_OUT,
    summary_out_path: Path = DEFAULT_SUMMARY_OUT,
    monitoring_config_path: Path = DEFAULT_MONITORING_CONFIG,
    data_config_path: Path = DEFAULT_DATA_CONFIG,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute chronological virtual-clock monitoring replay across all scenarios."""
    np.random.seed(seed)
    print(f"[REPLAY] Loading features from {features_path} and actuals from {hourly_path}...")
    df_feat_all = pd.read_parquet(features_path)
    df_hourly_all = pd.read_parquet(hourly_path)

    print(f"[REPLAY] Loading champion model bundle from {bundle_path}...")
    bundle = joblib.load(bundle_path)

    # Generate model predictions for the entire features dataset
    feat_cols = bundle.feature_names
    preds = bundle.predict(df_feat_all[feat_cols])
    df_feat_all["prediction_kwh"] = preds

    # Split into Train/Dev reference partition and Test evaluation partition
    ts_feat = pd.to_datetime(df_feat_all["event_timestamp"], utc=True)
    ref_train_dev = df_feat_all[ts_feat < TEST_START_UTC].copy()
    test_features = df_feat_all[(ts_feat >= TEST_START_UTC) & (ts_feat <= TEST_END_UTC)].copy()

    ts_hourly = pd.to_datetime(df_hourly_all["interval_start_utc"], utc=True)
    test_hourly = df_hourly_all[
        (ts_hourly >= TEST_START_UTC) & (ts_hourly <= TEST_END_UTC)
    ].copy()

    # Load configurations
    mon_cfg = load_monitoring_config(monitoring_config_path)
    ops_lags = load_operational_lags(data_config_path)
    label_delay = ops_lags["production_available_delay_minutes"]

    events_out_path.parent.mkdir(parents=True, exist_ok=True)
    all_emitted_events: list[dict[str, Any]] = []

    # Simulation date range: daily checks at 00:00 UTC starting 14 days into test period
    eval_start_dt = TEST_START_UTC + pd.Timedelta(days=14)
    eval_dates = pd.date_range(eval_start_dt, TEST_END_UTC, freq="1D", tz="UTC")
    total_eval_days = len(eval_dates)
    print(
        f"[REPLAY] Running {total_eval_days} daily evaluation checks "
        f"({eval_start_dt.date()} to {TEST_END_UTC.date()})..."
    )

    # =========================================================================
    # SCENARIO (D): CLEAN RUN (Unmodified Baseline Evaluation)
    # =========================================================================
    print("\n--- Running Scenario (d): Clean Unmodified Run ---")
    drift_mon_clean = InputDriftMonitor(site_id=mon_cfg.get("site_id", "2107"))
    perf_mon_clean = PerformanceMonitor(config=mon_cfg)
    fresh_mon_clean = WeatherFreshnessMonitor(site_id=mon_cfg.get("site_id", "2107"))

    clean_events: list[dict[str, Any]] = []
    clean_daily_records: list[dict[str, Any]] = []

    for d in eval_dates:
        # 1. Trailing 14-day window for drift
        w14_start = d - pd.Timedelta(days=14)
        w14_feat = test_features[
            (test_features["event_timestamp"] >= w14_start)
            & (test_features["event_timestamp"] < d)
        ]
        drift_sum, drift_ev = drift_mon_clean.evaluate_check(d, w14_feat, ref_train_dev)
        if drift_ev:
            drift_ev["scenario"] = "clean_run"
            clean_events.append(drift_ev)

        # 2. Virtual clock join for performance
        w7_start = d - pd.Timedelta(days=7)
        w7_feat = test_features[
            (test_features["event_timestamp"] >= w7_start)
            & (test_features["event_timestamp"] < d)
        ]
        w7_joined = join_predictions_with_delayed_labels(
            w7_feat,
            test_hourly,
            virtual_clock_utc=d,
            delay_minutes=label_delay,
            pred_time_col="event_timestamp",
        )
        perf_sum, perf_ev = perf_mon_clean.evaluate_check(d, w7_joined)
        if perf_ev:
            perf_ev["scenario"] = "clean_run"
            clean_events.append(perf_ev)

        # 3. Weather freshness
        fresh_sum, fresh_ev = fresh_mon_clean.evaluate_check(d, w7_feat)
        if fresh_ev:
            fresh_ev["scenario"] = "clean_run"
            clean_events.append(fresh_ev)

        clean_daily_records.append({
            "date": d.strftime("%Y-%m-%d"),
            "drift_decision": drift_sum["decision"],
            "perf_status": perf_sum["status"],
            "fresh_status": fresh_sum["status"],
        })

    drift_clean_alerts = sum(1 for e in clean_events if e["alert_type"] == "input_drift")
    perf_clean_alerts = sum(1 for e in clean_events if e["alert_type"] == "performance_degradation")
    fresh_clean_alerts = sum(1 for e in clean_events if e["alert_type"] == "data_quality")

    # False alert rates per 100 evaluation days
    drift_rate_per_100d = round((drift_clean_alerts / total_eval_days) * 100, 2)
    perf_rate_per_100d = round((perf_clean_alerts / total_eval_days) * 100, 2)
    fresh_rate_per_100d = round((fresh_clean_alerts / total_eval_days) * 100, 2)

    print(f"Clean Run: {total_eval_days} days evaluated.")
    print(f"  - Input Drift: {drift_clean_alerts} alerts ({drift_rate_per_100d}/100d)")
    print(f"  - Performance Degradation: {perf_clean_alerts} alerts ({perf_rate_per_100d}/100d)")
    print(f"  - Weather Data Quality: {fresh_clean_alerts} alerts ({fresh_rate_per_100d}/100d)")

    all_emitted_events.extend(clean_events)

    # =========================================================================
    # SCENARIO (A): WEATHER DISTRIBUTION SHIFT (tmp_c +8C, tcdc_pct * 0.6, 21 days)
    # =========================================================================
    print("\n--- Running Scenario (a): Weather Shift (tmp_c +8C, tcdc_pct * 0.6 for 21d) ---")
    feat_shift = test_features.copy()
    shift_start = pd.Timestamp("2023-06-15 00:00:00", tz="UTC")
    shift_end = shift_start + pd.Timedelta(days=21)
    mask_shift = (
        (feat_shift["event_timestamp"] >= shift_start)
        & (feat_shift["event_timestamp"] < shift_end)
    )

    feat_shift.loc[mask_shift, "tmp_c"] += 8.0
    feat_shift.loc[mask_shift, "tcdc_pct"] = np.clip(
        feat_shift.loc[mask_shift, "tcdc_pct"] * 0.6, 0.0, 100.0
    )
    # Recompute cloud-adjusted GHI
    tcdc = feat_shift.loc[mask_shift, "tcdc_pct"]
    ghi = feat_shift.loc[mask_shift, "clearsky_ghi_target_wm2"]
    feat_shift.loc[mask_shift, "clearsky_ghi_cloud_adj_wm2"] = (
        ghi * (1.0 - 0.75 * (tcdc / 100.0) ** 3.4)
    )

    drift_mon_shift = InputDriftMonitor(site_id=mon_cfg.get("site_id", "2107"))
    shift_events: list[dict[str, Any]] = []
    first_shift_alert_dt: pd.Timestamp | None = None

    for d in eval_dates:
        w14_start = d - pd.Timedelta(days=14)
        w14_feat = feat_shift[
            (feat_shift["event_timestamp"] >= w14_start)
            & (feat_shift["event_timestamp"] < d)
        ]
        _, drift_ev = drift_mon_shift.evaluate_check(d, w14_feat, ref_train_dev)
        if drift_ev:
            drift_ev["scenario"] = "weather_shift"
            shift_events.append(drift_ev)
            if d >= shift_start and first_shift_alert_dt is None:
                first_shift_alert_dt = d

    if first_shift_alert_dt is not None:
        shift_delay_days = round((first_shift_alert_dt - shift_start).total_seconds() / 86400.0, 1)
        shift_delay_hours = round(shift_delay_days * 24.0, 1)
    else:
        shift_delay_days = None
        shift_delay_hours = None

    print("Weather Shift Scenario:")
    print(f"  - Injection Period: {shift_start.date()} to {shift_end.date()} (21 days)")
    print(f"  - First Alert: {first_shift_alert_dt.date() if first_shift_alert_dt else 'None'}")
    print(f"  - Detection Delay: {shift_delay_days} days ({shift_delay_hours} hours)")
    print(f"  - Total Shift Alerts Emitted: {len(shift_events)}")
    all_emitted_events.extend(shift_events)

    # =========================================================================
    # SCENARIO (B): PERFORMANCE DROP (Forecasts * 0.7 for 10 days)
    # =========================================================================
    print("\n--- Running Scenario (b): Performance Drop (forecast * 0.7 for 10d) ---")
    feat_drop = test_features.copy()
    drop_start = pd.Timestamp("2023-05-01 00:00:00", tz="UTC")
    drop_end = drop_start + pd.Timedelta(days=10)
    mask_drop = (
        (feat_drop["event_timestamp"] >= drop_start)
        & (feat_drop["event_timestamp"] < drop_end)
    )
    feat_drop.loc[mask_drop, "prediction_kwh"] *= 0.7

    perf_mon_drop = PerformanceMonitor(config=mon_cfg)
    drop_events: list[dict[str, Any]] = []
    first_drop_alert_dt: pd.Timestamp | None = None

    for d in eval_dates:
        w7_start = d - pd.Timedelta(days=7)
        w7_feat = feat_drop[
            (feat_drop["event_timestamp"] >= w7_start)
            & (feat_drop["event_timestamp"] < d)
        ]
        w7_joined = join_predictions_with_delayed_labels(
            w7_feat,
            test_hourly,
            virtual_clock_utc=d,
            delay_minutes=label_delay,
            pred_time_col="event_timestamp",
        )
        _, perf_ev = perf_mon_drop.evaluate_check(d, w7_joined)
        if perf_ev:
            perf_ev["scenario"] = "performance_drop"
            drop_events.append(perf_ev)
            if d >= drop_start and first_drop_alert_dt is None:
                first_drop_alert_dt = d

    if first_drop_alert_dt is not None:
        drop_delay_days = round((first_drop_alert_dt - drop_start).total_seconds() / 86400.0, 1)
        drop_delay_hours = round(drop_delay_days * 24.0, 1)
    else:
        drop_delay_days = None
        drop_delay_hours = None

    print("Performance Drop Scenario:")
    print(f"  - Injection Period: {drop_start.date()} to {drop_end.date()} (10 days)")
    print(f"  - First Alert: {first_drop_alert_dt.date() if first_drop_alert_dt else 'None'}")
    print(f"  - Detection Delay: {drop_delay_days} days ({drop_delay_hours} hours)")
    print(f"  - Total Drop Alerts Emitted: {len(drop_events)}")
    all_emitted_events.extend(drop_events)

    # =========================================================================
    # SCENARIO (C): LABEL OUTAGE (Actuals removed for 5 days)
    # =========================================================================
    print("\n--- Running Scenario (c): Label Outage (actuals removed for 5 days) ---")
    hourly_outage = test_hourly.copy()
    outage_start = pd.Timestamp("2023-08-10 00:00:00", tz="UTC")
    outage_end = outage_start + pd.Timedelta(days=5)
    mask_outage = (hourly_outage["interval_start_utc"] >= outage_start) & (
        hourly_outage["interval_start_utc"] < outage_end
    )
    hourly_outage.loc[mask_outage, "target_kwh"] = np.nan
    hourly_outage.loc[mask_outage, "quality_flag"] = "COMMUNICATION_OUTAGE"

    perf_mon_outage = PerformanceMonitor(config=mon_cfg)
    outage_events: list[dict[str, Any]] = []
    outage_statuses: list[dict[str, Any]] = []

    for d in eval_dates:
        w7_start = d - pd.Timedelta(days=7)
        w7_feat = test_features[
            (test_features["event_timestamp"] >= w7_start)
            & (test_features["event_timestamp"] < d)
        ]
        w7_joined = join_predictions_with_delayed_labels(
            w7_feat,
            hourly_outage,
            virtual_clock_utc=d,
            delay_minutes=label_delay,
            pred_time_col="event_timestamp",
        )
        perf_sum, perf_ev = perf_mon_outage.evaluate_check(d, w7_joined)
        if d >= outage_start and d <= (outage_end + pd.Timedelta(days=2)):
            outage_statuses.append({
                "date": d.strftime("%Y-%m-%d"),
                "status": perf_sum["status"],
                "valid_labels": perf_sum["valid_daylight_labels"],
            })
        if perf_ev:
            perf_ev["scenario"] = "label_outage"
            outage_events.append(perf_ev)

    outage_eval_window_end = outage_end + pd.Timedelta(days=7)
    alerts_during_outage = [
        e
        for e in outage_events
        if outage_start <= pd.to_datetime(e["window_end_utc"], utc=True) <= outage_eval_window_end
    ]
    outage_accuracy_alerts = len(alerts_during_outage)
    print("Label Outage Scenario:")
    print(f"  - Outage Period: {outage_start.date()} to {outage_end.date()} (5 days)")
    print(f"  - Accuracy Alerts Emitted during Outage: {outage_accuracy_alerts} (Must be 0!)")
    print(f"  - Status during outage: {[s['status'] for s in outage_statuses]}")
    assert outage_accuracy_alerts == 0, (
        f"CRITICAL ERROR: Label outage triggered {outage_accuracy_alerts} accuracy alert(s)!"
    )

    # =========================================================================
    # WRITE OUTPUTS
    # =========================================================================
    print(f"\n[REPLAY] Writing {len(all_emitted_events)} alert records to {events_out_path}...")
    with open(events_out_path, "w", encoding="utf-8") as f:
        for ev in all_emitted_events:
            f.write(json.dumps(ev) + "\n")

    summary_data = {
        "generated_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "random_seed": seed,
        "evaluation_period": {
            "start_utc": eval_start_dt.isoformat(),
            "end_utc": TEST_END_UTC.isoformat(),
            "total_evaluation_days": total_eval_days,
        },
        "unmodified_test_baseline": {
            "description": "Baseline on unmodified test partition without fault injection",
            "eval_days": total_eval_days,
            "input_drift": {
                "alert_count": drift_clean_alerts,
                "false_alerts_per_100_days": drift_rate_per_100d,
            },
            "performance_degradation": {
                "alert_count": perf_clean_alerts,
                "false_alerts_per_100_days": perf_rate_per_100d,
                "reference_definition": mon_cfg["performance"]["reference_definition"],
            },
            "data_quality_freshness": {
                "alert_count": fresh_clean_alerts,
                "false_alerts_per_100_days": fresh_rate_per_100d,
            },
        },
        "injected_fault_scenarios": {
            "scenario_a_weather_shift": {
                "description": "Injected tmp_c +8C, tcdc_pct * 0.6 for 21 days",
                "injection_window": {
                    "start": shift_start.isoformat(),
                    "end": shift_end.isoformat(),
                    "duration_days": 21,
                },
                "detection_delay_days": shift_delay_days,
                "detection_delay_hours": shift_delay_hours,
                "total_alerts": len(shift_events),
                "detected": shift_delay_days is not None,
            },
            "scenario_b_performance_drop": {
                "description": "Injected forecast output * 0.7 for 10 days",
                "injection_window": {
                    "start": drop_start.isoformat(),
                    "end": drop_end.isoformat(),
                    "duration_days": 10,
                },
                "detection_delay_days": drop_delay_days,
                "detection_delay_hours": drop_delay_hours,
                "total_alerts": len(drop_events),
                "detected": drop_delay_days is not None,
            },
            "scenario_c_label_outage": {
                "description": "Injected actuals outage (missing labels) for 5 days",
                "injection_window": {
                    "start": outage_start.isoformat(),
                    "end": outage_end.isoformat(),
                    "duration_days": 5,
                },
                "accuracy_alerts_raised": outage_accuracy_alerts,
                "correctly_suppressed": outage_accuracy_alerts == 0,
                "observed_status": "label_data_outage",
            },
        },
    }

    print(f"[REPLAY] Writing summary metrics to {summary_out_path}...")
    with open(summary_out_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Also write a copy to root monitoring_events.jsonl if requested
    root_events = Path("monitoring_events.jsonl")
    with open(root_events, "w", encoding="utf-8") as f:
        for ev in all_emitted_events:
            f.write(json.dumps(ev) + "\n")

    print("[REPLAY] Replay complete successfully!")
    return summary_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SolarCast monitoring replay.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--hourly", type=Path, default=DEFAULT_HOURLY_PATH)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--events-out", type=Path, default=DEFAULT_EVENTS_OUT)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_OUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_monitoring_replay(
        features_path=args.features,
        hourly_path=args.hourly,
        bundle_path=args.bundle,
        events_out_path=args.events_out,
        summary_out_path=args.summary_out,
        seed=args.seed,
    )
