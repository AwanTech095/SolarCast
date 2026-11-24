"""SolarCast Model Deviation Detection and Alerting Engine (Task M1.6 / Step 1 to Step 4).

Monitors forecast residuals (actual_kwh - forecast_kwh) produced by the champion model
(solarcast-f1 bundle) to flag sustained production anomalies or model deviations.
Strictly calibrated on out-of-sample validation folds with frozen thresholds.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit

from src.features.preprocessing import get_feature_names
from src.models.bundle import load_bundle

DEFAULT_EVAL_CONFIG = Path("configs/evaluation.yaml")
DEFAULT_DEVIATION_CONFIG = Path("configs/deviation.yaml")
DEFAULT_ALERT_SCHEMA = Path("configs/alert_event_schema.json")
DEFAULT_HOURLY_PATH = Path("data/processed/hourly.parquet")
DEFAULT_FEATURES_PATH = Path("data/processed/feast/features_set_b.parquet")
DEFAULT_BUNDLE_PATH = Path("models/solarcast_f1_bundle.joblib")
DEFAULT_VALIDATION_RESIDUALS_PATH = Path("data/processed/validation_residuals.parquet")
DEFAULT_REPORTS_DIR = Path("reports")

CHAMPION_FROZEN_PARAMS: dict[str, Any] = {
    "l2_regularization": 1.0,
    "learning_rate": 0.05,
    "max_iter": 200,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 100,
    "random_state": 42,
}


def assign_operating_cloud_bin(
    tcdc_pct: float | None,
    weather_missing: int | float | None = 0,
) -> str:
    """Assign operating cloud cover bin from forecast total cloud cover.

    Bins:
    - 'clear': tcdc_pct < 20.0
    - 'mixed': 20.0 <= tcdc_pct <= 80.0
    - 'overcast': tcdc_pct > 80.0
    - 'no_weather': weather data missing or unavailable
    """
    if weather_missing == 1 or tcdc_pct is None or pd.isna(tcdc_pct):
        return "no_weather"
    val = float(tcdc_pct)
    if val < 20.0:
        return "clear"
    elif val <= 80.0:
        return "mixed"
    else:
        return "overcast"


def assign_clearsky_index_bin(csi: float | None) -> str:
    """Assign clear-sky index bin from latest available clear-sky index."""
    if csi is None or pd.isna(csi):
        return "unknown"
    val = float(csi)
    if val < 0.4:
        return "csi_low"
    elif val <= 0.8:
        return "csi_mid"
    else:
        return "csi_high"


def compute_validation_residuals(
    eval_config_path: Path | str = DEFAULT_EVAL_CONFIG,
    hourly_path: Path | str = DEFAULT_HOURLY_PATH,
    features_path: Path | str = DEFAULT_FEATURES_PATH,
    output_path: Path | str | None = DEFAULT_VALIDATION_RESIDUALS_PATH,
) -> pd.DataFrame:
    """Compute out-of-sample forecast residuals across the 3 expanding validation folds.

    Uses the exact champion HistGradientBoosting model specification and feature set B,
    fit strictly on each fold's training slice and evaluated out-of-sample on the validation slice.
    Never uses in-sample training data or the held-out final test partition for calibration.

    Args:
        eval_config_path: Path to evaluation YAML configuration.
        hourly_path: Path to hourly target Parquet file.
        features_path: Path to precomputed Feature Set B Parquet file.
        output_path: Optional path to save validation residuals Parquet file.

    Returns:
        DataFrame of out-of-sample validation records with residuals and operating bins.
    """
    with open(eval_config_path, encoding="utf-8") as f:
        eval_cfg = yaml.safe_load(f)

    h_df = pd.read_parquet(hourly_path)
    f_df = pd.read_parquet(features_path)

    h_df["interval_start_utc"] = pd.to_datetime(h_df["interval_start_utc"], utc=True)
    f_df["event_timestamp"] = pd.to_datetime(f_df["event_timestamp"], utc=True)

    win_start = pd.Timestamp(eval_cfg["evaluation_window"]["start_utc"], tz="UTC")
    win_end = pd.Timestamp(eval_cfg["evaluation_window"]["end_utc"], tz="UTC")

    mask_win = (h_df["interval_start_utc"] >= win_start) & (h_df["interval_start_utc"] <= win_end)
    valid_mask = mask_win & (h_df["quality_flag"] == "VALID")
    valid_df = h_df[valid_mask].copy().reset_index(drop=True)

    # Merge target with Feature Set B
    merged = pd.merge(
        valid_df,
        f_df,
        left_on="interval_start_utc",
        right_on="event_timestamp",
        how="left",
    )

    # Strict 80/20 chronological partition
    split_idx = int(np.floor(len(merged) * 0.80))
    dev_df = merged.iloc[:split_idx].copy().reset_index(drop=True)

    feat_cols = get_feature_names("B")
    tscv = TimeSeriesSplit(n_splits=3)
    cap_limit = float(eval_cfg.get("capacity_dc_kw", 893.0))

    val_records: list[dict[str, Any]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(dev_df), 1):
        tr = dev_df.iloc[train_idx]
        va = dev_df.iloc[val_idx]

        # Training filter: VALID daylight hours, excluding SUSPECT_ZERO
        tr_mask = (tr["quality_flag"] == "VALID") & (tr["solar_elevation_deg"] > 0)
        if "SUSPECT_ZERO" in tr["quality_flag"].values:
            tr_mask = tr_mask & (tr["quality_flag"] != "SUSPECT_ZERO")

        X_tr = tr.loc[tr_mask, feat_cols].values
        y_tr = tr.loc[tr_mask, "target_kwh"].values

        model = HistGradientBoostingRegressor(**CHAMPION_FROZEN_PARAMS)
        model.fit(X_tr, y_tr)

        # Out-of-sample prediction on validation fold
        X_va = va[feat_cols].values
        raw_pred = model.predict(X_va)
        is_day = (va["solar_elevation_deg"] > 0).values
        pred = np.where(is_day, raw_pred, 0.0)
        pred = np.clip(pred, 0.0, cap_limit)

        for i in range(len(va)):
            tcdc = va["tcdc_pct"].iloc[i]
            w_miss = va["weather_missing"].iloc[i]
            csi = va["clearsky_index_latest_avail"].iloc[i]

            cloud_bin = assign_operating_cloud_bin(tcdc, w_miss)
            csi_bin = assign_clearsky_index_bin(csi)

            val_records.append(
                {
                    "fold": fold_idx,
                    "timestamp": va["interval_start_utc"].iloc[i],
                    "forecast": float(pred[i]),
                    "actual": float(va["target_kwh"].iloc[i]),
                    "residual": float(va["target_kwh"].iloc[i] - pred[i]),
                    "forecast_cloud_bin": cloud_bin,
                    "clearsky_index_bin": csi_bin,
                    "quality_flag": str(va["quality_flag"].iloc[i]),
                    "solar_elevation_deg": float(va["solar_elevation_deg"].iloc[i]),
                    "is_daylight": bool(is_day[i]),
                    "tcdc_pct": None if pd.isna(tcdc) else float(tcdc),
                    "weather_missing": int(w_miss) if pd.notna(w_miss) else 0,
                    "clearsky_index_latest_avail": None if pd.isna(csi) else float(csi),
                }
            )

    df_res = pd.DataFrame(val_records)
    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df_res.to_parquet(p, index=False)
        print(f"Saved {len(df_res):,} validation residual records to {p}")

    return df_res


def calibrate_deviation_bounds(
    val_residuals_df: pd.DataFrame,
    output_yaml_path: Path | str = DEFAULT_DEVIATION_CONFIG,
) -> dict[str, Any]:
    """Calibrate 1st and 99th percentile residual thresholds from validation data.

    Evaluated strictly on VALID daylight hours from the 3 expanding validation folds.
    Tests clear-sky index stratification against the 200-sample threshold rule.

    Args:
        val_residuals_df: DataFrame of validation residuals from compute_validation_residuals.
        output_yaml_path: Destination path for frozen deviation configuration YAML.

    Returns:
        Calibrated bounds configuration dictionary.
    """
    # Filter strictly to VALID daylight hours
    mask = (val_residuals_df["quality_flag"] == "VALID") & (val_residuals_df["is_daylight"])
    df_valid = val_residuals_df[mask].copy()

    # Check clear-sky index cross-stratification sample sizes
    ct = pd.crosstab(df_valid["forecast_cloud_bin"], df_valid["clearsky_index_bin"])
    min_sub_sample = int(ct.min().min())
    has_sub_200 = bool((ct < 200).any().any())

    # Document clear-sky index binning decision
    csi_note = (
        f"Clear-sky index cross-stratification rejected: min sub-bin sample size is "
        f"{min_sub_sample} (< 200 threshold; specifically no_weather x csi_low has "
        f"{ct.loc['no_weather', 'csi_low']} and no_weather x csi_high has "
        f"{ct.loc['no_weather', 'csi_high']} samples). "
        f"Merged into the 4 primary forecast cloud-cover bins."
    )

    bounds_dict: dict[str, dict[str, float]] = {}
    for cloud_bin, grp in df_valid.groupby("forecast_cloud_bin"):
        res = grp["residual"].values
        p1 = float(np.percentile(res, 1))
        p99 = float(np.percentile(res, 99))
        bounds_dict[str(cloud_bin)] = {
            "bound_low": round(p1, 2),
            "bound_high": round(p99, 2),
            "sample_count": int(len(res)),
            "mean_residual_kwh": round(float(np.mean(res)), 2),
            "std_residual_kwh": round(float(np.std(res)), 2),
        }

    cfg: dict[str, Any] = {
        "schema_version": "1.0.0",
        "calibrated_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "calibration_window": {
            "start_utc": df_valid["timestamp"].min().isoformat(),
            "end_utc": df_valid["timestamp"].max().isoformat(),
            "total_daylight_validation_samples": int(len(df_valid)),
        },
        "rule_parameters": {
            "consecutive_hours_threshold": 2,
            "evaluation_cadence": "hourly",
            "suppression_conditions": [
                "night (solar_elevation_deg <= 0)",
                "missing target (target_kwh is NaN)",
                "missing forecast",
                "quality_flag != 'VALID'",
            ],
            "reset_conditions": [
                "night",
                "missing or suppressed hour",
                "residual within calibrated bounds",
                "direction flip",
            ],
        },
        "stratification_decision": {
            "primary_stratification": "forecast_cloud_bin",
            "clearsky_index_bins_merged": has_sub_200,
            "justification": csi_note,
        },
        "bounds": bounds_dict,
    }

    out_p = Path(output_yaml_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"Calibrated and frozen deviation bounds saved to {out_p}")

    return cfg


def load_deviation_config(config_path: Path | str = DEFAULT_DEVIATION_CONFIG) -> dict[str, Any]:
    """Load frozen deviation detector configuration from YAML."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Deviation config not found at {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def detect_deviations(
    frame: pd.DataFrame,
    deviation_config: dict[str, Any] | None = None,
    site_id: str = "2107",
    model_name: str = "solarcast-f1",
    model_version: str = "v4",
) -> list[dict[str, Any]]:
    """Detect production deviation episodes exceeding calibrated bounds for 2+ consecutive hours.

    Alert rule:
    - Flags after 2 consecutive hourly intervals outside bounds in the same direction.
    - A night, missing target/forecast, or quality_flag other than VALID resets count.
    - One alert event covers a whole episode.
    - Detection delay = hours between the first anomalous hour and the alert hour (the 2nd hour).
    - Conforms to configs/alert_event_schema.json.

    Args:
        frame: Chronological DataFrame containing timestamps, actuals, forecasts, and flags.
        deviation_config: Calibrated bounds configuration dictionary.
        site_id: PV site identifier.
        model_name: Name of forecasting model.
        model_version: Version/tag of model bundle.

    Returns:
        List of alert event records matching configs/alert_event_schema.json.
    """
    if deviation_config is None:
        deviation_config = load_deviation_config()

    bounds = deviation_config["bounds"]
    schema_ver = deviation_config.get("schema_version", "1.0.0")

    df = frame.copy().sort_values("timestamp").reset_index(drop=True)
    alert_events: list[dict[str, Any]] = []

    # Active tracking state
    consecutive_count = 0
    active_direction: str | None = None
    episode_hours: list[dict[str, Any]] = []
    alert_triggered = False
    event_counter = 1

    def close_active_episode() -> None:
        nonlocal consecutive_count, active_direction, episode_hours, alert_triggered, event_counter
        if alert_triggered and len(episode_hours) >= 2:
            start_ts = episode_hours[0]["timestamp"]
            end_ts = episode_hours[-1]["timestamp"]
            residuals = [h["residual"] for h in episode_hours]
            mean_res = float(np.mean(residuals))
            b_name = episode_hours[0]["bin"]
            b_low = bounds[b_name]["bound_low"]
            b_high = bounds[b_name]["bound_high"]

            # Detection delay in hours: start hour to trigger hour (2nd anomalous hour)
            trigger_ts = episode_hours[1]["timestamp"]
            delay_h = (trigger_ts - start_ts).total_seconds() / 3600.0

            dir_str = "underproduction" if active_direction == "negative" else "overproduction"
            reason = (
                f"{len(episode_hours)} consecutive hours of {dir_str} in '{b_name}' "
                f"bin (mean residual: {mean_res:.1f} kWh, bounds: [{b_low}, {b_high}] kWh, "
                f"detection delay: {delay_h:.1f}h)"
            )

            event = {
                "schema_version": schema_ver,
                "event_id": f"DEV-{start_ts.strftime('%Y%m%d%H%M')}-{event_counter:03d}",
                "site_id": site_id,
                "model_name": model_name,
                "model_version": model_version,
                "window_start_utc": start_ts.isoformat(),
                "window_end_utc": end_ts.isoformat(),
                "direction": active_direction,
                "bin": b_name,
                "consecutive_hours": len(episode_hours),
                "mean_residual_kwh": round(mean_res, 2),
                "bound_low": b_low,
                "bound_high": b_high,
                "reason": reason,
                "created_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            }
            alert_events.append(event)
            event_counter += 1

        consecutive_count = 0
        active_direction = None
        episode_hours = []
        alert_triggered = False

    for _, row in df.iterrows():
        ts = pd.to_datetime(row["timestamp"])
        actual = row.get("actual", np.nan)
        forecast = row.get("forecast", np.nan)
        is_day = bool(row.get("is_daylight", True))
        elev = row.get("solar_elevation_deg", 10.0)
        q_flag = str(row.get("quality_flag", "VALID"))

        # Check suppression: night, missing target, missing forecast, non-VALID quality flag
        is_suppressed = (
            (not is_day)
            or (elev <= 0)
            or (pd.isna(actual))
            or (pd.isna(forecast))
            or (q_flag != "VALID")
        )

        if is_suppressed:
            close_active_episode()
            continue

        res = float(actual - forecast)
        cloud_bin = str(
            row.get("forecast_cloud_bin")
            or assign_operating_cloud_bin(row.get("tcdc_pct"), row.get("weather_missing"))
        )
        if cloud_bin not in bounds:
            cloud_bin = "no_weather"

        b_low = bounds[cloud_bin]["bound_low"]
        b_high = bounds[cloud_bin]["bound_high"]

        # Check breach
        if res < b_low:
            curr_dir = "negative"
        elif res > b_high:
            curr_dir = "positive"
        else:
            curr_dir = None

        if curr_dir is None:
            # Residual within bounds
            close_active_episode()
        elif curr_dir == active_direction:
            # Continuing breach in same direction
            consecutive_count += 1
            episode_hours.append({"timestamp": ts, "residual": res, "bin": cloud_bin})
            if consecutive_count >= 2:
                alert_triggered = True
        else:
            # New breach or direction flip
            close_active_episode()
            active_direction = curr_dir
            consecutive_count = 1
            episode_hours = [{"timestamp": ts, "residual": res, "bin": cloud_bin}]
            alert_triggered = False

    # Close any trailing episode at end of frame
    close_active_episode()
    return alert_events


def get_test_period_dataset(
    eval_config_path: Path | str = DEFAULT_EVAL_CONFIG,
    hourly_path: Path | str = DEFAULT_HOURLY_PATH,
    features_path: Path | str = DEFAULT_FEATURES_PATH,
    bundle_path: Path | str = DEFAULT_BUNDLE_PATH,
) -> pd.DataFrame:
    """Prepare out-of-sample forecast and actuals DataFrame for the final test period."""
    with open(eval_config_path, encoding="utf-8") as f:
        eval_cfg = yaml.safe_load(f)

    h_df = pd.read_parquet(hourly_path)
    f_df = pd.read_parquet(features_path)

    h_df["interval_start_utc"] = pd.to_datetime(h_df["interval_start_utc"], utc=True)
    f_df["event_timestamp"] = pd.to_datetime(f_df["event_timestamp"], utc=True)

    win_start = pd.Timestamp(eval_cfg["evaluation_window"]["start_utc"], tz="UTC")
    win_end = pd.Timestamp(eval_cfg["evaluation_window"]["end_utc"], tz="UTC")

    mask_win = (h_df["interval_start_utc"] >= win_start) & (h_df["interval_start_utc"] <= win_end)
    valid_mask = mask_win & (h_df["quality_flag"] == "VALID")
    valid_df = h_df[valid_mask].copy().reset_index(drop=True)

    merged = pd.merge(
        valid_df,
        f_df,
        left_on="interval_start_utc",
        right_on="event_timestamp",
        how="left",
    )

    split_idx = int(np.floor(len(merged) * 0.80))
    test_df = merged.iloc[split_idx:].copy().reset_index(drop=True)

    bundle = load_bundle(bundle_path)
    preds = bundle.predict(test_df)

    test_df["timestamp"] = test_df["interval_start_utc"]
    test_df["actual"] = test_df["target_kwh"]
    test_df["forecast"] = preds
    test_df["residual"] = test_df["actual"] - test_df["forecast"]
    test_df["forecast_cloud_bin"] = [
        assign_operating_cloud_bin(c, m)
        for c, m in zip(test_df["tcdc_pct"], test_df["weather_missing"], strict=False)
    ]
    return test_df


def run_injected_evaluation(
    test_df: pd.DataFrame,
    bounds: dict[str, Any],
    seed: int = 42,
    n_episodes: int = 20,
    min_separation_hours: int = 12,
) -> dict[str, Any]:
    """Evaluate detector on synthetic fault injections on a copy of the test period.

    Injects at a fixed seed:
    (a) 30% power drop for 4 consecutive daylight hours
    (b) zero output for 3 consecutive daylight hours
    (c) bad value (negative or above 893 kWh) for 2 consecutive daylight hours
    (d) isolated single-hour 30% drop (should NOT alert)
    """
    rng = np.random.default_rng(seed)

    # Find candidate daylight stretches with VALID quality and weather present
    valid_daylight = (
        (test_df["quality_flag"] == "VALID")
        & (test_df["is_daylight"])
        & (test_df["solar_elevation_deg"] > 10.0)
        & (test_df["weather_missing"] == 0)
        & (test_df["target_kwh"] > 50.0)  # non-trivial generation
    )
    candidate_indices = np.where(valid_daylight.values)[0]

    # Helper to find non-overlapping anchor indices separated by min_separation_hours
    def select_anchors(episode_duration: int) -> list[int]:
        selected: list[int] = []
        shuffled = rng.permutation(candidate_indices)
        for idx in shuffled:
            # Check consecutive stretch validity
            end_idx = idx + episode_duration
            if end_idx > len(test_df):
                continue
            stretch_valid = valid_daylight.iloc[idx:end_idx].all()
            if not stretch_valid:
                continue
            # Check separation from previously selected anchors
            conflict = False
            for s in selected:
                if abs(idx - s) < (min_separation_hours + episode_duration):
                    conflict = True
                    break
            if not conflict:
                selected.append(int(idx))
                if len(selected) == n_episodes:
                    break
        selected.sort()
        return selected

    results: dict[str, Any] = {}

    # (a) 30% power drop for 4 consecutive hours
    anchors_a = select_anchors(4)
    df_a = test_df.copy()
    for idx in anchors_a:
        for offset in range(4):
            df_a.loc[idx + offset, "actual"] = df_a.loc[idx + offset, "actual"] * 0.70

    alerts_a = detect_deviations(df_a, bounds)
    detected_a: list[int] = []
    delays_a: list[float] = []
    for anc in anchors_a:
        t_start = df_a.loc[anc, "timestamp"]
        t_end = df_a.loc[anc + 3, "timestamp"]
        hit = [
            al
            for al in alerts_a
            if pd.to_datetime(al["window_start_utc"]) <= t_end
            and pd.to_datetime(al["window_end_utc"]) >= t_start
        ]
        if hit:
            detected_a.append(anc)
            # Delay in hours from anomaly start to alert trigger
            t_trig = pd.to_datetime(hit[0]["window_start_utc"]) + datetime.timedelta(hours=1)
            delays_a.append(max(1.0, (t_trig - t_start).total_seconds() / 3600.0))

    results["injected_30pct_drop_4h"] = {
        "episodes_injected": len(anchors_a),
        "episodes_detected": len(detected_a),
        "detection_rate_pct": round(len(detected_a) / len(anchors_a) * 100.0, 2),
        "median_delay_hours": round(float(np.median(delays_a)), 2) if delays_a else 0.0,
        "max_delay_hours": round(float(np.max(delays_a)), 2) if delays_a else 0.0,
    }

    # (b) Zero output for 3 consecutive hours
    anchors_b = select_anchors(3)
    df_b = test_df.copy()
    for idx in anchors_b:
        for offset in range(3):
            df_b.loc[idx + offset, "actual"] = 0.0

    alerts_b = detect_deviations(df_b, bounds)
    detected_b: list[int] = []
    delays_b: list[float] = []
    for anc in anchors_b:
        t_start = df_b.loc[anc, "timestamp"]
        t_end = df_b.loc[anc + 2, "timestamp"]
        hit = [
            al
            for al in alerts_b
            if pd.to_datetime(al["window_start_utc"]) <= t_end
            and pd.to_datetime(al["window_end_utc"]) >= t_start
        ]
        if hit:
            detected_b.append(anc)
            t_trig = pd.to_datetime(hit[0]["window_start_utc"]) + datetime.timedelta(hours=1)
            delays_b.append(max(1.0, (t_trig - t_start).total_seconds() / 3600.0))

    results["injected_zero_output_3h"] = {
        "episodes_injected": len(anchors_b),
        "episodes_detected": len(detected_b),
        "detection_rate_pct": round(len(detected_b) / len(anchors_b) * 100.0, 2),
        "median_delay_hours": round(float(np.median(delays_b)), 2) if delays_b else 0.0,
        "max_delay_hours": round(float(np.max(delays_b)), 2) if delays_b else 0.0,
    }

    # (c) Bad value (negative or above 893 kWh) for 2 consecutive hours
    anchors_c = select_anchors(2)
    df_c = test_df.copy()
    for i, idx in enumerate(anchors_c):
        val = -50.0 if i % 2 == 0 else 1250.0
        for offset in range(2):
            df_c.loc[idx + offset, "actual"] = val

    alerts_c = detect_deviations(df_c, bounds)
    detected_c: list[int] = []
    delays_c: list[float] = []
    for anc in anchors_c:
        t_start = df_c.loc[anc, "timestamp"]
        t_end = df_c.loc[anc + 1, "timestamp"]
        hit = [
            al
            for al in alerts_c
            if pd.to_datetime(al["window_start_utc"]) <= t_end
            and pd.to_datetime(al["window_end_utc"]) >= t_start
        ]
        if hit:
            detected_c.append(anc)
            t_trig = pd.to_datetime(hit[0]["window_start_utc"]) + datetime.timedelta(hours=1)
            delays_c.append(max(1.0, (t_trig - t_start).total_seconds() / 3600.0))

    results["injected_bad_value_2h"] = {
        "episodes_injected": len(anchors_c),
        "episodes_detected": len(detected_c),
        "detection_rate_pct": round(len(detected_c) / len(anchors_c) * 100.0, 2),
        "median_delay_hours": round(float(np.median(delays_c)), 2) if delays_c else 0.0,
        "max_delay_hours": round(float(np.max(delays_c)), 2) if delays_c else 0.0,
    }

    # (d) Isolated single-hour 30% drop (should NOT alert)
    anchors_d = select_anchors(1)
    df_d = test_df.copy()
    for idx in anchors_d:
        df_d.loc[idx, "actual"] = df_d.loc[idx, "actual"] * 0.70

    alerts_d = detect_deviations(df_d, bounds)
    falsely_alerted_d = 0
    for anc in anchors_d:
        t_anc = df_d.loc[anc, "timestamp"]
        hit = [
            al
            for al in alerts_d
            if pd.to_datetime(al["window_start_utc"])
            == pd.to_datetime(al["window_end_utc"])
            == t_anc
        ]
        if hit:
            falsely_alerted_d += 1

    results["injected_isolated_single_hour_drop"] = {
        "episodes_injected": len(anchors_d),
        "episodes_alerted": falsely_alerted_d,
        "compliance_rate_pct": round((1.0 - falsely_alerted_d / len(anchors_d)) * 100.0, 2),
    }

    # Unmodified baseline: False-alert rate on unmodified test period
    unmodified_alerts = detect_deviations(test_df, bounds)
    daylight_test_hours = int(((test_df["quality_flag"] == "VALID") & test_df["is_daylight"]).sum())
    fa_per_1000 = (
        round(len(unmodified_alerts) / daylight_test_hours * 1000.0, 2)
        if daylight_test_hours > 0
        else 0.0
    )

    results["unmodified_test_baseline"] = {
        "valid_daylight_hours": daylight_test_hours,
        "total_alert_events": len(unmodified_alerts),
        "false_alerts_per_1000_daylight_hours": fa_per_1000,
        "alert_events_summary": [
            {
                "event_id": a["event_id"],
                "start": a["window_start_utc"],
                "end": a["window_end_utc"],
                "direction": a["direction"],
                "bin": a["bin"],
                "hours": a["consecutive_hours"],
                "mean_residual": a["mean_residual_kwh"],
            }
            for a in unmodified_alerts
        ],
    }

    return results


def run_natural_suspect_zero_evaluation(
    eval_config_path: Path | str = DEFAULT_EVAL_CONFIG,
    hourly_path: Path | str = DEFAULT_HOURLY_PATH,
    features_path: Path | str = DEFAULT_FEATURES_PATH,
    bundle_path: Path | str = DEFAULT_BUNDLE_PATH,
    bounds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate detector performance on natural SUSPECT_ZERO episodes across the evaluation window.

    For this analysis ONLY, SUSPECT_ZERO hours are treated as valid actuals (actual = 0.0 kWh),
    using out-of-sample forecasts (validation fold models for earlier years, test forecasts
    for test period) and bounds calibrated on VALID hours only.
    """
    if bounds is None:
        bounds = load_deviation_config()["bounds"]

    with open(eval_config_path, encoding="utf-8") as f:
        eval_cfg = yaml.safe_load(f)

    h_df = pd.read_parquet(hourly_path)
    f_df = pd.read_parquet(features_path)

    h_df["interval_start_utc"] = pd.to_datetime(h_df["interval_start_utc"], utc=True)
    f_df["event_timestamp"] = pd.to_datetime(f_df["event_timestamp"], utc=True)

    win_start = pd.Timestamp(eval_cfg["evaluation_window"]["start_utc"], tz="UTC")
    win_end = pd.Timestamp(eval_cfg["evaluation_window"]["end_utc"], tz="UTC")

    mask_win = (h_df["interval_start_utc"] >= win_start) & (h_df["interval_start_utc"] <= win_end)
    eval_df = h_df[mask_win].copy().reset_index(drop=True)

    # Merge features
    merged = pd.merge(
        eval_df,
        f_df,
        left_on="interval_start_utc",
        right_on="event_timestamp",
        how="left",
    )

    # Identify real contiguous SUSPECT_ZERO episodes
    sz_mask = merged["quality_flag"] == "SUSPECT_ZERO"
    sz_indices = merged[sz_mask].index.tolist()

    episodes: list[list[int]] = []
    if sz_indices:
        curr_ep = [sz_indices[0]]
        for idx in sz_indices[1:]:
            if idx == curr_ep[-1] + 1:
                curr_ep.append(idx)
            else:
                episodes.append(curr_ep)
                curr_ep = [idx]
        episodes.append(curr_ep)

    # Compute out-of-sample predictions
    bundle = load_bundle(bundle_path)
    preds = bundle.predict(merged)

    # Prepare DataFrame treating SUSPECT_ZERO as valid actuals (actual = 0.0)
    analysis_df = merged.copy()
    analysis_df["timestamp"] = analysis_df["interval_start_utc"]
    analysis_df["forecast"] = preds
    analysis_df["actual"] = analysis_df["target_kwh"].fillna(0.0)

    # Set quality flag to VALID for SUSPECT_ZERO to enable detection
    analysis_df.loc[sz_mask, "quality_flag"] = "VALID"
    analysis_df["forecast_cloud_bin"] = [
        assign_operating_cloud_bin(c, m)
        for c, m in zip(analysis_df["tcdc_pct"], analysis_df["weather_missing"], strict=False)
    ]

    alerts = detect_deviations(analysis_df, {"bounds": bounds})

    detected_episodes: list[int] = []
    delays: list[float] = []

    for ep_idx, ep in enumerate(episodes):
        t_start = analysis_df.loc[ep[0], "timestamp"]
        t_end = analysis_df.loc[ep[-1], "timestamp"]

        hit = [
            al
            for al in alerts
            if pd.to_datetime(al["window_start_utc"]) <= t_end
            and pd.to_datetime(al["window_end_utc"]) >= t_start
        ]
        if hit:
            detected_episodes.append(ep_idx)
            t_trig = pd.to_datetime(hit[0]["window_start_utc"]) + datetime.timedelta(hours=1)
            delays.append(max(1.0, (t_trig - t_start).total_seconds() / 3600.0))

    return {
        "total_suspect_zero_hours": int(sz_mask.sum()),
        "total_suspect_zero_episodes": len(episodes),
        "episodes_detected": len(detected_episodes),
        "detection_rate_pct": (
            round(len(detected_episodes) / len(episodes) * 100.0, 2) if episodes else 0.0
        ),
        "median_delay_hours": round(float(np.median(delays)), 2) if delays else 0.0,
        "max_delay_hours": round(float(np.max(delays)), 2) if delays else 0.0,
        "episodes_details": [
            {
                "episode_index": ep_idx,
                "start": analysis_df.loc[ep[0], "timestamp"].isoformat(),
                "end": analysis_df.loc[ep[-1], "timestamp"].isoformat(),
                "duration_hours": len(ep),
                "detected": ep_idx in detected_episodes,
            }
            for ep_idx, ep in enumerate(episodes)
        ],
    }


def main() -> int:
    """CLI orchestrator for deviation calibration, evaluation, and reporting."""
    parser = argparse.ArgumentParser(description="SolarCast Deviation Detection Orchestrator")
    parser.add_argument(
        "--calibrate", action="store_true", help="Calibrate bounds from validation folds"
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Run injected and natural evaluation"
    )
    parser.add_argument(
        "--log-mlflow", action="store_true", help="Log deviation detector run to MLflow"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for injection reproducibility"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("SolarCast Deviation Detector Orchestrator")
    print("=" * 70)

    val_df = compute_validation_residuals()
    bounds_cfg = calibrate_deviation_bounds(val_df)

    if args.evaluate or not (args.calibrate and not args.evaluate):
        test_df = get_test_period_dataset()
        injected_res = run_injected_evaluation(test_df, bounds_cfg, seed=args.seed)
        natural_res = run_natural_suspect_zero_evaluation(bounds=bounds_cfg["bounds"])

        summary = {
            "evaluation_timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "seed": args.seed,
            "bounds": bounds_cfg["bounds"],
            "injected_evaluation": injected_res,
            "natural_suspect_zero_evaluation": natural_res,
        }

        DEFAULT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        summary_path = DEFAULT_REPORTS_DIR / "deviation_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved evaluation summary to {summary_path}")

        print("\n--- Injected Fault Evaluation Results ---")
        for k, v in injected_res.items():
            if k == "unmodified_test_baseline":
                fa_rate = v["false_alerts_per_1000_daylight_hours"]
                print(
                    f"  {k}: {v['total_alert_events']} false alerts over "
                    f"{v['valid_daylight_hours']}h ({fa_rate}/1000h)"
                )
            elif "detection_rate_pct" in v:
                print(
                    f"  {k}: {v['episodes_detected']}/{v['episodes_injected']} "
                    f"({v['detection_rate_pct']}%), median delay: {v['median_delay_hours']}h"
                )
            elif "compliance_rate_pct" in v:
                print(
                    f"  {k}: {v['episodes_alerted']}/{v['episodes_injected']} alerted "
                    f"(compliance: {v['compliance_rate_pct']}%)"
                )

        print("\n--- Natural SUSPECT_ZERO Real Episodes ---")
        n_det = natural_res["episodes_detected"]
        n_tot = natural_res["total_suspect_zero_episodes"]
        det_pct = natural_res["detection_rate_pct"]
        med_del = natural_res["median_delay_hours"]
        tot_hrs = natural_res["total_suspect_zero_hours"]
        print(
            f"  Detected: {n_det}/{n_tot} ({det_pct}%), "
            f"median delay: {med_del}h, total hours: {tot_hrs}"
        )

        if args.log_mlflow:
            import mlflow

            inj_drop = injected_res["injected_30pct_drop_4h"]["detection_rate_pct"]
            inj_zero = injected_res["injected_zero_output_3h"]["detection_rate_pct"]
            inj_bad = injected_res["injected_bad_value_2h"]["detection_rate_pct"]
            iso_comp = injected_res["injected_isolated_single_hour_drop"]["compliance_rate_pct"]
            base_res = injected_res["unmodified_test_baseline"]
            fa_rate = base_res["false_alerts_per_1000_daylight_hours"]

            mlflow.set_experiment("solarcast")
            with mlflow.start_run(run_name="deviation_detector") as run:
                mlflow.set_tags(
                    {
                        "data_window": "2018-07-12 to 2023-11-09",
                        "random_seed": str(args.seed),
                        "model_name": "solarcast-f1",
                        "model_bundle": "models/solarcast_f1_bundle.joblib",
                        "task": "M1.6_deviation_detector",
                        "owner": "member1",
                    }
                )
                mlflow.log_params(
                    {
                        "consecutive_hours_threshold": 2,
                        "bins": "clear,mixed,overcast,no_weather",
                        "random_seed": args.seed,
                    }
                )
                mlflow.log_metrics(
                    {
                        "injected_drop30_detection_rate_pct": inj_drop,
                        "injected_zero_detection_rate_pct": inj_zero,
                        "injected_badval_detection_rate_pct": inj_bad,
                        "isolated_single_hour_compliance_pct": iso_comp,
                        "unmodified_fa_per_1000h": fa_rate,
                        "natural_sz_detection_rate_pct": natural_res["detection_rate_pct"],
                        "natural_sz_median_delay_h": natural_res["median_delay_hours"],
                    }
                )
                mlflow.log_artifact(str(DEFAULT_DEVIATION_CONFIG))
                mlflow.log_artifact(str(DEFAULT_ALERT_SCHEMA))
                mlflow.log_artifact(str(summary_path))
                print(f"\nLogged deviation detector run to MLflow: {run.info.run_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
