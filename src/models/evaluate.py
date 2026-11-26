"""Evaluation metrics, candidate gate verification, and final test safeguards for SolarCast.

Provides:
- `compute_evaluation_metrics`: Daylight MAE, RMSE, capacity-normalized MAE, all-hour MAE/RMSE.
- `compute_monthly_daylight_mae`: Per-month daylight MAE breakdown.
- `evaluate_candidate_gate`: Verifies the 5% improvement threshold, capacity bound,
  and finite non-negative constraints.
- `write_evaluation_json`: Persists gate decisions and metric comparisons.
- `check_final_test_safeguard`: Enforces strict test-set protection.
- `log_final_test_audit`: Logs official access records when the test set is opened.
- `run_final_test_evaluation`: Unified evaluation orchestrator supporting both
  dry-run on the last validation fold and official one-time final test evaluation.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import TimeSeriesSplit

from src.features.preprocessing import build_features
from src.models.lineage import log_lineage

DEFAULT_CAPACITY_KWH = 893.0
DEFAULT_EVAL_CONFIG = Path("configs/evaluation.yaml")
DEFAULT_PARAMS_CONFIG = Path("configs/model_params.yaml")
DEFAULT_HOURLY_PATH = Path("data/processed/hourly.parquet")
DEFAULT_QH_PATH = Path("data/processed/quarter_hourly.parquet")
DEFAULT_WEATHER_PATH = Path("data/processed/hrrr_forecast_features.parquet")
DEFAULT_EVAL_JSON_PATH = Path("reports/evaluation.json")
DEFAULT_REPORTS_DIR = Path("reports")
FINAL_TEST_LOG_PATH = Path("reports/final_test_audit.json")


def get_git_commit() -> str:
    """Retrieve current Git commit SHA."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def compute_evaluation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    is_daylight: np.ndarray,
    capacity_denominator_kwh: float = DEFAULT_CAPACITY_KWH,
) -> dict[str, float]:
    """Compute primary daylight MAE, all-hour MAE, RMSE, and normalized MAE.

    Args:
        y_true: Ground truth target values (kWh).
        y_pred: Predicted values (kWh).
        is_daylight: Boolean array indicating daylight hours (elevation > 0).
        capacity_denominator_kwh: Capacity scaling factor in kWh (default 893.0).

    Returns:
        Dictionary of calculated metric values.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    is_daylight = np.asarray(is_daylight, dtype=bool)

    # Daylight metrics
    y_true_day = y_true[is_daylight]
    y_pred_day = y_pred[is_daylight]

    if len(y_true_day) > 0:
        daylight_mae = float(np.mean(np.abs(y_true_day - y_pred_day)))
        daylight_rmse = float(np.sqrt(np.mean((y_true_day - y_pred_day) ** 2)))
        norm_daylight_mae = float(daylight_mae / capacity_denominator_kwh)
    else:
        daylight_mae = np.nan
        daylight_rmse = np.nan
        norm_daylight_mae = np.nan

    # All-hour metrics
    all_hour_mae = float(np.mean(np.abs(y_true - y_pred)))
    all_hour_rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    return {
        "daylight_mae_kwh": daylight_mae,
        "daylight_rmse_kwh": daylight_rmse,
        "capacity_norm_daylight_mae": norm_daylight_mae,
        "all_hour_mae_kwh": all_hour_mae,
        "all_hour_rmse_kwh": all_hour_rmse,
    }


def compute_monthly_daylight_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: pd.DatetimeIndex,
    is_daylight: np.ndarray,
) -> pd.DataFrame:
    """Compute daylight MAE broken down by calendar month (YYYY-MM)."""
    df = pd.DataFrame({
        "month": pd.DatetimeIndex(timestamps).strftime("%Y-%m"),
        "y_true": np.asarray(y_true, dtype=float),
        "y_pred": np.asarray(y_pred, dtype=float),
        "is_daylight": np.asarray(is_daylight, dtype=bool),
    })
    day_df = df[df["is_daylight"]].copy()
    day_df["abs_err"] = (day_df["y_true"] - day_df["y_pred"]).abs()

    monthly = day_df.groupby("month")["abs_err"].agg(
        daylight_hours="count",
        daylight_mae="mean",
    ).reset_index()
    return monthly


def evaluate_candidate_gate(
    candidate_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    y_pred: np.ndarray,
    capacity_limit_kwh: float = DEFAULT_CAPACITY_KWH,
    required_improvement_pct: float = 5.0,
    max_capacity_norm_mae: float = 0.20,
) -> dict[str, Any]:
    """Evaluate whether candidate model satisfies the acceptance gate.

    Rules from configs/evaluation.yaml:
    1. At least 5% reduction in Daylight MAE relative to baseline champion.
    2. Capacity-normalized daylight MAE <= 20.0% (0.20).
    3. Predictions must be a nonempty 1D array, finite, non-negative and <= capacity.

    Missing, nonfinite or negative error metrics reject the candidate. Relative
    improvement requires a positive baseline MAE. Invalid policy settings raise
    ValueError so orchestration fails instead of silently weakening the gate.
    Invalid metric values are represented by None in the returned JSON record.
    """
    policy = {
        "capacity_limit_kwh": capacity_limit_kwh,
        "required_improvement_pct": required_improvement_pct,
        "max_capacity_norm_mae": max_capacity_norm_mae,
    }
    for name, value in policy.items():
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a finite number") from exc
        if not np.isfinite(number):
            raise ValueError(f"{name} must be a finite number")
        policy[name] = number
    capacity_limit_kwh = policy["capacity_limit_kwh"]
    required_improvement_pct = policy["required_improvement_pct"]
    max_capacity_norm_mae = policy["max_capacity_norm_mae"]
    if capacity_limit_kwh <= 0:
        raise ValueError("capacity_limit_kwh must be strictly positive")
    if not 0 <= required_improvement_pct <= 100:
        raise ValueError("required_improvement_pct must be between 0 and 100")
    if max_capacity_norm_mae < 0:
        raise ValueError("max_capacity_norm_mae must be non-negative")

    failure_reasons: list[str] = []

    def validate_metrics(
        metrics: dict[str, float], label: str, required: tuple[str, ...],
    ) -> dict[str, float | None]:
        """Validate supplied errors and retain a strict-JSON-safe audit record."""
        checked: dict[str, float | None] = {}
        for key in required:
            if key not in metrics:
                failure_reasons.append(f"Missing {label} metric: {key}")
        for key, value in metrics.items():
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError):
                number = float("nan")
            if not np.isfinite(number) or number < 0:
                failure_reasons.append(
                    f"Invalid {label} metric {key}: must be finite and non-negative"
                )
                checked[key] = None
            else:
                checked[key] = number
        return checked

    checked_baseline = validate_metrics(baseline_metrics, "baseline", ("daylight_mae_kwh",))
    checked_candidate = validate_metrics(
        candidate_metrics, "candidate", ("daylight_mae_kwh", "capacity_norm_daylight_mae")
    )
    baseline_mae = checked_baseline.get("daylight_mae_kwh")
    cand_mae = checked_candidate.get("daylight_mae_kwh")
    cand_norm_mae = checked_candidate.get("capacity_norm_daylight_mae")
    if baseline_mae == 0:
        failure_reasons.append("Baseline daylight MAE must be strictly positive")

    max_allowed_mae = None
    rel_improvement = None
    if baseline_mae is not None and baseline_mae > 0:
        max_allowed_mae = baseline_mae * (1.0 - required_improvement_pct / 100.0)
        if cand_mae is not None:
            improvement = ((baseline_mae - cand_mae) / baseline_mae) * 100.0
            if np.isfinite(improvement):
                rel_improvement = improvement
            else:
                failure_reasons.append("Relative MAE improvement is not finite")

    # Check 1: 5% Daylight MAE improvement
    if cand_mae is not None and max_allowed_mae is not None and cand_mae > max_allowed_mae:
        improvement_text = f"{rel_improvement:.2f}%" if rel_improvement is not None else "unknown"
        failure_reasons.append(
            f"Insufficient daylight MAE improvement: achieved {improvement_text} "
            f"(target >= {required_improvement_pct:.1f}%; "
            f"candidate {cand_mae:.2f} kWh > max allowed {max_allowed_mae:.2f} kWh)"
        )

    # Check 2: Capacity normalized bound
    if cand_norm_mae is not None and cand_norm_mae > max_capacity_norm_mae:
        failure_reasons.append(
            f"Capacity-normalized daylight MAE exceeded: {cand_norm_mae * 100:.2f}% "
            f"> max allowed {max_capacity_norm_mae * 100:.1f}%"
        )

    # Check 3: Nonempty one-dimensional predictions and physical output bounds
    try:
        preds = np.asarray(y_pred, dtype=float)
    except (TypeError, ValueError, OverflowError):
        failure_reasons.append("Predictions must be a numeric one-dimensional array")
    else:
        if preds.ndim != 1 or preds.size == 0:
            failure_reasons.append("Predictions must be a nonempty one-dimensional array")
        else:
            if not np.all(np.isfinite(preds)):
                failure_reasons.append("Predictions contain NaN or infinite values")
            if np.any(preds < -1e-5):
                min_val = float(np.min(preds))
                failure_reasons.append(
                    f"Predictions contain negative values (min: {min_val:.4f} kWh)"
                )
            if np.any(preds > (capacity_limit_kwh + 1e-3)):
                max_val = float(np.max(preds))
                failure_reasons.append(
                    f"Predictions exceed capacity limit {capacity_limit_kwh:.1f} kWh "
                    f"(max: {max_val:.2f} kWh)"
                )

    eligible = len(failure_reasons) == 0

    return {
        "eligible": eligible,
        "required_improvement_pct": required_improvement_pct,
        "achieved_improvement_pct": (
            round(rel_improvement, 2) if rel_improvement is not None else None
        ),
        "thresholds": {
            "max_daylight_mae_kwh": (
                round(max_allowed_mae, 2) if max_allowed_mae is not None else None
            ),
            "max_capacity_norm_daylight_mae": max_capacity_norm_mae,
            "capacity_limit_kwh": capacity_limit_kwh,
        },
        "baseline_metrics": {
            k: round(v, 4) if v is not None else None for k, v in checked_baseline.items()
        },
        "candidate_metrics": {
            k: round(v, 4) if v is not None else None for k, v in checked_candidate.items()
        },
        "failure_reasons": failure_reasons,
        "final_test_opened": False,
    }


def write_evaluation_json(
    gate_result: dict[str, Any],
    output_path: Path | str = DEFAULT_EVAL_JSON_PATH,
) -> Path:
    """Write gate evaluation results to evaluation.json."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(gate_result, f, indent=2)
    return path


def log_final_test_audit(
    reason: str,
    log_path: Path | str = FINAL_TEST_LOG_PATH,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Log official audit record when opening the final test set."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audit = {
        "final_test_opened": True,
        "opened_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "reason": reason.strip(),
        "git_commit": get_git_commit(),
        "metadata": metadata or {},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    return path


def check_final_test_safeguard(
    force: bool = False,
    reason: str | None = None,
    log_path: Path | str = FINAL_TEST_LOG_PATH,
) -> None:
    """Enforce safeguard preventing unauthorized opening of the final test set.

    Raises RuntimeError if the final test set was already opened unless --force
    is explicitly supplied with a non-empty documented reason.
    """
    path = Path(log_path)
    if path.exists():
        if not force:
            raise RuntimeError(
                "PROTECTED TEST SET VIOLATION: Final 20% test partition has already been opened! "
                f"See {path}. Re-evaluating requires passing --force and providing a reason."
            )
        if not reason or len(reason.strip()) < 10:
            raise ValueError(
                "FORCED OPENING REJECTED: A meaningful reason (>= 10 chars) must be logged "
                "when using --force to recompute the final test set."
            )


def run_final_test_evaluation(
    mode: str = "dry_run",
    force: bool = False,
    reason: str | None = None,
    eval_config_path: Path | str = DEFAULT_EVAL_CONFIG,
    params_config_path: Path | str = DEFAULT_PARAMS_CONFIG,
    hourly_path: Path | str = DEFAULT_HOURLY_PATH,
    quarter_hourly_path: Path | str = DEFAULT_QH_PATH,
    weather_path: Path | str = DEFAULT_WEATHER_PATH,
    reports_dir: Path | str = DEFAULT_REPORTS_DIR,
    val_summary_path: Path | str = Path("reports/set_a_vs_b_summary.json"),
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    """Execute end-to-end evaluation on either validation fold 3 or final test partition.

    Args:
        mode: 'dry_run' (Fold 3 of Train/Dev) or 'final_test' (the 20% held-out test partition).
        force: Required boolean flag to confirm opening final test set.
        reason: Documented reason string (required when force is True).
        eval_config_path: Path to configs/evaluation.yaml.
        params_config_path: Path to configs/model_params.yaml.
        hourly_path: Path to data/processed/hourly.parquet.
        quarter_hourly_path: Path to data/processed/quarter_hourly.parquet.
        weather_path: Path to data/processed/hrrr_forecast_features.parquet.
        reports_dir: Output reports directory.
        val_summary_path: Path to prior cross-validation summary JSON for generalization gap.
        tracking_uri: MLflow tracking URI.

    Returns:
        Structured dictionary of evaluation results and audit metadata.
    """
    start_time = time.time()
    rep_dir = Path(reports_dir)
    rep_dir.mkdir(parents=True, exist_ok=True)

    from src.models.baselines import (
        ClearSky15mPersistence,
        ClearSkyPersistence,
        Last15mPersistence,
        LatestHourPersistence,
        SameHourPreviousDayPersistence,
    )
    from src.models.tune import (
        build_hgb_model,
        build_ridge_pipeline,
        get_feature_columns_a,
        get_feature_columns_b,
    )

    with open(eval_config_path, encoding="utf-8") as f:
        eval_cfg = yaml.safe_load(f)

    cap_limit = float(eval_cfg.get("capacity_dc_kw", 893.0))
    delay_min = int(
        eval_cfg.get("operational_timing", {}).get("production_available_delay_minutes", 15)
    )

    sep_line = "=" * 95
    print(sep_line)
    print(f"SOLARCAST EVALUATION PIPELINE: MODE = {mode.upper()}")
    print(sep_line)

    # 1. Load Parquet datasets
    print("Loading datasets...")
    df_h = pd.read_parquet(hourly_path)
    df_h["interval_start_utc"] = pd.to_datetime(df_h["interval_start_utc"], utc=True)
    df_qh = pd.read_parquet(quarter_hourly_path)
    df_w = pd.read_parquet(weather_path) if Path(weather_path).exists() else pd.DataFrame()

    win_start = pd.Timestamp(eval_cfg["evaluation_window"]["start_utc"], tz="UTC")
    win_end = pd.Timestamp(eval_cfg["evaluation_window"]["end_utc"], tz="UTC")

    mask_win = (df_h["interval_start_utc"] >= win_start) & (df_h["interval_start_utc"] <= win_end)
    valid_mask = mask_win & (df_h["quality_flag"] == "VALID")
    valid_df = df_h[valid_mask].copy().reset_index(drop=True)

    # 80/20 chronological partition
    split_idx = int(np.floor(len(valid_df) * 0.80))
    dev_df = valid_df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = valid_df.iloc[split_idx:].copy().reset_index(drop=True)

    test_start = test_df["interval_start_utc"].min()
    test_end = test_df["interval_start_utc"].max()

    # Audit excluded hours in the test calendar range
    raw_test_window = df_h[
        (df_h["interval_start_utc"] >= test_start) & (df_h["interval_start_utc"] <= win_end)
    ]
    raw_test_count = len(raw_test_window)
    valid_test_count = int((raw_test_window["quality_flag"] == "VALID").sum())
    suspect_zero_test_count = int((raw_test_window["quality_flag"] == "SUSPECT_ZERO").sum())
    missing_data_test_count = int((raw_test_window["quality_flag"] == "MISSING_DATA").sum())
    total_excluded_test_count = suspect_zero_test_count + missing_data_test_count

    print(f"Evaluation Window: {win_start} to {win_end} ({len(valid_df):,} total valid hours)")
    dev_min_t = dev_df["interval_start_utc"].min()
    dev_max_t = dev_df["interval_start_utc"].max()
    print(f"Train/Dev Partition (80%): {len(dev_df):,} hours ({dev_min_t} to {dev_max_t})")
    print(f"Test Calendar Window: {test_start} to {win_end} ({raw_test_count:,} calendar hours)")
    print(f"  Valid Test Hours Included: {valid_test_count:,}")
    print(
        f"  Excluded Hours: {total_excluded_test_count} "
        f"({suspect_zero_test_count} SUSPECT_ZERO, {missing_data_test_count} MISSING_DATA)"
    )

    # 2. Configure Partition by Mode
    if mode == "dry_run":
        print("\n--- DRY RUN: Selecting Last Validation Fold (Fold 3 within Train/Dev) ---")
        tscv = TimeSeriesSplit(n_splits=3)
        splits = list(tscv.split(dev_df))
        train_idx, val_idx = splits[2]
        train_df = dev_df.iloc[train_idx].copy().reset_index(drop=True)
        eval_df = dev_df.iloc[val_idx].copy().reset_index(drop=True)
        partition_name = "last_validation_fold_3"
        f3_tr_min = train_df["interval_start_utc"].min()
        f3_tr_max = train_df["interval_start_utc"].max()
        f3_ev_min = eval_df["interval_start_utc"].min()
        f3_ev_max = eval_df["interval_start_utc"].max()
        print(f"Fold 3 Train Subset: {len(train_df):,} hours ({f3_tr_min} to {f3_tr_max})")
        print(f"Fold 3 Eval Subset:  {len(eval_df):,} hours ({f3_ev_min} to {f3_ev_max})")
        print("Note: The final test partition remains strictly untouched in dry-run mode.")
    elif mode == "final_test":
        print("\n--- OFFICIAL FINAL TEST RUN: Verifying Safeguards & Audit Log ---")
        if not force:
            raise RuntimeError(
                "Final test partition cannot be opened without --force and a documented --reason."
            )
        if not reason or len(reason.strip()) < 10:
            raise ValueError(
                "A documented reason (>= 10 chars) is strictly required to open the final test set."
            )
        check_final_test_safeguard(force=force, reason=reason)
        log_final_test_audit(
            reason=reason,
            metadata={
                "test_hours_valid": valid_test_count,
                "excluded_hours": total_excluded_test_count,
                "suspect_zero": suspect_zero_test_count,
                "missing_data": missing_data_test_count,
                "start_utc": str(test_start),
                "end_utc": str(test_end),
            },
        )
        train_df = dev_df.copy().reset_index(drop=True)
        eval_df = test_df.copy().reset_index(drop=True)
        partition_name = "final_test_partition"
        t_tr_min = train_df["interval_start_utc"].min()
        t_tr_max = train_df["interval_start_utc"].max()
        t_ev_min = eval_df["interval_start_utc"].min()
        t_ev_max = eval_df["interval_start_utc"].max()
        print(f"Train/Dev Partition (Fits): {len(train_df):,} hours ({t_tr_min} to {t_tr_max})")
        print(f"Final Test Partition (Eval): {len(eval_df):,} hours ({t_ev_min} to {t_ev_max})")
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'dry_run' or 'final_test'.")

    # 3. Extract Features
    feat_cfg = {"operational_timing": {"production_available_delay_minutes": delay_min}}
    print(f"\nExtracting features for training partition ({len(train_df):,} rows)...")
    feats_tr_a = build_features(
        train_df["interval_start_utc"],
        hourly_df=df_h,
        quarter_hourly_df=df_qh,
        config=feat_cfg,
        feature_set="A",
    )
    feats_tr_b = build_features(
        train_df["interval_start_utc"],
        hourly_df=df_h,
        quarter_hourly_df=df_qh,
        weather_df=df_w,
        config=feat_cfg,
        feature_set="B",
    )
    train_full_a = pd.concat([train_df, feats_tr_a.reset_index(drop=True)], axis=1)
    train_full_b = pd.concat([train_df, feats_tr_b.reset_index(drop=True)], axis=1)

    print(f"Extracting features for evaluation partition ({len(eval_df):,} rows)...")
    feats_ev_a = build_features(
        eval_df["interval_start_utc"],
        hourly_df=df_h,
        quarter_hourly_df=df_qh,
        config=feat_cfg,
        feature_set="A",
    )
    feats_ev_b = build_features(
        eval_df["interval_start_utc"],
        hourly_df=df_h,
        quarter_hourly_df=df_qh,
        weather_df=df_w,
        config=feat_cfg,
        feature_set="B",
    )
    eval_full_a = pd.concat([eval_df, feats_ev_a.reset_index(drop=True)], axis=1)
    eval_full_b = pd.concat([eval_df, feats_ev_b.reset_index(drop=True)], axis=1)

    cols_a = get_feature_columns_a()
    cols_b = get_feature_columns_b()

    # 4. Training Masks & Fits (Train/Dev Only)
    tr_mask = (train_full_b["quality_flag"] == "VALID") & (
        train_full_b["solar_elevation_target_deg"] > 0
    )
    if "SUSPECT_ZERO" in train_full_b["quality_flag"].values:
        tr_mask = tr_mask & (train_full_b["quality_flag"] != "SUSPECT_ZERO")

    y_train = train_full_b.loc[tr_mask, "target_kwh"].values
    X_train_a = train_full_a.loc[tr_mask, cols_a].values
    X_train_b = train_full_b.loc[tr_mask, cols_b].values

    print(f"Fitting ML models on {len(y_train):,} valid daylight training hours...")
    seed = 42

    ridge_params = {"alpha": 0.001, "fit_intercept": True}
    hgb_params = {
        "learning_rate": 0.05,
        "max_iter": 200,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 100,
        "l2_regularization": 1.0,
    }

    model_registry: dict[str, Any] = {
        # 5 Baselines
        "Latest_Hour_Persistence": {
            "type": "baseline",
            "instance": LatestHourPersistence(),
        },
        "Same_Hour_Previous_Day": {
            "type": "baseline",
            "instance": SameHourPreviousDayPersistence(),
        },
        "Clear_Sky_Persistence_Hourly": {
            "type": "baseline",
            "instance": ClearSkyPersistence(),
        },
        "Last_15m_Persistence": {
            "type": "baseline",
            "instance": Last15mPersistence(),
        },
        "Clear_Sky_Scaled_15m_Persistence": {
            "type": "baseline",
            "instance": ClearSky15mPersistence(delay_minutes=delay_min),
        },
        # 4 ML Models
        "Ridge_Set_A": {
            "type": "ml",
            "model": build_ridge_pipeline(ridge_params, random_seed=seed).fit(X_train_a, y_train),
            "cols": cols_a,
            "df_ev": eval_full_a,
            "feature_set": "A",
        },
        "Ridge_Set_B": {
            "type": "ml",
            "model": build_ridge_pipeline(ridge_params, random_seed=seed).fit(X_train_b, y_train),
            "cols": cols_b,
            "df_ev": eval_full_b,
            "feature_set": "B",
        },
        "HistGradientBoosting_Set_A": {
            "type": "ml",
            "model": build_hgb_model(hgb_params, random_seed=seed).fit(X_train_a, y_train),
            "cols": cols_a,
            "df_ev": eval_full_a,
            "feature_set": "A",
        },
        "HistGradientBoosting_Set_B": {
            "type": "ml",
            "model": build_hgb_model(hgb_params, random_seed=seed).fit(X_train_b, y_train),
            "cols": cols_b,
            "df_ev": eval_full_b,
            "feature_set": "B",
        },
    }

    # 5. Predictions on Evaluation Partition
    is_day_eval = (eval_full_b["solar_elevation_target_deg"] > 0).values
    y_true_eval = eval_full_b["target_kwh"].values
    times_eval = pd.DatetimeIndex(eval_full_b["interval_start_utc"])
    w_missing_eval = eval_full_b["weather_missing"].values == 1
    tcdc_eval = eval_full_b["tcdc_pct"].values

    predictions: dict[str, np.ndarray] = {}
    for name, spec in model_registry.items():
        if spec["type"] == "baseline":
            raw_p = spec["instance"].predict(eval_full_b)
        else:
            X_ev = spec["df_ev"][spec["cols"]].values
            raw_p = spec["model"].predict(X_ev)
        p_bounded = np.where(is_day_eval, raw_p, 0.0)
        p_bounded = np.clip(p_bounded, 0.0, cap_limit)
        predictions[name] = p_bounded

    # 6. Evaluation Masks
    primary_mask = is_day_eval & (~w_missing_eval)
    all_daylight_mask = is_day_eval
    all_hours_mask = np.ones(len(eval_full_b), dtype=bool)

    n_tot_hours = len(eval_full_b)
    n_all_daylight = int(all_daylight_mask.sum())
    n_primary_daylight = int(primary_mask.sum())
    n_weather_missing = int((all_daylight_mask & w_missing_eval).sum())

    pri_pct = n_primary_daylight / n_all_daylight
    miss_pct = n_weather_missing / n_all_daylight

    print("\n" + sep_line)
    print("EVALUATION SAMPLE COUNTS")
    print(sep_line)
    print(f"Total Hours Evaluated: {n_tot_hours:,}")
    print(f"All Daylight Hours:    {n_all_daylight:,}")
    print(f"Primary Daylight Hours (Weather Available): {n_primary_daylight:,} ({pri_pct:.2%})")
    print(f"Daylight Hours Missing Weather:             {n_weather_missing:,} ({miss_pct:.2%})")

    # August 2023 outage analysis
    aug23_start = pd.Timestamp("2023-08-16 15:00:00", tz="UTC")
    aug23_end = pd.Timestamp("2023-08-23 22:00:00", tz="UTC")
    aug23_mask = (times_eval >= aug23_start) & (times_eval <= aug23_end) & is_day_eval
    aug23_missing_count = int(aug23_mask.sum())
    print(
        f"  - Contiguous August 2023 Gap (2023-08-16 to 2023-08-23): "
        f"{aug23_missing_count} daylight hours"
    )

    # 7. Compute Metrics
    champion_baseline_name = "Clear_Sky_Scaled_15m_Persistence"
    hgb_set_a_name = "HistGradientBoosting_Set_A"

    primary_results: dict[str, dict[str, float]] = {}
    all_daylight_results: dict[str, dict[str, float]] = {}
    all_hour_results: dict[str, dict[str, float]] = {}

    for name, p in predictions.items():
        primary_results[name] = compute_evaluation_metrics(
            y_true_eval, p, primary_mask, capacity_denominator_kwh=cap_limit
        )
        all_daylight_results[name] = compute_evaluation_metrics(
            y_true_eval, p, all_daylight_mask, capacity_denominator_kwh=cap_limit
        )
        all_hour_results[name] = compute_evaluation_metrics(
            y_true_eval, p, all_hours_mask, capacity_denominator_kwh=cap_limit
        )

    base_pri_mae = primary_results[champion_baseline_name]["daylight_mae_kwh"]
    hgb_a_pri_mae = primary_results[hgb_set_a_name]["daylight_mae_kwh"]

    print("\n" + sep_line)
    print(f"PRIMARY EVALUATION: DAYLIGHT HOURS WITH WEATHER AVAILABLE (N = {n_primary_daylight:,})")
    print(sep_line)
    header = (
        f"{'Model':<35s} | {'Daylight MAE':<14s} | {'RMSE':<10s} | "
        f"{'Cap-MAE (%)':<12s} | {'vs Base (%)':<12s} | {'vs HGB A (%)':<12s}"
    )
    print(header)
    print("-" * len(header))
    for name in model_registry:
        m = primary_results[name]
        vs_base = ((base_pri_mae - m["daylight_mae_kwh"]) / base_pri_mae) * 100.0
        vs_a = (
            ((hgb_a_pri_mae - m["daylight_mae_kwh"]) / hgb_a_pri_mae) * 100.0
            if name != hgb_set_a_name
            else 0.0
        )
        cap_pct = m["capacity_norm_daylight_mae"] * 100
        print(
            f"{name:<35s} | {m['daylight_mae_kwh']:11.2f} kWh | "
            f"{m['daylight_rmse_kwh']:7.2f} kWh | {cap_pct:10.2f}% | "
            f"{vs_base:10.2f}% | {vs_a:10.2f}%"
        )

    base_all_mae = all_daylight_results[champion_baseline_name]["daylight_mae_kwh"]
    hgb_a_all_mae = all_daylight_results[hgb_set_a_name]["daylight_mae_kwh"]

    print("\n" + sep_line)
    print(f"ALL DAYLIGHT HOURS (INCLUDING MISSING WEATHER HOURS, N = {n_all_daylight:,})")
    print(sep_line)
    print(header)
    print("-" * len(header))
    for name in model_registry:
        m = all_daylight_results[name]
        vs_base = ((base_all_mae - m["daylight_mae_kwh"]) / base_all_mae) * 100.0
        vs_a = (
            ((hgb_a_all_mae - m["daylight_mae_kwh"]) / hgb_a_all_mae) * 100.0
            if name != hgb_set_a_name
            else 0.0
        )
        cap_pct = m["capacity_norm_daylight_mae"] * 100
        print(
            f"{name:<35s} | {m['daylight_mae_kwh']:11.2f} kWh | "
            f"{m['daylight_rmse_kwh']:7.2f} kWh | {cap_pct:10.2f}% | "
            f"{vs_base:10.2f}% | {vs_a:10.2f}%"
        )

    # 8. Cloud-Cover Bin Evaluation (on primary hours)
    print("\n" + sep_line)
    print("STRATIFIED DAYLIGHT MAE BY FORECAST TOTAL CLOUD COVER BIN (WEATHER PRESENT HOURS)")
    print(sep_line)
    cloud_bins = [
        ("Clear / Low Cloud (<20%)", (tcdc_eval < 20.0) & primary_mask),
        ("Mixed / Mid Cloud (20-80%)", (tcdc_eval >= 20.0) & (tcdc_eval <= 80.0) & primary_mask),
        ("Overcast / High Cloud (>80%)", (tcdc_eval > 80.0) & primary_mask),
    ]

    bin_hdr = (
        f"{'Cloud Cover Bin':<30s} | {'Hours':<8s} | {'Base Champ':<12s} | "
        f"{'HGB Set A':<12s} | {'HGB Set B':<12s} | {'Diff (A - B)':<14s} | {'B vs A (%)':<12s}"
    )
    print(bin_hdr)
    print("-" * len(bin_hdr))
    cloud_bin_dict: dict[str, Any] = {}
    for bin_name, bin_mask in cloud_bins:
        n_bin = int(bin_mask.sum())
        m_base = float(
            np.mean(np.abs(y_true_eval[bin_mask] - predictions[champion_baseline_name][bin_mask]))
        )
        m_a = float(
            np.mean(np.abs(y_true_eval[bin_mask] - predictions[hgb_set_a_name][bin_mask]))
        )
        m_b = float(
            np.mean(
                np.abs(
                    y_true_eval[bin_mask]
                    - predictions["HistGradientBoosting_Set_B"][bin_mask]
                )
            )
        )
        diff = m_a - m_b
        diff_pct = (diff / m_a) * 100.0 if m_a > 0 else 0.0
        cloud_bin_dict[bin_name] = {
            "hours": n_bin,
            "mae_base": m_base,
            "mae_hgb_a": m_a,
            "mae_hgb_b": m_b,
            "diff_b_vs_a": diff,
            "diff_pct": diff_pct,
        }
        print(
            f"{bin_name:<30s} | {n_bin:<8d} | {m_base:11.2f}  | "
            f"{m_a:11.2f}  | {m_b:11.2f}  | {diff:13.2f}  | {diff_pct:11.2f}%"
        )

    # 9. Monthly Breakdown
    print("\n" + sep_line)
    print("MONTHLY DAYLIGHT MAE BREAKDOWN (WEATHER PRESENT HOURS)")
    print(sep_line)
    m_base_df = compute_monthly_daylight_mae(
        y_true_eval, predictions[champion_baseline_name], times_eval, primary_mask
    )
    m_a_df = compute_monthly_daylight_mae(
        y_true_eval, predictions[hgb_set_a_name], times_eval, primary_mask
    )
    m_b_df = compute_monthly_daylight_mae(
        y_true_eval,
        predictions["HistGradientBoosting_Set_B"],
        times_eval,
        primary_mask,
    )

    monthly_table = pd.merge(
        m_base_df.rename(columns={"daylight_mae": "mae_base"}),
        m_a_df.rename(columns={"daylight_mae": "mae_a"}),
        on=["month", "daylight_hours"],
    )
    monthly_table = pd.merge(
        monthly_table,
        m_b_df.rename(columns={"daylight_mae": "mae_b"}),
        on=["month", "daylight_hours"],
    )
    monthly_table["diff_b_vs_a"] = monthly_table["mae_a"] - monthly_table["mae_b"]
    monthly_table["diff_pct"] = (
        monthly_table["diff_b_vs_a"] / monthly_table["mae_a"]
    ) * 100.0
    print(monthly_table.to_string(index=False))

    # 10. Generalization Gap Comparison (vs Validation Folds)
    val_comp: dict[str, Any] = {}
    val_key_map = {
        "Clear_Sky_Scaled_15m_Persistence": "Baseline_Champion",
        "Ridge_Set_A": "Ridge_Set_A",
        "Ridge_Set_B": "Ridge_Set_B",
        "HistGradientBoosting_Set_A": "HGB_Set_A",
        "HistGradientBoosting_Set_B": "HGB_Set_B",
    }
    if Path(val_summary_path).exists():
        with open(val_summary_path, encoding="utf-8") as f:
            val_summary = json.load(f)
        val_pri = val_summary.get("primary_results", {})

        print("\n" + sep_line)
        print("GENERALIZATION GAP COMPARISON (VALIDATION CV FOLDS VS EVALUATION PARTITION)")
        print(sep_line)
        gap_hdr = (
            f"{'Model':<35s} | {'Val MAE (kWh)':<15s} | "
            f"{'Eval MAE (kWh)':<15s} | {'Gap (kWh)':<12s} | {'Gap (%)':<10s}"
        )
        print(gap_hdr)
        print("-" * len(gap_hdr))
        for name in model_registry:
            v_key = val_key_map.get(name, name)
            if v_key in val_pri:
                v_mae = float(val_pri[v_key]["mae"])
                e_mae = primary_results[name]["daylight_mae_kwh"]
                gap_kwh = e_mae - v_mae
                gap_pct = (gap_kwh / v_mae) * 100.0
                val_comp[name] = {
                    "val_daylight_mae": v_mae,
                    "eval_daylight_mae": e_mae,
                    "gap_kwh": gap_kwh,
                    "gap_pct": gap_pct,
                }
                print(
                    f"{name:<35s} | {v_mae:14.2f}  | {e_mae:14.2f}  | "
                    f"{gap_kwh:+11.2f}  | {gap_pct:+9.2f}%"
                )

    print("\n" + sep_line)
    print("CRITICAL INTEGRITY STATEMENT")
    print(sep_line)
    print("No model was selected, tuned, or modified based on these evaluation results.")
    print("All feature extraction definitions, hyperparameters, and champion selection")
    print("were completed and frozen on the Train/Dev partition prior to this evaluation.")

    # 11. Write JSON Summary
    summary_filename = (
        "final_test_summary.json" if mode == "final_test" else "dry_run_last_fold_summary.json"
    )
    summary_path = rep_dir / summary_filename
    output_summary = {
        "mode": mode,
        "partition_name": partition_name,
        "evaluation_timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "git_commit": get_git_commit(),
        "reason": reason or ("dry_run_test_of_evaluation_code" if mode == "dry_run" else ""),
        "sample_counts": {
            "total_hours": n_tot_hours,
            "all_daylight_hours": n_all_daylight,
            "primary_daylight_hours": n_primary_daylight,
            "weather_missing_daylight_hours": n_weather_missing,
            "excluded_test_window_hours": total_excluded_test_count,
            "suspect_zero_test_hours": suspect_zero_test_count,
            "missing_data_test_hours": missing_data_test_count,
            "august_2023_outage_daylight_hours": aug23_missing_count,
        },
        "primary_results": primary_results,
        "all_daylight_results": all_daylight_results,
        "all_hour_results": all_hour_results,
        "cloud_bins": cloud_bin_dict,
        "monthly_breakdown": monthly_table.to_dict(orient="records"),
        "generalization_gap": val_comp,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(output_summary, f, indent=2, default=str)
    print(f"\nEvaluation summary written to: {summary_path}")

    # 12. MLflow Tracking
    if tracking_uri is None:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("solarcast-model-training")

    run_name = (
        "final_test_evaluation" if mode == "final_test" else "dry_run_last_fold_evaluation"
    )
    duration_sec = time.time() - start_time

    with mlflow.start_run(run_name=run_name) as run:
        log_lineage(run)
        mlflow.log_params({
            "eval_mode": mode,
            "partition": partition_name,
            "total_hours": n_tot_hours,
            "primary_daylight_hours": n_primary_daylight,
            "weather_missing_hours": n_weather_missing,
            "excluded_test_hours": total_excluded_test_count,
            "capacity_dc_kw": cap_limit,
            "delay_minutes": delay_min,
            "reason": reason or mode,
        })
        for name in model_registry:
            safe_name = name.lower()
            mlflow.log_metrics({
                f"{safe_name}_primary_daylight_mae": primary_results[name]["daylight_mae_kwh"],
                f"{safe_name}_primary_daylight_rmse": primary_results[name]["daylight_rmse_kwh"],
                f"{safe_name}_primary_cap_mae": primary_results[name]["capacity_norm_daylight_mae"],
                f"{safe_name}_all_daylight_mae": all_daylight_results[name]["daylight_mae_kwh"],
                f"{safe_name}_all_hour_mae": all_hour_results[name]["all_hour_mae_kwh"],
            })
        mlflow.log_metric("evaluation_duration_sec", float(np.round(duration_sec, 2)))
        mlflow.set_tags({
            "git_commit": get_git_commit(),
            "schema_version": "1.1.0",
            "eval_type": mode,
            "champion_model": "HistGradientBoosting_Set_B",
            "final_test_opened": str(mode == "final_test"),
        })
        mlflow.log_artifact(str(summary_path))
        print(f"MLflow run logged: {run_name} (Run ID: {run.info.run_id})")

    return output_summary


def main() -> int:
    """Check a saved candidate, or explicitly run a separate research benchmark."""
    parser = argparse.ArgumentParser(description="SolarCast Evaluation & Final Test Runner")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dry-run",
        action="store_true",
        help="Refit research benchmark on development Fold 3; NOT a saved-artifact release gate",
    )
    group.add_argument(
        "--candidate-manifest", type=Path,
        help="Verify and score the exact saved candidate without refitting or opening final test",
    )
    group.add_argument(
        "--final-test",
        action="store_true",
        help="Execute official one-time evaluation on the final 20% test partition",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Required flag confirming authorization to open the final test set",
    )
    parser.add_argument(
        "--reason",
        type=str,
        default=None,
        help="Documented reason string required when opening the final test set",
    )
    parser.add_argument("--eval-config", default=str(DEFAULT_EVAL_CONFIG))
    parser.add_argument("--params-config", default=str(DEFAULT_PARAMS_CONFIG))
    parser.add_argument("--hourly", default=str(DEFAULT_HOURLY_PATH))
    parser.add_argument("--quarter-hourly", default=str(DEFAULT_QH_PATH))
    parser.add_argument("--weather", default=str(DEFAULT_WEATHER_PATH))
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))

    args = parser.parse_args()
    if args.candidate_manifest is not None:
        if args.force or args.reason:
            parser.error("Candidate validation does not accept final-test authorization flags")
        from src.models.candidates import evaluate_candidate_artifact

        result = evaluate_candidate_artifact(args.candidate_manifest)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result["eligible"] is True else 2
    mode = "final_test" if args.final_test else "dry_run"

    run_final_test_evaluation(
        mode=mode,
        force=args.force,
        reason=args.reason,
        eval_config_path=args.eval_config,
        params_config_path=args.params_config,
        hourly_path=args.hourly,
        quarter_hourly_path=args.quarter_hourly,
        weather_path=args.weather,
        reports_dir=args.reports_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
