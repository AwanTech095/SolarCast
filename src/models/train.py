"""Training, cross-validation benchmarking, and MLflow logging pipeline for SolarCast.

Executes Step 2 (Model Training & Benchmarking), Step 3 (Gate Verification),
Step 4 (MLflow Tracking), and Step 5 (Model Bundling).
Supports both Feature Set A (23 features) and Feature Set B (29 features with HRRR weather).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import TimeSeriesSplit

from src.features.preprocessing import build_features
from src.models.baselines import ClearSky15mPersistence
from src.models.bundle import ModelBundle, save_bundle
from src.models.evaluate import (
    compute_evaluation_metrics,
    compute_monthly_daylight_mae,
    evaluate_candidate_gate,
    write_evaluation_json,
)
from src.models.lineage import log_lineage
from src.models.tune import (
    build_hgb_model,
    build_ridge_pipeline,
    get_feature_columns_a,
    get_feature_columns_b,
    tune_models,
)

DEFAULT_EVAL_CONFIG = Path("configs/evaluation.yaml")
DEFAULT_PARAMS_CONFIG = Path("configs/model_params.yaml")
DEFAULT_SCHEMA_CONFIG = Path("configs/feature_schema.json")
DEFAULT_HOURLY_PATH = Path("data/processed/hourly.parquet")
DEFAULT_QH_PATH = Path("data/processed/quarter_hourly.parquet")
DEFAULT_WEATHER_PATH = Path("data/processed/hrrr_forecast_features.parquet")
DEFAULT_REPORTS_DIR = Path("reports")
DEFAULT_MODELS_DIR = Path("models")
DEFAULT_PLOTS_DIR = Path("reports/plots")

BASELINE_CHAMPION_NAME = "Clear-Sky Scaled 15-Minute Persistence"


def compute_file_sha256(file_path: Path | str) -> str:
    """Compute SHA-256 hash of a file."""
    p = Path(file_path)
    if not p.exists():
        return "file_not_found"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit() -> str:
    """Retrieve current Git commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown_commit"


def generate_evaluation_plots(
    y_true_all: np.ndarray,
    y_pred_all: np.ndarray,
    timestamps_all: pd.DatetimeIndex,
    is_daylight_all: np.ndarray,
    output_dir: Path | str = DEFAULT_PLOTS_DIR,
    model_name: str = "HistGradientBoosting_Set_B",
) -> dict[str, Path]:
    """Generate diagnostic evaluation plots for candidate model."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: dict[str, Path] = {}

    y_true_day = y_true_all[is_daylight_all]
    y_pred_day = y_pred_all[is_daylight_all]

    # 1. Actual vs Predicted Scatter Plot
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true_day, y_pred_day, alpha=0.15, s=8, color="#1f77b4")
    max_val = max(float(np.max(y_true_day)), float(np.max(y_pred_day)), 893.0)
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="1:1 Perfect Line")
    ax.set_xlabel("Actual AC Energy (kWh)", fontsize=11)
    ax.set_ylabel("Predicted AC Energy (kWh)", fontsize=11)
    ax.set_title(f"{model_name} - Actual vs Predicted (Validation Daylight Hours)", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    p1 = out_dir / "actual_vs_predicted.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    plot_paths["actual_vs_predicted"] = p1

    # 2. Residuals Distribution
    residuals = y_true_day - y_pred_day
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(residuals, bins=50, color="#2ca02c", edgecolor="black", alpha=0.7)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Residual (Actual - Predicted, kWh)", fontsize=11)
    ax.set_ylabel("Count of Daylight Hours", fontsize=11)
    ax.set_title(
        f"{model_name} - Residuals Distribution (Mean: {np.mean(residuals):.2f} kWh)",
        fontsize=12,
    )
    ax.grid(True, linestyle=":", alpha=0.6)
    p2 = out_dir / "residuals.png"
    fig.tight_layout()
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    plot_paths["residuals"] = p2

    # 3. MAE by Hour of Day (UTC)
    df_eval = pd.DataFrame({
        "hour_utc": timestamps_all.hour,
        "is_daylight": is_daylight_all,
        "abs_err": np.abs(y_true_all - y_pred_all),
    })
    mae_by_hour = df_eval[df_eval["is_daylight"]].groupby("hour_utc")["abs_err"].mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(mae_by_hour.index, mae_by_hour.values, color="#ff7f0e", edgecolor="black", alpha=0.8)
    ax.set_xlabel("Hour of Day (UTC)", fontsize=11)
    ax.set_ylabel("Daylight MAE (kWh)", fontsize=11)
    ax.set_title(f"{model_name} - Daylight MAE by Hour of Day (UTC)", fontsize=12)
    ax.set_xticks(range(int(mae_by_hour.index.min()), int(mae_by_hour.index.max()) + 1))
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    p3 = out_dir / "error_by_hour.png"
    fig.tight_layout()
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    plot_paths["error_by_hour"] = p3

    # 4. MAE by Month
    m_df = compute_monthly_daylight_mae(y_true_all, y_pred_all, timestamps_all, is_daylight_all)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(m_df["month"], m_df["daylight_mae"], marker="o", color="#9467bd", linewidth=2)
    ax.set_xlabel("Validation Month", fontsize=11)
    ax.set_ylabel("Daylight MAE (kWh)", fontsize=11)
    ax.set_title(f"{model_name} - Monthly Daylight MAE Across Validation Folds", fontsize=12)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, linestyle=":", alpha=0.6)
    p4 = out_dir / "error_by_month.png"
    fig.tight_layout()
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    plot_paths["error_by_month"] = p4

    return plot_paths


def run_training_pipeline(
    eval_config_path: Path | str = DEFAULT_EVAL_CONFIG,
    params_config_path: Path | str = DEFAULT_PARAMS_CONFIG,
    hourly_path: Path | str = DEFAULT_HOURLY_PATH,
    quarter_hourly_path: Path | str = DEFAULT_QH_PATH,
    weather_path: Path | str = DEFAULT_WEATHER_PATH,
    reports_dir: Path | str = DEFAULT_REPORTS_DIR,
    models_dir: Path | str = DEFAULT_MODELS_DIR,
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    """Execute complete training, tuning, benchmarking, and MLflow logging pipeline."""
    start_time = time.time()
    rep_dir = Path(reports_dir)
    rep_dir.mkdir(parents=True, exist_ok=True)
    mod_dir = Path(models_dir)
    mod_dir.mkdir(parents=True, exist_ok=True)

    with open(eval_config_path, encoding="utf-8") as f:
        eval_cfg = yaml.safe_load(f)
    with open(params_config_path, encoding="utf-8") as f:
        params_cfg = yaml.safe_load(f)

    # 1. Hashes, Commit, and Schema Version
    data_hashes = {
        "hourly_parquet": compute_file_sha256(hourly_path),
        "quarter_hourly_parquet": compute_file_sha256(quarter_hourly_path),
        "hrrr_forecast_features_parquet": compute_file_sha256(weather_path),
    }
    git_commit = get_git_commit()
    feature_schema_version = "1.1.0"
    delay_min = int(
        eval_cfg.get("operational_timing", {}).get("production_available_delay_minutes", 15)
    )
    cap_limit = float(eval_cfg.get("capacity_dc_kw", 893.0))

    # 2. Load Datasets and enforce 80/20 chronological partition
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

    # Strict 80/20 Chronological Split
    split_idx = int(np.floor(len(valid_df) * 0.80))
    dev_df = valid_df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = valid_df.iloc[split_idx:].copy().reset_index(drop=True)

    t_dev_start = dev_df["interval_start_utc"].min()
    t_dev_end = dev_df["interval_start_utc"].max()
    t_test_start = test_df["interval_start_utc"].min()
    t_test_end = test_df["interval_start_utc"].max()

    print(f"Train/Dev Partition (80%): {len(dev_df):,} hours ({t_dev_start} to {t_dev_end})")
    print(
        f"Final Test Partition (20% - STRICTLY LOCKED): {len(test_df):,} hours "
        f"({t_test_start} to {t_test_end})"
    )

    # 3. Build Feature Matrices for Train/Dev
    feat_cfg = {"operational_timing": {"production_available_delay_minutes": delay_min}}
    print(f"Extracting Feature Set A (23 features, delay = {delay_min} min)...")
    feats_dev_a = build_features(
        dev_df["interval_start_utc"],
        hourly_df=df_h,
        quarter_hourly_df=df_qh,
        config=feat_cfg,
        feature_set="A",
    )
    dev_full_a = pd.concat([dev_df, feats_dev_a.reset_index(drop=True)], axis=1)
    cols_a = get_feature_columns_a()

    print(f"Extracting Feature Set B (29 features, delay = {delay_min} min)...")
    feats_dev_b = build_features(
        dev_df["interval_start_utc"],
        hourly_df=df_h,
        quarter_hourly_df=df_qh,
        weather_df=df_w,
        config=feat_cfg,
        feature_set="B",
    )
    dev_full_b = pd.concat([dev_df, feats_dev_b.reset_index(drop=True)], axis=1)
    cols_b = get_feature_columns_b()

    # 4. Tune Models on Set A and Set B
    print("\n--- Tuning Models on Feature Set A ---")
    tune_a = tune_models(dev_full_a, params_config_path=params_config_path, feature_set="A")

    print("\n--- Tuning Models on Feature Set B ---")
    tune_b = tune_models(dev_full_b, params_config_path=params_config_path, feature_set="B")

    # 5. Cross-Validation on Identical 3 Folds
    tscv = TimeSeriesSplit(n_splits=3)
    seed = int(params_cfg.get("random_seed", 42))
    baseline_model = ClearSky15mPersistence(delay_minutes=delay_min)

    models_to_benchmark = {
        "Baseline_Champion": {
            "type": "Baseline",
            "feature_set": "Baseline",
            "is_baseline": True,
            "params": {"delay_minutes": delay_min},
        },
        "Ridge_Set_A": {
            "type": "Ridge",
            "feature_set": "A",
            "is_baseline": False,
            "cols": cols_a,
            "df": dev_full_a,
            "params": tune_a["ridge"]["best_params"],
            "factory": lambda: build_ridge_pipeline(
                tune_a["ridge"]["best_params"], random_seed=seed
            ),
        },
        "HistGradientBoosting_Set_A": {
            "type": "HistGradientBoostingRegressor",
            "feature_set": "A",
            "is_baseline": False,
            "cols": cols_a,
            "df": dev_full_a,
            "params": tune_a["hist_gradient_boosting"]["best_params"],
            "factory": lambda: build_hgb_model(
                tune_a["hist_gradient_boosting"]["best_params"], random_seed=seed
            ),
        },
        "Ridge_Set_B": {
            "type": "Ridge",
            "feature_set": "B",
            "is_baseline": False,
            "cols": cols_b,
            "df": dev_full_b,
            "params": tune_b["ridge"]["best_params"],
            "factory": lambda: build_ridge_pipeline(
                tune_b["ridge"]["best_params"], random_seed=seed
            ),
        },
        "HistGradientBoosting_Set_B": {
            "type": "HistGradientBoostingRegressor",
            "feature_set": "B",
            "is_baseline": False,
            "cols": cols_b,
            "df": dev_full_b,
            "params": tune_b["hist_gradient_boosting"]["best_params"],
            "factory": lambda: build_hgb_model(
                tune_b["hist_gradient_boosting"]["best_params"], random_seed=seed
            ),
        },
    }

    predictions = {name: [] for name in models_to_benchmark}
    val_targets = []
    val_times = []
    val_daylight = []
    val_weather_missing = []

    for fold_num, (train_idx, val_idx) in enumerate(tscv.split(dev_df), 1):
        print(f"Evaluating fold {fold_num}/3 (Train: {len(train_idx)}, Val: {len(val_idx)})...")
        val_sub_b = dev_full_b.iloc[val_idx]
        y_val = val_sub_b["target_kwh"].values
        is_day = (val_sub_b["solar_elevation_target_deg"] > 0).values
        w_miss = val_sub_b["weather_missing"].values

        val_targets.append(y_val)
        val_times.append(val_sub_b["interval_start_utc"].values)
        val_daylight.append(is_day)
        val_weather_missing.append(w_miss)

        # Baseline
        preds_base = baseline_model.predict(val_sub_b)
        predictions["Baseline_Champion"].append(preds_base)

        # ML models
        for name, spec in models_to_benchmark.items():
            if spec.get("is_baseline"):
                continue
            df_curr = spec["df"]
            cols_curr = spec["cols"]

            tr_sub = df_curr.iloc[train_idx]
            v_sub = df_curr.iloc[val_idx]

            tr_mask = (
                (tr_sub["quality_flag"] == "VALID")
                & (tr_sub["solar_elevation_target_deg"] > 0)
            )
            if "SUSPECT_ZERO" in tr_sub["quality_flag"].values:
                tr_mask = tr_mask & (tr_sub["quality_flag"] != "SUSPECT_ZERO")

            X_tr = tr_sub.loc[tr_mask, cols_curr].values
            y_tr = tr_sub.loc[tr_mask, "target_kwh"].values

            clf = spec["factory"]()
            clf.fit(X_tr, y_tr)

            X_v = v_sub[cols_curr].values
            raw_v = clf.predict(X_v)
            preds_v = np.where(is_day, raw_v, 0.0)
            preds_v = np.clip(preds_v, 0.0, cap_limit)
            predictions[name].append(preds_v)

    # Concatenate across folds
    y_true_concat = np.concatenate(val_targets)
    times_concat = pd.DatetimeIndex(np.concatenate(val_times))
    daylight_concat = np.concatenate(val_daylight)
    w_missing_concat = np.concatenate(val_weather_missing)

    preds_concat = {k: np.concatenate(v) for k, v in predictions.items()}

    # Primary mask: daylight hours where weather is present
    primary_mask = daylight_concat & (w_missing_concat == 0)

    # Compute metrics for all models on primary hours
    benchmark_records: dict[str, dict[str, Any]] = {}
    for name, spec in models_to_benchmark.items():
        m_day = compute_evaluation_metrics(
            y_true_concat, preds_concat[name], primary_mask, cap_limit
        )
        monthly_df = compute_monthly_daylight_mae(
            y_true_concat, preds_concat[name], times_concat, primary_mask
        )
        benchmark_records[name] = {
            "type": spec["type"],
            "feature_set": spec["feature_set"],
            "params": spec["params"],
            "factory": spec.get("factory"),
            "cols": spec.get("cols", cols_b),
            "metrics": m_day,
            "monthly_mae_df": monthly_df,
            "y_true": y_true_concat,
            "y_pred": preds_concat[name],
            "timestamps": times_concat,
            "is_daylight": primary_mask,
        }

    base_metrics = benchmark_records["Baseline_Champion"]["metrics"]
    hgb_a_metrics = benchmark_records["HistGradientBoosting_Set_A"]["metrics"]
    hgb_b_metrics = benchmark_records["HistGradientBoosting_Set_B"]["metrics"]

    # 6. Model Selection Rule: Set A vs Set B
    diff_mae = hgb_a_metrics["daylight_mae_kwh"] - hgb_b_metrics["daylight_mae_kwh"]
    diff_mae_pct = (diff_mae / hgb_a_metrics["daylight_mae_kwh"]) * 100.0

    n_pri = primary_mask.sum()
    n_tot = daylight_concat.sum()
    print("\n=======================================================")
    print("MODEL VALIDATION BENCHMARK (IDENTICAL DAYLIGHT HOURS)")
    print(f"Weather Present Hours N = {n_pri:,} / Total Daylight = {n_tot:,}")
    print("=======================================================")
    base_cap = base_metrics["capacity_norm_daylight_mae"] * 100
    hgb_a_cap = hgb_a_metrics["capacity_norm_daylight_mae"] * 100
    hgb_b_cap = hgb_b_metrics["capacity_norm_daylight_mae"] * 100
    b_mae = base_metrics["daylight_mae_kwh"]
    a_mae = hgb_a_metrics["daylight_mae_kwh"]
    s_mae = hgb_b_metrics["daylight_mae_kwh"]
    print(f"Baseline Champion: Daylight MAE = {b_mae:.2f} kWh ({base_cap:.2f}%)")
    print(f"HGB Set A:         Daylight MAE = {a_mae:.2f} kWh ({hgb_a_cap:.2f}%)")
    print(f"HGB Set B:         Daylight MAE = {s_mae:.2f} kWh ({hgb_b_cap:.2f}%)")
    print(f"Set B vs Set A:    Difference = {diff_mae:+.2f} kWh ({diff_mae_pct:+.2f}%)")

    if diff_mae_pct >= 1.0:
        best_candidate_name = "HistGradientBoosting_Set_B"
        selected_feature_set = "B"
        selected_cols = cols_b
        selected_dev_df = dev_full_b
        print("Selection: Set B improves upon Set A by >= 1.0%. Selected: HGB Set B.")
    else:
        best_candidate_name = "HistGradientBoosting_Set_A"
        selected_feature_set = "A"
        selected_cols = cols_a
        selected_dev_df = dev_full_a
        print("Selection: Difference < 1.0%. Selected champion: HistGradientBoosting (Set A).")

    best_candidate = benchmark_records[best_candidate_name]

    # 7. Evaluate Candidate Gate
    gate_eval = evaluate_candidate_gate(
        candidate_metrics=best_candidate["metrics"],
        baseline_metrics=base_metrics,
        y_pred=best_candidate["y_pred"],
        capacity_limit_kwh=cap_limit,
        required_improvement_pct=5.0,
        max_capacity_norm_mae=0.20,
    )
    gate_eval["candidate_model_name"] = best_candidate_name
    gate_eval["baseline_champion_name"] = BASELINE_CHAMPION_NAME

    # Write evaluation.json
    eval_json_path = rep_dir / "evaluation.json"
    write_evaluation_json(gate_eval, eval_json_path)
    write_evaluation_json(gate_eval, Path("evaluation.json"))
    print(f"\nCandidate Gate Status: {'ELIGIBLE' if gate_eval['eligible'] else 'REJECTED'}")
    impr_pct = gate_eval["achieved_improvement_pct"]
    print(f"  Improvement over Baseline Champion: {impr_pct:.2f}% (Threshold >= 5.0%)")

    # 8. Train Final Champion Model on complete Train/Dev dataset (VALID daylight hours)
    print(f"\nTraining final {best_candidate_name} on complete Train/Dev dataset...")
    full_day_mask = (
        (selected_dev_df["quality_flag"] == "VALID")
        & (selected_dev_df["solar_elevation_target_deg"] > 0)
    )
    if "SUSPECT_ZERO" in selected_dev_df["quality_flag"].values:
        full_day_mask = full_day_mask & (selected_dev_df["quality_flag"] != "SUSPECT_ZERO")

    X_full_train = selected_dev_df.loc[full_day_mask, selected_cols].values
    y_full_train = selected_dev_df.loc[full_day_mask, "target_kwh"].values

    final_model = best_candidate["factory"]()
    final_model.fit(X_full_train, y_full_train)

    # 9. Generate Diagnostic Plots
    plots = generate_evaluation_plots(
        y_true_all=best_candidate["y_true"],
        y_pred_all=best_candidate["y_pred"],
        timestamps_all=best_candidate["timestamps"],
        is_daylight_all=best_candidate["is_daylight"],
        output_dir=DEFAULT_PLOTS_DIR,
        model_name=best_candidate_name,
    )

    # 10. Save Model Bundle
    bundle = ModelBundle(
        estimator=final_model,
        feature_names=selected_cols,
        training_window={
            "start_utc": str(t_dev_start),
            "end_utc": str(t_dev_end),
            "training_samples": str(len(y_full_train)),
        },
        data_hashes=data_hashes,
        mlflow_run_id=None,
        capacity_limit_kwh=cap_limit,
        model_name="solarcast-f1",
        feature_schema_version=feature_schema_version,
        feature_set=selected_feature_set,
        needs_weather=(selected_feature_set == "B"),
    )
    bundle_path = mod_dir / "solarcast_f1_bundle.joblib"
    save_bundle(bundle, bundle_path)
    print(f"Model bundle saved to {bundle_path}")

    # 11. MLflow Tracking & Artifact Logging
    if tracking_uri is None:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = "solarcast-model-training"
    mlflow.set_experiment(experiment_name)

    training_duration = time.time() - start_time
    status_tag = "candidate" if gate_eval["eligible"] else "rejected"

    for name, rec in benchmark_records.items():
        if rec.get("feature_set") == "Baseline":
            continue
        run_name = f"{name}_15mDelay"
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            log_lineage(run)
            improvement = float(
                (base_metrics["daylight_mae_kwh"] - rec["metrics"]["daylight_mae_kwh"])
                / base_metrics["daylight_mae_kwh"]
                * 100.0
            )
            mlflow.log_params({
                "model_type": rec["type"],
                "feature_set": rec["feature_set"],
                "feature_count": len(rec["cols"]),
                "random_seed": seed,
                "delay_minutes": delay_min,
                **rec["params"],
            })
            mlflow.log_metrics({
                **rec["metrics"],
                "training_duration_sec": float(np.round(training_duration, 2)),
                "improvement_over_baseline_pct": improvement,
            })
            mlflow.set_tags({
                "git_commit": git_commit,
                "data_hash_hourly": data_hashes["hourly_parquet"],
                "data_hash_qh": data_hashes["quarter_hourly_parquet"],
                "data_hash_hrrr": data_hashes["hrrr_forecast_features_parquet"],
                "feature_schema_version": feature_schema_version,
                "status": status_tag if name == best_candidate_name else "rejected",
                "is_best_candidate": str(name == best_candidate_name),
            })

            # If best candidate, log plots, evaluation.json, and bundle
            if name == best_candidate_name:
                bundle.mlflow_run_id = run_id
                save_bundle(bundle, bundle_path)
                log_lineage(run)

                for _p_name, p_file in plots.items():
                    mlflow.log_artifact(str(p_file), artifact_path="plots")
                mlflow.log_artifact(str(eval_json_path), artifact_path="evaluation")
                mlflow.log_artifact(str(bundle_path), artifact_path="bundle")

                try:
                    mlflow.sklearn.log_model(
                        final_model,
                        artifact_path="model",
                        registered_model_name="solarcast-f1",
                    )
                except Exception as reg_err:
                    print(f"MLflow model registration notice: {reg_err}")

    return {
        "best_candidate_name": best_candidate_name,
        "selected_feature_set": selected_feature_set,
        "best_candidate_metrics": best_candidate["metrics"],
        "gate_evaluation": gate_eval,
        "benchmark_records": benchmark_records,
        "bundle_path": str(bundle_path),
        "evaluation_json_path": str(eval_json_path),
        "training_duration_sec": training_duration,
        "data_hashes": data_hashes,
        "git_commit": git_commit,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="SolarCast model training and evaluation")
    parser.add_argument("--eval-config", default=str(DEFAULT_EVAL_CONFIG))
    parser.add_argument("--params-config", default=str(DEFAULT_PARAMS_CONFIG))
    parser.add_argument("--hourly", default=str(DEFAULT_HOURLY_PATH))
    parser.add_argument("--quarter-hourly", default=str(DEFAULT_QH_PATH))
    parser.add_argument("--weather", default=str(DEFAULT_WEATHER_PATH))
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    args = parser.parse_args()

    run_training_pipeline(
        eval_config_path=args.eval_config,
        params_config_path=args.params_config,
        hourly_path=args.hourly,
        quarter_hourly_path=args.quarter_hourly,
        weather_path=args.weather,
        reports_dir=args.reports_dir,
        models_dir=args.models_dir,
    )
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
