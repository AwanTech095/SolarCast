"""Hyperparameter tuning and cross-validation search for SolarCast (Task M1.4).

Executes 10-iteration randomized search for:
1. Ridge Regression (with Median Imputer + Standard Scaler fit strictly per fold)
2. HistGradientBoostingRegressor (native NaN handling)

Uses identical 3 expanding-window chronological folds as baseline evaluation.
Selects optimal hyperparameters based on mean validation Daylight MAE across folds.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.evaluate import compute_evaluation_metrics

DEFAULT_PARAMS_CONFIG = Path("configs/model_params.yaml")
DEFAULT_EVAL_CONFIG = Path("configs/evaluation.yaml")
CAPACITY_LIMIT_KWH = 893.0


def get_feature_columns_a(schema_path: Path | str = "configs/feature_schema.json") -> list[str]:
    """Return the canonical 23 feature column names for Feature Set A from feature schema."""
    p = Path(schema_path)
    if p.exists():
        import json

        with open(p, encoding="utf-8") as f:
            schema = json.load(f)
        return [item["name"] for item in schema.get("features", [])]

    return [
        "power_kw_lag_0m",
        "power_kw_lag_15m",
        "power_kw_lag_30m",
        "power_kw_lag_45m",
        "power_kw_last_4_mean",
        "energy_kwh_latest_avail_hour",
        "energy_kwh_prev_day_same_hour",
        "energy_kwh_prev_week_same_hour",
        "energy_kwh_rolling_mean_6h",
        "energy_kwh_rolling_mean_24h",
        "hour_sin",
        "hour_cos",
        "day_of_year_sin",
        "day_of_year_cos",
        "solar_elevation_target_deg",
        "clearsky_ghi_target_wm2",
        "clearsky_energy_target_kwh",
        "clearsky_index_latest_avail",
        "production_age_minutes",
        "power_kw_missing_count",
        "latest_hour_missing",
        "prev_day_missing",
        "prev_week_missing",
    ]


def sample_hyperparameter_configs(
    search_space: dict[str, list[Any]],
    n_iter: int = 10,
    random_seed: int = 42,
) -> list[dict[str, Any]]:
    """Sample n_iter configurations without replacement from hyperparameter space."""
    keys = sorted(search_space.keys())
    values = [search_space[k] for k in keys]
    all_combos = [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*values)]

    rng = np.random.RandomState(random_seed)
    if len(all_combos) <= n_iter:
        return all_combos

    indices = rng.choice(len(all_combos), size=n_iter, replace=False)
    return [all_combos[i] for i in indices]


def build_ridge_pipeline(params: dict[str, Any], random_seed: int = 42) -> Pipeline:
    """Construct Ridge pipeline with SimpleImputer and StandardScaler."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("regressor", Ridge(**params, random_state=random_seed)),
    ])


def build_hgb_model(
    params: dict[str, Any],
    random_seed: int = 42,
) -> HistGradientBoostingRegressor:
    """Construct HistGradientBoostingRegressor model with native NaN handling."""
    return HistGradientBoostingRegressor(**params, random_state=random_seed)


def evaluate_model_on_folds(
    model_factory: Any,
    dev_full: pd.DataFrame,
    feature_cols: list[str],
    tscv: TimeSeriesSplit,
    is_pipeline: bool = False,
    capacity_limit: float = CAPACITY_LIMIT_KWH,
) -> dict[str, Any]:
    """Train on training daylight folds, predict with night-zeroing, and compute fold metrics."""
    fold_metrics: list[dict[str, float]] = []

    for _fold_num, (train_idx, val_idx) in enumerate(tscv.split(dev_full), 1):
        train_df = dev_full.iloc[train_idx]
        val_df = dev_full.iloc[val_idx]

        # Filter training to valid, non-suspect zero, daylight hours
        train_day_mask = (
            (train_df["quality_flag"] == "VALID")
            & (train_df.get("is_daylight", train_df["solar_elevation_target_deg"] > 0))
        )
        if "quality_flag" in train_df and "SUSPECT_ZERO" in train_df["quality_flag"].values:
            train_day_mask = train_day_mask & (train_df["quality_flag"] != "SUSPECT_ZERO")

        X_train = train_df.loc[train_day_mask, feature_cols].values
        y_train = train_df.loc[train_day_mask, "target_kwh"].values

        # Instantiate fresh model
        model = model_factory()
        model.fit(X_train, y_train)

        # Validation predictions
        X_val = val_df[feature_cols].values
        y_val_true = val_df["target_kwh"].values
        is_day_val = (
            val_df["is_daylight"].values
            if "is_daylight" in val_df
            else (val_df["solar_elevation_target_deg"] > 0).values
        )

        raw_preds = model.predict(X_val)
        # Apply physical constraints: 0 at night, clip to [0, capacity]
        preds = np.where(is_day_val, raw_preds, 0.0)
        preds = np.clip(preds, 0.0, capacity_limit)

        m = compute_evaluation_metrics(y_val_true, preds, is_day_val, capacity_limit)
        fold_metrics.append(m)

    mean_daylight_mae = float(np.mean([m["daylight_mae_kwh"] for m in fold_metrics]))
    mean_daylight_rmse = float(np.mean([m["daylight_rmse_kwh"] for m in fold_metrics]))
    mean_cap_norm = float(np.mean([m["capacity_norm_daylight_mae"] for m in fold_metrics]))
    mean_all_hour_mae = float(np.mean([m["all_hour_mae_kwh"] for m in fold_metrics]))

    return {
        "mean_daylight_mae": mean_daylight_mae,
        "mean_daylight_rmse": mean_daylight_rmse,
        "mean_capacity_norm_daylight_mae": mean_cap_norm,
        "mean_all_hour_mae": mean_all_hour_mae,
        "fold_metrics": fold_metrics,
    }


def tune_models(
    dev_full: pd.DataFrame,
    params_config_path: Path | str = DEFAULT_PARAMS_CONFIG,
    n_splits: int = 3,
) -> dict[str, Any]:
    """Execute random search for Ridge and HistGradientBoostingRegressor."""
    with open(params_config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("random_seed", 42))
    n_iter = int(cfg.get("n_search_iter", 10))
    feature_cols = get_feature_columns_a()
    tscv = TimeSeriesSplit(n_splits=n_splits)

    models_cfg = cfg.get("models", {})

    # 1. Tune Ridge
    print(f"\n--- Tuning Ridge Regression ({n_iter} configurations) ---")
    ridge_space = models_cfg.get("ridge", {}).get("search_space", {"alpha": [0.1, 1.0, 10.0]})
    ridge_configs = sample_hyperparameter_configs(ridge_space, n_iter=n_iter, random_seed=seed)

    best_ridge_params = None
    best_ridge_score = float("inf")
    best_ridge_result = None
    ridge_trials = []

    for idx, r_params in enumerate(ridge_configs, 1):
        res = evaluate_model_on_folds(
            model_factory=lambda p=r_params: build_ridge_pipeline(p, random_seed=seed),
            dev_full=dev_full,
            feature_cols=feature_cols,
            tscv=tscv,
            is_pipeline=True,
        )
        score = res["mean_daylight_mae"]
        ridge_trials.append({"params": r_params, "score": score, "metrics": res})
        cap_pct = res["mean_capacity_norm_daylight_mae"] * 100
        print(f"  [Ridge Trial {idx:02d}/{n_iter}] alpha={r_params.get('alpha')}: "
              f"MAE = {score:.2f} kWh ({cap_pct:.2f}% cap)")
        if score < best_ridge_score:
            best_ridge_score = score
            best_ridge_params = r_params
            best_ridge_result = res

    # 2. Tune HistGradientBoostingRegressor
    print(f"\n--- Tuning HistGradientBoostingRegressor ({n_iter} configurations) ---")
    hgb_space = models_cfg.get("hist_gradient_boosting", {}).get("search_space", {
        "learning_rate": [0.05, 0.1],
        "max_iter": [100],
    })
    hgb_configs = sample_hyperparameter_configs(hgb_space, n_iter=n_iter, random_seed=seed)

    best_hgb_params = None
    best_hgb_score = float("inf")
    best_hgb_result = None
    hgb_trials = []

    for idx, h_params in enumerate(hgb_configs, 1):
        res = evaluate_model_on_folds(
            model_factory=lambda p=h_params: build_hgb_model(p, random_seed=seed),
            dev_full=dev_full,
            feature_cols=feature_cols,
            tscv=tscv,
            is_pipeline=False,
        )
        score = res["mean_daylight_mae"]
        hgb_trials.append({"params": h_params, "score": score, "metrics": res})
        cap_pct = res["mean_capacity_norm_daylight_mae"] * 100
        print(f"  [HGB Trial {idx:02d}/{n_iter}] lr={h_params.get('learning_rate')}, "
              f"iter={h_params.get('max_iter')}, leaves={h_params.get('max_leaf_nodes')}: "
              f"MAE = {score:.2f} kWh ({cap_pct:.2f}% cap)")
        if score < best_hgb_score:
            best_hgb_score = score
            best_hgb_params = h_params
            best_hgb_result = res

    return {
        "feature_set": "A",
        "feature_count": len(feature_cols),
        "ridge": {
            "best_params": best_ridge_params,
            "best_daylight_mae": best_ridge_score,
            "best_result": best_ridge_result,
            "trials": ridge_trials,
        },
        "hist_gradient_boosting": {
            "best_params": best_hgb_params,
            "best_daylight_mae": best_hgb_score,
            "best_result": best_hgb_result,
            "trials": hgb_trials,
        },
    }
