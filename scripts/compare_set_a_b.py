"""Benchmark Feature Set A vs Feature Set B on identical validation hours."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import TimeSeriesSplit

from src.features.preprocessing import build_features, get_feature_names
from src.models.baselines import ClearSky15mPersistence
from src.models.evaluate import compute_evaluation_metrics, compute_monthly_daylight_mae
from src.models.tune import (
    build_hgb_model,
    build_ridge_pipeline,
    get_feature_columns_a,
    get_feature_columns_b,
    tune_models,
)

def main():
    start_time = time.time()
    with open("configs/evaluation.yaml", encoding="utf-8") as f:
        eval_cfg = yaml.safe_load(f)
    with open("configs/model_params.yaml", encoding="utf-8") as f:
        params_cfg = yaml.safe_load(f)

    seed = int(params_cfg.get("random_seed", 42))
    cap_limit = float(eval_cfg.get("capacity_dc_kw", 893.0))
    delay_min = int(eval_cfg.get("operational_timing", {}).get("production_available_delay_minutes", 15))

    print("Loading datasets...")
    df_h = pd.read_parquet("data/processed/hourly.parquet")
    df_qh = pd.read_parquet("data/processed/quarter_hourly.parquet")
    df_w = pd.read_parquet("data/processed/hrrr_forecast_features.parquet")

    df_h["interval_start_utc"] = pd.to_datetime(df_h["interval_start_utc"], utc=True)
    win_start = pd.Timestamp(eval_cfg["evaluation_window"]["start_utc"], tz="UTC")
    win_end = pd.Timestamp(eval_cfg["evaluation_window"]["end_utc"], tz="UTC")

    mask_win = (df_h["interval_start_utc"] >= win_start) & (df_h["interval_start_utc"] <= win_end)
    valid_mask = mask_win & (df_h["quality_flag"] == "VALID")
    valid_df = df_h[valid_mask].copy().reset_index(drop=True)

    # Strict 80/20 chronological partition
    split_idx = int(np.floor(len(valid_df) * 0.80))
    dev_df = valid_df.iloc[:split_idx].copy().reset_index(drop=True)

    print(f"Dev dataset: {len(dev_df):,} hours ({dev_df['interval_start_utc'].min()} to {dev_df['interval_start_utc'].max()})")

    print("\n--- Extracting Feature Set A (23 features) ---")
    feat_cfg = {"operational_timing": {"production_available_delay_minutes": delay_min}}
    feats_a = build_features(
        dev_df["interval_start_utc"],
        hourly_df=df_h,
        quarter_hourly_df=df_qh,
        config=feat_cfg,
        feature_set="A",
    )
    dev_full_a = pd.concat([dev_df, feats_a.reset_index(drop=True)], axis=1)
    cols_a = get_feature_columns_a()

    print("\n--- Extracting Feature Set B (29 features) ---")
    feats_b = build_features(
        dev_df["interval_start_utc"],
        hourly_df=df_h,
        quarter_hourly_df=df_qh,
        weather_df=df_w,
        config=feat_cfg,
        feature_set="B",
    )
    dev_full_b = pd.concat([dev_df, feats_b.reset_index(drop=True)], axis=1)
    cols_b = get_feature_columns_b()

    # 1. Tune Set A
    print("\n================ Tuning Set A ================")
    tune_a = tune_models(dev_full_a, feature_set="A")

    # 2. Tune Set B
    print("\n================ Tuning Set B ================")
    tune_b = tune_models(dev_full_b, feature_set="B")

    print("\nBest Hyperparameters:")
    print("  Ridge Set A:", tune_a["ridge"]["best_params"], f"Daylight MAE: {tune_a['ridge']['best_daylight_mae']:.2f}")
    print("  HGB Set A:  ", tune_a["hist_gradient_boosting"]["best_params"], f"Daylight MAE: {tune_a['hist_gradient_boosting']['best_daylight_mae']:.2f}")
    print("  Ridge Set B:", tune_b["ridge"]["best_params"], f"Daylight MAE: {tune_b['ridge']['best_daylight_mae']:.2f}")
    print("  HGB Set B:  ", tune_b["hist_gradient_boosting"]["best_params"], f"Daylight MAE: {tune_b['hist_gradient_boosting']['best_daylight_mae']:.2f}")

    # 3. Cross-Validation on Identical Folds
    tscv = TimeSeriesSplit(n_splits=3)
    baseline_model = ClearSky15mPersistence(delay_minutes=delay_min)

    # Models dictionary
    models = {
        "Baseline_Champion": {"type": "baseline"},
        "Ridge_Set_A": {"factory": lambda: build_ridge_pipeline(tune_a["ridge"]["best_params"], random_seed=seed), "cols": cols_a, "df": dev_full_a},
        "HGB_Set_A": {"factory": lambda: build_hgb_model(tune_a["hist_gradient_boosting"]["best_params"], random_seed=seed), "cols": cols_a, "df": dev_full_a},
        "Ridge_Set_B": {"factory": lambda: build_ridge_pipeline(tune_b["ridge"]["best_params"], random_seed=seed), "cols": cols_b, "df": dev_full_b},
        "HGB_Set_B": {"factory": lambda: build_hgb_model(tune_b["hist_gradient_boosting"]["best_params"], random_seed=seed), "cols": cols_b, "df": dev_full_b},
    }

    # Generate predictions across folds
    predictions = {name: [] for name in models}
    val_targets = []
    val_times = []
    val_daylight = []
    val_weather_missing = []
    val_tcdc = []
    val_tmp = []

    for fold_num, (train_idx, val_idx) in enumerate(tscv.split(dev_df), 1):
        print(f"\nEvaluating fold {fold_num}/3 (Train: {len(train_idx)}, Val: {len(val_idx)})...")
        # Target and metadata
        val_sub_b = dev_full_b.iloc[val_idx]
        y_val = val_sub_b["target_kwh"].values
        is_day = (val_sub_b["solar_elevation_target_deg"] > 0).values
        w_miss = val_sub_b["weather_missing"].values
        tcdc = val_sub_b["tcdc_pct"].values
        tmp = val_sub_b["tmp_c"].values

        val_targets.append(y_val)
        val_times.append(val_sub_b["interval_start_utc"].values)
        val_daylight.append(is_day)
        val_weather_missing.append(w_miss)
        val_tcdc.append(tcdc)
        val_tmp.append(tmp)

        # Baseline predictions
        preds_base = baseline_model.predict(val_sub_b)
        predictions["Baseline_Champion"].append(preds_base)

        # Train and predict for each ML model
        for name, spec in models.items():
            if spec.get("type") == "baseline":
                continue
            df_curr = spec["df"]
            cols_curr = spec["cols"]

            tr_sub = df_curr.iloc[train_idx]
            v_sub = df_curr.iloc[val_idx]

            tr_mask = (tr_sub["quality_flag"] == "VALID") & (tr_sub["solar_elevation_target_deg"] > 0)
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
    tcdc_concat = np.concatenate(val_tcdc)
    tmp_concat = np.concatenate(val_tmp)

    preds_concat = {k: np.concatenate(v) for k, v in predictions.items()}

    # Masks for evaluation
    primary_mask = daylight_concat & (w_missing_concat == 0)
    all_daylight_mask = daylight_concat

    print(f"\nTotal validation daylight hours: {all_daylight_mask.sum():,}")
    print(f"Primary validation daylight hours (weather present): {primary_mask.sum():,} ({primary_mask.sum()/all_daylight_mask.sum():.2%})")
    print(f"Validation daylight hours with missing weather: {(daylight_concat & (w_missing_concat == 1)).sum():,}")

    def calc_metrics(y_t, y_p, mask):
        yt = y_t[mask]
        yp = y_p[mask]
        mae = float(np.mean(np.abs(yt - yp)))
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        cap_mae = float(mae / cap_limit * 100.0)
        return {"mae": mae, "rmse": rmse, "cap_norm_mae": cap_mae, "n": int(mask.sum())}

    print("\n==========================================================================================")
    print("PRIMARY COMPARISON: VALIDATION DAYLIGHT HOURS WITH WEATHER PRESENT (N = %d)" % primary_mask.sum())
    print("==========================================================================================")
    print(f"{'Model':<25s} | {'Daylight MAE (kWh)':<18s} | {'RMSE (kWh)':<12s} | {'Cap-MAE (%)':<12s} | {'vs Base Champ (%)':<18s} | {'vs Set A HGB (%)':<18s}")
    print("-" * 115)

    base_primary = calc_metrics(y_true_concat, preds_concat["Baseline_Champion"], primary_mask)
    hgb_a_primary = calc_metrics(y_true_concat, preds_concat["HGB_Set_A"], primary_mask)

    primary_results = {}
    for name in models:
        m = calc_metrics(y_true_concat, preds_concat[name], primary_mask)
        primary_results[name] = m
        vs_base = (base_primary["mae"] - m["mae"]) / base_primary["mae"] * 100.0
        vs_a = (hgb_a_primary["mae"] - m["mae"]) / hgb_a_primary["mae"] * 100.0 if name != "HGB_Set_A" else 0.0
        print(f"{name:<25s} | {m['mae']:18.2f} | {m['rmse']:12.2f} | {m['cap_norm_mae']:11.2f}% | {vs_base:17.2f}% | {vs_a:17.2f}%")

    print("\n==========================================================================================")
    print("ALL VALIDATION DAYLIGHT HOURS (INCLUDING MISSING WEATHER HOURS, N = %d)" % all_daylight_mask.sum())
    print("==========================================================================================")
    print(f"{'Model':<25s} | {'Daylight MAE (kWh)':<18s} | {'RMSE (kWh)':<12s} | {'Cap-MAE (%)':<12s} | {'vs Base Champ (%)':<18s} | {'vs Set A HGB (%)':<18s}")
    print("-" * 115)

    base_all = calc_metrics(y_true_concat, preds_concat["Baseline_Champion"], all_daylight_mask)
    hgb_a_all = calc_metrics(y_true_concat, preds_concat["HGB_Set_A"], all_daylight_mask)

    all_results = {}
    for name in models:
        m = calc_metrics(y_true_concat, preds_concat[name], all_daylight_mask)
        all_results[name] = m
        vs_base = (base_all["mae"] - m["mae"]) / base_all["mae"] * 100.0
        vs_a = (hgb_a_all["mae"] - m["mae"]) / hgb_a_all["mae"] * 100.0 if name != "HGB_Set_A" else 0.0
        print(f"{name:<25s} | {m['mae']:18.2f} | {m['rmse']:12.2f} | {m['cap_norm_mae']:11.2f}% | {vs_base:17.2f}% | {vs_a:17.2f}%")

    # Cloud-cover bin evaluation on primary hours
    print("\n==========================================================================================")
    print("STRATIFIED DAYLIGHT MAE BY FORECAST TOTAL CLOUD COVER BIN (WEATHER PRESENT HOURS)")
    print("==========================================================================================")
    bins = [
        ("Clear / Low Cloud (<20%)", (tcdc_concat < 20.0) & primary_mask),
        ("Mixed / Mid Cloud (20-80%)", (tcdc_concat >= 20.0) & (tcdc_concat <= 80.0) & primary_mask),
        ("Overcast / High Cloud (>80%)", (tcdc_concat > 80.0) & primary_mask),
    ]

    print(f"{'Cloud Cover Bin':<30s} | {'Hours':<8s} | {'Base Champ':<12s} | {'HGB Set A':<12s} | {'HGB Set B':<12s} | {'B vs A (kWh)':<14s} | {'B vs A (%)':<12s}")
    print("-" * 110)
    for bin_name, bin_mask in bins:
        n_bin = int(bin_mask.sum())
        m_base = calc_metrics(y_true_concat, preds_concat["Baseline_Champion"], bin_mask)["mae"]
        m_a = calc_metrics(y_true_concat, preds_concat["HGB_Set_A"], bin_mask)["mae"]
        m_b = calc_metrics(y_true_concat, preds_concat["HGB_Set_B"], bin_mask)["mae"]
        diff = m_a - m_b
        diff_pct = (m_a - m_b) / m_a * 100.0
        print(f"{bin_name:<30s} | {n_bin:<8d} | {m_base:12.2f} | {m_a:12.2f} | {m_b:12.2f} | {diff:13.2f} | {diff_pct:11.2f}%")

    # Monthly performance breakdown
    print("\n==========================================================================================")
    print("MONTHLY DAYLIGHT MAE BREAKDOWN (WEATHER PRESENT HOURS)")
    print("==========================================================================================")
    m_base_df = compute_monthly_daylight_mae(y_true_concat, preds_concat["Baseline_Champion"], times_concat, primary_mask)
    m_a_df = compute_monthly_daylight_mae(y_true_concat, preds_concat["HGB_Set_A"], times_concat, primary_mask)
    m_b_df = compute_monthly_daylight_mae(y_true_concat, preds_concat["HGB_Set_B"], times_concat, primary_mask)

    monthly_comp = pd.merge(m_base_df.rename(columns={"daylight_mae": "mae_base"}), m_a_df.rename(columns={"daylight_mae": "mae_a"}), on=["month", "daylight_hours"])
    monthly_comp = pd.merge(monthly_comp, m_b_df.rename(columns={"daylight_mae": "mae_b"}), on=["month", "daylight_hours"])
    monthly_comp["diff_b_vs_a"] = monthly_comp["mae_a"] - monthly_comp["mae_b"]
    monthly_comp["diff_pct"] = monthly_comp["diff_b_vs_a"] / monthly_comp["mae_a"] * 100.0
    print(monthly_comp.to_string(index=False))

    # Permutation importance for selected model
    # Model selection rule:
    # "select champion by validation MAE on identical hours (prefer Set A if difference < 1%)."
    mae_diff = hgb_a_primary["mae"] - primary_results["HGB_Set_B"]["mae"]
    mae_diff_pct = mae_diff / hgb_a_primary["mae"] * 100.0
    print("\n==========================================================================================")
    print("MODEL SELECTION DECISION")
    print("==========================================================================================")
    print(f"HGB Set A Daylight MAE: {hgb_a_primary['mae']:.2f} kWh")
    print(f"HGB Set B Daylight MAE: {primary_results['HGB_Set_B']['mae']:.2f} kWh")
    print(f"Difference (A - B): {mae_diff:+.2f} kWh ({mae_diff_pct:+.2f}%)")

    if mae_diff_pct >= 1.0:
        champion_model_name = "HGB_Set_B"
        champion_feature_set = "B"
        print("Set B improves upon Set A by >= 1.0%. Selected champion: HistGradientBoosting (Set B).")
    else:
        champion_model_name = "HGB_Set_A"
        champion_feature_set = "A"
        print("Difference between Set B and Set A is < 1.0% (negligible or negative). Preferring Set A (simpler, zero-external-weather dependency).")

    # Permutation Importance computation for champion model
    print("\n==========================================================================================")
    print(f"COMPUTING PERMUTATION IMPORTANCE FOR {champion_model_name} (5 REPEATS ACROSS 3 FOLDS)")
    print("==========================================================================================")
    champ_spec = models[champion_model_name]
    champ_df = champ_spec["df"]
    champ_cols = champ_spec["cols"]

    rng = np.random.RandomState(seed)
    n_repeats = 5
    feature_importances = {col: [] for col in champ_cols}

    for fold_num, (train_idx, val_idx) in enumerate(tscv.split(dev_df), 1):
        tr_sub = champ_df.iloc[train_idx]
        v_sub = champ_df.iloc[val_idx]

        tr_mask = (tr_sub["quality_flag"] == "VALID") & (tr_sub["solar_elevation_target_deg"] > 0)
        if "SUSPECT_ZERO" in tr_sub["quality_flag"].values:
            tr_mask = tr_mask & (tr_sub["quality_flag"] != "SUSPECT_ZERO")

        X_tr = tr_sub.loc[tr_mask, champ_cols].values
        y_tr = tr_sub.loc[tr_mask, "target_kwh"].values

        model = champ_spec["factory"]()
        model.fit(X_tr, y_tr)

        v_day_mask = (v_sub["solar_elevation_target_deg"] > 0).values
        # If set B, evaluate permutation on weather present hours
        if champion_feature_set == "B":
            v_eval_mask = v_day_mask & (v_sub["weather_missing"].values == 0)
        else:
            v_eval_mask = v_day_mask

        X_v = v_sub[champ_cols].values.copy()
        y_v = v_sub["target_kwh"].values

        # Base predictions
        base_raw = model.predict(X_v)
        base_pred = np.where(v_day_mask, base_raw, 0.0)
        base_pred = np.clip(base_pred, 0.0, cap_limit)
        base_mae = float(np.mean(np.abs(y_v[v_eval_mask] - base_pred[v_eval_mask])))

        # Permute each feature
        for col_idx, col_name in enumerate(champ_cols):
            fold_col_diffs = []
            for _ in range(n_repeats):
                X_perm = X_v.copy()
                perm_idx = rng.permutation(len(X_perm))
                X_perm[:, col_idx] = X_perm[perm_idx, col_idx]

                perm_raw = model.predict(X_perm)
                perm_pred = np.where(v_day_mask, perm_raw, 0.0)
                perm_pred = np.clip(perm_pred, 0.0, cap_limit)
                perm_mae = float(np.mean(np.abs(y_v[v_eval_mask] - perm_pred[v_eval_mask])))
                fold_col_diffs.append(perm_mae - base_mae)

            feature_importances[col_name].append(float(np.mean(fold_col_diffs)))

    # Average importance across folds
    avg_importances = [
        {"feature": col, "mae_increase_kwh": float(np.mean(diffs))}
        for col, diffs in feature_importances.items()
    ]
    avg_importances.sort(key=lambda x: x["mae_increase_kwh"], reverse=True)

    print(f"\nTop 10 Features by Permutation Importance ({champion_model_name}):")
    print(f"{'Rank':<5s} | {'Feature Name':<32s} | {'Daylight MAE Increase (kWh)':<28s}")
    print("-" * 70)
    for r, item in enumerate(avg_importances[:10], 1):
        print(f"{r:<5d} | {item['feature']:<32s} | +{item['mae_increase_kwh']:<26.2f}")

    # Save summary data for writing reports
    output_summary = {
        "primary_results": primary_results,
        "all_results": all_results,
        "cloud_bins": {
            bin_name: {
                "n_hours": int(bin_mask.sum()),
                "mae_base": float(calc_metrics(y_true_concat, preds_concat["Baseline_Champion"], bin_mask)["mae"]),
                "mae_a": float(calc_metrics(y_true_concat, preds_concat["HGB_Set_A"], bin_mask)["mae"]),
                "mae_b": float(calc_metrics(y_true_concat, preds_concat["HGB_Set_B"], bin_mask)["mae"]),
            }
            for bin_name, bin_mask in bins
        },
        "monthly_comp": monthly_comp.to_dict(orient="records"),
        "tune_a": tune_a,
        "tune_b": tune_b,
        "champion_model_name": champion_model_name,
        "champion_feature_set": champion_feature_set,
        "mae_diff": float(mae_diff),
        "mae_diff_pct": float(mae_diff_pct),
        "permutation_importance": avg_importances,
    }

    with open("reports/set_a_vs_b_summary.json", "w", encoding="utf-8") as f:
        json.dump(output_summary, f, indent=2, default=str)
    print("\nSummary written to reports/set_a_vs_b_summary.json")

if __name__ == "__main__":
    main()
