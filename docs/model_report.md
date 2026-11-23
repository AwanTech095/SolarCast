# SolarCast Model Report: Mode F1 Candidate Models & Benchmarking (Feature Set A vs B)

**Author**: Member 1 (Data, Features, Models, MLflow)  
**Date**: November 23, 2026  
**Commit**: `m1/hrrr-forecast-features`  
**Status**: Complete — Gate Passed (Eligible, Selected Champion: HistGradientBoosting Set B)

---

## 1. Executive Summary

This report documents the training, hyperparameter optimization, cross-validation benchmarking, Feature Set A vs Feature Set B comparison, and gate evaluation for **SolarCast Mode F1** (advance next-hour solar AC active energy forecasting in kWh for PV site 2107, 893 kW DC capacity).

Evaluation was conducted under strict operational constraints with `production_available_delay_minutes = 15` and `weather_availability_lag_hours = 2`. Models were trained and evaluated exclusively on the **80% Train/Dev partition** (37,090 hourly intervals: 2018-07-12 00:00 UTC to 2022-10-17 00:00 UTC) across **3 expanding-window chronological folds**. The **final 20% test partition** (9,273 hourly intervals: 2022-10-17 01:00 UTC to 2023-11-09 23:00 UTC) remains **strictly locked** (`final_test_opened: false`).

### Key Results on Identical Primary Validation Daylight Hours ($N = 13,374$):
- **Baseline Champion**: **Clear-Sky Scaled 15-Minute Persistence** ($kt_{15} \times \text{ClearSky}_{\text{target}}$): Daylight MAE **42.09 kWh** (4.71% capacity), RMSE 67.52 kWh.
- **Ridge Regression Set A (23 features)**: Daylight MAE **43.61 kWh** (4.88% capacity), RMSE 64.17 kWh (-3.62% vs baseline champion, rejected).
- **Ridge Regression Set B (29 features)**: Daylight MAE **42.62 kWh** (4.77% capacity), RMSE 62.29 kWh (-1.27% vs baseline champion, rejected).
- **HistGradientBoosting Set A (23 features)**: Daylight MAE **32.24 kWh** (3.61% capacity), RMSE 54.88 kWh (+23.40% vs baseline champion).
- **HistGradientBoosting Set B (29 features, Selected Winner)**: Daylight MAE **31.43 kWh** (3.52% capacity), RMSE 53.35 kWh (**+25.33% improvement** vs baseline champion).

### Model Selection Decision:
Feature Set B improves upon Feature Set A by **+0.81 kWh (+2.52% relative improvement)** in Daylight MAE and **1.53 kWh** in RMSE. Because the improvement satisfies the pre-registered candidate selection rule ($\ge 1.0\%$ improvement), **HistGradientBoosting (Feature Set B)** is selected as the winning champion model. The candidate gate criteria ($\ge 5.0\%$ improvement and $\le 20.0\%$ capacity MAE) are fully satisfied (**25.33% improvement**, **3.52% cap-MAE**), qualifying the model as **ELIGIBLE** in `evaluation.json`.

---

## 2. Evaluation Protocol & Data Partitioning

### 2.1 Dataset & Time Horizon
- **Plant**: PVDAQ Site 2107 (Farm Solar Array, Arbuckle, CA; DC Capacity: 893 kW, Nominal AC: 662.4 kW).
- **Evaluation Window**: 2018-07-12 00:00 UTC to 2023-11-09 23:00 UTC (46,363 VALID hourly records).
- **Target Variable**: Active AC Energy in kWh over $[H, H+1\text{h})$ UTC.
- **Filtering**: Target hours must have `quality_flag == "VALID"` and not `SUSPECT_ZERO`. Daylight evaluation strictly requires midpoint solar elevation $> 0^\circ$. Night hours ($\le 0^\circ$) are deterministically predicted as $0.0\text{ kWh}$.

### 2.2 Chronological Split & Leakage Safeguards
- **Train/Dev Partition (80%)**: 37,090 hours (2018-07-12 00:00 UTC to 2022-10-17 00:00 UTC).
- **Final Test Partition (20%)**: 9,273 hours (2022-10-17 01:00 UTC to 2023-11-09 23:00 UTC).
  - **Locked**: Never used for tuning, selection, or threshold estimation. Guarded by `check_final_test_safeguard()` in code.
- **Cross-Validation**: 3 expanding-window chronological folds (`TimeSeriesSplit(n_splits=3)`) within Train/Dev:
  - Fold 1: Train 9,274 hrs $\to$ Val 9,272 hrs (2019-08 to 2020-08)
  - Fold 2: Train 18,546 hrs $\to$ Val 9,272 hrs (2020-08 to 2021-09)
  - Fold 3: Train 27,818 hrs $\to$ Val 9,272 hrs (2021-09 to 2022-10)

### 2.3 Operational Production & Weather Cutoff Conventions
At forecast issue time $H$, latest plant and weather data adhere strictly to operational latency:
- **Production cutoff**: $\text{Cutoff} = H - 15\text{ minutes}$.
  - Latest usable 15-minute interval: $[H - 30\text{m}, H - 15\text{m})$.
  - Latest fully available hour: $[H - 2\text{h}, H - 1\text{h})$.
- **Weather availability constraint**: $\text{weather\_available\_at\_utc} \le H$.
  - Nominal HRRR release lag is 2 hours ($\text{init} \le H - 2\text{h}$).
  - When nominal run is unavailable, cascading fallback up to 8 hours age is permitted.
  - Future weather is strictly zeroed or rejected by the pipeline.

---

## 3. Feature Sets A and B

The feature definitions are tracked under `configs/feature_schema.json` (Schema Version `1.1.0`):

| Group | Features | Count | Description |
|---|---|:---:|---|
| **Set A (Base)** | Lags `0m`, `15m`, `30m`, `45m`, `last_4_mean`; Hourly lags `latest_avail`, `prev_day`, `prev_week`; Rolling means `6h`, `24h`; Harmonics `hour_sin`, `hour_cos`, `day_of_year_sin`, `day_of_year_cos`; pvlib `solar_elevation_target_deg`, `clearsky_ghi_target_wm2`, `clearsky_energy_target_kwh`, `clearsky_index_latest_avail`; Telemetry metadata `production_age_minutes`, `power_kw_missing_count`, `latest_hour_missing`, `prev_day_missing`, `prev_week_missing` | 23 | Past-only plant generation history, astronomical calendar encodings, and pvlib clear-sky models. |
| **Set B (Weather-Enhanced)** | All 23 features from Set A plus: `tmp_c` (2m air temperature), `tcdc_pct` (total cloud cover %), `clearsky_ghi_cloud_adj_wm2` (Kasten-Czeplak cloud-scaled clear-sky GHI), `weather_run_age_hours`, `weather_fallback_used`, `weather_missing` | 29 | Incorporates NOAA HRRR NWP point forecasts and physical cloud attenuation. |

---

## 4. Hyperparameter Tuning

Tuning was conducted over 10 randomized configurations per model across the 3 cross-validation folds:

### 4.1 Ridge Regression (Pipeline with MedianImputer + StandardScaler)
- **Search Space**: `alpha` $\in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]$, `fit_intercept` $\in \{\text{True}, \text{False}\}$.
- **Optimal Hyperparameters (Set A & Set B)**: `alpha = 0.001`, `fit_intercept = True`.
- **Validation Daylight MAE**: 43.52 kWh (Set A) $\to$ 42.67 kWh (Set B).

### 4.2 HistGradientBoostingRegressor (Native NaN Handling)
- **Search Space**: `learning_rate` $\in [0.01, 0.03, 0.05, 0.1, 0.15, 0.2]$, `max_iter` $\in [50, 100, 150, 200]$, `max_leaf_nodes` $\in [15, 31, 63]$, `min_samples_leaf` $\in [10, 20, 50]$, `l2_regularization` $\in [0.0, 1.0, 10.0]$.
- **Optimal Hyperparameters (Set A & Set B)**:
  - `learning_rate`: `0.05`
  - `max_iter`: `200`
  - `max_leaf_nodes`: `31`
  - `min_samples_leaf`: `100`
  - `l2_regularization`: `1.0`
- **Validation Daylight MAE**: 32.24 kWh (Set A) $\to$ 31.53 kWh (Set B).

---

## 5. Model Validation Benchmark (Set A vs Set B Comparison)

### 5.1 Primary Evaluation: Identical Daylight Hours with Weather Available ($N = 13,374$)

| Model | Feature Set | Daylight MAE (kWh) | Daylight RMSE (kWh) | Cap-Norm MAE (%) | vs Baseline Champion (%) | vs Set A HGB (%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Clear-Sky Scaled 15m Persistence** | Baseline | 42.09 | 67.52 | 4.71% | *Reference* | -30.55% |
| **Ridge Regression** | Set A (23) | 43.61 | 64.17 | 4.88% | -3.62% | -35.28% |
| **Ridge Regression** | Set B (29) | 42.62 | 62.29 | 4.77% | -1.27% | -32.21% |
| **HistGradientBoosting** | Set A (23) | 32.24 | 54.88 | 3.61% | +23.40% | *Reference* |
| **HistGradientBoosting** | **Set B (29)** | **31.43** | **53.35** | **3.52%** | **+25.33%** | **+2.52%** |

### 5.2 All Daylight Hours Including Missing Weather Hours ($N = 13,914$)
In the 540 validation daylight hours where HRRR weather was missing/unobtainable (`weather_missing == 1`), Set B gracefully routed NaNs via native tree branching:
- **Baseline Champion**: Daylight MAE 42.18 kWh, RMSE 67.62 kWh
- **HistGradientBoosting Set A**: Daylight MAE 32.24 kWh, RMSE 54.83 kWh
- **HistGradientBoosting Set B**: Daylight MAE **31.53 kWh**, RMSE **53.37 kWh** (+2.20% improvement over Set A)

---

## 6. Detailed Stratification: Does Weather Help?

To understand how and where weather forecasts help, performance was stratified by forecast cloud cover and calendar month.

### 6.1 Stratified Performance by Forecast Cloud Cover Bin ($N = 13,374$)

| Cloud Cover Regime | Hours ($N$) | Baseline Champion (kWh) | HGB Set A (kWh) | HGB Set B (kWh) | Diff (A - B) (kWh) | Relative Gain (%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Clear / Low Cloud ($< 20\%$)** | 9,843 (73.6%) | 36.66 | 25.55 | **24.12** | **+1.43** | **+5.59%** |
| **Mixed / Mid Cloud ($20\% - 80\%$)** | 1,664 (12.4%) | 59.77 | **49.46** | 54.26 | **-4.80** | **-9.70%** |
| **Overcast / High Cloud ($> 80\%$)** | 1,867 (14.0%) | 54.94 | 52.17 | **49.60** | **+2.58** | **+4.94%** |

#### Critical Finding & Physical Explanation:
1. **Clear Sky ($< 20\%$)**: Weather features provide a consistent **+5.59%** gain. The 2m temperature feature enables the tree model to account for PV panel thermal derating (efficiency loss at high cell temperatures during hot California summer afternoons).
2. **Overcast Sky ($> 80\%$)**: Weather features provide a **+4.94%** gain. The Kasten-Czeplak cloud adjustment (`clearsky_ghi_cloud_adj_wm2`) and `tcdc_pct` signal strong frontal cloud attenuation ahead of time.
3. **Mixed Clouds ($20\% - 80\%$) — Degradation**: Weather features make the model **-9.70% WORSE** (MAE increases by 4.80 kWh).
   - *Hypothesis*: timing errors of forecast cloud on broken cloud; not tested. In scattered or convective cloud regimes, a 3-km numerical weather model may misjudge the timing of cloud transit directly over an 893 kW collector field. If HRRR predicts 60% cloud cover while sun shines through a gap, or predicts clear sky while a cumulus cloud shades the array, weather features may pull the prediction in the wrong direction. By contrast, Set A's recent 15-minute telemetry (`power_kw_lag_0m`) directly senses the current ground-level shading state.

### 6.2 Monthly Breakdown
Across the 39 out-of-fold validation months:
- **Set B beats Set A in 30 out of 39 months (76.9% of months)**. Major gains occur during frontal winter rain/cloud events (e.g., Dec 2021: +11.44%, Nov 2019: +9.64%, Mar 2020: +9.25%, Feb 2022: +6.44%).
- **Set A beats Set B in 9 months (23.1% of months)**, primarily during spring and late summer transition months with broken convective cloudiness (e.g., Sept 2020: -4.84%, Aug 2020: -3.98%, Apr 2022: -3.60%, May 2020: -3.09%, Sept 2021: -3.38%).

---

## 7. Permutation Feature Importance for Selected Model (HGB Set B)

Computed across 5 permutation iterations on out-of-fold validation data across all 3 folds:

| Rank | Feature Name | Daylight MAE Increase (kWh) | Category | Explanation |
|:---:|---|:---:|:---:|---|
| **1** | `power_kw_lag_0m` | **+144.04** | Telemetry | **Most Recent Inverter Power**: Active power ending at $H-15\text{m}$ is by far the single dominant predictor. Shuffling it destroys immediate plant state. |
| **2** | `hour_sin` | **+51.94** | Astronomy | **Diurnal Cycle Timing**: Captures solar elevation ramp direction (morning vs afternoon). |
| **3** | `energy_kwh_prev_day_same_hour` | **+8.26** | History | **Yesterday's Generation**: Captures multi-day weather persistence. |
| **4** | `clearsky_ghi_target_wm2` | **+6.53** | Clear-Sky | **Theoretical Irradiance Ceiling**: pvlib extraterrestrial clear-sky baseline. |
| **5** | `clearsky_ghi_cloud_adj_wm2` | **+4.92** | **Weather (NWP)** | **Cloud-Adjusted Irradiance**: Kasten-Czeplak parameterization combining clear-sky GHI with HRRR total cloud cover. Top-ranked weather feature. |
| **6** | `hour_cos` | **+4.23** | Astronomy | **Diurnal Phase**: Resolves exact time of day symmetry. |
| **7** | `tcdc_pct` | **+3.29** | **Weather (NWP)** | **Total Cloud Cover Forecast**: Direct numerical forecast percentage from HRRR. |
| **8** | `clearsky_index_latest_avail` | **+3.05** | Telemetry | **Past Clearness Index**: Observed cloudiness in the latest fully available hour $[H-2\text{h}, H-1\text{h})$. |
| **9** | `energy_kwh_prev_week_same_hour` | **+2.62** | History | **Seasonal Day-of-Week Reference**. |
| **10** | `power_kw_lag_30m` | **+2.53** | Telemetry | **Short-Term Ramp Rate**: Captures rapid cloud entry/exit over 30 minutes. |
| **17** | `tmp_c` | **+0.50** | **Weather (NWP)** | **Ambient Temperature**: Modulates PV module temperature and thermal derating. |
| **24–29** | Metadata flags (`weather_missing`, `weather_run_age_hours`, etc.) | **$\approx 0.00$** | Metadata | Informative routing indicators for edge cases. |

---

## 8. Acceptance Gate Evaluation (`evaluation.json`)

The formal candidate acceptance criteria defined in `configs/evaluation.yaml`:
1. **Daylight MAE Improvement**: $\ge 5.0\%$ reduction vs baseline champion.
2. **Capacity-Normalized MAE**: $\le 20.0\%$ (0.20).
3. **Physical Feasibility**: Strictly finite, non-negative, and $\le 893.0\text{ kWh}$.
4. **Final Test Safeguard**: Final test partition unopened (`final_test_opened: false`).

```json
{
  "eligible": true,
  "required_improvement_pct": 5.0,
  "achieved_improvement_pct": 25.33,
  "thresholds": {
    "max_daylight_mae_kwh": 39.98,
    "max_capacity_norm_daylight_mae": 0.2,
    "capacity_limit_kwh": 893.0
  },
  "baseline_metrics": {
    "daylight_mae_kwh": 42.0891,
    "daylight_rmse_kwh": 67.5237,
    "capacity_norm_daylight_mae": 0.0471,
    "all_hour_mae_kwh": 21.15,
    "all_hour_rmse_kwh": 47.8266
  },
  "candidate_metrics": {
    "daylight_mae_kwh": 31.4262,
    "daylight_rmse_kwh": 53.3511,
    "capacity_norm_daylight_mae": 0.0352,
    "all_hour_mae_kwh": 15.8267,
    "all_hour_rmse_kwh": 37.7508
  },
  "failure_reasons": [],
  "final_test_opened": false,
  "candidate_model_name": "HistGradientBoosting_Set_B",
  "baseline_champion_name": "Clear-Sky Scaled 15-Minute Persistence"
}
```
**Gate Decision**: **ELIGIBLE**.

---

## 9. Model Artifacts & MLflow Lineage

- **MLflow Tracking URI**: `file:./mlruns`
- **Experiment**: `solarcast-model-training`
- **Winning Candidate Run**: `HistGradientBoosting_Set_B_15mDelay`
- **Registered Model**: `solarcast-f1` (Version 4, status: `candidate`)
- **Benchmark Comparison Runs**:
  - `Ridge_Set_A_15mDelay` (status: `rejected`)
  - `HistGradientBoosting_Set_A_15mDelay` (status: `rejected`)
  - `Ridge_Set_B_15mDelay` (status: `rejected`)
- **Model Bundle**: `models/solarcast_f1_bundle.joblib`
  - Dataclass: `ModelBundle`
  - `feature_set`: `"B"`
  - `needs_weather`: `True`
  - `feature_schema_version`: `"1.1.0"`
  - `feature_names`: 29 canonical features
  - Bounded inference: deterministic zeroing at night, clipping to $[0.0, 893.0\text{ kWh}]$, optional standalone `weather_df` merging.
- **Diagnostic Plots (`reports/plots/`)**:
  - `actual_vs_predicted.png`: Out-of-fold validation scatter with 1:1 diagonal.
  - `residuals.png`: Error histogram centered near zero with symmetric dispersion.
  - `error_by_hour.png`: Diurnal MAE profile showing zero nighttime error.
  - `error_by_month.png`: 39-month out-of-fold MAE tracking.
