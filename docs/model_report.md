# SolarCast Model Report: Mode F1 Candidate Models & Benchmarking

**Author**: Member 1 (Data, Features, Models, MLflow)  
**Date**: November 23, 2026  
**Commit**: `b829b00` (working tree)  
**Status**: Complete — Gate Passed (Eligible)

---

## 1. Executive Summary

This report documents the training, hyperparameter optimization, cross-validation benchmarking, and gate evaluation for **SolarCast Mode F1** (advance next-hour solar AC active energy forecasting in kWh for PV site 2107, 893 kW DC capacity).

Evaluation was conducted under strict operational constraints with `production_available_delay_minutes = 15`. Models were trained and evaluated exclusively on the **80% Train/Dev partition** (37,090 hourly intervals: 2018-07-12 00:00 UTC to 2022-10-17 00:00 UTC) across **3 expanding-window chronological folds**. The **final 20% test partition** (9,273 hourly intervals: 2022-10-17 01:00 UTC to 2023-11-09 23:00 UTC) remains **strictly locked** (`final_test_opened: false`).

Both candidate models comfortably beat the baseline champion (**Clear-Sky Persistence**, Daylight MAE 75.62 kWh):
- **Ridge Regression (Pipeline)**: Daylight MAE **43.52 kWh** (4.87% capacity), **42.45% improvement**.
- **HistGradientBoostingRegressor (Winner)**: Daylight MAE **32.24 kWh** (3.61% capacity), **57.36% improvement**.

The candidate gate required $\ge 5.0\%$ daylight MAE improvement over baseline and capacity-normalized daylight MAE $\le 20.0\%$. **HistGradientBoosting** achieved **57.36% improvement** and **3.61% cap-MAE**, passing all gate requirements and being registered as model `solarcast-f1` (version 1) in MLflow.

---

## 2. Evaluation Protocol & Data Partitioning

### 2.1 Dataset & Time Horizon
- **Plant**: PVDAQ Site 2107 (Denver, CO; DC Capacity: 893 kW).
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

### 2.3 Operational Production Delay
At forecast issue time $H$, latest plant data requires an operational acquisition delay:
$$\text{Cutoff} = H - 15\text{ minutes}$$
- Latest usable 15-minute interval: $[H - 30\text{m}, H - 15\text{m})$
- Latest fully available hour: $[H - 2\text{h}, H - 1\text{h})$

---

## 3. Feature Set A (23 Production & Clear-Sky Features)

Feature Set A utilizes only production telemetry, astronomical positions, and pvlib clear-sky models (no weather forecast data yet; pending HRRR extraction):

| # | Feature Name | Description | Units |
|---|---|---|---|
| 1–4 | `power_kw_lag_0m` to `45m` | 15-minute active power at $H-15\text{m}, H-30\text{m}, H-45\text{m}, H-60\text{m}$ | kW |
| 5 | `power_kw_last_4_mean` | Average active power across the last 4 available 15-minute intervals | kW |
| 6 | `energy_kwh_latest_avail_hour` | Energy of latest fully completed hour $[H-2\text{h}, H-1\text{h})$ | kWh |
| 7 | `energy_kwh_prev_day_same_hour` | Energy of target hour on previous day $[H-24\text{h}, H-23\text{h})$ | kWh |
| 8 | `energy_kwh_prev_week_same_hour` | Energy of target hour on previous week $[H-168\text{h}, H-167\text{h})$ | kWh |
| 9 | `energy_kwh_rolling_mean_6h` | Past-only 6-hour rolling mean of hourly energy $[H-7\text{h}, H-1\text{h})$ | kWh |
| 10 | `energy_kwh_rolling_mean_24h` | Past-only 24-hour rolling mean of hourly energy $[H-25\text{h}, H-1\text{h})$ | kWh |
| 11–12 | `hour_sin`, `hour_cos` | Harmonic encoding of target hour of day | - |
| 13–14 | `day_of_year_sin`, `cos` | Harmonic encoding of day of year | - |
| 15 | `solar_elevation_target_deg` | Solar elevation angle at midpoint of target hour ($H+30\text{m}$) | deg |
| 16 | `clearsky_ghi_target_wm2` | pvlib Ineichen clear-sky GHI at midpoint ($H+30\text{m}$) | $\text{W/m}^2$ |
| 17 | `clearsky_energy_target_kwh` | Clear-sky expected energy ($\text{GHI} \times P95\text{ ratio}$) | kWh |
| 18 | `clearsky_index_latest_avail` | Clear-sky index of latest available hour $[H-2\text{h}, H-1\text{h})$ | - |
| 19 | `production_age_minutes` | Latency of latest available meter reading (nominal 15m) | min |
| 20–23 | `power_kw_missing_count`, flags | Missingness count (0–4) and indicator flags for lags | counts / flags |

---

## 4. Hyperparameter Tuning Results

Tuning was conducted over 10 randomized configurations per model across the 3 cross-validation folds:

### 4.1 Ridge Regression (Pipeline with MedianImputer + StandardScaler)
- **Search Space**: `alpha` $\in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]$, `fit_intercept` $\in \{\text{True}, \text{False}\}$.
- **Optimal Hyperparameters**: `alpha = 0.001`, `fit_intercept = True`.
- **Validation Daylight MAE**: **43.52 kWh** (stable across `alpha` $\le 1.0$).

### 4.2 HistGradientBoostingRegressor (Native NaN Handling)
- **Search Space**: `learning_rate` $\in [0.01, 0.03, 0.05, 0.1, 0.15, 0.2]$, `max_iter` $\in [50, 100, 150, 200]$, `max_leaf_nodes` $\in [15, 31, 63]$, `min_samples_leaf` $\in [10, 20, 50]$.
- **Optimal Hyperparameters**:
  - `learning_rate`: `0.05`
  - `max_iter`: `200`
  - `max_leaf_nodes`: `31`
  - `min_samples_leaf`: `20`
- **Validation Daylight MAE**: **32.24 kWh** (3.61% capacity).

---

## 5. Model Validation Benchmark

The table below summarizes performance across the 3 expanding folds (27,816 validation hours, of which 13,873 are daylight hours):

| Model Name | Daylight MAE (kWh) | Daylight RMSE (kWh) | Cap-Norm Daylight MAE | All-Hour MAE (kWh) | All-Hour RMSE (kWh) | Improvement vs Baseline |
|---|---|---|---|---|---|---|
| **Clear-Sky Persistence** (Baseline Champion) | 75.62 | 113.80 | 8.47% | 37.87 | 80.53 | *Reference* |
| **Ridge Regression** | 43.52 | 64.04 | 4.87% | 21.81 | 45.29 | +42.45% |
| **HistGradientBoosting** (Winner) | **32.24** | **54.76** | **3.61%** | **16.18** | **38.73** | **+57.36%** |

*Note: All-hour metrics evaluate over all hours (predicting 0.0 at night with zero night error).*

### Key Observations:
1. **HistGradientBoosting Dominance**: Non-linear tree interactions between clear-sky energy, recent 15-minute power slopes, and diurnal harmonics yield a **25.9% error reduction over Ridge** (32.24 kWh vs 43.52 kWh) and **57.36% over Clear-Sky Persistence**.
2. **Nighttime Zero Guarantee**: Enforcing exact zero for night hours eliminates false nighttime generation, cutting all-hour MAE in half.
3. **Physical Bound Clipping**: All predictions are clipped to $[0, 893.0\text{ kWh}]$, ensuring 0 negative values and 0 capacity exceedances.

---

## 6. Monthly Performance Breakdown

Validation daylight MAE (kWh) across representative months on the out-of-fold validation set:

| Month | Daylight Hours | Clear-Sky Persistence MAE | Ridge MAE | HistGradientBoosting MAE |
|---|---|---|---|---|
| 2019-08 | 389 | 49.32 | 27.29 | **18.56** |
| 2019-12 | 302 | 92.40 | 61.66 | **43.11** |
| 2020-01 | 276 | 110.15 | 66.62 | **59.66** |
| 2020-04 | 397 | 85.12 | 48.26 | **37.99** |
| 2020-07 | 446 | 36.88 | 25.94 | **12.11** |
| 2021-01 | 294 | 112.44 | 70.70 | **55.07** |
| 2021-06 | 437 | 42.10 | 35.18 | **19.95** |
| 2021-12 | 302 | 104.70 | 60.21 | **42.71** |
| 2022-07 | 447 | 41.50 | 32.33 | **20.36** |

- **Summer Performance**: Exceptional accuracy in June–August (MAE ~12–20 kWh, ~1.4%–2.2% capacity) where clear skies dominate Denver summer days.
- **Winter Performance**: Higher error in December–February (MAE ~42–60 kWh, ~4.7%–6.7% capacity) driven by cloud cover unpredictability in the absence of numerical weather forecasts. Adding forecast cloud cover (HRRR) is expected to significantly reduce winter errors.

---

## 7. Acceptance Gate Evaluation (`evaluation.json`)

The formal candidate acceptance criteria defined in `configs/evaluation.yaml`:
1. **Daylight MAE Improvement**: $\ge 5.0\%$ reduction vs baseline champion ($\le 71.84\text{ kWh}$).
2. **Capacity-Normalized MAE**: $\le 20.0\%$ (0.20).
3. **Physical Feasibility**: Strictly finite, non-negative, and $\le 893.0\text{ kWh}$.

### Outcome:
```json
{
  "eligible": true,
  "required_improvement_pct": 5.0,
  "achieved_improvement_pct": 57.36,
  "thresholds": {
    "max_daylight_mae_kwh": 71.84,
    "max_capacity_norm_daylight_mae": 0.2,
    "capacity_limit_kwh": 893.0
  },
  "baseline_metrics": {
    "daylight_mae_kwh": 75.6201,
    "daylight_rmse_kwh": 113.8039,
    "capacity_norm_daylight_mae": 0.0847,
    "all_hour_mae_kwh": 37.868,
    "all_hour_rmse_kwh": 80.5283
  },
  "candidate_metrics": {
    "daylight_mae_kwh": 32.2446,
    "daylight_rmse_kwh": 54.7602,
    "capacity_norm_daylight_mae": 0.0361,
    "all_hour_mae_kwh": 16.1815,
    "all_hour_rmse_kwh": 38.7329
  },
  "failure_reasons": [],
  "final_test_opened": false,
  "candidate_model_name": "HistGradientBoosting",
  "baseline_champion_name": "Clear-Sky Persistence"
}
```
**Gate Decision**: **ELIGIBLE**.

---

## 8. Diagnostic Plots

Diagnostic artifacts generated and tracked in MLflow (`reports/plots/`):
- `actual_vs_predicted.png`: Out-of-fold scatter plot showing tight clustering along the 1:1 diagonal with daylight MAE annotation.
- `residuals.png`: Error distribution centered at zero ($\mu = 0.52\text{ kWh}$) with symmetric tails.
- `error_by_hour.png`: MAE profile by UTC hour, confirming 0 nighttime error and peak error during solar noon convective hours.
- `error_by_month.png`: Seasonal MAE progression highlighting winter cloud challenges.

---

## 9. Model Artifacts & MLflow Lineage

- **MLflow Tracking URI**: `file:./mlruns`
- **Experiment**: `solarcast-model-training`
- **Candidate Run**: `HistGradientBoosting_SetA_15mDelay` (`88c574915ce7417c83a4911cc4b1f127`)
- **Registered Model**: `solarcast-f1` (Version 1, status: `candidate`)
- **Model Bundle**: `models/solarcast_f1_bundle.joblib` (439 KB) containing estimator, feature schema, data hashes, and bounded inference methods.
