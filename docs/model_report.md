# SolarCast Model Report: Mode F1 Candidate Models & Benchmarking

**Author**: Member 1 (Data, Features, Models, MLflow)  
**Date**: November 23, 2026  
**Commit**: `5af9c87` (working tree)  
**Status**: Complete — Gate Passed (Eligible)

---

## 1. Executive Summary

This report documents the training, hyperparameter optimization, cross-validation benchmarking, and gate evaluation for **SolarCast Mode F1** (advance next-hour solar AC active energy forecasting in kWh for PV site 2107, 893 kW DC capacity).

Evaluation was conducted under strict operational constraints with `production_available_delay_minutes = 15`. Models were trained and evaluated exclusively on the **80% Train/Dev partition** (37,090 hourly intervals: 2018-07-12 00:00 UTC to 2022-10-17 00:00 UTC) across **3 expanding-window chronological folds**. The **final 20% test partition** (9,273 hourly intervals: 2022-10-17 01:00 UTC to 2023-11-09 23:00 UTC) remains **strictly locked** (`final_test_opened: false`).

To ensure a fair and rigorous comparison, candidate models are benchmarked against baselines endowed with the **same information set**, specifically incorporating recent 15-minute telemetry:
- **Baseline Champion**: **Clear-Sky Scaled 15-Minute Persistence** ($kt_{15} \times \text{ClearSky}_{\text{target}}$): Daylight MAE **42.18 kWh** (4.72% capacity), RMSE 67.61 kWh.
- **Ridge Regression (Pipeline)**: Daylight MAE **43.52 kWh** (4.87% capacity) — achieves a -3.16% change (does not beat the 15-minute scaled baseline, though it beats hourly baselines by 42.45%).
- **HistGradientBoostingRegressor (Winner)**: Daylight MAE **32.24 kWh** (3.61% capacity), achieving a **23.56% improvement** over the champion baseline.

The candidate gate requires $\ge 5.0\%$ daylight MAE improvement over the baseline champion ($\le 40.07\text{ kWh}$) and capacity-normalized daylight MAE $\le 20.0\%$. **HistGradientBoosting** achieved **23.56% improvement** and **3.61% cap-MAE**, passing all gate criteria and being registered as model `solarcast-f1` (version 2) in MLflow.

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

## 5. Model Validation Benchmark (Fair Information Set)

To prevent unfair comparisons between models using 15-minute readings and baselines using 2-hour old hourly readings, all five baselines and both candidate models were evaluated on the exact same 3 expanding-window validation folds:

| Model | Daylight MAE (kWh) | Daylight RMSE (kWh) | Cap-Norm Daylight MAE | All-Hour MAE (kWh) | Improvement vs Champion Baseline | Gate Eligibility |
|---|---|---|---|---|---|---|
| **Clear-Sky Scaled 15-Min Persistence** (Baseline Champion) | **42.18** | **67.61** | **4.72%** | **21.15** | *Reference* | N/A |
| **Last 15-Minute Persistence** | 88.07 | 112.82 | 9.86% | 44.10 | -108.79% | N/A |
| **Clear-Sky Persistence (Hourly)** | 75.62 | 113.80 | 8.47% | 37.87 | -79.27% | N/A |
| **Same-Hour Previous Day** | 75.65 | 128.37 | 8.47% | 37.86 | -79.33% | N/A |
| **Persistence (Latest Available Hour)** | 175.73 | 216.05 | 19.68% | 87.95 | -316.59% | N/A |
| **Ridge Regression** | 43.52 | 64.04 | 4.87% | 21.81 | -3.16% | REJECTED |
| **HistGradientBoosting** (Winner) | **32.24** | **54.76** | **3.61%** | **16.18** | **+23.56%** | **ELIGIBLE** |

### Key Observations:
1. **The Fair Baseline Champion**: Clear-sky scaling of the latest 15-minute reading cuts the persistence baseline error nearly in half (from 75.62 kWh down to 42.18 kWh).
2. **Linear Ridge Underperforms Non-linear Baseline**: Ridge Regression (43.52 kWh) cannot match the multiplicative interaction of the 15-minute clear-sky baseline (42.18 kWh), failing the gate.
3. **HistGradientBoosting Wins Decisively**: HistGradientBoosting (32.24 kWh) outperforms the 15-minute clear-sky baseline by **23.56%** (and beats Ridge by 25.9%), proving that gradient boosted trees successfully discover non-linear combinations of recent generation slopes and astronomical coordinates that simple heuristics cannot capture.

---

## 6. Feature Importance Analysis (Permutation Importance)

Permutation feature importance was evaluated for `HistGradientBoostingRegressor` on the out-of-fold validation sets across all 3 folds (shuffling each feature 5 times and measuring the degradation in Daylight MAE):

| Rank | Feature Name | Daylight MAE Increase (kWh) | Plain-Language Explanation |
|:---:|---|:---:|---|
| **1** | `power_kw_lag_0m` | **+116.92** | **Most Recent Inverter Power**: Active power in the 15-minute interval right before cutoff is by far the single most critical predictor. Shuffling it destroys the recent weather/generation baseline, increasing error by 116.9 kWh. |
| **2** | `hour_sin` | **+40.97** | **Diurnal Cycle Timing**: Captures whether the solar plant is on an upward morning ramp, midday plateau, or evening ramp. |
| **3** | `energy_kwh_prev_day_same_hour` | **+15.03** | **Yesterday's Generation Baseline**: Provides multi-day weather persistence (e.g. continuous clear summer heat dome vs cloudy monsoon week). |
| **4** | `clearsky_ghi_target_wm2` | **+11.33** | **Clear-Sky Irradiance Ceiling**: Theoretical astronomical maximum solar irradiance reaching the horizontal plane at target hour midpoint. |
| **5** | `power_kw_lag_30m` | **+7.67** | **Power Rate-of-Change**: Allows the model to compute recent ramps (cloud incoming vs cloud clearing) across 30 minutes. |
| **6** | `hour_cos` | **+7.62** | **Diurnal Phase Alignment**: Complements `hour_sin` to resolve exact solar time without ambiguity. |
| **7** | `energy_kwh_prev_week_same_hour` | **+5.77** | **Seasonal Reference**: Tracks the seasonal shift in generation for that specific solar hour. |
| **8** | `clearsky_index_latest_avail` | **+3.91** | **Atmospheric Clearness Ratio**: Measures cloud attenuation observed during the last fully available hour. |
| **9** | `power_kw_lag_15m` | **+2.99** | **Intermediate Lag**: Refines the short-term generation slope between lag 0m and lag 30m. |
| **10** | `energy_kwh_rolling_mean_24h` | **+2.84** | **Daily Energy Regime**: Quantifies overall plant productivity over the past 24 hours. |

---

## 7. Monthly Performance Breakdown

Validation daylight MAE (kWh) across representative months on the out-of-fold validation set:

| Month | Daylight Hours | Clear-Sky 15m Baseline (kWh) | Ridge MAE (kWh) | HistGradientBoosting MAE (kWh) |
|---|---|---|---|---|
| 2019-08 | 389 | 24.12 | 27.29 | **18.56** |
| 2019-12 | 302 | 52.19 | 61.66 | **43.11** |
| 2020-01 | 276 | 68.45 | 66.62 | **59.66** |
| 2020-04 | 397 | 45.80 | 48.26 | **37.99** |
| 2020-07 | 446 | 16.32 | 25.94 | **12.11** |
| 2021-01 | 294 | 67.80 | 70.70 | **55.07** |
| 2021-06 | 437 | 25.10 | 35.18 | **19.95** |
| 2021-12 | 302 | 53.40 | 60.21 | **42.71** |
| 2022-07 | 447 | 26.50 | 32.33 | **20.36** |

---

## 8. Acceptance Gate Evaluation (`evaluation.json`)

The formal candidate acceptance criteria defined in `configs/evaluation.yaml`:
1. **Daylight MAE Improvement**: $\ge 5.0\%$ reduction vs baseline champion ($\le 40.07\text{ kWh}$).
2. **Capacity-Normalized MAE**: $\le 20.0\%$ (0.20).
3. **Physical Feasibility**: Strictly finite, non-negative, and $\le 893.0\text{ kWh}$.

### Outcome:
```json
{
  "eligible": true,
  "required_improvement_pct": 5.0,
  "achieved_improvement_pct": 23.56,
  "thresholds": {
    "max_daylight_mae_kwh": 40.07,
    "max_capacity_norm_daylight_mae": 0.2,
    "capacity_limit_kwh": 893.0
  },
  "baseline_metrics": {
    "daylight_mae_kwh": 42.1841,
    "daylight_rmse_kwh": 67.6055,
    "capacity_norm_daylight_mae": 0.0472,
    "all_hour_mae_kwh": 21.15,
    "all_hour_rmse_kwh": 47.8138
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
  "baseline_champion_name": "Clear-Sky Scaled 15-Minute Persistence"
}
```
**Gate Decision**: **ELIGIBLE**.

---

## 9. Diagnostic Plots

Diagnostic artifacts generated and tracked in MLflow (`reports/plots/`):
- `actual_vs_predicted.png`: Out-of-fold scatter plot showing tight clustering along the 1:1 diagonal with daylight MAE annotation.
- `residuals.png`: Error distribution centered at zero ($\mu = 0.52\text{ kWh}$) with symmetric tails.
- `error_by_hour.png`: MAE profile by UTC hour, confirming 0 nighttime error and peak error during solar noon convective hours.
- `error_by_month.png`: Seasonal MAE progression highlighting winter cloud challenges.

---

## 10. Model Artifacts & MLflow Lineage

- **MLflow Tracking URI**: `file:./mlruns`
- **Experiment**: `solarcast-model-training`
- **Candidate Run**: `HistGradientBoosting_SetA_15mDelay` (`7b64f8c9349e4943bbeb094672fabb80`)
- **Registered Model**: `solarcast-f1` (Version 2, status: `candidate`)
- **Rejected Run**: `Ridge_SetA_15mDelay` (`01f1b17d58694fcf9afc91d071e27e0a`, status: `rejected`, improvement: -3.16%)
- **Model Bundle**: `models/solarcast_f1_bundle.joblib` (439 KB) containing estimator, feature schema, data hashes, and bounded inference methods.
