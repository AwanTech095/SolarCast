# SolarCast Final Test Evaluation Report: Mode F1 Models on 20% Held-Out Test Partition

**Author**: Member 1 (Data, Features, Models, MLflow)  
**Date**: November 23, 2026  
**Evaluation Window**: `2022-10-17 01:00:00+00:00` to `2023-11-09 23:00:00+00:00` (9,273 valid test hours)  
**Safeguard Status**: Opened Once (`reports/final_test_audit.json`)  
**Audit Reason**: `"one-time final evaluation after model selection"`  
**Git Commit**: `763b39104ba9993987590d51e879dc16a011174f`  
**MLflow Run ID**: `620faa341a8a4c14824f31e40b157fd6`  
**Tracking URI**: `file:./mlruns` (Experiment: `solarcast-model-training`)  

---

## 1. Executive Summary & Integrity Statement

This document reports the official, one-time final evaluation of SolarCast Mode F1 models on the strictly held-out **final 20% test partition** (latest chronological partition: October 17, 2022 01:00 UTC through November 9, 2023 23:00 UTC).

### Critical Integrity & Governance Statement
> **No model was selected, tuned, or modified based on these evaluation results.**  
> All feature definitions (configs/feature_schema.json v1.1.0), hyperparameter search spaces, trained estimators, and champion model selection (`HistGradientBoosting_Set_B`) were established, frozen, and committed exclusively on the 80% Train/Dev development partition (2018-07-12 to 2022-10-17 UTC) prior to opening this test set. The models evaluated were fitted strictly on the Train/Dev partition; no test rows were ever included in model fitting.

### Key Evaluation Findings
1. **Winning Model Confirmation**: The pre-selected champion model **`HistGradientBoosting_Set_B`** (Feature Set B, 29 features with NOAA HRRR numerical weather predictions) decisively confirmed its superiority on the final test partition:
   - **Primary Daylight Hours (Weather Available, $N = 4,387$)**: Achieves **32.69 kWh Daylight MAE** (**3.66% capacity-normalized MAE**) and **56.85 kWh RMSE**.
   - **Improvement vs Baseline Champion**: Beats the champion baseline (`Clear_Sky_Scaled_15m_Persistence`, 43.48 kWh MAE) by **+24.83%** (exceeding the 5.0% course gate by nearly $5\times$).
   - **Improvement vs Feature Set A**: Beats the weather-free model (`HistGradientBoosting_Set_A`, 33.99 kWh MAE) by **+1.31 kWh (+3.84% relative gain)**, confirming that forecast weather provides authentic, generalized predictive skill on unseen data.
2. **All Daylight Hours Resilience ($N = 4,601$)**:
   - Evaluated across all daylight hours, including 214 hours where forecast weather was unavailable (cause unknown; observed only as missing files/chunks in the archive), `HistGradientBoosting_Set_B` achieves **33.48 kWh Daylight MAE** (**3.75% capacity-normalized MAE**), outperforming the baseline champion by **+24.33%** and Set A by **+3.46%**.
3. **Generalization Gap**:
   - The test Daylight MAE of `HistGradientBoosting_Set_B` (32.69 kWh) degrades by only **+1.26 kWh (+4.01%)** compared to its 3-fold cross-validation performance on Train/Dev (31.43 kWh). This is the smallest generalization gap among all evaluated machine learning models.

---

## 2. Test Window Accounting & Exclusions

The evaluation window was determined by reserving the latest 20% of valid chronological time following the strict project rule (`AGENTS.md`).

| Category | Hours | Percentage | Description / Accounting |
|---|:---:|:---:|---|
| **Total Test Calendar Window** | **9,335** | 100.00% | Hourly intervals from `2022-10-17 01:00 UTC` to `2023-11-09 23:00 UTC`. |
| **Excluded Test Hours** | **62** | 0.66% | Removed per quality assurance criteria before evaluation. |
| — *Suspect Zero Outages (`SUSPECT_ZERO`)* | *35* | *0.37%* | Daytime hours with solar elevation $> 5^\circ$ where meter reported persistent $0.0\text{ kW}$ generation (cause unknown; observed only as persistent zero daytime generation during periods with solar elevation $> 5^\circ$). |
| — *Missing Plant Telemetry (`MISSING_DATA`)* | *27* | *0.29%* | Hours with fewer than 4 valid 15-minute telemetry intervals. |
| **Valid Evaluated Test Hours** | **9,273** | 99.34% | Clean test dataset evaluated across all models. |
| **Nighttime Hours ($\le 0^\circ$ solar elevation)** | **4,672** | 50.38% | Deterministically predicted as $0.0\text{ kWh}$ across all models. |
| **Total Daylight Hours ($> 0^\circ$ solar elevation)** | **4,601** | 49.62% | Daylight hours subject to solar generation forecasting. |
| — *Primary Daylight Hours (Weather Available)* | **4,387** | 95.35% | Daylight hours with valid NOAA HRRR forecast data ($H - 2\text{h}$ operational latency). |
| — *Daylight Hours Missing Weather* | **214** | 4.65% | Daylight hours lacking weather (including 101 daylight hours from the 2023-08-16 to 2023-08-23 period: cause unknown; observed only as missing files/chunks in the archive). |

---

## 3. Comprehensive Model Benchmark Results

All five project baselines and four machine learning model configurations were evaluated in a single atomic evaluation event. Models were trained strictly on the 37,090 hours of the Train/Dev partition (18,603 valid daylight training hours).

### 3.1 Primary Evaluation: Daylight Hours with Weather Available ($N = 4,387$)

The primary benchmark evaluates performance on identical daylight hours where all features (telemetry, solar geometry, and forecast weather) are present.

| Model / Pipeline | Feature Set | Daylight MAE (kWh) | Daylight RMSE (kWh) | Cap-Norm MAE (%) | vs Baseline Champion (%) | vs Set A HGB (%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Latest Hour Persistence** | 1 | 174.71 | 215.77 | 19.56% | -301.80% | -413.98% |
| **Same-Hour Previous Day** | 1 | 83.24 | 144.52 | 9.32% | -91.43% | -144.88% |
| **Clear-Sky Persistence (Hourly)** | 2 | 72.52 | 110.34 | 8.12% | -66.77% | -113.34% |
| **Last 15-Minute Persistence** | 1 | 90.80 | 116.43 | 10.17% | -108.82% | -167.13% |
| **Clear-Sky Scaled 15m Persistence** | Baseline Champ | **43.48** | **72.28** | **4.87%** | *Reference (0.0%)* | -27.92% |
| **Ridge Regression** | Set A (23) | 47.01 | 70.00 | 5.26% | -8.11% | -38.29% |
| **Ridge Regression** | Set B (29) | 45.29 | 66.93 | 5.07% | -4.17% | -33.25% |
| **HistGradientBoosting** | Set A (23) | 33.99 | 58.92 | 3.81% | +21.83% | *Reference (0.0%)* |
| **HistGradientBoosting (Champion)** | **Set B (29)** | **32.69** | **56.85** | **3.66%** | **+24.83%** | **+3.84%** |

*Note: Capacity-normalized MAE is computed relative to the 893 kW DC plant rating ($32.69 / 893 = 3.66\%$).*

### 3.2 All Daylight Hours (Including Missing Weather Hours, $N = 4,601$)

This evaluation tests system robustness in the presence of real-world forecast archive outages. Set B pipelines handle missing weather natively (using native NaN histogram binning for HistGradientBoosting and Median Imputation for Ridge).

| Model / Pipeline | Feature Set | Daylight MAE (kWh) | Daylight RMSE (kWh) | Cap-Norm MAE (%) | vs Baseline Champion (%) | vs Set A HGB (%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Latest Hour Persistence** | 1 | 175.41 | 216.39 | 19.64% | -296.52% | -405.85% |
| **Same-Hour Previous Day** | 1 | 83.51 | 144.79 | 9.35% | -88.78% | -140.84% |
| **Clear-Sky Persistence (Hourly)** | 2 | 73.12 | 111.08 | 8.19% | -65.29% | -110.87% |
| **Last 15-Minute Persistence** | 1 | 91.56 | 117.48 | 10.25% | -106.97% | -164.04% |
| **Clear-Sky Scaled 15m Persistence** | Baseline Champ | **44.24** | **73.58** | **4.95%** | *Reference (0.0%)* | -27.57% |
| **Ridge Regression** | Set A (23) | 47.62 | 71.02 | 5.33% | -7.65% | -37.33% |
| **Ridge Regression** | Set B (29) | 46.19 | 68.18 | 5.17% | -4.42% | -33.22% |
| **HistGradientBoosting** | Set A (23) | 34.68 | 59.99 | 3.88% | +21.61% | *Reference (0.0%)* |
| **HistGradientBoosting (Champion)** | **Set B (29)** | **33.48** | **58.02** | **3.75%** | **+24.33%** | **+3.46%** |

### 3.3 All-Hour Metrics (All 9,273 Test Hours)

All models strictly output $0.0\text{ kWh}$ during nighttime intervals ($\text{solar\_elevation} \le 0^\circ$), resulting in exact zero error at night.

| Model | Feature Set | All-Hour MAE (kWh) | All-Hour RMSE (kWh) | Capacity-Normalized MAE (%) |
|---|:---:|:---:|:---:|:---:|
| **Latest Hour Persistence** | 1 | 87.10 | 152.43 | 9.75% |
| **Same-Hour Previous Day** | 1 | 41.50 | 101.99 | 4.65% |
| **Clear-Sky Persistence (Hourly)** | 2 | 36.34 | 78.25 | 4.07% |
| **Last 15-Minute Persistence** | 1 | 45.49 | 82.76 | 5.09% |
| **Clear-Sky Scaled 15m Persistence** | Baseline Champ | 22.01 | 51.83 | 2.47% |
| **Ridge Regression** | Set A (23) | 23.69 | 50.03 | 2.65% |
| **Ridge Regression** | Set B (29) | 22.98 | 48.03 | 2.57% |
| **HistGradientBoosting** | Set A (23) | 17.27 | 42.26 | 1.93% |
| **HistGradientBoosting (Champion)** | **Set B (29)** | **16.67** | **40.87** | **1.87%** |

---

## 4. Stratified Analysis by Forecast Cloud-Cover Bin

To evaluate where numerical weather forecasts provide the highest marginal value, test daylight hours with weather available were stratified by forecast total cloud cover (`tcdc_pct`):

| Cloud Cover Stratum | Hours | Baseline Champ MAE (kWh) | HGB Set A MAE (kWh) | HGB Set B MAE (kWh) | Difference (A - B) (kWh) | Set B Improvement vs Set A (%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Clear / Low Cloud (`< 20%`)** | 2,781 (63.4%) | 32.20 | 21.72 | **20.38** | +1.34 | **+6.15%** |
| **Mixed / Mid Cloud (`20% - 80%`)** | 579 (13.2%) | 70.38 | **56.35** | 58.16 | -1.81 | **-3.21%** |
| **Overcast / High Cloud (`> 80%`)** | 1,027 (23.4%) | 58.87 | 54.62 | **51.64** | +2.98 | **+5.46%** |

### Detailed Stratum Insights
1. **Clear Skies (`< 20%` TCDC, 63.4% of hours)**:
   - Feature Set B achieves **20.38 kWh MAE** (2.28% capacity MAE), improving over Set A by **+6.15%**. Incorporating forecast temperature (`tmp_c`) captures inverter efficiency and module thermal degradation during hot summer conditions.
2. **Overcast & High Cloud (`> 80%` TCDC, 23.4% of hours)**:
   - Feature Set B achieves **51.64 kWh MAE**, beating Set A by **+2.98 kWh (+5.46%)** and beating the baseline champion by **+7.23 kWh (+12.28%)**. Strong, widespread synoptic storm systems are reliably detected in advance by the HRRR model, suppressing false high-generation predictions that plague persistence models.
3. **Broken / Mixed Cloud (`20% - 80%` TCDC, 13.2% of hours)**:
   - Feature Set B exhibits a minor degradation of **-1.81 kWh (-3.21%)** relative to Set A (58.16 kWh vs 56.35 kWh).
   - **Hypothesis**: timing errors of forecast cloud on broken cloud; not tested. In broken cloud regimes, a 3-km NWP grid cell may experience spatial-temporal phase timing errors (e.g. cloud passage simulated earlier or later than actual arrival at the array). Set A's recent 15-minute telemetry senses immediate ground-level conditions, whereas mistimed NWP cloud signals can misguide tree splits. However, on the test set this difference was smaller (-3.21% vs -9.70% on validation), and is balanced by the 6.15% gain across the majority 63.4% clear-sky regime.

---

## 5. Per-Month Performance Breakdown

Monthly daylight MAE breakdown on the test partition (October 2022 to November 2023, weather-present hours):

| Month | Daylight Hours | Baseline Champ MAE (kWh) | HGB Set A MAE (kWh) | HGB Set B MAE (kWh) | Diff (A - B) (kWh) | Set B vs Set A (%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **2022-10** | 155 | 36.58 | **27.82** | 28.79 | -0.97 | -3.48% |
| **2022-11** | 297 | 51.79 | 41.34 | **40.52** | +0.83 | +2.00% |
| **2022-12** | 294 | 51.51 | 46.22 | **42.91** | +3.31 | **+7.16%** |
| **2023-01** | 294 | 46.85 | 37.22 | **34.70** | +2.52 | **+6.77%** |
| **2023-02** | 291 | 60.24 | 49.99 | **47.91** | +2.08 | +4.16% |
| **2023-03** | 317 | 64.79 | 60.76 | **56.90** | +3.86 | **+6.35%** |
| **2023-04** | 354 | 37.51 | 27.53 | **27.44** | +0.09 | +0.31% |
| **2023-05** | 402 | 43.87 | 30.68 | **29.91** | +0.77 | +2.50% |
| **2023-06** | 437 | 40.17 | 26.42 | **26.20** | +0.22 | +0.84% |
| **2023-07** | 447 | 34.15 | **24.01** | 24.54 | -0.53 | -2.23% |
| **2023-08** | 320 | 32.57 | 20.32 | **19.59** | +0.73 | +3.60% |
| **2023-09** | 364 | 26.88 | 21.92 | **20.86** | +1.05 | +4.80% |
| **2023-10** | 336 | 40.04 | 32.61 | **30.37** | +2.25 | **+6.89%** |
| **2023-11** | 79 | 67.39 | 55.74 | **49.15** | +6.58 | **+11.81%** |

### Monthly Trends
- **Consistency**: Feature Set B outperforms Feature Set A in **12 out of 14 test months** (85.7% of calendar months).
- **Winter Months**: Substantial gains occur during overcast winter months (e.g. November 2023: **+11.81%**, note small sample of 79 daylight hours; December 2022: **+7.16%**; January 2023: **+6.77%**; March 2023: **+6.35%**). During these overcast periods, persistence baselines degrade (MAE $> 60\text{ kWh}$), whereas HRRR NWP signals alert the model to cloud deck formation hours before arrival.
- **Summer Stability**: During clear summer months (June–September), Daylight MAE remains stable at **19.5–26.2 kWh** (~2.2%–2.9% capacity).

---

## 6. Generalization Gap Analysis

The generalization gap measures the difference in performance between the 3-fold expanding cross-validation on the Train/Dev partition and the final held-out test partition.

| Model Pipeline | Validation CV Daylight MAE (kWh) | Final Test Daylight MAE (kWh) | Generalization Gap $\Delta\text{MAE}$ (kWh) | Generalization Gap $\Delta\%$ |
|---|:---:|:---:|:---:|:---:|
| **Clear-Sky Scaled 15m Persistence** | 42.09 | 43.48 | +1.39 | +3.31% |
| **Ridge Regression Set A** | 43.61 | 47.01 | +3.40 | +7.79% |
| **Ridge Regression Set B** | 42.62 | 45.29 | +2.67 | +6.27% |
| **HistGradientBoosting Set A** | 32.24 | 33.99 | +1.75 | +5.44% |
| **HistGradientBoosting Set B (Champion)** | **31.43** | **32.69** | **+1.26** | **+4.01%** |

### Assessment of Overfitting & Model Stability
- **Minimal Gap**: Champion model `HistGradientBoosting_Set_B` exhibits a generalization gap of **+1.26 kWh (+4.01%)**. This confirms that hyperparameter tuning on the training partition avoided severe overfitting.
- **Model Comparison**: The validation-to-test gap is +1.26 kWh (+4.01%) for HGB B, +1.75 kWh (+5.44%) for HGB A, and +1.39 kWh (+3.31%) for the baseline champion; differences are small.
- **Linear Model Degradation**: Ridge regression showed larger degradation (+6.27% to +7.79%), confirming that non-linear gradient boosted trees handle seasonal interactions and solar geometry far more effectively.

---

## 7. Analysis of Missing Weather

In the test partition, **214 daylight hours** (4.65% of total daylight hours) lacked valid forecast weather:
- **Contiguous August 2023 Gap**: 101 contiguous daylight hours occurred between `2023-08-16 15:00 UTC` and `2023-08-23 22:00 UTC` (cause unknown; observed only as missing files/chunks in the archive).
- **Scattered Missing Hours**: The remaining 113 daylight hours were scattered 1-to-2 hour periods (cause unknown; observed only as missing files/chunks in the archive).

### Pipeline Resilience Under Missing Weather
- In operational deployment, when forecast weather is unavailable, `HistGradientBoosting_Set_B` utilizes scikit-learn's built-in **native NaN histogram binning**, routing missing values to default split directions learned during training.
- Comparing subsets:
  - Primary daylight (weather present, $N = 4,387$): Set B Daylight MAE = **32.69 kWh** (vs Set A 33.99 kWh).
  - All daylight ($N = 4,601$): Set B Daylight MAE = **33.48 kWh** (vs Set A 34.68 kWh).
  - Shift: **+0.79 kWh** for Set B, while Set A (which does not use weather at all) shows **+0.69 kWh** between the same subsets (33.99 to 34.68 kWh). The shift reflects differing hour subsets rather than a pure degradation from missing weather alone.
- Across the full 4,601 daylight hours, Set B maintains a **24.33% improvement** over the baseline champion (44.24 kWh) and beats Set A (34.68 kWh) by **3.46%** with native NaN handling.

---

## 8. Final Gate Verification & Compliance Audit

| Requirement / Gate Criteria | Threshold / Requirement | Final Test Observed | Compliance Status |
|---|---|:---:|:---:|
| **Candidate Daylight MAE Improvement** | $\ge 5.0\%$ improvement vs Baseline Champion | **+24.83%** (32.69 vs 43.48 kWh) | **PASSED** (Exceeded by $4.96\times$) |
| **Capacity-Normalized Daylight MAE** | $\le 20.0\%$ of plant DC rating (893 kW) | **3.66%** | **PASSED** (Far below limit) |
| **Non-negative Predictions** | $\min(\hat{y}) \ge 0.0\text{ kWh}$ | $0.0\text{ kWh}$ | **PASSED** (0 negative predictions) |
| **Physical Plant Capacity Bound** | $\max(\hat{y}) \le 893.0\text{ kWh}$ | $781.4\text{ kWh}$ | **PASSED** (0 over-capacity predictions) |
| **Nighttime Determinism** | $\hat{y} = 0.0$ when solar elevation $\le 0^\circ$ | $0.0\text{ kWh}$ identically | **PASSED** (Strictly zero error at night) |
| **Single Opening Integrity** | Test set opened exactly once with logged audit | Audit record in `final_test_audit.json` | **PASSED** (One single audited event) |

---

## 9. Conclusion & Recommendations for Member 2 & Member 3

1. **Production Champion Selected**: `solarcast_f1_bundle.joblib` (Model: `HistGradientBoostingRegressor` with Feature Set B, Schema `1.1.0`) is conclusively verified and ready for production serving.
2. **Serving Contract**: The bundle encapsulates `feature_set = "B"` and `needs_weather = True`. When serving next-hour predictions at hour $H$:
   - Telemetry must be queried up to $H - 15\text{m}$.
   - NOAA HRRR forecast point must be queried for the target hour with valid release $\le H$.
   - If HRRR is unavailable, pass `NaN` for weather columns; the bundle handles missing weather gracefully without error.
3. **Deviation Monitoring (Task M1.6)**: The test residuals ($\text{actual} - \hat{y}_{\text{F1}}$) establish the empirical baseline for residual calibration and plant deviation alerts.
