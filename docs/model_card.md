# Model Card: SolarCast Mode F1 (`solarcast-f1`)

## 1. Model Details
- **Model Name**: `solarcast-f1`
- **Version**: `1.1.0` (Feature Set B with NOAA HRRR NWP Weather)
- **Model Type**: Gradient Boosted Decision Trees (`HistGradientBoostingRegressor` from `scikit-learn`)
- **Developer**: SolarCast MLOps Team — Member 1 (Data, Features, Models)
- **Release Date**: November 23, 2026
- **License**: Course Internal / Educational
- **Framework & Dependencies**: Python 3.11, `scikit-learn==1.5.2`, `joblib==1.4.2`, `pvlib==0.11.1`, `numpy==1.26.4`, `pandas==2.2.3`, `mlflow==2.17.2`.

---

## 2. Intended Use & Scope
- **Primary Task**: Next-hour solar AC active energy forecast in kilowatt-hours (kWh) over the interval $[H, H+1\text{h})$ UTC.
- **Plant / Site**: PVDAQ Site 2107 (Farm Solar Array, Arbuckle, CA; Nameplate DC Capacity: 893 kW, Nominal AC: 662.4 kW).
- **Forecast Issue Timing**: Issued at hour $H$ UTC with an operational plant telemetry cutoff of $H - 15\text{ minutes}$ UTC and weather availability constraint of $\text{available\_at\_utc} \le H$ UTC (2-hour nominal latency).
- **Out of Scope**:
  - Intra-hour (e.g., 5-minute or 15-minute) dispatch forecasting.
  - Multi-day or day-ahead forecasting.
  - Direct application to other PV sites without site-specific retraining and geographical recalibration.
  - Curtailment forecasting or inverter failure diagnosis (production deviation monitoring handles post-hoc discrepancy flagging).

---

## 3. Factors & Input Features

The model uses **Feature Set B** (29 features), defined in `configs/feature_schema.json` (Schema Version `1.1.0`):
1. **Recent Production Lags (15-min intervals)**:
   - `power_kw_lag_0m`: Power during $[H-30\text{m}, H-15\text{m})$
   - `power_kw_lag_15m`: Power during $[H-45\text{m}, H-30\text{m})$
   - `power_kw_lag_30m`: Power during $[H-60\text{m}, H-45\text{m})$
   - `power_kw_lag_45m`: Power during $[H-75\text{m}, H-60\text{m})$
   - `power_kw_last_4_mean`: Mean of the 4 intervals
2. **Historical Production Lags & Rolling Means**:
   - `energy_kwh_latest_avail_hour`: Latest fully completed hour $[H-2\text{h}, H-1\text{h})$
   - `energy_kwh_prev_day_same_hour`: Same hour yesterday $[H-24\text{h}, H-23\text{h})$
   - `energy_kwh_prev_week_same_hour`: Same hour last week $[H-168\text{h}, H-167\text{h})$
   - `energy_kwh_rolling_mean_6h`: 6-hour past rolling mean $[H-7\text{h}, H-1\text{h})$
   - `energy_kwh_rolling_mean_24h`: 24-hour past rolling mean $[H-25\text{h}, H-1\text{h})$
3. **Temporal & Solar Geometry Features**:
   - `hour_sin`, `hour_cos`: Diurnal cycle
   - `day_of_year_sin`, `day_of_year_cos`: Annual seasonal cycle
   - `solar_elevation_target_deg`: Solar elevation at midpoint $H+30\text{m}$
4. **Physical Clear-Sky Benchmarks**:
   - `clearsky_ghi_target_wm2`: pvlib Ineichen model GHI at midpoint $H+30\text{m}$
   - `clearsky_energy_target_kwh`: Expected clear-sky energy ($\text{GHI} \times P95\text{ ratio}$)
   - `clearsky_index_latest_avail`: Recent atmospheric transmission index $[H-2\text{h}, H-1\text{h})$
5. **Operational Telemetry Flags**:
   - `production_age_minutes`: Latency of plant data (nominally 15 min)
   - `power_kw_missing_count`: Count of missing readings in last 4 intervals (0 to 4)
   - `latest_hour_missing`, `prev_day_missing`, `prev_week_missing`: Binary availability flags
6. **NOAA HRRR Forecast Weather & Cloud Adjustment Features**:
   - `tmp_c`: 2-meter air temperature forecast ($^\circ\text{C}$) averaged over leads F1 and F2
   - `tcdc_pct`: Total cloud cover forecast (%) averaged over leads F1 and F2
   - `clearsky_ghi_cloud_adj_wm2`: Kasten-Czeplak cloud-scaled clear-sky GHI ($\text{GHI} \times (1 - 0.75 \times (\text{TCDC}/100)^{3.4})$)
   - `weather_run_age_hours`: Age of initialization run (nominal 2.0 h, higher if fallback)
   - `weather_fallback_used`: Flag (1 if fallback run $>2\text{h}$ old used, 0 otherwise)
   - `weather_missing`: Flag (1 if forecast weather is missing or out-of-bounds, 0 otherwise)

---

## 4. Training Data & Preprocessing

- **Data Sources**:
  - Inverter telemetry: PVDAQ Site 2107 (`data/processed/hourly.parquet` and `quarter_hourly.parquet`).
  - Forecast weather: NOAA HRRR MesoWest Zarr archive (`data/processed/hrrr_forecast_features.parquet`).
- **Development Partition (80%)**: 37,090 hours (2018-07-12 00:00 UTC to 2022-10-17 00:00 UTC).
- **Training Samples**: 18,603 daylight hours (`solar_elevation_target_deg > 0`, `quality_flag == "VALID"`, not `SUSPECT_ZERO`).
- **Final Test Set (20%)**: 9,273 valid hours (2022-10-17 01:00 UTC to 2023-11-09 23:00 UTC) opened once on 2026-11-23 under strict audit logging for one-time evaluation (see [docs/final_test_report.md](file:///c:/projects/SolarCast/docs/final_test_report.md)).
- **Handling of Missing Values**: Built-in native histogram binning support in `HistGradientBoostingRegressor` (NaNs treated as missing values, avoiding imputation distortion).
- **Value Range Sanitization**: Any out-of-range weather values (TCDC outside $[0, 100]\%$, TMP outside $[-40, 60]^\circ\text{C}$, such as upstream fill value $-10000$ in July 2020) are sanitized to `NaN` with `weather_missing = 1`.
- **Physical Bounding**:
  - Target hours with solar elevation $\le 0^\circ$ are deterministically set to $0.0\text{ kWh}$.
  - Predictions during daylight are strictly clipped to $[0.0, 893.0\text{ kWh}]$.

---

## 5. Evaluation & Performance Metrics

### 5.1 Out-of-Fold Cross-Validation on Train/Dev (3 Folds, N = 13,374 Daylight Hours with Weather)

| Metric | Selected Champion (`solarcast-f1` Set B) | Candidate Set A (No Weather) | Champion Baseline (`Clear-Sky Scaled 15-Min Persistence`) | Improvement vs Baseline Champion | Improvement vs Set A |
|---|:---:|:---:|:---:|:---:|:---:|
| **Daylight MAE** | **31.43 kWh** | 32.24 kWh | 42.09 kWh | **+25.33%** | **+2.52%** |
| **Daylight RMSE** | **53.35 kWh** | 54.88 kWh | 67.52 kWh | **+21.00%** | **+2.79%** |
| **Capacity-Normalized Daylight MAE** | **3.52%** | 3.61% | 4.71% | **-1.19 pp** | **-0.09 pp** |
| **All-Hour MAE** | **15.83 kWh** | 16.18 kWh | 21.15 kWh | **+25.17%** | **+2.20%** |
| **All-Hour RMSE** | **37.75 kWh** | 38.73 kWh | 47.83 kWh | **+21.07%** | **+2.53%** |

### 5.2 One-Time Final Test Evaluation (Latest 20% Held-Out Time, N = 4,387 Primary Daylight Hours)

| Metric | Selected Champion (`solarcast-f1` Set B) | Candidate Set A (No Weather) | Champion Baseline (`Clear-Sky Scaled 15-Min Persistence`) | Improvement vs Baseline Champion | Improvement vs Set A |
|---|:---:|:---:|:---:|:---:|:---:|
| **Daylight MAE** | **32.69 kWh** | 33.99 kWh | 43.48 kWh | **+24.83%** | **+3.84%** |
| **Daylight RMSE** | **56.85 kWh** | 58.92 kWh | 72.28 kWh | **+21.36%** | **+3.51%** |
| **Capacity-Normalized Daylight MAE** | **3.66%** | 3.81% | 4.87% | **-1.21 pp** | **-0.15 pp** |
| **All-Hour MAE** | **16.67 kWh** | 17.27 kWh | 22.01 kWh | **+24.25%** | **+3.45%** |
| **All-Hour RMSE** | **40.87 kWh** | 42.26 kWh | 51.83 kWh | **+21.14%** | **+3.28%** |

- **All Daylight Hours ($N = 4,601$, including 214 missing-weather hours)**: Daylight MAE **33.48 kWh** (3.75% capacity), outperforming baseline champion by **+24.33%** and Set A by **+3.46%**.
- **Generalization Gap**: $\Delta\text{MAE} = +1.26\text{ kWh}$ ($+4.01\%$) between validation CV (31.43 kWh) and final test (32.69 kWh), demonstrating minimal overfitting.

### Acceptance Gate Status:
- $\ge 5.0\%$ Daylight MAE Improvement: **PASSED** (24.83% on final test, 25.33% on validation CV vs champion baseline)
- Capacity-Normalized Daylight MAE $\le 20.0\%$: **PASSED** (3.66% on final test, 3.52% on validation CV)
- Non-negative, finite, bounded outputs: **PASSED** (0 violations)
- Final Test Safeguard: **PASSED** (`final_test_opened: true`, audited once on 2026-11-23 in `reports/final_test_audit.json`)
- Gate Decision: **ELIGIBLE** (recorded in `reports/evaluation.json` and `./evaluation.json`).

---

## 6. Seasonal & Diurnal Error Characteristics

- **Diurnal Profile**:
  - Night hours: Zero error (identically 0 kWh).
  - Morning / Evening low sun: Low MAE (< 15 kWh).
  - Midday solar noon: Peak MAE (~50–65 kWh) due to rapid cloud passing and convective buildup.
- **Seasonal Profile**:
  - Summer (Jun–Aug): High stability, Daylight MAE ~11–20 kWh (1.2%–2.2% capacity).
  - Winter (Dec–Feb): Daylight MAE ~38–57 kWh (4.3%–6.4% capacity) driven by frontal storm events. Weather features deliver strong winter gains (e.g. +11.4% in Dec 2021).

---

## 7. Limitations & Caveats

1. **Mixed Cloud Cover Degradation (-9.70%)**:
   - In broken, mixed-cloud conditions (20% to 80% forecast cloud cover, 12.4% of daylight hours), Feature Set B is **9.70% worse** than Feature Set A (54.26 kWh vs 49.46 kWh Daylight MAE).
   - This degradation stems from NWP spatial/temporal phase timing errors: 3-km grid numerical weather forecasts frequently miss the exact timing of cumulus cloud transit over a compact PV site. Set A's recent 15-minute telemetry directly senses the real surface shadow, making telemetry more reliable than NWP in broken cloud fields.
2. **Operational Latency & Missing Weather**:
   - The model relies on NOAA HRRR forecasts available with a 2-hour operational latency ($\text{init} \le H - 2\text{h}$).
   - Missing runs occur in 4.09% of daylight hours in the public archive. When weather is unobtainable, the model falls back to native NaN routing with a minor 0.1 kWh error increase.
3. **Single Site Calibration**:
   - Trained exclusively on PVDAQ Site 2107 in Arbuckle, CA (Sacramento Valley). Direct transfer to coastal marine layer or high-elevation microclimates requires site-specific recalibration.
4. **Final Test Safeguard & Reproducibility**:
   - The 20% final test set was opened exactly once on November 23, 2026 under strict audit logging (`reports/final_test_audit.json`).
   - No models were selected or tuned on test results; all estimators were fit strictly on the Train/Dev partition.
