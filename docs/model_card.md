# Model Card: SolarCast Mode F1 (`solarcast-f1`)

## 1. Model Details
- **Model Name**: `solarcast-f1`
- **Version**: `1.0.0`
- **Model Type**: Gradient Boosted Decision Trees (`HistGradientBoostingRegressor` from `scikit-learn`)
- **Developer**: SolarCast MLOps Team — Member 1 (Data, Features, Models)
- **Release Date**: November 23, 2026
- **License**: Course Internal / Educational
- **Framework & Dependencies**: Python 3.11, `scikit-learn==1.5.2`, `joblib==1.4.2`, `pvlib==0.11.1`, `numpy==1.26.4`, `pandas==2.2.3`, `mlflow==2.17.2`.

---

## 2. Intended Use & Scope
- **Primary Task**: Next-hour solar AC active energy forecast in kilowatt-hours (kWh) over the interval $[H, H+1\text{h})$ UTC.
- **Plant / Site**: PVDAQ Site 2107 (Denver, Colorado, USA; Nameplate DC Capacity: 893 kW).
- **Forecast Issue Timing**: Issued at hour $H$ UTC with an operational telemetry cutoff of $H - 15\text{ minutes}$ UTC.
- **Out of Scope**:
  - Intra-hour (e.g., 5-minute or 15-minute) dispatch forecasting.
  - Multi-day or day-ahead forecasting.
  - Direct application to other PV sites without site-specific retraining and geographical recalibration.
  - Curtailment forecasting or inverter failure diagnosis (production deviation monitoring handles post-hoc discrepancy flagging).

---

## 3. Factors & Input Features

The model uses **Feature Set A** (23 features), defined in `configs/feature_schema.json`:
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

---

## 4. Training Data & Preprocessing

- **Data Source**: PVDAQ Site 2107 validated inverter telemetry (`data/processed/hourly.parquet` and `quarter_hourly.parquet`).
- **Development Partition (80%)**: 37,090 hours (2018-07-12 00:00 UTC to 2022-10-17 00:00 UTC).
- **Training Samples**: 18,603 daylight hours (`solar_elevation_target_deg > 0`, `quality_flag == "VALID"`, not `SUSPECT_ZERO`).
- **Locked Test Set (20%)**: 9,273 hours (2022-10-17 01:00 UTC to 2023-11-09 23:00 UTC) strictly unopened.
- **Handling of Missing Values**: Built-in native histogram binning support in `HistGradientBoostingRegressor` (NaNs treated as missing values, avoiding imputation distortion).
- **Physical Bounding**:
  - Target hours with solar elevation $\le 0^\circ$ are deterministically set to $0.0\text{ kWh}$.
  - Predictions during daylight are strictly clipped to $[0.0, 893.0\text{ kWh}]$.

---

## 5. Evaluation & Performance Metrics

Evaluation on the 3-fold chronological cross-validation partition (13,873 daylight validation hours):

| Metric | Candidate (`solarcast-f1`) | Baseline (`Clear-Sky Persistence`) | Improvement |
|---|---|---|---|
| **Daylight MAE** | **32.24 kWh** | 75.62 kWh | **+57.36%** |
| **Daylight RMSE** | **54.76 kWh** | 113.80 kWh | **+51.88%** |
| **Capacity-Normalized Daylight MAE** | **3.61%** | 8.47% | **-4.86 pp** |
| **All-Hour MAE** | **16.18 kWh** | 37.87 kWh | **+57.27%** |
| **All-Hour RMSE** | **38.73 kWh** | 80.53 kWh | **+51.90%** |

### Acceptance Gate Status:
- $\ge 5.0\%$ Daylight MAE Improvement: **PASSED** (57.36%)
- Capacity-Normalized Daylight MAE $\le 20.0\%$: **PASSED** (3.61%)
- Non-negative, finite, bounded outputs: **PASSED** (0 violations)
- Gate Decision: **ELIGIBLE** (recorded in `reports/evaluation.json` and `./evaluation.json`).

---

## 6. Seasonal & Diurnal Error Characteristics

- **Diurnal Profile**:
  - Night hours: Zero error (identically 0 kWh).
  - Morning / Evening low sun: Low MAE (< 15 kWh).
  - Midday solar noon: Highest absolute MAE (~50–65 kWh) due to rapid cloud passing and convective buildup.
- **Seasonal Profile**:
  - Summer (Jun–Aug): High stability, Daylight MAE ~12–20 kWh (1.3%–2.2% capacity).
  - Winter (Dec–Feb): Daylight MAE ~40–60 kWh (4.5%–6.7% capacity) driven by frontal cloud systems.

---

## 7. Limitations & Caveats

1. **Single Site Calibration**: Trained exclusively on PVDAQ Site 2107 in Denver, CO. Direct transfer to coastal, tropical, or high-humidity climates requires recalibration.
2. **Missing Weather Forecasts**: Feature Set A relies on production lags and clear-sky models without numerical weather predictions (NWP). Integrating HRRR forecast cloud cover (TCDC/LCDC/MCDC/HCDC) and 2m temperature will address cloud passage dynamics.
3. **Data Quality Dependency**: Degradation occurs if telemetry exceeds the 15-minute cutoff or if prolonged inverter outages occur (indicated by `power_kw_missing_count > 0` and missingness flags).
4. **Final Test Partition Reserve**: The reported metrics are out-of-fold cross-validation estimates. Final test set metrics remain sealed until final team sign-off.
