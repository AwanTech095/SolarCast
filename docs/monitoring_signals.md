# SolarCast Operational Monitoring Signals & Alerting Engine

This document provides the technical specification, statistical foundations, empirical validation evidence,
and operational runbook for the SolarCast monitoring engine (Mode F1 next-hour solar AC energy forecasting, PVDAQ Site 2107).

---

## 1. Architecture & Monitored Signals Overview

The SolarCast monitoring stack operates on a virtual-clock daily evaluation cadence, evaluating four distinct operational signals
and emitting standardized JSON events conforming to [configs/alert_event_schema.json](file:///c:/projects/SolarCast/configs/alert_event_schema.json) (v1.1.0).

```
+----------------------------------------------------------------------------------------------------+
|                                    SOLARCAST OPERATIONAL MONITORS                                  |
+------------------------------------+------------------------------------+--------------------------+
|  Monitor                           | Implementation                     | Alert Type               |
+------------------------------------+------------------------------------+--------------------------+
|  1. Input Feature Drift            | src/models/drift.py                | input_drift              |
|  2. Performance Degradation        | src/models/performance_monitor.py  | performance_degradation  |
|  3. Weather Freshness & Quality    | src/models/drift.py                | data_quality             |
|  4. Production Residual Deviations | src/models/deviation.py            | residual_deviation       |
+------------------------------------+------------------------------------+--------------------------+
```

All monitors output structured alerts to [reports/monitoring_events.jsonl](file:///c:/projects/SolarCast/reports/monitoring_events.jsonl)
and record metrics in [reports/monitoring_summary.json](file:///c:/projects/SolarCast/reports/monitoring_summary.json).

---

## 2. Monitor 1: Input Feature Drift (`src/models/drift.py`)

### 2.1 Monitored Features Selection & Rationale
To maintain high statistical power without excessive false discovery, exactly 8 key numeric features were selected from [configs/feature_schema.json](file:///c:/projects/SolarCast/configs/feature_schema.json) (Feature Set B):

1. `power_kw_lag_0m` (kW): Top permutation importance feature (+116.92 kWh); captures immediate pre-cutoff generation state.
2. `energy_kwh_prev_day_same_hour` (kWh): Diurnal persistence lag; tracks day-over-day solar production continuity.
3. `energy_kwh_rolling_mean_24h` (kWh): 24-hour moving energy baseline; captures multi-day weather front regimes.
4. `clearsky_ghi_target_wm2` (W/m²): Deterministic astronomical clear-sky irradiance; verifies solar geometry and temporal alignment.
5. `clearsky_index_latest_avail` (dimensionless): Historical atmospheric transparency ratio [0, 2]; monitors preceding sky clarity.
6. `tmp_c` (°C): NOAA HRRR NWP 2-meter ambient temperature; impacts silicon PV module efficiency (-0.4%/°C above 25°C).
7. `tcdc_pct` (%): NOAA HRRR NWP total cloud cover forecast; primary physical driver of atmospheric solar extinction.
8. `clearsky_ghi_cloud_adj_wm2` (W/m²): Physical Kasten-Czeplak cloud-adjusted irradiance combining geometry and forecast cloud cover.

### 2.2 Population Stability Index (PSI) Formulation
The Population Stability Index measures distribution divergence across $B = 10$ quantile bins established on the reference distribution:

$$\text{PSI} = \sum_{b=1}^{B} (A_b - E_b) \times \ln\left(\frac{A_b}{E_b}\right)$$

where $A_b$ is the proportion of observations in bin $b$ from the current window, and $E_b$ is the expected proportion from the reference distribution. An epsilon floor ($\epsilon = 10^{-4}$) is applied before re-normalizing to eliminate division by zero.

- **Hand-Computed Verification Example**:
  - Reference: 10 uniform bins with $E_b = 0.10$ each.
  - Current: 8 bins with $A_b = 0.05$, 2 bins with $A_b = 0.30$.
  - Bins 1..8: $8 \times (0.05 - 0.10) \times \ln(0.05 / 0.10) = 8 \times (-0.05) \times (-0.693147) = 0.277259$.
  - Bins 9..10: $2 \times (0.30 - 0.10) \times \ln(0.30 / 0.10) = 2 \times 0.20 \times 1.098612 = 0.439445$.
  - Total $\text{PSI} = 0.277259 + 0.439445 = \mathbf{0.716704}$. Verified in `tests/test_monitoring.py::test_psi_against_hand_computed_example`.

### 2.3 Season-Matched Reference Slicing
Solar arrays experience severe deterministic seasonality: winter solar elevation peaks at $27^\circ$ while summer peaks at $74^\circ$. Using a static all-year reference causes false drift alarms every winter and summer.
- **Matching Rule**: For an evaluation day in calendar month $m$, the reference distribution is subsetted from historical Train/Dev daylight observations (`event_timestamp < 2022-10-17`) matching calendar months $\{m-1, m, m+1\}$ (wrapping December/January) and the exact daylight hours present in the current window.
- **Reference Sample Size**: Yields 3,600 to 5,900 valid seasonal reference hours per comparison.

### 2.4 Drift Decision & Alerting Rule
- **Evaluation Cadence**: Daily at 00:00 UTC over a rolling 14-day daylight window.
- **Sample Size Requirement**: $\ge 200$ valid rows per comparison. If valid rows $< 200$, decision is `insufficient_data` (no alert).
- **Drift Threshold**: $\text{PSI} > 0.20$ in at least 2 monitored features.
- **Persistence Requirement**: Condition must be breached on **2 consecutive daily checks**.
- **Missing Weather**: Excluded per-feature; missing weather counts are reported as data-quality indicators rather than drift.

---

## 3. Monitor 2: Delayed-Label Performance Monitor (`src/models/performance_monitor.py`)

### 3.1 Virtual Clock Label Visibility
In real-world operations, utility meter data is subject to network transmission latency.
- Forecast intervals $[H, H+1\text{h})$ close at $H + 1\text{h}$.
- Ground-truth production telemetry arrives at $H + 1\text{h} + 15\text{m}$ (from `configs/data.yaml` `production_available_delay_minutes: 15`).
- **Causality Enforcement**: At virtual clock time $T$, an actual observation is accessible if and only if:

$$\text{actual\_available\_at\_utc} = \text{interval\_end\_utc} + 15\text{m} \le T$$

Any observations with arrival times after $T$ are masked as `np.nan` and flagged as `UNAVAILABLE_PENDING`.

### 3.2 Reference Definition: Fixed vs. Month-Specific Empirical Evaluation
The team plan proposed alerting when 7-day Daylight MAE $> 1.25 \times \text{reference\_MAE}$. We empirically compared two reference definitions on the unmodified validation partition (Train/Dev, 1,551 days):

| Calendar Month | Daylight Hours | Validation Daylight MAE (kWh) | Fixed Ref Alerts (31.43 kWh) | Month-Specific Ref Alerts |
|:---:|:---:|:---:|:---:|:---:|
| **1 (Jan)** | 822 | **50.64** | **57** | **0** |
| **2 (Feb)** | 792 | **39.06** | **28** | **5** |
| **3 (Mar)** | 1013 | **44.01** | **32** | **0** |
| **4 (Apr)** | 1017 | 37.65 | 39 | 12 |
| **5 (May)** | 1286 | 29.91 | 23 | 23 |
| **6 (Jun)** | 1231 | 23.25 | 0 | 10 |
| **7 (Jul)** | 1290 | 16.67 | 0 | 20 |
| **8 (Aug)** | 1612 | 21.84 | 3 | 17 |
| **9 (Sep)** | 1428 | 28.51 | 3 | 5 |
| **10 (Oct)** | 1131 | 29.62 | 20 | 24 |
| **11 (Nov)** | 847 | **36.66** | **36** | **11** |
| **12 (Dec)** | 905 | **39.56** | **17** | **0** |
| **Full Train/Dev** | **13,374** | **31.43 (Weighted Mean)** | **258 days in alert** | **127 days in alert (-50.8%)** |

- **Empirical Finding**: A fixed reference of 31.43 kWh ($1.25 \times 31.43 = 39.29\text{ kWh}$) produces severe seasonal distortion: it triggers **170 false alert days** across winter/spring months (Nov–Mar) where normal baseline error is naturally 37–51 kWh due to low sun angles and frequent cloud cover, while remaining completely deaf in summer (0 alerts in Jun/Jul).
- **Decision**: Based strictly on validation evidence, Definition (b) **Month-Specific Reference MAE** was selected and frozen in [configs/monitoring.yaml](file:///c:/projects/SolarCast/configs/monitoring.yaml).

### 3.3 Degradation Decision & Alerting Rule
- **Evaluation Window**: Trailing 7 days $[T - 7\text{d}, T)$.
- **Minimum Sample Size**: $\ge 30$ valid daylight labels (`target_kwh.notna()` and `quality_flag == 'VALID'`).
- **Data Outage Handling**: If total daylight predictions $\ge 30$ but valid labels $< 30$, status is recorded as `label_data_outage` (or `insufficient_labels`). **No accuracy alert is emitted**, and missing labels are never treated as zero error.
- **Alert Trigger**: 7-day Daylight MAE $> 1.25 \times \max(\text{month\_ref\_mae}, 1.0\text{ kWh})$ on **2 consecutive daily checks**.

---

## 4. Monitor 3: Weather Freshness & Quality

### 4.1 Freshness Classification Policy
From [configs/data.yaml](file:///c:/projects/SolarCast/configs/data.yaml) operational policies:
- `weather_availability_lag_hours: 2`: Nominal HRRR publication delay.
- `max_fallback_hours: 8.0`: Maximum acceptable fallback to an older NWP run.

Classification criteria for issue hour $H$:
1. `missing`: `weather_missing == 1` or `pd.isna(weather_run_age_hours)`.
2. `fresh`: `weather_run_age_hours <= 2.0`.
3. `fallback`: `2.0 < weather_run_age_hours <= 8.0`.
4. `stale`: `weather_run_age_hours > 8.0`.
5. **Future Weather Forbidden**: If `weather_run_age_hours < 0`, a `ValueError` is raised immediately.

### 4.2 Data Quality Alerting Rule
- **Window**: Trailing 7 days $[T - 7\text{d}, T)$.
- **Sample Floor**: $\ge 30$ daylight hours.
- **Alert Condition**: Non-fresh weather share (`(fallback + stale + missing) / total_hours`) $> 15\%$. Emits `data_quality` event.

---

## 5. Standardized Alert Event Schema (`configs/alert_event_schema.json` v1.1.0)

Schema v1.1.0 supports all four operational alert types while maintaining backward compatibility with existing residual deviation pipelines:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "schema_version": "1.1.0",
  "event_id": "PERF-202305040000-001",
  "site_id": "2107",
  "alert_type": "performance_degradation",
  "model_name": "solarcast-f1",
  "model_version": "v2",
  "window_start_utc": "2023-04-27T00:00:00+00:00",
  "window_end_utc": "2023-05-04T00:00:00+00:00",
  "reason": "Model performance degradation: 7-day daylight MAE (44.82 kWh) exceeded 1.25x reference (29.91 kWh, threshold 37.39 kWh) on 2 consecutive daily checks (ratio: 1.50).",
  "created_at_utc": "2026-11-24T07:51:50.000000+00:00",
  "performance_details": {
    "metric_name": "daylight_mae_kwh",
    "mae_7d_kwh": 44.82,
    "reference_mae_kwh": 29.91,
    "reference_definition": "month_specific",
    "degradation_ratio": 1.4985,
    "consecutive_checks": 2,
    "valid_label_count": 92
  }
}
```

---

## 6. Historical Replay Results (`scripts/replay_monitoring.py`)

The virtual-clock replay was executed over the locked final test partition (375 daily evaluations, 2022-10-31 to 2023-11-09) across four controlled scenarios. Injected data was evaluated on isolated copies with fixed random seed (`seed = 42`).

### 6.1 Unmodified Test Baseline Results (Real Data)
On the unmodified final test period without fault injection, the monitors established the following baseline alert rates:

| Monitor Signal | Alert Type | Total Alerts Emitted | False Alerts per 100 Days | Operational Interpretation |
|:---|:---|:---:|:---:|:---|
| **Input Feature Drift** | `input_drift` | 33 | **8.80** | Seasonal transition boundaries (late autumn and spring). |
| **Performance Degradation** | `performance_degradation` | 102 | **27.20** | Concentrated in anomalous 2022/2023 winter atmospheric rivers. |
| **Weather Freshness** | `data_quality` | 51 | **13.60** | MesoWest NOAA HRRR upstream archive dropout episodes. |

### 6.2 Injected Fault Evaluation (Controlled Synthetic Experiments)
Injected fault experiments were conducted on isolated data copies to verify detector sensitivity and suppression guarantees:

| Scenario | Injected Anomaly | Evaluation Period | Detection Status | Detection Delay | Key Verification Result |
|:---|:---|:---:|:---:|:---:|:---|
| **(a) Weather Shift** | `tmp_c` +8°C, `tcdc_pct` $\times 0.6$ | 2023-06-15 to 2023-07-06 (21d) | **Detected** | **4.0 days** (96.0 h) | 2 features breached PSI > 0.20 (`tmp_c` 1.40, `energy_kwh_rolling_mean_24h` 0.60). |
| **(b) Performance Drop** | Forecasts multiplied by 0.7 | 2023-05-01 to 2023-05-11 (10d) | **Detected** | **3.0 days** (72.0 h) | 7-day MAE surged to 44.82 kWh vs 37.39 kWh threshold. |
| **(c) Label Outage** | Actuals removed (NaN / OUTAGE) | 2023-08-10 to 2023-08-15 (5d) | **Correctly Suppressed** | N/A | **0 accuracy alerts emitted**. Status correctly transitioned to `label_data_outage`. |

---

## 7. MLflow Tracking Lineage

All monitoring replay runs, parameters, metrics, and JSON summary artifacts are tracked in MLflow:
- **Experiment**: `solarcast`
- **Run Name**: `monitoring_replay`
- **Run ID**: `16ffd7429be642928941e95f932fe8b3`
- **Logged Artifacts**:
  - `reports/monitoring_summary.json`
  - `reports/monitoring_events.jsonl`
  - `configs/monitoring.yaml`
  - `configs/alert_event_schema.json`

To view the run in the MLflow UI:
```powershell
.\.venv\Scripts\mlflow ui --backend-store-uri file:./mlruns
```

---

## 8. Known Limitations & Operational Boundaries

1. **Single-Site Calibration**: Thresholds (PSI 0.20, degradation ratio 1.25, and monthly reference values) are calibrated strictly on PVDAQ Site 2107 (893 kW DC fixed-tilt, Arbuckle, CA). Applying these thresholds to other solar sites requires localized calibration against historical data.
2. **One Test Year Scope**: The final test partition spans exactly one calendar year (2022-10 to 2023-11). Inter-annual climate variability (such as extreme El Niño / atmospheric river seasons) can cause temporary elevations in prediction error without underlying model flaw.
3. **Thresholds are Proposals Frozen from Validation**: The thresholds documented here are operational proposals established on validation evidence. Member 2's automated retraining DAG may adjust cooldowns, label count minimums, or sensitivity multipliers based on pipeline capacity.
4. **Alerts are Flags, Not Equipment Diagnoses**: An alert signals a statistically significant empirical shift; it does not identify the physical root cause (e.g., inverter hardware trip, soiling, grid curtailment, or wildfire smoke). Root-cause attribution requires field inverter telemetry.
5. **200-Row Sample Floor Limitation**: At latitude $39^\circ\text{N}$, a 14-day daylight window contains $\ge 200$ hours only during peak summer (June/July). During winter, spring, and autumn, 14 days contains only 120–180 daylight hours, causing the drift monitor to report `insufficient_data` by design. Extending the window to 21 or 28 days would ensure winter coverage.
