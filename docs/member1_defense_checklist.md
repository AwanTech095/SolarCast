# Member 1 Oral Defense Preparation Checklist

**Project:** SolarCast (Mode F1: Next-Hour Solar AC Energy Forecasting & Deviation Detection)  
**Package Owner:** Member 1 (Data, Features, Feast, Models, MLflow)  
**Site:** PVDAQ Site 2107 (Farm Solar Array, Arbuckle, CA; 893 kW DC, 662.4 kW Nominal AC)  
**Champion Model:** `solarcast-f1` (v1.1.0, Feature Set B, `HistGradientBoostingRegressor`)

---

## 1. Core Demonstration Checklist

### 1.1 Recover a Versioned Dataset
- **Plain Explanation:** All raw and processed datasets are cryptographically tracked with immutable SHA-256 checksums in `configs/data_manifest.json` and versioned using DVC (owned by Member 2). Re-running the ingestion pipeline or checking hashes against the manifest confirms data integrity with zero data tampering or silent file drift. If raw data files are missing, the pipeline retrieves them directly from the DOE/NREL Open Energy Data Initiative (OEDI) S3 bucket (`s3://oedi-data-lake/pvdaq/2023-solar-data-prize/2107_OEDI/`).
- **Exact Command / File:**
  ```powershell
  .\.venv\Scripts\python -m src.data.ingestion --config configs/data.yaml
  ```
- **Expected Output:**
  ```
  INFO Starting SolarCast ingestion for Site 2107...
  INFO Verified data/raw/2107_meter_15m_data.csv (6319525 bytes, SHA-256: 7f6c7cf8...)
  INFO Saved data manifest to configs/data_manifest.json (9 artifacts recorded)
  ```

---

### 1.2 Explain kW versus kWh
- **Plain Explanation:** Power in kilowatts (kW) measures the instantaneous rate of energy transfer, whereas energy in kilowatt-hours (kWh) measures the accumulation of active power over time ($E = \int P(t)\,dt$). In the PVDAQ Site 2107 revenue telemetry (`data/raw/2107_meter_15m_data.csv`), readings are 15-minute average active power in kW. Because each 15-minute interval represents one-quarter of an hour ($\Delta t = 15/60 = 0.25\text{ h}$), energy over that quarter-hour is $E_{15} = P \times 0.25\text{ kWh}$. For an hourly interval $[H, H+1\text{h})$, the target AC energy is the sum of its four constituent quarter-hourly energies ($E_1 = \sum_{i=1}^4 P_i \times 0.25 = 0.25 \sum P_i$). If any of the four 15-minute intervals is missing, the target is set to `NaN` with quality flag `INCOMPLETE_INTERVALS` because missing telemetry must never be confused with zero production.
- **Exact Command / File:**
  File: `src/data/aggregation.py` (see `aggregate_to_hourly`)  
  Test:
  ```powershell
  .\.venv\Scripts\pytest tests/test_data.py -k test_kw_to_kwh_arithmetic -v
  ```
- **Expected Output:**
  ```
  tests/test_data.py::TestHourlyAggregation::test_kw_to_kwh_arithmetic PASSED [100%]
  1 passed, 22 deselected in 3.39s
  ```

---

### 1.3 Show Why Each Feature Was Available at Issue Time
- **Plain Explanation:** Mode F1 forecasts next-hour energy over interval $[H, H+1\text{h})$ UTC at issue time $H$ UTC. Operational plant revenue meter telemetry incurs a nominal 15-minute transmission delay, enforcing a strict telemetry cutoff of $H - 15\text{m}$ UTC: the latest usable 15-minute interval is $[H-30\text{m}, H-15\text{m})$ (`power_kw_lag_0m`) and the latest fully available hour is $[H-2\text{h}, H-1\text{h})$ (`energy_kwh_latest_avail_hour`). Numerical weather predictions from the NOAA HRRR MesoWest archive require 2 hours for assimilation, simulation, and dissemination, meaning only runs initialized at or before $H - 2\text{h}$ UTC are eligible (`weather_available_at_utc <= H`). In `data/processed/feast/features_set_b.parquet`, three explicit audit columns (`production_cutoff_utc`, `weather_available_at_utc`, and `source_event_time_max_utc`) record the operational source timestamps, and 100% of rows mathematically satisfy $\text{audit\_timestamp} \le \text{event\_timestamp}$.
- **Exact Command / File:**
  File: `src/features/preprocessing.py` and `src/features/build_feature_table.py`  
  Test:
  ```powershell
  .\.venv\Scripts\pytest tests/test_features.py -k "test_no_leakage_after_cutoff_via_perturbation or test_weather_perturbation_after_cutoff" -v
  ```
- **Expected Output:**
  ```
  tests/test_features.py::TestFeatureIntegrityAndLeakage::test_no_leakage_after_cutoff_via_perturbation PASSED [ 50%]
  tests/test_features.py::TestFeatureIntegrityAndLeakage::test_weather_perturbation_after_cutoff PASSED [100%]
  2 passed, 8 deselected in 2.60s
  ```

---

### 1.4 Retrieve Historical Feast Features
- **Plain Explanation:** Feast provides point-in-time correct historical feature retrieval via as-of joins against the offline Parquet store (`data/processed/feast/features_set_b.parquet`), evaluating a sliding lookback window $[t_{\text{entity}} - \text{ttl}, t_{\text{entity}}]$ (`ttl = 7 days`). This guarantees that backtesting and offline training evaluate only features that existed prior to or at the target timestamp, eliminating future lookahead bias. In our skew test (Demo a of `scripts/demo_feast.py`), 20 random issue times across 2019–2023 were retrieved through Feast offline store and compared against features computed by `build_features()`, verifying zero training-serving skew (maximum absolute discrepancy identically $0.00\text{e}+00$).
- **Exact Command / File:**
  File: `src/features/feast_provider.py`  
  Command (run in dedicated `.venv-feast`):
  ```powershell
  .\.venv-feast\Scripts\python -c "from src.features.feast_provider import get_historical_features; import pandas as pd; df = get_historical_features(pd.DataFrame({'site_id': ['2107'], 'event_timestamp': [pd.Timestamp('2023-06-15 15:00:00+00:00')]})); print('Retrieved:', df[['site_id', 'power_kw_lag_0m', 'tcdc_pct', 'tmp_c']].to_dict(orient='records'))"
  ```
- **Expected Output:**
  ```
  Retrieved: [{'site_id': '2107', 'power_kw_lag_0m': 105.6, 'tcdc_pct': 1.0, 'tmp_c': 22.85}]
  ```

---

### 1.5 Compare MLflow Runs
- **Plain Explanation:** All model experiments, hyperparameter tuning runs, cross-validation benchmarks, and final test evaluations are logged to MLflow experiments `solarcast-model-training` and `solarcast`. Comparing runs demonstrates why `HistGradientBoosting_Set_B_15mDelay` was selected as champion over `Ridge_Set_B_15mDelay` and `HistGradientBoosting_Set_A_15mDelay`: across 13,374 identical validation daylight hours, HGB Set B achieved a Daylight MAE of 31.43 kWh (+25.33% over baseline champion 42.09 kWh, and +2.52% over Set A 32.24 kWh), decisively passing the $\ge 5.0\%$ acceptance gate recorded in `reports/evaluation.json`.
- **Exact Command / File:**
  File: `reports/set_a_vs_b_summary.json` and `evaluation.json`  
  Command:
  ```powershell
  .\.venv\Scripts\python -c "import json; d=json.load(open('reports/set_a_vs_b_summary.json')); print('HGB Set B MAE:', round(d['primary_results']['HGB_Set_B']['mae'], 2), 'kWh vs Baseline:', round(d['primary_results']['Baseline_Champion']['mae'], 2), 'kWh')"
  ```
- **Expected Output:**
  ```
  HGB Set B MAE: 31.43 kWh vs Baseline: 42.09 kWh
  ```

---

### 1.6 Explain the Rejected Ridge Model
- **Plain Explanation:** Ridge Regression (L2-penalized linear regression) was benchmarked as an interpretable baseline candidate. Ridge was rejected because it failed the acceptance gate on validation (Daylight MAE 43.52 kWh on Feature Set A vs champion baseline 42.18 kWh, representing a $-3.16\%$ regression where a $\ge 5.0\%$ improvement was required; on Set B, Ridge reached 42.62 kWh vs champion baseline 42.09 kWh, also failing the gate). The physical explanation often offered—that linear models cannot capture non-linear solar elevation geometry, cloud attenuation thresholds, or inverter clipping ceilings—is a domain hypothesis, not experimentally tested or isolated. The empirical basis for rejection is strictly that Ridge failed the validation acceptance gate.
- **Exact Command / File:**
  File: `reports/evaluation.json` and `reports/set_a_vs_b_summary.json`  
  Command:
  ```powershell
  .\.venv\Scripts\python -c "import json; d=json.load(open('reports/set_a_vs_b_summary.json')); print('Ridge Set B MAE:', round(d['primary_results']['Ridge_Set_B']['mae'], 2), 'kWh; Baseline Champion MAE:', round(d['primary_results']['Baseline_Champion']['mae'], 2), 'kWh')"
  ```
- **Expected Output:**
  ```
  Ridge Set B MAE: 42.62 kWh; Baseline Champion MAE: 42.09 kWh
  ```

---

### 1.7 Explain a Flagged Deviation
- **Plain Explanation:** The production deviation engine monitors plant telemetry by comparing ground-truth production against advance F1 forecasts ($\text{residual} = \text{actual\_kwh} - \text{forecast\_kwh}$) after each hourly interval closes. To avoid false alarms, an alert event triggers only after **2 consecutive hourly intervals** breach calibrated 1st or 99th percentile bounds in the same direction within an operating cloud bin (`clear`, `mixed`, `overcast`, `no_weather`). For example, on 2023-04-01 during the baseline test period (`DEV-202304012000-001`), two consecutive daylight hours experienced severe underproduction during a missing-weather interval, with a mean residual of $-302.54\text{ kWh}$ against the lower bound of $-155.80\text{ kWh}$. The detector emitted an alert record conforming to `configs/alert_event_schema.json`. This signals an empirical output discrepancy, not an equipment diagnosis: without inverter field telemetry, the root cause could be soiling, inverter clipping, or localized cloud cover.
- **Exact Command / File:**
  File: `configs/deviation.yaml` and `reports/deviation_summary.json`  
  Command:
  ```powershell
  .\.venv\Scripts\python -c "import json; d=json.load(open('reports/deviation_summary.json')); print(d['injected_evaluation']['unmodified_test_baseline']['alert_events_summary'][0])"
  ```
- **Expected Output:**
  ```
  {'event_id': 'DEV-202304012000-001', 'start': '2023-04-01T20:00:00+00:00', 'end': '2023-04-01T21:00:00+00:00', 'direction': 'negative', 'bin': 'no_weather', 'hours': 2, 'mean_residual': -302.54}
  ```

---

## 2. Known Limitations & Operational Caveats

1. **Assumed Operational Latencies**: The pipeline assumes a fixed 2-hour HRRR forecast publication lag ($H - 2\text{h}$) and a 15-minute plant meter transmission cutoff ($H - 15\text{m}$). Real-world utility networks may experience variable ingest jitter or delayed telemetry uploads.
2. **Single-Site Parameterization**: Models, feature scaling, and deviation thresholds are calibrated strictly on PVDAQ Site 2107 (893 kW fixed ground mount, Arbuckle, CA). The model cannot be deployed to other geographical sites or single-axis tracker systems without retraining and local threshold recalibration.
3. **Weather Gaps Cause Unknown**: In the public MesoWest NOAA HRRR Zarr archive, 4.09% of daylight validation hours had missing forecast runs (e.g., 101 daylight hours in August 2023). These gaps are observed only as missing files/chunks in the upstream archive; the physical root cause upstream is unknown.
4. **Mixed-Cloud Degradation Hypothesis Not Tested**: In broken/mixed cloud regimes (20% to 80% forecast cloud cover), Feature Set B was 9.70% worse than Set A on validation (-3.21% on test). The hypothesis that NWP 3-km grid cells have spatial and temporal phase timing errors over localized arrays was not empirically verified with localized all-sky camera imagers.
5. **Detector Misses Small Forecasts**: Near sunrise and sunset ($5^\circ$ to $10^\circ$ solar elevation), expected generation is small (15–35 kWh). Even a complete 100% plant trip produces a residual of only $-15$ to $-35\text{ kWh}$, which cannot breach the conservative $-140.66\text{ kWh}$ clear-sky threshold.
