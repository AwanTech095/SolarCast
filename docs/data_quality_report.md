# SolarCast Data Quality & Aggregation Report (Task M1.2)

**Dataset:** System 2107 — Farm Solar Array (Arbuckle, CA)  
**Package:** Member 1 (Data, Features, Feast, Models, MLflow)  
**Date:** 2026-11-21  
**Source Artifacts:**
- Raw Revenue Meter: `data/raw/2107_meter_15m_data.csv` (240,062 rows, 6.32 MB)
- Quarantined Records: `data/processed/quarantine.parquet` (1 row)
- Aggregated Canonical Dataset: `data/processed/hourly.parquet` (60,096 rows)
- Data Manifests: `data/raw/manifest.json` and `configs/data_manifest.json`

---

## 1. Executive Summary

This report documents the validation, quarantine, timezone localization, and hourly energy aggregation executed under Task M1.2. The raw 15-minute average power series (kW) from the revenue-grade meter at PV site 2107 was audited with Pandera validation schemas, transformed from naive local civil time (`America/Los_Angeles`) to continuous standard UTC, and aggregated to hourly energy ($E = \sum P \times 0.25\,\text{kWh}$).

---

## 2. Step 1: Validation and Quarantine Audit

### 2.1 Quality Screening Rules Enforced
1. **Capacity Ceiling:** Readings exceeding DC nameplate capacity ($893.0\,\text{kW}$) are quarantined with reason `above_893_kw`. No silent clipping is performed.
2. **Physical Non-Negativity:** Negative generation values ($< 0.0\,\text{kW}$) are quarantined with reason `negative_power`.
3. **Finiteness:** Null (`NaN`), unparseable, or infinite (`inf`) power readings are quarantined with reason `non_finite`.
4. **Uniqueness:** Repeated timestamps are quarantined with reason `duplicate_timestamp`.

### 2.2 Quarantine Findings
- **Total Raw 15-Minute Records:** 240,062
- **Valid Records Retained:** 240,061
- **Quarantined Records:** **1 row** (isolated in `data/processed/quarantine.parquet`)

| Timestamp (Local) | Measured Power (kW) | Nameplate DC Limit (kW) | Quarantine Reason |
|---|---|---|---|
| `2018-02-18 12:30:00` | 1,783.68 | 893.00 | `above_893_kw` |

- **Duplicate Timestamps:** 0
- **Negative Values:** 0
- **Null / NaN Values in Raw Data:** 0

---

## 3. Step 2: Timezone Localization and DST Handling

### 3.1 Interval Alignment & Civil Time
- Raw meter timestamps in `measured_on` are naive local civil time (`America/Los_Angeles`) with Daylight Saving Time (DST).
- The meter is left-aligned (`time_interval = 'L'`); timestamp $t$ represents the start of interval $[t, t + 15\,\text{min})$.

### 3.2 Spring-Forward Mechanics (March)
- Civil clocks advance by 1 hour (skipping 02:00:00 to 02:45:00 local time).
- In the raw data, consecutive readings jump from `01:45:00 PST` (09:45 UTC) to `03:00:00 PDT` (10:00 UTC).
- In UTC, the elapsed difference between 09:45 UTC and 10:00 UTC is exactly **15 minutes**. No intervals are lost in UTC.

### 3.3 Autumn Fall-Back Mechanics (November)
- Civil clocks fall back by 1 hour (repeating 01:00:00 to 01:59:59 local time).
- The plant data logger recorded the `01:00` to `01:45` local labels only **once**.
- **Resolution Policy:** Ambiguous fall-back labels are localized to the **first occurrence** (Daylight Time / PDT, UTC-7, corresponding to 08:00–08:45 UTC). The second occurrence (Standard Time / PST, UTC-8, corresponding to 09:00–09:45 UTC) is unrecorded and treated strictly as missing data without imputation.

---

## 4. Step 2: Hourly Aggregation Metrics

### 4.1 Aggregation Rule
- **Formula:** $\text{target\_kwh} = \sum_{i=1}^{4} P_i \times 0.25\,\text{h}$
- **Completeness Requirement:** Configured `min_intervals = 4`. All 4 fifteen-minute intervals in a UTC hour must be present and valid. If fewer than 4 intervals are available, `target_kwh` is assigned `NaN` alongside an explanatory `quality_flag`.

### 4.2 Overall Dataset Metrics
| Metric | Value | Notes |
|---|---|---|
| **Total Hourly Rows** | **60,096** | Continuous UTC grid: `2017-01-01 08:00:00 UTC` to `2023-11-10 07:00:00 UTC` |
| **Valid Target Hours** | **60,013** | 99.8619% complete (`quality_flag == 'VALID'`) |
| **NaN Target Hours** | **83** | **0.1381%** missingness |
| **Missing Data Hours** | **80** | `quality_flag == 'MISSING_DATA'` (0 intervals recorded) |
| **Incomplete Interval Hours** | **3** | `quality_flag == 'INCOMPLETE_INTERVALS'` (1–3 intervals recorded) |

### 4.3 Incomplete Interval Audit (3 Hours)
| UTC Hour Start | Recorded Intervals | Reason for Incompleteness |
|---|---|---|
| `2017-01-01 08:00:00+00:00` | 3 (08:15, 08:30, 08:45) | Raw meter file begins at `00:15` local time; `00:00` is absent |
| `2018-02-18 20:00:00+00:00` | 3 (20:00, 20:15, 20:45) | `20:30 UTC` (12:30 local) quarantined ($1,783.68\,\text{kW} > 893\,\text{kW}$) |
| `2023-11-10 07:00:00+00:00` | 3 (07:00, 07:15, 07:30) | Raw meter file ends at `23:30` local time; `23:45` is absent |

### 4.4 Hours Lost to Fall DST Transition (7 Hours)
Each autumn, exactly 1 hourly row (4 fifteen-minute intervals) at `09:00:00 UTC` is lost due to single-logging of the civil fall-back hour:

| Year | Fall DST Date | Missing UTC Hour | Quality Flag | Target kWh |
|---|---|---|---|---|
| 2017 | 2017-11-05 | `2017-11-05 09:00:00+00:00` | `MISSING_DATA` | `NaN` |
| 2018 | 2018-11-04 | `2018-11-04 09:00:00+00:00` | `MISSING_DATA` | `NaN` |
| 2019 | 2019-11-03 | `2019-11-03 09:00:00+00:00` | `MISSING_DATA` | `NaN` |
| 2020 | 2020-11-01 | `2020-11-01 09:00:00+00:00` | `MISSING_DATA` | `NaN` |
| 2021 | 2021-11-07 | `2021-11-07 09:00:00+00:00` | `MISSING_DATA` | `NaN` |
| 2022 | 2022-11-06 | `2022-11-06 09:00:00+00:00` | `MISSING_DATA` | `NaN` |
| 2023 | 2023-11-05 | `2023-11-05 09:00:00+00:00` | `MISSING_DATA` | `NaN` |

*Note on remaining missing hours:* The remaining 73 `MISSING_DATA` hours stem from two recorded multi-day plant maintenance outages (July 15–17, 2021: ~50 hours; May 11–12, 2023: ~23 hours). Total: $7 + 50 + 23 = 80$ hours.

### 4.5 Daytime Zeros by Year
Daytime is defined per course requirements as solar elevation $> 0^\circ$ calculated using `pvlib.solarposition.get_solarposition`:

| Year | Daytime Zeros (Evaluation at Hour Start) | Daytime Zeros (Evaluation at Hour Midpoint) | Primary Causes |
|---|---|---|---|
| **2017** | 364 | 318 | Morning sunrise threshold, winter cloud/rain |
| **2018** | 256 | 188 | Winter rain/fog, inverter startup |
| **2019** | 181 | 140 | Heavy cloud cover |
| **2020** | 202 | 156 | Smoke / wildfire dimming, overcast skies |
| **2021** | 200 | 157 | Scheduled plant maintenance outage (July 15–17) |
| **2022** | 412 | 383 | Extended winter storms, dense valley fog |
| **2023** | 129 | 97 | Maintenance outage (May 11–12), cloud events |
| **Total** | **1,744** | **1,439** | |

---

## 5. Canonical Output Schema Verification

The generated parquet file `data/processed/hourly.parquet` strictly adheres to `hourly_schema` (Pandera):

| Column Name | Data Type | Constraint / Checks | Description |
|---|---|---|---|
| `site_id` | `String` | Non-null, `"2107"` | Unique identifier for PV facility |
| `interval_start_utc` | `datetime64[ns, UTC]` | Non-null, 1h step | Start of hourly measurement window |
| `interval_end_utc` | `datetime64[ns, UTC]` | `start + 1h` | End of hourly measurement window |
| `target_kwh` | `Float64` | $[0.0, 893.0]$ or `NaN` | Integrated AC energy production |
| `capacity_kw` | `Float64` | `893.0` | DC nameplate capacity for normalization |
| `capacity_ac_observed_kw` | `Float64` | `708.48` | Operational AC inverter peak limit |
| `latitude` | `Float64` | `38.996306` | Plant geographic latitude |
| `longitude` | `Float64` | `-122.134111` | Plant geographic longitude |
| `solar_elevation_deg` | `Float64` | $[-90.0, 90.0]$ | Solar elevation angle via pvlib |
| `is_daylight` | `Bool` | `solar_elevation_deg > 0.0` | Binary daylight filter for primary MAE |
| `quality_flag` | `String` | `VALID`, `INCOMPLETE_INTERVALS`, `MISSING_DATA` | Data completeness classification |
