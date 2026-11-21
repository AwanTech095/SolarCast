# SolarCast Data Quality & Aggregation Report (Task M1.2)

**Dataset:** System 2107 — Farm Solar Array (Arbuckle, CA)  
**Package:** Member 1 (Data, Features, Feast, Models, MLflow)  
**Date:** 2026-11-21 (Updated: Midpoint Solar Elevation & Zero Generation Audit)  
**Source Artifacts:**
- Raw Revenue Meter: `data/raw/2107_meter_15m_data.csv` (240,062 rows, 6.32 MB)
- Quarantined Records: `data/processed/quarantine.parquet` (1 row)
- Aggregated Canonical Dataset: `data/processed/hourly.parquet` (60,096 rows)
- Data Manifests: `data/raw/manifest.json` and `configs/data_manifest.json`

---

## 1. Executive Summary

This report documents data validation, anomaly quarantine, timezone localization, hourly energy aggregation, and a systematic zero-generation audit for PV site 2107. All timestamps are standard UTC. The target hourly energy ($E = \sum P \times 0.25\,\text{kWh}$) is evaluated alongside astronomical solar elevation computed at interval midpoint ($t + 30\,\text{min}$) via `pvlib.solarposition.get_solarposition`.

---

## 2. Validation and Quarantine Audit

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

## 3. Timezone Localization and DST Handling

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

## 4. Hourly Aggregation Metrics

### 4.1 Aggregation Rule
- **Formula:** $\text{target\_kwh} = \sum_{i=1}^{4} P_i \times 0.25\,\text{h}$
- **Completeness Requirement:** Configured `min_intervals = 4`. All 4 fifteen-minute intervals in a UTC hour must be present and valid. If fewer than 4 intervals are available, `target_kwh` is assigned `NaN` alongside an explanatory `quality_flag`.

### 4.2 Overall Dataset Metrics
| Metric | Value | Notes |
|---|---|---|
| **Total Hourly Rows** | **60,096** | Continuous UTC grid: `2017-01-01 08:00:00 UTC` to `2023-11-10 07:00:00 UTC` |
| **Valid Target Hours** | **59,565** | 99.1164% complete (`quality_flag == 'VALID'`) |
| **Suspect Zero Hours** | **448** | 0.7455% of hours (`quality_flag == 'SUSPECT_ZERO'`; target=0.0 unchanged) |
| **Missing Data Hours** | **80** | 0.1331% of hours (`quality_flag == 'MISSING_DATA'`; 0 intervals recorded) |
| **Incomplete Interval Hours** | **3** | 0.0050% of hours (`quality_flag == 'INCOMPLETE_INTERVALS'`; 1–3 intervals) |
| **Total NaN Target Hours** | **83** | **0.1381%** missingness ($80 + 3$) |

### 4.3 Incomplete Interval Audit (3 Hours)
| UTC Hour Start | Recorded Intervals | Observed Detail |
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

*Note on remaining missing hours:* The remaining 73 `MISSING_DATA` hours stem from two multi-day gaps: July 15–17, 2021 (50 hours) and May 11–12, 2023 (23 hours). Cause unknown; observed only as missing data in raw meter file (no log or metadata explanation is provided in the source dataset). Total: $7 + 50 + 23 = 80$ hours.

---

## 5. Daytime Zero Generation Audit

Daylight is defined by solar elevation at the midpoint of each hourly interval ($t + 30\,\text{min}$) $> 0^\circ$ (`is_daylight`). Total daylight zero rows (`target_kwh == 0.0` and `is_daylight == True`): **1,439 hours**.

### 5.1 Daylight Zeros by Solar Elevation Bin and Year
| Year | 0–2° | 2–5° | 5–10° | 10–20° | >20° | Total Daytime Zeros |
|---|---|---|---|---|---|---|
| **2017** | 36 | 36 | 96 | 68 | 82 | **318** |
| **2018** | 31 | 24 | 77 | 36 | 20 | **188** |
| **2019** | 27 | 20 | 39 | 33 | 21 | **140** |
| **2020** | 30 | 32 | 51 | 28 | 15 | **156** |
| **2021** | 32 | 31 | 62 | 30 | 2 | **157** |
| **2022** | 41 | 53 | 89 | 71 | 129 | **383** |
| **2023** | 17 | 12 | 30 | 20 | 18 | **97** |
| **Total** | **214** | **208** | **444** | **286** | **287** | **1,439** |

*Note:* For all daylight zero occurrences, cause unknown; observed only as zero production in the revenue meter records.

### 5.2 Runs of $\ge 2$ Consecutive Hours with Elevation $> 10^\circ$ and Target $== 0$
Audit identified **55 runs** where solar elevation at interval midpoint exceeded $10^\circ$ and the plant produced exactly $0.0\,\text{kWh}$ for 2 or more consecutive hours:

| Run # | Start Interval (UTC) | End Interval (UTC) | Length (Hours) |
|---|---|---|---|
| 1 | `2017-01-01 17:00:00+00:00` | `2017-01-01 23:00:00+00:00` | 7 |
| 2 | `2017-01-02 17:00:00+00:00` | `2017-01-02 23:00:00+00:00` | 7 |
| 3 | `2017-01-03 17:00:00+00:00` | `2017-01-03 23:00:00+00:00` | 7 |
| 4 | `2017-01-04 17:00:00+00:00` | `2017-01-04 23:00:00+00:00` | 7 |
| 5 | `2017-01-05 17:00:00+00:00` | `2017-01-05 23:00:00+00:00` | 7 |
| 6 | `2017-01-06 17:00:00+00:00` | `2017-01-06 23:00:00+00:00` | 7 |
| 7 | `2017-01-07 17:00:00+00:00` | `2017-01-07 23:00:00+00:00` | 7 |
| 8 | `2017-01-08 17:00:00+00:00` | `2017-01-08 23:00:00+00:00` | 7 |
| 9 | `2017-01-09 17:00:00+00:00` | `2017-01-09 23:00:00+00:00` | 7 |
| 10 | `2017-01-10 17:00:00+00:00` | `2017-01-10 23:00:00+00:00` | 7 |
| 11 | `2017-01-11 17:00:00+00:00` | `2017-01-11 23:00:00+00:00` | 7 |
| 12 | `2017-01-12 17:00:00+00:00` | `2017-01-12 23:00:00+00:00` | 7 |
| 13 | `2017-01-13 17:00:00+00:00` | `2017-01-13 23:00:00+00:00` | 7 |
| 14 | `2017-01-14 17:00:00+00:00` | `2017-01-14 23:00:00+00:00` | 7 |
| 15 | `2017-01-15 17:00:00+00:00` | `2017-01-15 23:00:00+00:00` | 7 |
| 16 | `2017-01-16 17:00:00+00:00` | `2017-01-16 21:00:00+00:00` | 5 |
| 17 | `2017-11-08 16:00:00+00:00` | `2017-11-08 18:00:00+00:00` | 3 |
| 18 | `2018-04-07 23:00:00+00:00` | `2018-04-08 01:00:00+00:00` | 3 |
| 19 | `2018-04-08 15:00:00+00:00` | `2018-04-08 19:00:00+00:00` | 5 |
| 20 | `2018-04-08 21:00:00+00:00` | `2018-04-08 22:00:00+00:00` | 2 |
| 21 | `2018-04-09 00:00:00+00:00` | `2018-04-09 01:00:00+00:00` | 2 |
| 22 | `2018-04-22 15:00:00+00:00` | `2018-04-22 19:00:00+00:00` | 5 |
| 23 | `2018-12-19 18:00:00+00:00` | `2018-12-19 19:00:00+00:00` | 2 |
| 24 | `2019-02-22 16:00:00+00:00` | `2019-02-23 00:00:00+00:00` | 9 |
| 25 | `2019-02-23 16:00:00+00:00` | `2019-02-24 00:00:00+00:00` | 9 |
| 26 | `2019-10-09 15:00:00+00:00` | `2019-10-09 21:00:00+00:00` | 7 |
| 27 | `2020-01-09 17:00:00+00:00` | `2020-01-09 23:00:00+00:00` | 7 |
| 28 | `2020-01-18 17:00:00+00:00` | `2020-01-18 23:00:00+00:00` | 7 |
| 29 | `2020-04-21 00:00:00+00:00` | `2020-04-21 01:00:00+00:00` | 2 |
| 30 | `2020-09-09 15:00:00+00:00` | `2020-09-09 18:00:00+00:00` | 4 |
| 31 | `2020-11-13 16:00:00+00:00` | `2020-11-13 17:00:00+00:00` | 2 |
| 32 | `2021-07-27 14:00:00+00:00` | `2021-07-27 15:00:00+00:00` | 2 |
| 33 | `2022-02-18 22:00:00+00:00` | `2022-02-19 00:00:00+00:00` | 3 |
| 34 | `2022-02-19 16:00:00+00:00` | `2022-02-20 00:00:00+00:00` | 9 |
| 35 | `2022-02-20 16:00:00+00:00` | `2022-02-21 00:00:00+00:00` | 9 |
| 36 | `2022-02-21 16:00:00+00:00` | `2022-02-22 00:00:00+00:00` | 9 |
| 37 | `2022-02-22 16:00:00+00:00` | `2022-02-23 00:00:00+00:00` | 9 |
| 38 | `2022-02-23 16:00:00+00:00` | `2022-02-24 00:00:00+00:00` | 9 |
| 39 | `2022-02-24 16:00:00+00:00` | `2022-02-25 00:00:00+00:00` | 9 |
| 40 | `2022-02-25 16:00:00+00:00` | `2022-02-26 00:00:00+00:00` | 9 |
| 41 | `2022-02-26 16:00:00+00:00` | `2022-02-27 00:00:00+00:00` | 9 |
| 42 | `2022-02-27 16:00:00+00:00` | `2022-02-28 00:00:00+00:00` | 9 |
| 43 | `2022-02-28 16:00:00+00:00` | `2022-03-01 00:00:00+00:00` | 9 |
| 44 | `2022-03-01 16:00:00+00:00` | `2022-03-02 00:00:00+00:00` | 9 |
| 45 | `2022-03-02 16:00:00+00:00` | `2022-03-03 00:00:00+00:00` | 9 |
| 46 | `2022-03-03 16:00:00+00:00` | `2022-03-04 00:00:00+00:00` | 9 |
| 47 | `2022-03-04 16:00:00+00:00` | `2022-03-05 00:00:00+00:00` | 9 |
| 48 | `2022-03-05 16:00:00+00:00` | `2022-03-06 00:00:00+00:00` | 9 |
| 49 | `2022-03-06 16:00:00+00:00` | `2022-03-07 00:00:00+00:00` | 9 |
| 50 | `2022-03-07 15:00:00+00:00` | `2022-03-07 18:00:00+00:00` | 4 |
| 51 | `2022-11-01 16:00:00+00:00` | `2022-11-01 17:00:00+00:00` | 2 |
| 52 | `2022-12-03 16:00:00+00:00` | `2022-12-03 23:00:00+00:00` | 8 |
| 53 | `2023-03-18 15:00:00+00:00` | `2023-03-19 00:00:00+00:00` | 10 |
| 54 | `2023-03-19 15:00:00+00:00` | `2023-03-20 00:00:00+00:00` | 10 |
| 55 | `2023-04-07 15:00:00+00:00` | `2023-04-07 16:00:00+00:00` | 2 |

*Note:* In each occurrence, cause unknown; observed only as zero production in the revenue meter.

### 5.3 Frozen Value Audit (Runs of $\ge 3$ Identical Non-Zero Values)
Screening for possible stuck sensor or frozen telemetry values ($\ge 3$ consecutive hours of identical positive readings) identified exactly **1 run**:
- **Start:** `2017-01-19 03:00:00+00:00`
- **End:** `2017-01-19 05:00:00+00:00`
- **Length:** 3 hours
- **Target Value:** `0.1216 kWh` (nighttime idle reading)
- **Daytime Frozen Values:** Zero extended identical non-zero readings were observed during daylight hours.

### 5.4 The `SUSPECT_ZERO` Quality Flag
- **Definition:** Assigned to hours where `high_sun` is `True` ($\text{elevation} > 5^\circ$) and $\text{target\_kwh} == 0.0$ sitting inside a run of 2 or more consecutive such hours.
- **Handling:** Target value remains untouched ($0.0\,\text{kWh}$), but the flag is preserved in `quality_flag` for filtering or ablation in model training.
- **Total Flagged:** **448 hours** (2017: 140, 2018: 28, 2019: 29, 2020: 27, 2021: 6, 2022: 194, 2023: 24).

---

## 6. Canonical Output Schema Verification

The generated parquet file `data/processed/hourly.parquet` adheres strictly to `hourly_schema` (Pandera):

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
| `solar_elevation_deg` | `Float64` | $[-90.0, 90.0]$ | Astronomical solar elevation at interval midpoint |
| `is_daylight` | `Bool` | `solar_elevation_deg > 0.0` | Binary daylight filter for primary MAE |
| `high_sun` | `Bool` | `solar_elevation_deg > 5.0` | High-sun filter for anomaly screening |
| `quality_flag` | `String` | `VALID`, `INCOMPLETE_INTERVALS`, `MISSING_DATA`, `SUSPECT_ZERO` | Data completeness and quality classification |
