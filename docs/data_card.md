# SolarCast Data Card: PV Generation & Weather Feasibility (Task M1.1)

**Package:** Member 1 (Data, Features, Feast, Models, MLflow)  
**Date:** 2026-11-16  
**Status:** In Progress (Feasibility audit of candidate PV systems on OEDI / PVDAQ & NASA POWER)  
**Branch:** `m1/data-feasibility`  

---

## 1. Executive Summary & Purpose

Task M1.1 evaluates candidate photovoltaic (PV) generation datasets from the National Renewable Energy Laboratory (NREL) Open Energy Data Initiative (OEDI) data lake (`s3://oedi-data-lake/pvdaq/`) under the DOE 2023 Solar Data Prize, alongside collocated meteorological data from the NASA POWER Hourly Point API (`https://power.larc.nasa.gov/api/temporal/hourly/point`).

The primary objective is to select **one** PV site satisfying all MLOps course requirements:
- Defensible AC power / energy units without ambiguous scaling.
- Explicit interval alignment (left/center/right) and known timestamp conventions (UTC vs. local civil time with DST).
- Overlapping multi-year history (>= 2 years) covering complete seasonal cycles.
- Low missingness / gap percentage.
- Transparent licensing allowing educational reuse with citation.

---

## 2. Candidate 1: System 2107 — Farm Solar Array (Arbuckle, CA)

### 2.1 System Specifications & Metadata
- **System ID:** `2107` (Public Name: `Farm Solar Array`, Site ID: `7211`, Site Public Name: `FSA_1`)
- **Location:** Arbuckle, Colusa County, California, United States (`West` region)
- **Coordinates:** Latitude `38.996306° N`, Longitude `-122.134111° W`, Elevation `10.0 m`
- **Climate Type:** Köppen `Csa` (Hot-summer Mediterranean climate)
- **Nameplate Capacity:** `893.0 kW DC` (commercial ground mount)
- **Mount & Tracking:** Fixed ground-mount (`mnt_fss_1`, manufacturer Schletter), Azimuth `180.0°` (true south), Tilt `25.0°`, Tracking: `False` (fixed tilt)
- **Meters & Inverters:**
  - Meter: 1 revenue-grade meter (`meter_fss_1`, ID `3133`, `time_interval = 'L'`)
  - Inverters: 24 string inverters (ABB `TRIO-27.6-TL-OUTD-S1B`, IDs `29943`..`29966`, each 27.6 kW AC nominal, `time_interval = 'L'`)

### 2.2 Channels & Measurement Units
- **Primary AC Channel:** `meter_revenue_grade_ac_output_meter_149578` in dedicated file `2107_meter_15m_data.csv`.
  - Common Name: `AC power`
  - Engineering Units: `kW` (instantaneous power)
  - Scale Factor: `None` (1.0), Offset: `None` (0.0)
  - Energy Conversion: Hourly energy $E_{\text{kWh}} = \sum_{i=1}^4 P_{i, \text{kW}} \times 0.25\,\text{h}$
- **Secondary Channels:** 24 inverter AC power channels `inv_01_ac_power_inv_149583` .. `inv_24_ac_power_inv_149698` (`kW`), DC current/voltage per inverter, plane-of-array irradiance (`poa_irradiance_o_149574` in `W/m²`), ambient temperature (`ambient_temperature_o_149575` in `°F`), and wind speed (`wind_speed_o_149576` in `m/s`).

### 2.3 Temporal Structure & Timestamp Behavior
- **Sampling Resolution:** 15 minutes (`0.25 h`) for revenue-grade meter; 5 minutes for electrical inverter CSV.
- **Interval Alignment:** `time_interval = 'L'` (Left-aligned). The timestamp represents the start of the 15-minute measurement interval $[t, t + 15\,\text{min})$.
- **Timezone & Daylight Saving Time (DST):**
  - Metadata specifies `timezone_code: PST8PDT` (Pacific Standard Time / Pacific Daylight Time).
  - Empirical verification across spring-forward transitions (e.g. 2019-03-10, 2020-03-08, 2021-03-14) reveals exactly 92 intervals (skipping `02:00:00` to `02:45:00`, creating a 75-minute timestamp step from `01:45:00` to `03:00:00`).
  - Fall-back transitions (e.g. 2019-11-03, 2020-11-01) record exactly 96 intervals.
  - **Critical Finding:** Timestamps in `measured_on` are in **local civil time with DST**, NOT UTC. Ingestion pipeline must localize to `America/Los_Angeles` and convert to UTC (`interval_start_utc`).

### 2.4 Data Coverage, Quality & Gaps
- **Date Range:** `2017-01-01 00:15:00` to `2023-11-09 23:30:00` (nearly 7 full years in primary prize dataset; extended to 2024–2025 in supplemental files).
- **Interval Completeness:**
  - Total records in `2107_meter_15m_data.csv`: 240,062 rows.
  - Expected 15-minute intervals over span: 240,381 intervals.
  - Missing intervals: 319 intervals (**0.13% gap rate**, >99.8% complete).
  - Outages > 1 hour: Only 10 occurrences in ~7 years (two multi-day maintenance outages: 2021-07-15 to 2021-07-18 and 2023-05-11 to 2023-05-13; remaining 8 are the 75-minute spring DST clock adjustments).
- **Physical Validity:** Min AC power: `0.0 kW`, Max: `1,783.68 kW`, Mean: `153.09 kW`, Std: `222.52 kW`. Zero null values. Zero negative generation values.

### 2.5 Storage Footprint & Access
- **S3 Location:** `s3://oedi-data-lake/pvdaq/2023-solar-data-prize/2107_OEDI/` (anonymous S3 access, no credentials required).
- **Files & Sizes:**
  - `2107_meter_15m_data.csv`: **6.32 MB** (clean site-level revenue meter, 2017–2023).
  - `2107_electrical_data.csv`: 438.57 MB (inverter-level 5-minute data).
  - Total 2107 directory on S3: 14 files, ~1.09 GB.
- **Licence & Terms:** US Department of Energy (DOE) Solar Data Prize 2023 / NREL OEDI open data. Publicly available with required citation: DOI `10.25984/1846021`.
