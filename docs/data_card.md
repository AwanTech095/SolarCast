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

---

## 3. Candidate 2: System 2105 — Maui Ocean Center (Maui, HI)

### 3.1 System Specifications & Metadata
- **System ID:** `2105` (Public Name: `Maui Ocean Center`, Site ID: `7209`, Site Public Name: `MUFFC_1`)
- **Location:** Maui, Hawaii, United States (`Pacific` region)
- **Coordinates:** Latitude `20.884134° N`, Longitude `-156.340543° W`, Elevation `10.0 m`
- **Climate Type:** Köppen `Af` (Tropical rainforest / maritime)
- **Nameplate Capacity:** `110.0 kW DC` (commercial rooftop)
- **Mount & Tracking:** Commercial rooftop (`moc_mount`), Tracking: `False`, Azimuth: `varies`, Tilt: `varies` (panels face multiple orientations across building roof faces).
- **Meters & Inverters:**
  - Meter: Metadata registers metric `meter_ac_output_(kwatts)_meter_150161` (`kW`), but no dedicated site-level meter CSV exists in the data directory.
  - Inverters: Split across 11 individual inverter CSV files (`2105_inv01_data.csv` to `2105_inv11_data.csv`), each tracking strings `inv_string01` .. `inv_string11`.

### 3.2 Channels & Measurement Units
- **Primary AC Channels:** `inv_stringXX_ac_output_(kwh)_inv_YYYYYY` across 11 inverter CSV files.
  - Common Name: `AC energy`
  - Engineering Units: `kWh` (cumulative active energy meter reading)
  - Scale Factor: `None` (1.0), Offset: `None` (0.0)
  - Energy Conversion: Requires cumulative meter differencing ($\Delta E = E_t - E_{t-1}$) with meter reset and counter rollover detection, evaluated across 11 unsynchronized series and summed to site level.
- **Secondary Channels:** AC voltage (`V`), DC voltage (`V`), inverter temperature (`°C`), power factor, and two environmental sensors (`2105_environment_1_data.csv` and `2105_environment_2_data.csv` recording ambient temperature in `°C`).

### 3.3 Temporal Structure & Timestamp Behavior
- **Sampling Resolution:** Stated as `5-15 min`, but empirical records show uneven event-driven reporting.
- **Interval Alignment:** Unspecified in metadata (`Meters: {}`, `Inverters: {}`, metric `time_interval = None`).
- **Timezone & Daylight Saving Time (DST):**
  - Metadata specifies `timezone_code: 10` (Hawaii Standard Time, HST, UTC-10).
  - Hawaii **does not observe Daylight Saving Time**.
  - Timestamps in `measured_on` are constant standard local civil time (UTC-10) without seasonal clock shifts. Offset to UTC is a fixed $+10$ hours year-round.

### 3.4 Data Coverage, Quality & Gaps
- **Date Range:** `2018-12-15 00:15:00` to `2023-11-15 12:21:50` (4.92 years).
- **Completeness & Gaps:**
  - Moderate to high missingness. Nighttime non-producing records are frequently dropped rather than reported as zeros, leading to irregular multi-hour steps between consecutive timestamps.
  - Inverter file timestamps are not strictly locked to a uniform 15-minute grid.
- **Physical Validity & Geometry:**
  - Complex multi-plane geometry (varying azimuth and tilt angles) causes inter-row and building shading that violates simple single-diode or fixed clear-sky assumptions.

### 3.5 Storage Footprint & Access
- **S3 Location:** `s3://oedi-data-lake/pvdaq/2023-solar-data-prize/2105_OEDI/` (anonymous S3 access).
- **Files & Sizes:** 16 files totaling **207.75 MB** on S3.
- **Licence & Terms:** DOE Solar Data Prize 2023 / NREL OEDI open data (DOI `10.25984/1846021`).
