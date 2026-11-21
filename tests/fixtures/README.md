# Test Fixtures: Hourly Generation Sample

This directory contains deterministic sample data fixtures for offline test suites, CI/CD validation, and integration tests across project packages (Members 1, 2, and 3).

## 1. File Details
- **File:** `hourly_sample.csv`
- **Rows:** 720 (30 consecutive days: `2022-07-01 00:00:00+00:00` to `2022-07-31 00:00:00+00:00`)
- **Size:** ~95.2 KB (< 200 KB)
- **Columns:**
  - `site_id`: PV site identifier (`"2107"`)
  - `interval_start_utc`: UTC interval start timestamp
  - `interval_end_utc`: UTC interval end timestamp (`start + 1h`)
  - `target_kwh`: Integrated hourly AC energy production
  - `capacity_kw`: DC nameplate capacity (`893.0` kW)
  - `capacity_ac_observed_kw`: Maximum observed AC inverter limit (`708.48` kW)
  - `latitude`: Geographic latitude (`38.996306`)
  - `longitude`: Geographic longitude (`-122.134111`)
  - `solar_elevation_deg`: Midpoint astronomical solar elevation via pvlib
  - `is_daylight`: Boolean indicator (`solar_elevation_deg > 0.0`)
  - `high_sun`: Boolean indicator (`solar_elevation_deg > 5.0`)
  - `quality_flag`: Completeness and data quality indicator (`VALID`, `SUSPECT_ZERO`, `MISSING_DATA`, `INCOMPLETE_INTERVALS`)

## 2. Dataset Attribution & Provenance
This file is a public-data excerpt from the **NLR/NREL PVDAQ 2023 Solar Data Prize Datasets** hosted on the Open Energy Data Initiative (OEDI) data lake.

- **Source System:** System 2107 (Farm Solar Array, Arbuckle, CA)
- **Attribution / DOI:** [10.25984/1846021](https://doi.org/10.25984/1846021)
- **Licence:** US Department of Energy (DOE) Open Data Policy / AWS Open Data Sponsorship Program. Educational and research reuse permitted with attribution.
