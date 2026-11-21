# NOAA HRRR Forecast Data Feasibility Report (Task M1.1b)

**Author:** Emad (Member 1: Data and ML Engineering)  
**Date:** 2026-11-22  
**Target PV Plant:** System 2107 (Farm Solar Array, Arbuckle, CA; $38.996306^\circ\text{N}, -122.134111^\circ\text{W}$, 10 m elevation)  
**Candidate Weather Sources Audited:**  
1. NOAA HRRR via MesoWest S3 Zarr Archive (`s3://hrrrzarr/`, region `us-west-1`, anonymous)  
2. NOAA HRRR via AWS Big Data Program GRIB2 Archive (`s3://noaa-hrrr-bdp-pds/`, anonymous)  
3. pvlib Clear-Sky Model (Deterministic Astronomical Irradiance Benchmark)  
4. NASA POWER Hourly API (Observed Weather Benchmark for Retrospective Mode E1)  

---

## Executive Summary & Gate Recommendation

| Criterion | Evaluation Result | Status |
|---|---|---|
| **Spatial Extraction** | 334 meters from plant coordinates to nearest HRRR grid point | **Verified** (PASS) |
| **Field Semantics** | Instantaneous point forecasts; trapezoidal averaging maps to hourly intervals | **Verified** (PASS) |
| **Operational Release Latency** | Dissemination ~50–80 min post-cycle; 2-hour lag ensures availability before issue time | **Assumed** (PASS) |
| **HRRR Zarr Coverage (2018–2023)** | 24/24 runs/day available across HRRR v3 and v4 | **Verified** (PASS) |
| **HRRR Zarr Coverage (2017–mid 2018)** | 0 forecast runs exist in Zarr (HRRR v2 archive has analysis files only) | **Verified** (FAIL) |
| **HRRR Zarr Forecast Radiation** | **Omitted across all years** due to upstream `iris-grib #265` night bug | **Verified** (FATAL BLOCKER) |
| **NOAA GRIB2 Forecast Radiation** | Radiation exists in GRIB2, but requires ~189 GB download and binary decoders | **Verified** (HIGH RISK) |
| **Physical Irradiance Correlation** | pvlib Clear-Sky GHI: $r = 0.945$; HRRR Analysis DSWRF: $r = 0.927$ | **Verified** (PASS) |
| **Gate Recommendation** | **CONDITIONAL** (Alternative Weather Strategy Required for Mode F1) | **DECISION** |

### Gate Decision: CONDITIONAL (Rejection of HRRR Zarr Irradiance)

- **Cloud-Optimized HRRR Zarr for Forecast Irradiance is a NO-GO:**  
  The community cloud-optimized Zarr mirror (`s3://hrrrzarr/`) completely lacks forecast solar radiation (`surface/DSWRF/surface/DSWRF`). Official MesoWest documentation confirms: *"Solar radiation is always affected by this bug at night and thus all forecasts for these variables are missing in the Zarr archive."* All forecast DSWRF chunk queries return fill values (`-10000.0` / NaN).
- **Direct GRIB2 Ingestion is an Impractical Route for Windows/Course Constraints:**  
  While official GRIB2 files at `s3://noaa-hrrr-bdp-pds/` contain forecast DSWRF, byte-range downloading over 2017–2023 requires ~189 GB of transfer and binary GRIB2 decoders (`eccodes` / `wgrib2`), which conflict with our pinned Windows Python 3.11 environment. Furthermore, HRRR v2 (2017 to mid-2018) exhibits differing file structures and schema changes.
- **Recommended Action for Member 1 & Team:**  
  1. **Adopt a Hybrid Forecast Feature Set:** Use HRRR forecast near-surface temperature (2m TMP) from Zarr (which is complete, verified, and rapidly extractable from July 2018 onward) combined with **pvlib clear-sky irradiance (GHI/DNI/DHI)** and historical production lag/rolling features. Clear-sky GHI demonstrates an exceptional $r = 0.945$ linear correlation with daytime plant production.
  2. **Retain Mode E1 Retrospective Benchmark with NASA POWER:** Keep NASA POWER observed solar insolation and temperature for retrospective performance benchmarking as planned.
  3. **Strict Evaluation of Weather Value:** Train both (a) production-and-time-only models and (b) weather-enhanced models (with forecast temperature and clear-sky index) on identical eligible timestamps to honestly test whether forecast weather meets the 5% improvement threshold.

---

## 1. Spatial Grid Mapping

The System 2107 photovoltaic array is located in Arbuckle, Colusa County, California ($38.996306^\circ\text{N}, -122.134111^\circ\text{W}$, elevation 10 m).

Using the native HRRR Lambert Conformal Conic coordinate reference system:
- Standard projection parameters (`hrrrzarr/grid/projparams.json`):
  `+proj=lcc +lat_0=38.5 +lon_0=262.5 +lat_1=38.5 +lat_2=38.5 +R=6371229 +units=m +no_defs`
- Nearest 2D grid index: **`index_x = 198`**, **`index_y = 642`** (grid shape: $1059 \times 1799$) [**Verified**].
- Coordinate of nearest grid center: **$38.999203^\circ\text{N}, -122.133110^\circ\text{W}$** [**Verified**].
- Distance from plant to grid center: **`0.334 km`** (334 meters, well within the 3 km HRRR grid cell) [**Verified**].
- MesoWest Zarr chunk tile: **`chunk_id = 1.4`** (`chunk_x = 1`, `chunk_y = 4`), with in-chunk spatial offsets `in_chunk_x = 48`, `in_chunk_y = 42` (chunk dimensions: $150 \times 150$) [**Verified**].

---

## 2. Field Semantics & Target Interval Mapping

The NOAA HRRR GRIB2 index file (`s3://noaa-hrrr-bdp-pds/hrrr.20220613/conus/hrrr.t00z.wrfsfcf01.grib2.idx`) was audited [**Verified**]:
```text
71:44441839:d=2022061300:TMP:2 m above ground:1 hour fcst:
123:88776819:d=2022061300:DSWRF:surface:1 hour fcst:
```

### Key Semantics Findings
1. **Instantaneous Values:**  
   Both near-surface temperature (`TMP:2 m above ground`) and downward shortwave radiation flux (`DSWRF:surface`) are labeled `"N hour fcst"`, representing **instantaneous point-in-time forecasts** at the valid forecast timestamp, NOT time-interval averages (which are designated `"N-M hour ave fcst"`).
2. **Interval Mapping to Hourly Production:**  
   SolarCast's target production interval is $[H, H+1)$ UTC (e.g., [14:00, 15:00) UTC). Because the target energy (kWh) represents integrated production over that hour, instantaneous forecast points at the boundary timestamps $H$ (lead $F_1$) and $H+1$ (lead $F_2$) must be mapped using **trapezoidal numerical integration**:
   $$\overline{\text{DSWRF}}_{[H, H+1)} = \frac{\text{DSWRF}_{F_1} + \text{DSWRF}_{F_2}}{2} \quad (\text{W/m}^2)$$
   $$\overline{T}_{[H, H+1)} = \frac{T_{F_1} + T_{F_2}}{2} - 273.15 \quad (^\circ\text{C})$$
   Assuming radiation was an hourly average would cause a phase shift and misestimate solar ramp rates during sunrise and sunset.

---

## 3. Operational Release Timing & Information Availability

### Mode F1 Information Boundary
For an issue hour $H$ predicting production for $[H, H+1)$:
- Prediction issue time: **$H$ UTC** (e.g., 14:00 UTC).
- Forecast availability rule: `weather_available_at_utc <= prediction_issue_time_utc`.
- Latest eligible run initialization: $I \le H - \text{lag}$.
- Configured default lag: **`weather_availability_lag_hours = 2`** ($I = H - 2\text{h}$) [**Verified** in configs].
- Associated leads: $F_1 = H - I = 2$ (`F02`) and $F_2 = (H+1) - I = 3$ (`F03`).

### Availability Evidence vs Assumption
- **Release Latency Assumption:** [**Assumed from documented operational latency**]  
  NCEP operational documentation and National Weather Service Service Change Notices (SCN 18-18 and SCN 20-91) indicate that numerical model execution and CONUS GRIB2 file dissemination complete between 50 and 80 minutes after nominal model run cycle time. Assuming a run initialized at $I$ is available at $I + 2\text{h}$ guarantees that all 18 hourly forecast files have been posted to S3 before prediction issue time $H$.
- **Sensitivity Analysis (Lag 2 vs Lag 3):** [**Verified**]  
  Running the identical extraction with `weather_availability_lag_hours = 3` ($I = H - 3\text{h}$, leads $F_1 = 3, F_2 = 4$) across the 104 daylight hours of the sample week resulted in **81 differing forecast temperature values (77.9%)**, with a mean absolute temperature update of $0.65^\circ\text{C}$. This proves that model updates actively ingest new assimilations, and that enforcing a strict 2-hour lag provides fresher operational intelligence without future leakage.
- **Production History Timing:** [**Assumed**]  
  Configured delay: `production_available_delay_minutes = 15`. For issue time $H$, the latest plant interval is $[H - 30\text{m}, H - 15\text{m})$, available exactly at $H$.
- **Actual Label Timing:** [**Assumed**]  
  For interval $[H, H+1)$, the interval closes at $H+1$ and actual revenue meter data arrives at $H+1 + 15\text{m}$.

---

## 4. Archive Coverage Probes (HRRR v2, v3, v4)

Nine sample dates across the 2017–2023 production period were probed on `s3://hrrrzarr/` to evaluate multi-year archive completeness [**Verified**]:

| Date | HRRR Version | Forecast Runs Found | Analysis Runs Found | Forecast Chunk Shape | Notes & Archive Status |
|---|---|---|---|---|---|
| **2017-01-15** | HRRR v2 | **0 / 24** | 22 / 24 | N/A | **Zero forecast runs** in Zarr archive; 2 missing analysis runs |
| **2017-07-15** | HRRR v2 | **0 / 24** | 24 / 24 | N/A | **Zero forecast runs** in Zarr archive; analysis runs present |
| **2018-01-15** | HRRR v2 | **0 / 24** | 24 / 24 | N/A | **Zero forecast runs** in Zarr archive; analysis runs present |
| **2018-07-15** | HRRR v3 | **24 / 24** | 24 / 24 | (36, 150, 150) | HRRR v3 deployed 2018-07-12; 36-h forecasts on 00/06/12/18z |
| **2019-06-15** | HRRR v3 | **24 / 24** | 24 / 24 | (36, 150, 150) | Complete 24 forecast runs |
| **2020-06-15** | HRRR v3 | **24 / 24** | 24 / 24 | (36, 150, 150) | Complete 24 forecast runs |
| **2021-06-15** | HRRR v4 | **24 / 24** | 24 / 24 | (48, 150, 150) | HRRR v4 deployed 2020-12-02; extended to 48-h forecasts |
| **2022-12-15** | HRRR v4 | **24 / 24** | 24 / 24 | (48, 150, 150) | Complete 24 forecast runs |
| **2023-06-15** | HRRR v4 | **24 / 24** | 24 / 24 | (48, 150, 150) | Complete 24 forecast runs |

### Findings on Version Differences
1. **Pre-July 2018 Deficit:** The community Zarr archive contains **no forecast runs prior to 2018-07-12** (HRRR v2). Retrospective forecasts for 2017 and early 2018 exist only in raw GRIB2 archives.
2. **Forecast Length Evolution:** In HRRR v3, major cycle runs (00z, 06z, 12z, 18z) extend to 36 forecast hours, while intermediate runs extend to 18 hours. In HRRR v4, major cycles extend to 48 forecast hours.

---

## 5. The Fatal Downward Shortwave Radiation Bug (`iris-grib #265`)

A deep audit of `hrrrzarr/sfc/YYYYMMDD/YYYYMMDD_HHz_fcst.zarr/surface/DSWRF/surface/DSWRF` revealed that:
1. The Zarr array metadata file `.zarray` exists, with `shape = (48, 1059, 1799)` and `fill_value = -10000.0`.
2. However, the store contains **exactly 0 data chunk files** (only `.zarray` and `.zattrs`).
3. Reading `DSWRF[:, 642, 198]` via Zarr returns `-10000.0` across all 48 forecast leads.
4. Official MesoWest HRRR Zarr documentation explicitly documents this root cause:
   > *"Missing data due to iris-grib #265: Sometimes you'll see a variable that has zarr array metadata generated for a given date/time, but is missing any actual data... Unfortunately, meaningful forecast data is missing in some instances because if one forecast hour is set to a constant, the entire forecast zarr will be missing. As a result, you must go to the GRIB2 files to get the data for the hours where it does exist. **Solar radiation is always affected by this bug at night and thus all forecasts for these variables are missing in the Zarr archive.**"*

In contrast, **Analysis runs (`_anl.zarr`)** do contain valid DSWRF chunks during daytime (e.g. `921.0 W/m²` on 2022-06-13 18:00 UTC), because analysis files represent a single timestamp ($F00$) rather than a sequence crossing nighttime. However, analysis fields represent retrospective observations, not advance forecasts.

---

## 6. Physical Plausibility & Correlation Analysis

Using the sample week (`2022-06-13` to `2022-06-19`, 168 hours total, 104 daylight hours, **0 flagged rows, 100% VALID** in `data/processed/hourly.parquet`), we evaluated the physical alignment of measured AC energy against weather variables [**Verified**]:

![HRRR and PV Correlation Plots](file:///c:/projects/SolarCast/docs/hrrr_pv_correlation.png)

1. **Measured PV vs pvlib Clear-Sky GHI:**  
   $$r = 0.945 \quad (N = 104)$$  
   Clear-sky GHI demonstrates an extraordinary linear relationship with System 2107 AC production during sunny June days, perfectly reproducing sunrise, solar noon peak (~545 kWh), and sunset decline.
2. **Measured PV vs HRRR Analysis DSWRF (F00):**  
   $$r = 0.927 \quad (N = 104)$$  
   Confirms that when numerical downward shortwave radiation is available, HRRR radiation models plant output with high physical fidelity.
3. **Measured PV vs HRRR Forecast 2m Temperature:**  
   $$r = 0.382 \quad (N = 104)$$  
   Near-surface temperature correlates moderately with production due to shared diurnal heating, but temperature alone cannot substitute for solar radiation.

---

## 7. Storage, Ingestion Cost & Runtime

| Extraction Pipeline | Bytes per daylight issue hour | Request Count / Hour | Est. 1-Year Transfer (4,500 daylight h) | Est. 7-Year Transfer (2017–2023) | External Dependencies |
|---|---|---|---|---|---|
| **HRRR Zarr (TMP only)** | ~371 KB | 1 GET | ~1.67 GB | ~11.7 GB (2018–2023) | pure Python (`zarr`, `s3fs`) |
| **HRRR Zarr (if DSWRF worked)** | ~742 KB | 2 GETs | ~3.34 GB | ~23.4 GB | pure Python (`zarr`, `s3fs`) |
| **HRRR GRIB2 Byte-Range** | ~6.0 MB (4 byte ranges) | 6 HTTP GETs | ~27.0 GB | ~189.0 GB | `eccodes` / `wgrib2` C binaries |

---

## 8. Sample Forecast-Release Alignment Preview

A 48-row daylight preview table has been generated and saved to [`docs/data_feasibility_preview.csv`](file:///c:/projects/SolarCast/docs/data_feasibility_preview.csv). Below is an excerpt of the first 5 daylight hours on 2022-06-13 [**Verified**]:

| prediction_issue_time_utc | target_start_utc | weather_run_init_utc | weather_available_at_utc | forecast_irradiance_wm2 | forecast_temperature_c | production_feature_cutoff_utc | actual_energy_kwh | quality_flags |
|---|---|---|---|---|---|---|---|---|
| 2022-06-13T00:00:00Z | 2022-06-13T00:00:00Z | 2022-06-12T22:00:00Z | 2022-06-13T00:00:00Z | NaN | 25.60 | 2022-06-12T23:45:00Z | 330.24 | VALID;HRRR_DSWRF_MISSING_ZARR_BUG |
| 2022-06-13T01:00:00Z | 2022-06-13T01:00:00Z | 2022-06-12T23:00:00Z | 2022-06-13T01:00:00Z | NaN | 24.48 | 2022-06-13T00:45:00Z | 139.04 | VALID;HRRR_DSWRF_MISSING_ZARR_BUG |
| 2022-06-13T02:00:00Z | 2022-06-13T02:00:00Z | 2022-06-13T00:00:00Z | 2022-06-13T02:00:00Z | NaN | 23.10 | 2022-06-13T01:45:00Z | 23.04 | VALID;HRRR_DSWRF_MISSING_ZARR_BUG |
| 2022-06-13T13:00:00Z | 2022-06-13T13:00:00Z | 2022-06-13T11:00:00Z | 2022-06-13T13:00:00Z | NaN | 16.35 | 2022-06-13T12:45:00Z | 8.32 | VALID;HRRR_DSWRF_MISSING_ZARR_BUG |
| 2022-06-13T14:00:00Z | 2022-06-13T14:00:00Z | 2022-06-13T12:00:00Z | 2022-06-13T14:00:00Z | NaN | 17.60 | 2022-06-13T13:45:00Z | 23.20 | VALID;HRRR_DSWRF_MISSING_ZARR_BUG |

---

## 9. Conclusion & Next Steps for Team

1. **Feasibility Gate Conclusion:**  
   NOAA HRRR through the community Zarr mirror cannot serve as an advance solar irradiance source due to the missing forecast DSWRF chunks. However, HRRR provides valid, verified forecast near-surface temperature (2m TMP).
2. **Path to M1.3 (Features & Feast):**  
   Member 1 will construct the feature store incorporating:
   - Astronomical clear-sky irradiance (pvlib GHI, DNI, DHI, solar elevation, solar azimuth) evaluated at interval midpoint.
   - HRRR forecast 2m temperature with operational 2-hour latency.
   - Historical production lags ($t-1\text{h}$, $t-2\text{h}$, $t-24\text{h}$) and past-only rolling statistics.
   - NASA POWER historical observations retained strictly for retrospective benchmark analysis (Mode E1).
